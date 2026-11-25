//! Indirect execution's arguments, bounded before the GPU reads them. The
//! title's own compute passes write most of them; a pass we drop or
//! translate wrongly leaves whatever the memory held, and a draw of four
//! billion vertices or a dispatch past the host's limits runs until the
//! host resets the display driver. So active records are copied and clamped
//! on the GPU into a device-local scratch buffer, just before execution, by a
//! compute shader of ours (HLSL compiled by the system's compiler; a 5.1
//! shader with its root signature inside), leaving the title's argument and
//! count buffers untouched.
//!
//! When a count buffer is given, threads at or past `min(max, count)` return
//! immediately without charging the list's budget or reading uninitialized
//! tail slots. Each list has a budget (2^27 vertex instances or threads,
//! groups counted as 64 threads, mesh groups as 128): active records past it
//! run empty, saturating without 32-bit wraparound so a list of thousands of
//! garbage records cannot run for seconds either.
//!
//! shortcut: the bounds (2^24 vertices, 2^26 vertex instances, 65,535
//! groups per dimension and 2^20 per dispatch, one-dimensional mesh
//! dispatches, the budget) cut legitimate work beyond them; raise them if
//! the title needs more.

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::s;

use crate::host::{HostList, host, transition};

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=6, b0), SRV(t0), SRV(t1), UAV(u0), UAV(u1)"
cbuffer Record : register(b0) {
    uint kind;
    uint stride;
    uint records;
    uint args_offset;
    uint has_count;
    uint count_offset;
};
ByteAddressBuffer src_args : register(t0);
ByteAddressBuffer count_buf : register(t1);
RWByteAddressBuffer dst_args : register(u0);
// The list's work so far (vertex instances, threads): past the budget,
// records run empty.
RWByteAddressBuffer budget : register(u1);
static const uint BUDGET = 1u << 27;
// Groups of one dispatch; more run empty.
static const uint GROUPS = 1u << 20;
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint eff_count = records;
    if (has_count != 0) {
        eff_count = min(eff_count, count_buf.Load(count_offset));
    }
    if (id.x >= eff_count) return;
    uint src_at = args_offset + id.x * stride;
    uint dst_at = id.x * stride;
    uint cost;
    uint2 draw_counts = uint2(0, 0);
    uint3 groups = uint3(0, 0, 0);
    if (kind <= 1) {
        // Draws: vertex or index count, then instances.
        uint count = min(src_args.Load(src_at), 1u << 24);
        uint instances = min(src_args.Load(src_at + 4), (1u << 26) / max(count, 1u));
        cost = count * instances;
        draw_counts = uint2(count, instances);
    } else if (kind == 2) {
        uint3 g = min(src_args.Load3(src_at), uint3(65535u, 65535u, 65535u));
        // x * y fits 32 bits; the whole count may not.
        if (g.z != 0 && g.x * g.y > GROUPS / g.z) {
            g = uint3(0, 0, 0);
        }
        cost = g.x * g.y * g.z * 64u;
        groups = g;
    } else {
        // Mesh dispatches: the console's are one-dimensional.
        uint3 g = src_args.Load3(src_at);
        g = uint3(min(g.x, 65535u), min(g.y, 1u), min(g.z, 1u));
        cost = g.x * g.y * g.z * 128u;
        groups = g;
    }
    bool over = false;
    if (cost > 0) {
        uint before;
        budget.InterlockedOr(0, 0, before);
        [allow_uav_condition]
        while (before < BUDGET) {
            uint next = (BUDGET - before <= cost) ? BUDGET : (before + cost);
            uint prev;
            budget.InterlockedCompareExchange(0, before, next, prev);
            if (prev == before) break;
            before = prev;
        }
        over = before >= BUDGET || BUDGET - before < cost;
    }
    if (kind <= 1) {
        if (over) {
            draw_counts = uint2(0, 0);
        }
        dst_args.Store2(dst_at, draw_counts);
        for (uint off = 8; off < stride; off += 4) {
            dst_args.Store(dst_at + off, src_args.Load(src_at + off));
        }
    } else {
        if (over) {
            groups = uint3(0, 0, 0);
        }
        dst_args.Store3(dst_at, groups);
        for (uint off = 12; off < stride; off += 4) {
            dst_args.Store(dst_at + off, src_args.Load(src_at + off));
        }
    }
}
"#;

struct Clamp {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Clamp {}
// SAFETY: as above.
unsafe impl Sync for Clamp {}

fn clamp() -> Option<&'static Clamp> {
    static CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| match make() {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(error = %e, "no argument clamp: indirect execution dropped");
                None
            }
        })
        .as_ref()
}

fn make() -> windows::core::Result<Clamp> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    make_for(device)
}

fn make_for(device: &ID3D12Device) -> windows::core::Result<Clamp> {
    make_from(device, SOURCE)
}

fn make_from(device: &ID3D12Device, source: &str) -> windows::core::Result<Clamp> {
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            s!("clamp"),
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(
                    errors.GetBufferPointer().cast::<u8>(),
                    errors.GetBufferSize(),
                )
            };
            tracing::warn!("argument clamp: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob's own bytes; the root signature is inside the shader.
    unsafe {
        let bytes =
            std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        })?;
        Ok(Clamp { root, pso })
    }
}

/// Minimum record byte stride for a supported PC indirect argument `kind`.
fn min_stride(kind: u32) -> Option<u32> {
    match kind {
        0 => Some(16),
        1 => Some(20),
        2 | 10 => Some(12),
        _ => None,
    }
}

/// Validates record alignment and returns the scratch byte size `stride * records`.
fn validate_layout(
    (kind, stride, records): (u32, u32, u32),
    src_offset: u64,
    count_offset: Option<u64>,
) -> Option<u64> {
    let min = min_stride(kind)?;
    if stride < min || !stride.is_multiple_of(4) || !src_offset.is_multiple_of(4) {
        return None;
    }
    if count_offset.is_some_and(|off| !off.is_multiple_of(4)) {
        return None;
    }
    let bytes = u64::from(stride).checked_mul(u64::from(records))?;
    (bytes <= u64::from(u32::MAX)).then_some(bytes)
}

/// Records, in `list`, the non-destructive clamp of up to `records` records of
/// `kind` (PC argument numbering) `stride` bytes apart from `src` (and bounded
/// by the 32-bit count in `count` when given) into a bump-allocated slice of a
/// device-local scratch buffer, leaving that scratch buffer in
/// `INDIRECT_ARGUMENT` at `Bounded::offset`. Replaces the list's compute root
/// signature and pipeline (callers bind theirs after). `None` if the clamp or
/// alignment is not available.
///
/// The argument and count buffers are read in place (root SRVs, and the
/// count by `ExecuteIndirect`). One the list's shaders already wrote is in
/// `UNORDERED_ACCESS`, from which a buffer cannot be promoted, so it moves
/// to [`READ`] here and back in [`done`]; the others are promoted from
/// `COMMON`.
pub(crate) fn bound(
    list: &HostList,
    src: (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    spec: (u32, u32, u32),
) -> Option<Bounded> {
    let bytes = validate_layout(spec, src.1, count.map(|(_, off)| off))?;
    let (c, budget, (dst, offset)) = (clamp()?, list.budget()?, list.indirect_scratch(bytes)?);
    let written = written_among(src.0, count.map(|(b, _)| b), |b| list.gpu_written(b));
    record_clamp(
        c,
        &list.list,
        (&dst, offset),
        budget,
        src,
        count,
        spec,
        &written,
    )
    .then_some(Bounded {
        scratch: dst,
        offset,
        written,
    })
}

/// A clamp recorded by [`bound`]: the scratch buffer and byte offset
/// `ExecuteIndirect` reads the arguments from, and the title buffers moved out
/// of unordered access until [`done`].
pub(crate) struct Bounded {
    pub(crate) scratch: ID3D12Resource,
    pub(crate) offset: u64,
    written: Vec<ID3D12Resource>,
}

/// The state written argument and count buffers are read in: by the clamp
/// and, for the count, by `ExecuteIndirect`.
const READ: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATES(
    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE.0 | D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT.0,
);

/// The distinct buffers among `src` and `count` that `written` (shaders of
/// the list wrote them through unordered access) names; once each, as the
/// count may be in the argument buffer.
fn written_among(
    src: &ID3D12Resource,
    count: Option<&ID3D12Resource>,
    written: impl Fn(&ID3D12Resource) -> bool,
) -> Vec<ID3D12Resource> {
    use windows::core::Interface;
    let mut out: Vec<ID3D12Resource> = Vec::new();
    for b in std::iter::once(src).chain(count) {
        if !out.iter().any(|o| o.as_raw() == b.as_raw()) && written(b) {
            out.push(b.clone());
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn record_clamp(
    c: &Clamp,
    l: &ID3D12GraphicsCommandList,
    (dst, dst_offset): (&ID3D12Resource, u64),
    budget: u64,
    (src, src_offset): (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    (kind, stride, records): (u32, u32, u32),
    written: &[ID3D12Resource],
) -> bool {
    let Some(bytes) = validate_layout(
        (kind, stride, records),
        src_offset,
        count.map(|(_, off)| off),
    ) else {
        return false;
    };
    if !dst_offset.is_multiple_of(4) {
        return false;
    }
    // SAFETY: live D3D12 buffer resources.
    let (src_base, dst_base) = unsafe { (src.GetGPUVirtualAddress(), dst.GetGPUVirtualAddress()) };
    let Some(dst_gpu) = dst_base.checked_add(dst_offset) else {
        return false;
    };
    if !src_base.is_multiple_of(4) || !dst_gpu.is_multiple_of(4) || !budget.is_multiple_of(4) {
        return false;
    }
    let (src_gpu, args_offset) = match src_offset
        .checked_add(bytes)
        .filter(|&end| end <= u64::from(u32::MAX))
    {
        Some(_) => (src_base, src_offset as u32),
        None => {
            let Some(rebased) = src_base.checked_add(src_offset) else {
                return false;
            };
            (rebased, 0)
        }
    };
    let (count_gpu, has_count, count_offset) = match count {
        Some((buf, off)) => {
            // SAFETY: a live D3D12 buffer resource.
            let base = unsafe { buf.GetGPUVirtualAddress() };
            if !base.is_multiple_of(4) {
                return false;
            }
            match off.checked_add(4).filter(|&end| end <= u64::from(u32::MAX)) {
                Some(_) => (base, 1u32, off as u32),
                None => {
                    let Some(rebased) = base.checked_add(off) else {
                        return false;
                    };
                    (rebased, 1u32, 0u32)
                }
            }
        }
        None => (src_gpu, 0u32, 0u32),
    };
    let constants = [kind, stride, records, args_offset, has_count, count_offset];
    // SAFETY: an open list; our root signature and pipeline; live source,
    // count, scratch destination, and budget buffers. `src` and `count` are
    // promoted from `COMMON` unless `written` names them (in
    // `UNORDERED_ACCESS`, moved to `READ`), while `dst` transitions
    // `COMMON -> UNORDERED_ACCESS -> INDIRECT_ARGUMENT`.
    unsafe {
        // Earlier shader writes (the title's argument and count passes) first.
        let mut before = vec![
            uav_barrier(),
            transition(
                dst,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            ),
        ];
        before.extend(
            written
                .iter()
                .map(|b| transition(b, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, READ)),
        );
        l.ResourceBarrier(&before);
        if records > 0 {
            l.SetComputeRootSignature(&c.root);
            l.SetPipelineState(&c.pso);
            l.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
            l.SetComputeRootShaderResourceView(1, src_gpu);
            l.SetComputeRootShaderResourceView(2, count_gpu);
            l.SetComputeRootUnorderedAccessView(3, dst_gpu);
            l.SetComputeRootUnorderedAccessView(4, budget);
            // Experiment: VICTORY_CLAMP_GROUP is the clamp file's threads per group.
            let group = std::env::var("VICTORY_CLAMP_GROUP")
                .ok()
                .and_then(|s| s.parse::<u32>().ok())
                .unwrap_or(64);
            l.Dispatch(records.div_ceil(group), 1, 1);
        }
        l.ResourceBarrier(&[
            uav_barrier(),
            transition(
                dst,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            ),
        ]);
    }
    true
}

/// After indirect execution (or if preparation failed after [`bound`]):
/// returns the scratch buffer to the common state for the next call, and the
/// title buffers [`bound`] moved out of unordered access back to it (where
/// the list's later shaders expect them).
pub(crate) fn done(list: &HostList, bounded: &Bounded) {
    done_on(&list.list, &bounded.scratch, &bounded.written);
}

fn done_on(list: &ID3D12GraphicsCommandList, scratch: &ID3D12Resource, written: &[ID3D12Resource]) {
    let mut after = vec![transition(
        scratch,
        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
        D3D12_RESOURCE_STATE_COMMON,
    )];
    after.extend(
        written
            .iter()
            .map(|b| transition(b, READ, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)),
    );
    // SAFETY: an open list, a live scratch buffer in INDIRECT_ARGUMENT, and
    // live written buffers in READ (moved there by `record_clamp`).
    unsafe { list.ResourceBarrier(&after) };
}

fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: std::mem::ManuallyDrop::new(None),
            }),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Foundation::HANDLE;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
    use windows::Win32::Graphics::Dxgi::*;
    use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
    use windows::core::Interface;

    const WORKER_CS: &str = r#"
#define RS "UAV(u0)"
RWByteAddressBuffer hits : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    uint prev;
    hits.InterlockedAdd(0, 1, prev);
}
"#;

    struct WarpHarness {
        device: ID3D12Device,
        queue: ID3D12CommandQueue,
        allocator: ID3D12CommandAllocator,
        list: ID3D12GraphicsCommandList,
        fence: ID3D12Fence,
        submitted: u64,
        event: HANDLE,
        info_queue: Option<ID3D12InfoQueue>,
        clamp: Clamp,
        worker: Clamp,
    }

    impl WarpHarness {
        fn new() -> Option<Self> {
            // SAFETY: plain DXGI WARP query and D3D12 test harness setup.
            unsafe {
                let debug_on = std::env::var_os("VICTORY_D3D12_DEBUG").is_some();
                let mut debug_enabled = false;
                if debug_on {
                    let mut layer: Option<ID3D12Debug> = None;
                    if D3D12GetDebugInterface(&mut layer).is_ok()
                        && let Some(layer) = &layer
                    {
                        layer.EnableDebugLayer();
                        debug_enabled = true;
                        if std::env::var_os("VICTORY_D3D12_GBV").is_some()
                            && let Ok(d1) = layer.cast::<ID3D12Debug1>()
                        {
                            d1.SetEnableGPUBasedValidation(true);
                        }
                    }
                }
                let factory: IDXGIFactory4 =
                    CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
                // Experiment: VICTORY_TEST_HW picks the first hardware adapter.
                let hardware = std::env::var_os("VICTORY_TEST_HW").is_some();
                let adapter: IDXGIAdapter = if hardware {
                    let mut found = None;
                    let mut i = 0;
                    while let Ok(a) = factory.EnumAdapters1(i) {
                        let d = a.GetDesc1().ok()?;
                        if d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0 {
                            let name = String::from_utf16_lossy(&d.Description);
                            eprintln!("adapter: {}", name.trim_end_matches('\0'));
                            found = Some(a.cast::<IDXGIAdapter>().ok()?);
                            break;
                        }
                        i += 1;
                    }
                    found?
                } else {
                    factory.EnumWarpAdapter().ok()?
                };
                let desc = adapter.cast::<IDXGIAdapter1>().ok()?.GetDesc1().ok()?;
                if !hardware && desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0 {
                    return None;
                }
                let mut device: Option<ID3D12Device> = None;
                D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).ok()?;
                let device = device?;
                let info_queue = debug_enabled
                    .then(|| device.cast::<ID3D12InfoQueue>().ok())
                    .flatten();
                if debug_enabled {
                    assert!(
                        info_queue.is_some(),
                        "ID3D12InfoQueue must be available when the D3D12 debug layer is enabled"
                    );
                }
                if let Some(q) = &info_queue {
                    q.ClearStoredMessages();
                }
                let queue: ID3D12CommandQueue = device
                    .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                        Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                        ..Default::default()
                    })
                    .ok()?;
                let allocator: ID3D12CommandAllocator = device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                    .ok()?;
                let list: ID3D12GraphicsCommandList = device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                    .ok()?;
                let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
                let event = CreateEventW(None, false, false, None).ok()?;
                // Experiment: VICTORY_CLAMP_HLSL replaces the clamp's source.
                let clamp = match std::env::var_os("VICTORY_CLAMP_HLSL") {
                    Some(path) => {
                        let source = std::fs::read_to_string(&path).expect("the clamp file reads");
                        make_from(&device, &source).expect("the clamp file compiles")
                    }
                    None => make_for(&device).expect("clamp shader must compile on WARP"),
                };
                let worker = compile_cs(&device, WORKER_CS).expect("worker shader must compile");
                Some(Self {
                    device,
                    queue,
                    allocator,
                    list,
                    fence,
                    submitted: 0,
                    event,
                    info_queue,
                    clamp,
                    worker,
                })
            }
        }

        /// CPU-mapped L0 write-back buffer matching `Shadow`'s heap type.
        fn shared_buffer(&self, size: u64) -> (ID3D12Resource, *mut u8, u64) {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_CUSTOM,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
                MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: size.max(4),
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            };
            let mut buffer: Option<ID3D12Resource> = None;
            let mut data = std::ptr::null_mut();
            // SAFETY: plain buffer creation and persistent CPU mapping.
            unsafe {
                self.device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        D3D12_RESOURCE_STATE_COMMON,
                        None,
                        &mut buffer,
                    )
                    .unwrap();
                let buffer = buffer.unwrap();
                buffer.Map(0, None, Some(&mut data)).unwrap();
                let gpu = buffer.GetGPUVirtualAddress();
                (buffer, data.cast(), gpu)
            }
        }

        /// Device-local `DEFAULT`-heap UAV scratch buffer.
        fn default_uav(&self, size: u64) -> ID3D12Resource {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: size.max(64 << 10),
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            };
            let mut buffer: Option<ID3D12Resource> = None;
            // SAFETY: plain device-local buffer creation.
            unsafe {
                self.device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        D3D12_RESOURCE_STATE_COMMON,
                        None,
                        &mut buffer,
                    )
                    .unwrap();
            }
            buffer.unwrap()
        }

        fn signature(&self, kind: u32, stride: u32) -> ID3D12CommandSignature {
            let arg = D3D12_INDIRECT_ARGUMENT_DESC {
                Type: D3D12_INDIRECT_ARGUMENT_TYPE(kind as i32),
                ..Default::default()
            };
            let desc = D3D12_COMMAND_SIGNATURE_DESC {
                ByteStride: stride,
                NumArgumentDescs: 1,
                pArgumentDescs: &arg,
                NodeMask: 0,
            };
            let mut sig: Option<ID3D12CommandSignature> = None;
            // SAFETY: valid single-argument command signature description.
            unsafe {
                self.device
                    .CreateCommandSignature(&desc, None::<&ID3D12RootSignature>, &mut sig)
                    .unwrap();
            }
            sig.unwrap()
        }

        fn drain_debug_errors(&self) -> Vec<String> {
            let Some(q) = &self.info_queue else {
                return Vec::new();
            };
            let mut errors = Vec::new();
            // SAFETY: querying messages stored in our `ID3D12InfoQueue`.
            unsafe {
                let count = q.GetNumStoredMessages();
                for i in 0..count {
                    let mut len = 0usize;
                    if q.GetMessage(i, None, &mut len).is_err() || len == 0 {
                        continue;
                    }
                    let mut buf = vec![0u8; len];
                    let msg = buf.as_mut_ptr().cast::<D3D12_MESSAGE>();
                    if q.GetMessage(i, Some(msg), &mut len).is_err() {
                        continue;
                    }
                    let m = &*msg;
                    if matches!(
                        m.Severity,
                        D3D12_MESSAGE_SEVERITY_CORRUPTION | D3D12_MESSAGE_SEVERITY_ERROR
                    ) {
                        let text = std::slice::from_raw_parts(
                            m.pDescription.cast::<u8>(),
                            m.DescriptionByteLength.saturating_sub(1),
                        );
                        errors.push(String::from_utf8_lossy(text).into_owned());
                    }
                }
                q.ClearStoredMessages();
            }
            errors
        }

        fn submit_and_wait(&mut self) {
            self.submitted += 1;
            // SAFETY: closing, executing, waiting on fence, and resetting our list.
            unsafe {
                self.list.Close().unwrap();
                let cmd: ID3D12CommandList = self.list.cast().unwrap();
                self.queue.ExecuteCommandLists(&[Some(cmd)]);
                self.queue.Signal(&self.fence, self.submitted).unwrap();
                self.fence
                    .SetEventOnCompletion(self.submitted, self.event)
                    .unwrap();
                WaitForSingleObject(self.event, INFINITE);
                self.allocator.Reset().unwrap();
                self.list.Reset(&self.allocator, None).unwrap();
            }
            let errors = self.drain_debug_errors();
            assert!(
                errors.is_empty(),
                "D3D12 debug/GBV errors:\n{}",
                errors.join("\n")
            );
        }
    }

    fn compile_cs(device: &ID3D12Device, source: &str) -> windows::core::Result<Clamp> {
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: source and entry names outlive `D3DCompile`.
        unsafe {
            D3DCompile(
                source.as_ptr().cast(),
                source.len(),
                s!("worker"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut code,
                None,
            )?;
            let code = code.unwrap();
            let bytes = std::slice::from_raw_parts(
                code.GetBufferPointer().cast::<u8>(),
                code.GetBufferSize(),
            );
            let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
            let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: bytes.as_ptr().cast(),
                    BytecodeLength: bytes.len(),
                },
                ..Default::default()
            })?;
            Ok(Clamp { root, pso })
        }
    }

    /// The clamp compiles with the system's compiler and the host takes it.
    #[test]
    fn argument_clamp_builds() {
        if super::host().is_none() {
            return; // no host GPU here
        }
        assert!(super::clamp().is_some());
    }

    #[test]
    fn rejects_unaligned_or_undersized_layouts() {
        assert_eq!(validate_layout((0, 16, 4), 0, None), Some(64));
        assert_eq!(validate_layout((1, 20, 4), 64, Some(128)), Some(80));
        assert_eq!(validate_layout((2, 12, 0), 0, Some(0)), Some(0));
        assert_eq!(validate_layout((10, 16, 2), 8, None), Some(32));
        // Undersized strides for each kind.
        assert_eq!(validate_layout((0, 12, 1), 0, None), None);
        assert_eq!(validate_layout((1, 16, 1), 0, None), None);
        assert_eq!(validate_layout((2, 8, 1), 0, None), None);
        assert_eq!(validate_layout((10, 8, 1), 0, None), None);
        // Unknown kind.
        assert_eq!(validate_layout((3, 16, 1), 0, None), None);
        // Unaligned stride, source offset, or count offset.
        assert_eq!(validate_layout((0, 18, 1), 0, None), None);
        assert_eq!(validate_layout((0, 16, 1), 2, None), None);
        assert_eq!(validate_layout((0, 16, 1), 0, Some(6)), None);
        // Total byte size exceeding 32-bit ByteAddressBuffer indexing.
        assert_eq!(validate_layout((1, 20, u32::MAX), 0, None), None);
    }

    /// WARP D3D12 execution test proving:
    /// (a) `countBuffer` with a small count (`3`) and `4,093` garbage tail slots
    ///     (`max = 4096`) does NOT blow `BUDGET` or zero valid dispatches in
    ///     this or subsequent `ExecuteIndirect` calls in the same list,
    /// (b) `src_args` (and `count_buf`) are NOT mutated in-place, and
    /// (c) full `stride` bytes (`stride = 24` > `12`) are copied for active
    ///     records into `dst_args` at offset `0`.
    #[test]
    fn warp_count_buffer_ignores_garbage_tail_and_preserves_source() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const KIND_DISPATCH: u32 = 2;
        const STRIDE: u32 = 24; // 6 dwords: 3 dispatch group counts + 3 extra payload dwords
        const MAX: u32 = 4096;
        const ARGS_OFFSET: usize = 64; // non-zero offset with sentinel prefix
        const COUNT_OFFSET_1: usize = 128;
        const COUNT_OFFSET_2: usize = 192;

        let src_bytes = ARGS_OFFSET + (STRIDE as usize) * (MAX as usize) + 64;
        let (src_buf, src_ptr, _) = warp.shared_buffer(src_bytes as u64);
        let (count_buf, count_ptr, _) = warp.shared_buffer(512);
        let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (_hits_buf, hits_ptr, hits_gpu) = warp.shared_buffer(256);
        let (readback_buf, readback_ptr, _) = warp.shared_buffer((STRIDE * 8) as u64);
        let dst_buf = warp.default_uav(u64::from(STRIDE) * u64::from(MAX) * 2);
        let sig = warp.signature(KIND_DISPATCH, STRIDE);

        // Fill the entire source buffer with non-zero sentinel bytes, then
        // populate 3 valid records at `ARGS_OFFSET` and 4,093 huge garbage
        // records in `3..MAX` that would exhaust `BUDGET` ~2,000x over if read.
        let mut initial_src = vec![0xA5u8; src_bytes];
        let write_record = |buf: &mut [u8], idx: usize, words: [u32; 6]| {
            let base = ARGS_OFFSET + idx * (STRIDE as usize);
            for (w, val) in words.into_iter().enumerate() {
                buf[base + w * 4..base + (w + 1) * 4].copy_from_slice(&val.to_le_bytes());
            }
        };
        write_record(
            &mut initial_src,
            0,
            [3, 2, 1, 0x1111_2222, 0x3333_4444, 0x5555_6666],
        );
        write_record(
            &mut initial_src,
            1,
            [5, 1, 1, 0x7777_8888, 0x9999_AAAA, 0xBBBB_CCCC],
        );
        // Record 2 exceeds 65,535 in X -> clamped to (65535, 1, 1).
        write_record(
            &mut initial_src,
            2,
            [100_000, 1, 1, 0xDDDD_EEEE, 0xFFFF_0001, 0x0123_4567],
        );
        for slot in 3..MAX as usize {
            // Each garbage slot would clamp to 65535 * 16 * 1 = 1,048,560 groups
            // (`cost = 67,107,840 ~= BUDGET / 2`), exhausting `BUDGET` immediately!
            write_record(
                &mut initial_src,
                slot,
                [0xFFFF_FFFF, 16, 1, 0xDEAD_BEEF, 0xBAD0_C0DE, slot as u32],
            );
        }

        // Second indirect call's source records (at offset 0 of a second buffer,
        // or later in `count_buf`): put 2 active records + garbage tail in a
        // second range to run back-to-back in the SAME command list.
        let mut initial_count = vec![0x5Au8; 512];
        initial_count[COUNT_OFFSET_1..COUNT_OFFSET_1 + 4].copy_from_slice(&3u32.to_le_bytes());
        initial_count[COUNT_OFFSET_2..COUNT_OFFSET_2 + 4].copy_from_slice(&2u32.to_le_bytes());

        // SAFETY: buffers are mapped with exact capacities above.
        unsafe {
            std::ptr::copy_nonoverlapping(initial_src.as_ptr(), src_ptr, src_bytes);
            std::ptr::copy_nonoverlapping(initial_count.as_ptr(), count_ptr, 512);
            (budget_ptr as *mut u32).write_volatile(0);
            (hits_ptr as *mut u32).write_volatile(0);
            std::ptr::write_bytes(readback_ptr, 0, (STRIDE * 8) as usize);
        }

        // Record Call #1 (`eff_count = 3` out of `MAX = 4096`) at offset 0,
        // copy the first 5 records of `dst_buf` out for inspection, then record
        // Call #2 (`eff_count = 2` out of `MAX = 4096`) at a non-zero bump
        // offset in the SAME command list!
        let call2_offset = (u64::from(STRIDE) * u64::from(MAX)).next_multiple_of(16);
        assert!(record_clamp(
            &warp.clamp,
            &warp.list,
            (&dst_buf, 0),
            budget_gpu,
            (&src_buf, ARGS_OFFSET as u64),
            Some((&count_buf, COUNT_OFFSET_1 as u64)),
            (KIND_DISPATCH, STRIDE, MAX),
            &[],
        ));
        // SAFETY: open command list, valid worker PSO/root and buffers.
        unsafe {
            warp.list.SetComputeRootSignature(&warp.worker.root);
            warp.list.SetPipelineState(&warp.worker.pso);
            warp.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
            warp.list.ExecuteIndirect(
                &sig,
                MAX,
                &dst_buf,
                0,
                Some(&count_buf),
                COUNT_OFFSET_1 as u64,
            );
        }
        done_on(&warp.list, &dst_buf, &[]);

        // Snapshot `dst_buf` after Call #1 (from `COMMON`, copy 5 * STRIDE bytes,
        // then return `dst_buf` to `COMMON` before Call #2).
        // SAFETY: open command list; `dst_buf` returned to `COMMON` by `done_on`.
        unsafe {
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            warp.list
                .CopyBufferRegion(&readback_buf, 0, &dst_buf, 0, u64::from(STRIDE * 5));
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_COMMON,
            )]);
        }

        // Call #2 in the SAME command list at `call2_offset` (`eff_count = 2` ->
        // records 0 and 1: 6 + 5 = 11 groups). Would run empty if Call #1's tail
        // blew `BUDGET`!
        assert!(record_clamp(
            &warp.clamp,
            &warp.list,
            (&dst_buf, call2_offset),
            budget_gpu,
            (&src_buf, ARGS_OFFSET as u64),
            Some((&count_buf, COUNT_OFFSET_2 as u64)),
            (KIND_DISPATCH, STRIDE, MAX),
            &[],
        ));
        // SAFETY: open command list, valid worker PSO/root and buffers.
        unsafe {
            warp.list.SetComputeRootSignature(&warp.worker.root);
            warp.list.SetPipelineState(&warp.worker.pso);
            warp.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
            warp.list.ExecuteIndirect(
                &sig,
                MAX,
                &dst_buf,
                call2_offset,
                Some(&count_buf),
                COUNT_OFFSET_2 as u64,
            );
        }
        done_on(&warp.list, &dst_buf, &[]);

        warp.submit_and_wait();

        // SAFETY: GPU execution finished; read mapped host results.
        let (hits, spent, after_src, after_count, clamped_words) = unsafe {
            let hits = (hits_ptr as *const u32).read_volatile();
            let spent = (budget_ptr as *const u32).read_volatile();
            let after_src = std::slice::from_raw_parts(src_ptr.cast_const(), src_bytes).to_vec();
            let after_count = std::slice::from_raw_parts(count_ptr.cast_const(), 512).to_vec();
            let words = std::slice::from_raw_parts(
                readback_ptr.cast::<u32>().cast_const(),
                (STRIDE as usize / 4) * 5,
            )
            .to_vec();
            (hits, spent, after_src, after_count, words)
        };

        // Call #1: (3*2*1) + (5*1*1) + (65535*1*1) = 65,546 groups.
        // Call #2: (3*2*1) + (5*1*1) = 11 groups.
        let expected_groups = (6 + 5 + 65_535) + (6 + 5);
        assert_eq!(
            hits, expected_groups,
            "valid indirect dispatches must all execute"
        );
        assert_eq!(
            spent,
            expected_groups * 64,
            "budget must reflect only active records below countBuffer"
        );
        assert_eq!(
            after_src, initial_src,
            "src_args must never be mutated in-place"
        );
        assert_eq!(
            after_count, initial_count,
            "count_buf must never be mutated"
        );

        // Check `dst_args` records 0..3: clamped group counts + preserved extra stride words.
        assert_eq!(
            &clamped_words[0..6],
            &[3, 2, 1, 0x1111_2222, 0x3333_4444, 0x5555_6666]
        );
        assert_eq!(
            &clamped_words[6..12],
            &[5, 1, 1, 0x7777_8888, 0x9999_AAAA, 0xBBBB_CCCC]
        );
        assert_eq!(
            &clamped_words[12..18],
            &[65_535, 1, 1, 0xDDDD_EEEE, 0xFFFF_0001, 0x0123_4567]
        );
        // Inactive slot 3 was never copied from `src_args`.
        assert_ne!(clamped_words[21], 0xDEAD_BEEF);
    }

    /// An argument buffer a compute pass of the same list wrote through
    /// unordered access (holding its own count, so named once) is moved to
    /// `READ` for the clamp and the count read, sees that pass's write, and
    /// is back in `UNORDERED_ACCESS` for the list's next writer.
    #[test]
    fn warp_reads_arguments_written_earlier_in_the_list() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const COUNT_OFFSET: usize = 64;
        let (src_buf, src_ptr, src_gpu) = warp.shared_buffer(256);
        let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (_hits_buf, hits_ptr, hits_gpu) = warp.shared_buffer(256);
        let dst_buf = warp.default_uav(12 * 4);
        let sig = warp.signature(2, 12);
        // SAFETY: mapped buffers of the sizes above.
        unsafe {
            let words = src_ptr.cast::<u32>();
            for (i, w) in [2u32, 1, 1].into_iter().enumerate() {
                words.add(i).write_volatile(w);
            }
            words.add(COUNT_OFFSET / 4).write_volatile(1);
            (budget_ptr as *mut u32).write_volatile(0);
            (hits_ptr as *mut u32).write_volatile(0);
        }
        let written = written_among(&src_buf, Some(&src_buf), |_| true);
        assert_eq!(written.len(), 1, "arguments and count share one buffer");
        let produce = |warp: &WarpHarness| {
            // SAFETY: an open list; the worker adds one to the first word.
            unsafe {
                warp.list.SetComputeRootSignature(&warp.worker.root);
                warp.list.SetPipelineState(&warp.worker.pso);
                warp.list.SetComputeRootUnorderedAccessView(0, src_gpu);
                warp.list.Dispatch(1, 1, 1);
            }
        };
        // The producer promotes the buffer to UNORDERED_ACCESS: X becomes 3.
        produce(&warp);
        assert!(record_clamp(
            &warp.clamp,
            &warp.list,
            (&dst_buf, 0),
            budget_gpu,
            (&src_buf, 0),
            Some((&src_buf, COUNT_OFFSET as u64)),
            (2, 12, 4),
            &written,
        ));
        // SAFETY: an open list; worker pipeline and live buffers.
        unsafe {
            warp.list.SetComputeRootSignature(&warp.worker.root);
            warp.list.SetPipelineState(&warp.worker.pso);
            warp.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
            warp.list
                .ExecuteIndirect(&sig, 4, &dst_buf, 0, Some(&src_buf), COUNT_OFFSET as u64);
        }
        done_on(&warp.list, &dst_buf, &written);
        // Back in UNORDERED_ACCESS: a later writer of the list: X becomes 4.
        produce(&warp);
        warp.submit_and_wait();
        // SAFETY: the GPU finished; mapped results.
        let (hits, x) = unsafe {
            (
                (hits_ptr as *const u32).read_volatile(),
                (src_ptr as *const u32).read_volatile(),
            )
        };
        assert_eq!(hits, 3, "the clamp must read the producer's write");
        assert_eq!(x, 4, "the list's later writer must still run");
    }

    /// WARP D3D12 test proving `BUDGET` saturates without 32-bit unsigned
    /// wraparound across >64 huge indexed-draw records (`count = None`), while
    /// preserving `StartIndexLocation`, `BaseVertexLocation`,
    /// `StartInstanceLocation`, and custom stride padding, and leaving
    /// `src_args` untouched.
    #[test]
    fn warp_budget_saturates_without_wraparound_and_preserves_draw_fields() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const KIND_INDEXED: u32 = 1;
        const STRIDE: u32 = 24; // 5 indexed-draw words + 1 trailing custom word
        const RECORDS: u32 = 128; // 2 waves of 64: old InterlockedAdd wrapped 2^32 at 64!
        const BUDGET_CAP: u32 = 1 << 27;

        let total_bytes = (STRIDE * RECORDS) as usize;
        let (src_buf, src_ptr, _) = warp.shared_buffer(total_bytes as u64);
        let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (readback_buf, readback_ptr, _) = warp.shared_buffer(total_bytes as u64);
        let dst_buf = warp.default_uav(total_bytes as u64);

        let mut initial_src = vec![0u8; total_bytes];
        for i in 0..RECORDS as usize {
            let words: [u32; 6] = [
                u32::MAX,                    // IndexCount -> clamps to 1 << 24
                u32::MAX,                    // InstanceCount -> clamps to 4 (cost = 1 << 26)
                1000 + i as u32,             // StartIndexLocation
                (-200i32 + i as i32) as u32, // BaseVertexLocation
                5000 + i as u32,             // StartInstanceLocation
                0xCAFE_0000 | (i as u32),    // Extra stride word
            ];
            let base = i * (STRIDE as usize);
            for (w, val) in words.into_iter().enumerate() {
                initial_src[base + w * 4..base + (w + 1) * 4].copy_from_slice(&val.to_le_bytes());
            }
        }

        // SAFETY: mapped buffers have sufficient capacity.
        unsafe {
            std::ptr::copy_nonoverlapping(initial_src.as_ptr(), src_ptr, total_bytes);
            (budget_ptr as *mut u32).write_volatile(0);
            std::ptr::write_bytes(readback_ptr, 0, total_bytes);
        }

        assert!(record_clamp(
            &warp.clamp,
            &warp.list,
            (&dst_buf, 0),
            budget_gpu,
            (&src_buf, 0),
            None,
            (KIND_INDEXED, STRIDE, RECORDS),
            &[],
        ));
        done_on(&warp.list, &dst_buf, &[]);

        // SAFETY: open command list; copy clamped scratch buffer for verification.
        unsafe {
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            warp.list
                .CopyBufferRegion(&readback_buf, 0, &dst_buf, 0, total_bytes as u64);
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_COMMON,
            )]);
        }
        warp.submit_and_wait();

        // SAFETY: GPU work completed.
        let (spent, after_src, out_words) = unsafe {
            let spent = (budget_ptr as *const u32).read_volatile();
            let after_src = std::slice::from_raw_parts(src_ptr.cast_const(), total_bytes).to_vec();
            let out_words = std::slice::from_raw_parts(
                readback_ptr.cast::<u32>().cast_const(),
                total_bytes / 4,
            )
            .to_vec();
            (spent, after_src, out_words)
        };

        assert_eq!(
            spent, BUDGET_CAP,
            "budget must saturate at BUDGET without wrapping"
        );
        assert_eq!(after_src, initial_src, "src_args must remain untouched");

        let mut kept = 0usize;
        let mut zeroed = 0usize;
        for i in 0..RECORDS as usize {
            let rec = &out_words[i * 6..(i + 1) * 6];
            if rec[0] == (1 << 24) && rec[1] == 4 {
                kept += 1;
            } else {
                assert_eq!((rec[0], rec[1]), (0, 0), "excess draw must be zeroed");
                zeroed += 1;
            }
            assert_eq!(rec[2], 1000 + i as u32);
            assert_eq!(rec[3], (-200i32 + i as i32) as u32);
            assert_eq!(rec[4], 5000 + i as u32);
            assert_eq!(rec[5], 0xCAFE_0000 | (i as u32));
        }
        assert_eq!((kept, zeroed), (2, (RECORDS as usize) - 2));
    }

    /// Exercises `HostList::indirect_scratch`, `bound`, `done`, `Host::execute`,
    /// and `HostList::reset` end-to-end on the host device.
    #[test]
    fn host_list_reuses_and_grows_scratch_across_submissions() {
        let Some(host) = super::host() else {
            return;
        };
        let Ok(list) = host.new_list() else {
            return;
        };
        let Some((src_buf, src_ptr)) = host.shared_buffer(256 << 10) else {
            return;
        };
        let Some((count_buf, count_ptr)) = host.shared_buffer(256) else {
            return;
        };
        // SAFETY: mapped host buffers.
        unsafe {
            std::ptr::write_bytes(src_ptr, 0x7F, 256 << 10);
            (count_ptr as *mut u32).write_volatile(2);
        }

        // First two calls fit inside the initial 64 KiB scratch buffer and
        // bump-allocate non-overlapping offsets in the same `ID3D12Resource`.
        let b1 = bound(&list, (&src_buf, 0), Some((&count_buf, 0)), (0, 16, 64)).unwrap();
        done(&list, &b1);
        let b2 = bound(&list, (&src_buf, 64), Some((&count_buf, 0)), (1, 20, 128)).unwrap();
        done(&list, &b2);
        assert_eq!(
            b1.scratch.as_raw(),
            b2.scratch.as_raw(),
            "scratch buffer within capacity must be reused"
        );
        assert_eq!(b1.offset, 0);
        assert_eq!(b2.offset, 16 * 64, "second call bump-allocates past first");

        // A call needing >64 KiB grows the scratch buffer mid-list while keeping
        // `b1` alive until reset.
        let b3 = bound(&list, (&src_buf, 0), Some((&count_buf, 0)), (1, 20, 5000)).unwrap();
        done(&list, &b3);
        assert_ne!(
            b3.scratch.as_raw(),
            b1.scratch.as_raw(),
            "larger request must grow scratch buffer"
        );
        assert_eq!(b3.offset, 0);

        list.close();
        host.execute(&[&list]);
        list.reset();

        // After reset, the bump offset resets to 0 and requests up to the grown
        // high-water mark reuse `b3` with zero new allocations.
        let b4 = bound(&list, (&src_buf, 0), Some((&count_buf, 0)), (1, 20, 4096)).unwrap();
        done(&list, &b4);
        assert_eq!(
            b4.scratch.as_raw(),
            b3.scratch.as_raw(),
            "grown scratch buffer must persist across reset"
        );
        assert_eq!(b4.offset, 0, "bump offset must reset to 0 after list reset");
        list.close();
        host.execute(&[&list]);
        list.reset();
    }

    // ---------------------------------------------------------------
    // Experiment (not for upstream): clamp variants from a file, on WARP
    // or on the hardware adapter, checked against the budget policy and
    // timed.
    // ---------------------------------------------------------------

    const SENTINEL: u32 = 0xDEAD_BEEF;
    const CAP: u32 = 1 << 27;

    /// One clamp of draw records (`kind` 0) `stride` bytes apart, from
    /// budget `start`: the seconds its submission took, the scratch words
    /// after, the budget word after.
    fn run_clamp(
        h: &mut WarpHarness,
        stride: u32,
        src_words: &[u32],
        count: Option<u32>,
        start: u32,
    ) -> (f64, Vec<u32>, u32) {
        let bytes = src_words.len() * 4;
        let records = (bytes / stride as usize) as u32;
        let (src_buf, src_ptr, _) = h.shared_buffer(bytes as u64);
        let (fill_buf, fill_ptr, _) = h.shared_buffer(bytes as u64);
        let (readback_buf, readback_ptr, _) = h.shared_buffer(bytes as u64);
        let (_budget_buf, budget_ptr, budget_gpu) = h.shared_buffer(256);
        let (count_buf, count_ptr, _) = h.shared_buffer(256);
        let dst_buf = h.default_uav(bytes as u64);
        // SAFETY: mapped buffers of these sizes; an open list.
        unsafe {
            std::ptr::copy_nonoverlapping(src_words.as_ptr(), src_ptr.cast::<u32>(), src_words.len());
            for i in 0..src_words.len() {
                fill_ptr.cast::<u32>().add(i).write_volatile(SENTINEL);
            }
            budget_ptr.cast::<u32>().write_volatile(start);
            count_ptr.cast::<u32>().write_volatile(count.unwrap_or(0));
            h.list.CopyBufferRegion(&dst_buf, 0, &fill_buf, 0, bytes as u64);
        }
        h.submit_and_wait();
        assert!(record_clamp(
            &h.clamp,
            &h.list,
            (&dst_buf, 0),
            budget_gpu,
            (&src_buf, 0),
            count.map(|_| (&count_buf, 0)),
            (0, stride, records),
            &[],
        ));
        done_on(&h.list, &dst_buf, &[]);
        let at = std::time::Instant::now();
        h.submit_and_wait();
        let took = at.elapsed().as_secs_f64();
        // SAFETY: a live device; an open list; mapped buffers.
        unsafe {
            if let Err(e) = h.device.GetDeviceRemovedReason() {
                panic!("device removed after {took:.3} s with {records} records: {e}");
            }
            h.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            h.list.CopyBufferRegion(&readback_buf, 0, &dst_buf, 0, bytes as u64);
            h.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_COMMON,
            )]);
        }
        h.submit_and_wait();
        // SAFETY: the GPU work is done.
        unsafe {
            let out = std::slice::from_raw_parts(readback_ptr.cast::<u32>().cast_const(), src_words.len()).to_vec();
            let after_src = std::slice::from_raw_parts(src_ptr.cast::<u32>().cast_const(), src_words.len());
            assert_eq!(after_src, src_words, "the source is untouched");
            (took, out, budget_ptr.cast::<u32>().read_volatile())
        }
    }

    /// What the policy allows of a clamp's output; panics on a violation.
    /// Returns (admitted positive-cost records, rejected ones, groups where
    /// a positive-cost record was admitted after a rejected one).
    fn check_policy(
        name: &str,
        stride: u32,
        src: &[u32],
        out: &[u32],
        count: Option<u32>,
        start: u32,
        after: u32,
    ) -> (usize, usize, usize) {
        let words = (stride / 4) as usize;
        let records = src.len() / words;
        let eff = count.map_or(records, |c| records.min(c as usize));
        let (mut admitted, mut rejected, mut unordered) = (0usize, 0usize, 0usize);
        let mut spent = 0u64;
        let mut all = 0u64;
        let mut group_rejected = false;
        for i in 0..records {
            if i % 64 == 0 {
                group_rejected = false;
            }
            let (s, o) = (&src[i * words..(i + 1) * words], &out[i * words..(i + 1) * words]);
            if i >= eff {
                assert!(o.iter().all(|&w| w == SENTINEL), "{name}: record {i} past the count was written");
                continue;
            }
            assert_eq!(&o[2..], &s[2..], "{name}: record {i} lost its other words");
            let c = s[0].min(1 << 24);
            let n = s[1].min((1 << 26) / c.max(1));
            let cost = u64::from(c) * u64::from(n);
            all += cost;
            if (o[0], o[1]) == (c, n) {
                if cost > 0 {
                    admitted += 1;
                    spent += cost;
                    if group_rejected {
                        unordered += 1;
                    }
                }
            } else {
                assert_eq!((o[0], o[1]), (0, 0), "{name}: record {i} is neither kept nor emptied");
                assert!(cost > 0, "{name}: record {i} costs nothing and was emptied");
                rejected += 1;
                group_rejected = true;
            }
        }
        assert!(after <= CAP, "{name}: the budget word passed the budget ({after:#x})");
        assert!(u64::from(start.min(CAP)) + spent <= u64::from(CAP), "{name}: admitted work passes the budget");
        if rejected == 0 {
            assert_eq!(u64::from(after), u64::from(start) + spent, "{name}: nothing rejected, budget is the sum");
        } else {
            assert_eq!(after, CAP, "{name}: a record was rejected, the budget is spent");
        }
        if u64::from(start) + all <= u64::from(CAP) {
            assert_eq!(rejected, 0, "{name}: everything fits, nothing may be rejected");
        }
        (admitted, rejected, unordered)
    }

    fn xorshift(state: &mut u64) -> u32 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        (*state >> 16) as u32
    }

    /// Experiment: the budget policy of the clamp under test. Run with
    /// VICTORY_EXP=1 (and VICTORY_CLAMP_HLSL / VICTORY_TEST_HW as wanted).
    #[test]
    fn exp_policy() {
        if std::env::var_os("VICTORY_EXP").is_none() {
            return;
        }
        let Some(mut h) = WarpHarness::new() else {
            return;
        };
        let draws = |f: &dyn Fn(usize) -> (u32, u32), n: usize, stride: u32| -> Vec<u32> {
            let words = (stride / 4) as usize;
            let mut v = vec![0u32; n * words];
            for i in 0..n {
                let (c, k) = f(i);
                v[i * words] = c;
                v[i * words + 1] = k;
                for w in 2..words {
                    v[i * words + w] = 0x1000_0000 + (i * 8 + w) as u32;
                }
            }
            v
        };
        let mut case = |name: &str, stride: u32, src: Vec<u32>, count: Option<u32>, start: u32| {
            let (_, out, after) = run_clamp(&mut h, stride, &src, count, start);
            let got = check_policy(name, stride, &src, &out, count, start, after);
            eprintln!("policy {name}: admitted {} rejected {} unordered {} budget {after:#x}", got.0, got.1, got.2);
            (got, after)
        };
        // Everything fits, with a partial last group.
        let (got, after) = case("fits", 16, draws(&|i| (1 + (i as u32 % 7), 1 + (i as u32 % 3)), 1000, 16), None, 0);
        assert_eq!(got.1, 0);
        assert!(after > 0);
        // Garbage: two records of 2^26 fit exactly.
        let (got, _) = case("garbage", 20, draws(&|_| (u32::MAX, u32::MAX), 128, 20), None, 0);
        assert_eq!((got.0, got.1), (2, 126));
        // One group whose plain sum is 2^32.
        let (got, _) = case("overflow", 16, draws(&|_| (1 << 24, 4), 64, 16), None, 0);
        assert_eq!((got.0, got.1), (2, 62));
        // An exact fit at lane 5, a rejected lane 6, a free lane 7, a rejected lane 8.
        let (got, after) = case(
            "exact",
            16,
            draws(&|i| if i == 7 { (9, 0) } else { (100, 1) }, 9, 16),
            None,
            CAP - 600,
        );
        assert_eq!((got.0, got.1, after), (6, 2, CAP));
        // A spent budget.
        let (got, after) = case("spent", 16, draws(&|i| (i as u32 % 2, 1), 200, 16), None, CAP);
        assert_eq!((got.0, got.1, after), (0, 100, CAP));
        // A count in the middle of a group; garbage past it is not charged.
        let (got, after) = case(
            "count",
            24,
            draws(&|i| if i < 70 { (3, 2) } else { (u32::MAX, u32::MAX) }, 200, 24),
            Some(70),
            5,
        );
        assert_eq!((got.0, got.1, after), (70, 0, 5 + 70 * 6));
        // A count of zero and one past the records.
        let (got, after) = case("count0", 16, draws(&|_| (u32::MAX, u32::MAX), 130, 16), Some(0), 7);
        assert_eq!((got.0, got.1, after), (0, 0, 7));
        let (got, _) = case("count-over", 16, draws(&|_| (2, 2), 130, 16), Some(u32::MAX), 0);
        assert_eq!((got.0, got.1), (130, 0));
        // Random mixes that run the budget out part of the way through.
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        for round in 0..6 {
            let n = 1 + (xorshift(&mut seed) % 5000) as usize;
            let mut costs = Vec::with_capacity(n);
            for _ in 0..n {
                let r = xorshift(&mut seed);
                costs.push(match r % 16 {
                    0 => (0, r >> 8),
                    1 => (r >> 4, r >> 9),
                    2 => (1 << 12, 1 << 6),
                    _ => (1 + (r >> 8) % 4096, 1 + (r >> 20) % 16),
                });
            }
            let start = match round % 3 {
                0 => 0,
                1 => CAP / 2,
                _ => CAP - 100_000,
            };
            case(&format!("random{round}"), 16, draws(&|i| costs[i], n, 16), None, start);
        }
    }

    /// Experiment: seconds for the clamp of N draws of one vertex instance
    /// each, which all fit. VICTORY_BENCH=1024,4096 lists N; a run over
    /// VICTORY_BENCH_STOP_MS (default 80) ends the test before larger N.
    #[test]
    fn exp_bench() {
        let Some(list) = std::env::var("VICTORY_BENCH").ok() else {
            return;
        };
        let stop = std::env::var("VICTORY_BENCH_STOP_MS")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(80.0);
        let Some(mut h) = WarpHarness::new() else {
            return;
        };
        // Warm the pipeline up.
        run_clamp(&mut h, 16, &[1, 1, 0, 0].repeat(64), None, 0);
        for n in list.split(',').filter_map(|s| s.trim().parse::<usize>().ok()) {
            let src = [1u32, 1, 0, 0].repeat(n);
            let mut times = Vec::new();
            for _ in 0..3 {
                let (took, out, after) = run_clamp(&mut h, 16, &src, None, 0);
                let got = check_policy("bench", 16, &src, &out, None, 0, after);
                assert_eq!((got.0, got.1, after as usize), (n, 0, n));
                times.push(took * 1000.0);
                if took * 1000.0 > stop {
                    break;
                }
            }
            eprintln!("bench records={n} ms={times:.2?}");
            if times.iter().any(|&t| t > stop) {
                eprintln!("bench stopped: over {stop} ms");
                break;
            }
        }
    }
}
