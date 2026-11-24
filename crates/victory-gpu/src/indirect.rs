//! Indirect execution's arguments, bounded before the GPU reads them. The
//! title's own compute passes write most of them; a pass we drop or
//! translate wrongly leaves whatever the memory held, and a draw of four
//! billion vertices or a dispatch past the host's limits runs until the
//! host resets the display driver. So every record is clamped on the GPU,
//! just before its execution, by a compute shader of ours (HLSL compiled by
//! the system's compiler; a 5.1 shader with its root signature inside).
//!
//! Each list also has a budget (2^27 vertex instances or threads, groups
//! counted as 64 threads, mesh groups as 128): records past it run empty,
//! so a list of thousands of garbage records cannot run for seconds
//! either.
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
#define RS "RootConstants(num32BitConstants=4, b0), UAV(u0), UAV(u1)"
cbuffer Record : register(b0) { uint kind; uint stride; uint records; uint unused; };
RWByteAddressBuffer args : register(u0);
// The list's work so far (vertex instances, threads): past the budget,
// records run empty.
RWByteAddressBuffer budget : register(u1);
static const uint BUDGET = 1u << 27;
// Groups of one dispatch; more run empty.
static const uint GROUPS = 1u << 20;
// One thread walks the records in order, so the budget needs no atomics.
// The two ways of sharing the word between threads that were tried both
// failed: a plain add wraps it after enough rejected records and lets work
// in again (64 draw records of 2^26 bring it back to zero), and reserving
// by compare-exchange retries more the more records fit (65,536 small ones
// ran an RTX 4080 past the host's time-out, and the driver was reset).
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    uint seen = min(budget.Load(0), BUDGET);
    [loop]
    for (uint i = 0; i < records; i++) {
        uint at = i * stride;
        uint cost;
        if (kind <= 1) {
            // Draws: vertex or index count, then instances.
            uint count = min(args.Load(at), 1u << 24);
            uint instances = min(args.Load(at + 4), (1u << 26) / max(count, 1u));
            cost = count * instances;
            args.Store2(at, uint2(count, instances));
        } else if (kind == 2) {
            uint3 g = min(args.Load3(at), uint3(65535, 65535, 65535));
            // x * y fits 32 bits; the whole count may not.
            if (g.z != 0 && g.x * g.y > GROUPS / g.z) {
                g = uint3(0, 0, 0);
            }
            cost = g.x * g.y * g.z * 64;
            args.Store3(at, g);
        } else {
            // Mesh dispatches: the console's are one-dimensional.
            uint3 g = args.Load3(at);
            g = uint3(min(g.x, 65535u), min(g.y, 1u), min(g.z, 1u));
            cost = g.x * g.y * g.z * 128;
            args.Store3(at, g);
        }
        // The first record that does not fit takes what is left, so those
        // after it run empty too.
        if (cost > BUDGET - seen) {
            // Draws keep their layout with no vertices; groups become zero.
            args.Store(at, 0u);
            seen = BUDGET;
        } else {
            seen += cost;
        }
    }
    budget.Store(0, seen);
}
"#;

/// Records clamped by one call; more are refused.
// shortcut: what the clamp's one thread walks in about 25 ms on an RTX 4080
// (2^20 take 0.4 s); an execution of more records is dropped. If the title
// has those, clamp only as many as its count buffer says.
const RECORDS: u32 = 1 << 16;

/// Whether the clamp can address `records` records `stride` bytes apart at
/// host address `gpu`: 4-byte aligned, records holding the three words the
/// clamp writes, and offsets that fit its 32 bits.
fn addressable(gpu: u64, stride: u32, records: u32) -> bool {
    gpu.is_multiple_of(4)
        && stride.is_multiple_of(4)
        && stride >= 12
        && records <= RECORDS
        && u64::from(stride) * u64::from(records) <= 1 << 32
}

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
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            SOURCE.as_ptr().cast(),
            SOURCE.len(),
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
                std::slice::from_raw_parts(errors.GetBufferPointer().cast::<u8>(), errors.GetBufferSize())
            };
            tracing::warn!("argument clamp: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob's own bytes; the root signature is inside the shader.
    unsafe {
        let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
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

/// Records, in `list`, the clamp of `records` records of `kind` (PC
/// argument numbering) `stride` bytes apart at host address `gpu` in
/// `buffer`, and leaves the buffer ready to be read as arguments. Replaces
/// the list's compute root signature and pipeline (callers bind theirs
/// after). False if the clamp is not available or cannot address the
/// records; nothing is recorded then.
pub(crate) fn bound(
    list: &HostList,
    buffer: &ID3D12Resource,
    gpu: u64,
    (kind, stride, records): (u32, u32, u32),
) -> bool {
    if !addressable(gpu, stride, records) {
        return false;
    }
    let (Some(c), Some(budget)) = (clamp(), list.budget()) else {
        return false;
    };
    let l = &list.list;
    // SAFETY: an open list; our root signature, pipeline and a live buffer
    // the caller's list holds.
    unsafe {
        // Earlier shader writes (the title's argument passes) first.
        l.ResourceBarrier(&[uav_barrier()]);
        record(l, c, gpu, budget, (kind, stride, records));
        l.ResourceBarrier(&[
            uav_barrier(),
            transition(
                buffer,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            ),
        ]);
    }
    true
}

/// The clamp itself on `l`: `records` records at host address `gpu`,
/// against the budget word at host address `budget`.
///
/// # Safety
/// An open list; both addresses in live buffers shaders may write.
unsafe fn record(
    l: &ID3D12GraphicsCommandList,
    c: &Clamp,
    gpu: u64,
    budget: u64,
    (kind, stride, records): (u32, u32, u32),
) {
    let constants = [kind, stride, records, 0];
    // SAFETY: the caller's; `constants` covers the parameter.
    unsafe {
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
        l.SetComputeRootUnorderedAccessView(1, gpu);
        l.SetComputeRootUnorderedAccessView(2, budget);
        l.Dispatch(1, 1, 1);
    }
}

/// After the execution: the buffer back to unordered access, where the
/// title's passes keep using it.
pub(crate) fn done(list: &HostList, buffer: &ID3D12Resource) {
    // SAFETY: an open list and a live buffer.
    unsafe {
        list.list.ResourceBarrier(&[transition(
            buffer,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )])
    };
}

fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: std::mem::ManuallyDrop::new(None) }),
        },
    }
}

#[cfg(test)]
mod tests {
    /// The clamp compiles with the system's compiler and the host takes it.
    #[test]
    fn argument_clamp_builds() {
        if super::host().is_none() {
            return; // no host GPU here
        }
        assert!(super::clamp().is_some());
    }

    /// Runs the clamp alone over `records` records of `kind` whose words
    /// are all `fill`, `calls` times against one budget word: the records
    /// left live (first word not zero) and the budget word after. `None`
    /// without a host GPU.
    fn clamped(kind: u32, stride: u32, records: u32, calls: u32, fill: u32) -> Option<(usize, u32)> {
        let host = super::host()?;
        let c = super::clamp().expect("the clamp builds");
        let len = (stride * records) as usize;
        let (args, data) = host.shared_buffer(len as u64).expect("an argument buffer");
        let (budget, word) = host.shared_buffer(256).expect("a budget word");
        let list = host.new_list().expect("a list");
        // SAFETY: our mapped buffers, written before the list runs and read
        // after it has; an open list, closed before it is executed.
        unsafe {
            for at in (0..len).step_by(4) {
                data.add(at).cast::<u32>().write_volatile(fill);
            }
            word.cast::<u32>().write_volatile(0);
            for _ in 0..calls {
                list.list.ResourceBarrier(&[super::uav_barrier()]);
                super::record(
                    &list.list,
                    c,
                    args.GetGPUVirtualAddress(),
                    budget.GetGPUVirtualAddress(),
                    (kind, stride, records),
                );
            }
            list.list.Close().expect("the list closes");
            host.execute(&[&list]);
            let live = (0..len)
                .step_by(stride as usize)
                .filter(|&at| data.add(at).cast::<u32>().read_volatile() != 0)
                .count();
            Some((live, word.cast::<u32>().read_volatile()))
        }
    }

    /// Rejected records do not wrap the budget: of 128 garbage draws (2^26
    /// vertex instances each once clamped) two run, where a wrapping add let
    /// four; of 600 garbage mesh dispatches 16, not 31.
    #[test]
    fn rejected_records_do_not_wrap_the_budget() {
        let Some(draws) = clamped(0, 16, 128, 1, u32::MAX) else {
            return; // no host GPU here
        };
        assert_eq!(draws, (2, 1 << 27));
        assert_eq!(clamped(10, 12, 600, 1, u32::MAX), Some((16, 1 << 27)));
    }

    /// Records that fit all run: the most one call takes, one vertex
    /// instance each.
    #[test]
    fn records_within_the_budget_all_run() {
        let Some(small) = clamped(0, 16, super::RECORDS, 1, 1) else {
            return; // no host GPU here
        };
        assert_eq!(small, (1 << 16, 1 << 16));
    }

    /// The budget is the list's: a second execution of the same records
    /// finds it spent, and they all run empty.
    #[test]
    fn a_spent_budget_stays_spent() {
        let Some(twice) = clamped(1, 20, 64, 2, u32::MAX) else {
            return; // no host GPU here
        };
        assert_eq!(twice, (0, 1 << 27));
    }

    #[test]
    fn records_the_clamp_cannot_address_are_refused() {
        use super::addressable;
        assert!(addressable(0x1000, 16, 1 << 16));
        assert!(addressable(0x1000, 12, 0));
        assert!(!addressable(0x1002, 16, 1), "address not 4-byte aligned");
        assert!(!addressable(0x1000, 18, 1), "stride not 4-byte aligned");
        assert!(!addressable(0x1000, 8, 1), "a record shorter than three words");
        assert!(!addressable(0x1000, 16, (1 << 16) + 1), "too many records");
        assert!(!addressable(0x1000, 1 << 17, 1 << 16), "offsets past 32 bits");
    }
}
