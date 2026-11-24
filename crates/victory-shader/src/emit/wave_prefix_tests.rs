//! Readback of wave prefixes: synthetic wave32 compute programs whose
//! whole-wave section leaves the prefix of one operation over the lanes'
//! own inputs, lowered to DXIL (`wavePrefixOp`), validated by `dxil.dll`
//! when present, and run on every hardware adapter (the answer is the
//! driver's, so one GPU does not speak for another) and on WARP. Each of
//! the 32 lanes loads its own dword of a structured buffer and stores the
//! prefix to its own dword of a raw UAV, which is compared with the
//! console's instructions run on the CPU.
//!
//! DXIL documents `wavePrefixOp` for sums and products only; no compiler
//! emits the minimum and maximum kinds and `dxil.dll` does not check the
//! kind, so only a driver says what they do.

use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::core::Interface;

use super::host::{Dispatch, Host, Root, Stride, View};
use super::{HostShader, lower, validator};
use crate::dxbc::{Container, FourCC};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const NAN: u32 = 0x7FC0_0000;
/// What every lane stores before the section: a lane that then stores
/// nothing is told from one that stores 0. Not an input, an identity or a
/// NaN.
const SENTINEL: u32 = 0xDEAD_BEEF;

/// A scanned operation: its VOP2 opcode, its identity, and the `row_shr`
/// distances of the scan inside each row of 16 lanes.
#[derive(Clone, Copy)]
struct Scan {
    name: &'static str,
    opcode: u32,
    ident: u32,
    steps: &'static [u32],
    float: bool,
}

/// The distances of `a_wave_prefix_of_min_lowers`: fine for an operation
/// that may count a lane twice.
const EVERY: &[u32] = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
/// Distances that count each lane once, as a sum needs.
const DOUBLING: &[u32] = &[1, 2, 4, 8];

const fn scan(name: &'static str, opcode: u32, ident: u32, steps: &'static [u32]) -> Scan {
    Scan {
        name,
        opcode,
        ident,
        steps,
        float: opcode == 0x03 || opcode == 0x0F || opcode == 0x10,
    }
}

const SUM_U32: Scan = scan("v_add_nc_u32", 0x25, 0, DOUBLING);
const SUM_F32: Scan = scan("v_add_f32", 0x03, 0, DOUBLING);
const MIN_U32: Scan = scan("v_min_u32", 0x13, u32::MAX, EVERY);
const MAX_U32: Scan = scan("v_max_u32", 0x14, 0, EVERY);
const MIN_I32: Scan = scan("v_min_i32", 0x11, 0x7FFF_FFFF, EVERY);
const MAX_I32: Scan = scan("v_max_i32", 0x12, 0x8000_0000, EVERY);
const MIN_F32: Scan = scan("v_min_f32", 0x0F, 0x7F80_0000, EVERY);
const MAX_F32: Scan = scan("v_max_f32", 0x10, 0xFF80_0000, EVERY);

impl Scan {
    /// The operation on two lanes' dwords, as the console computes it: a
    /// float minimum or maximum keeps the operand that is not a NaN.
    fn apply(&self, a: u32, b: u32) -> u32 {
        let (fa, fb) = (f32::from_bits(a), f32::from_bits(b));
        let pick = |min: bool| match (fa.is_nan(), fb.is_nan()) {
            (true, _) => b,
            (_, true) => a,
            _ if (fa < fb) == min => a,
            _ => b,
        };
        match self.opcode {
            0x25 => a.wrapping_add(b),
            0x03 => (fa + fb).to_bits(),
            0x13 => a.min(b),
            0x14 => a.max(b),
            0x11 => (a as i32).min(b as i32) as u32,
            0x12 => (a as i32).max(b as i32) as u32,
            0x0F => pick(true),
            _ => pick(false),
        }
    }

    /// The program: lane `v0` loads dword `v0` of the structured buffer at
    /// entry 0 of the table at s0 and stores [`SENTINEL`] to dword `v0` of
    /// the raw UAV at entry 1, a whole-wave section scans what it loaded,
    /// and the lane stores the prefix over the sentinel.
    ///
    /// The inclusive section is that of `a_wave_prefix_of_min_lowers`. The
    /// exclusive one first moves every lane's term one lane up (lane 16's
    /// from lane 15 by `v_readlane`), the identity into lane 0, and scans
    /// that.
    fn program(&self, inclusive: bool) -> Vec<u32> {
        // The identity as a VOP source: an inline constant or a literal.
        let ident = |words: &mut Vec<u32>, inst: u32| match self.ident {
            0 => words.push(inst | 0x80),
            u32::MAX => words.push(inst | 0xC1),
            k => words.extend([inst | 0xFF, k]),
        };
        // `op v<r>, <src0 with DPP>, v<r>`.
        let op = |r: u32| self.opcode << 25 | r << 17 | r << 9 | 0xFA;
        let scan = |words: &mut Vec<u32>, r: u32, lane15: u32| {
            for &n in self.steps {
                // row_shr:n, lanes the shift leaves without a source kept.
                words.extend([op(r), 0xFF00_0000 | (0x110 + n) << 8 | r]);
            }
            words.extend([
                0xD760_0000 | lane15, // v_readlane_b32 s<lane15>, v<r>, 15
                0x0001_1F00 | r,
                0x7E04_0200 | lane15, // v_mov_b32 v2, s<lane15>
                op(r),                // op v<r>, v2 quad_perm:[0,1,2,3] row_mask:0x2, v<r>
                0x2F00_E402,
            ]);
        };
        let mut words = vec![
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0602,
            0xFA00_0000, // s_load_dwordx4 s[24:27], s[4:5], 0x0
            0xF408_0A02,
            0xFA00_0020, // s_load_dwordx4 s[40:43], s[4:5], 0x20
            0x3420_0082, // v_lshlrev_b32 v16, 2, v0
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0xE030_1000,
            0x8006_0C10, // buffer_load_dword v12, v16, s[24:27], 0 offen
            0xBF8C_3F70, // s_waitcnt vmcnt(0)
            0x7E08_02FF, // v_mov_b32 v4, SENTINEL
            SENTINEL,
            0xE070_1000,
            0x800A_0410, // buffer_store_dword v4, v16, s[40:43], 0 offen
            0xBE84_3DC1, // s_or_saveexec_b32 s4, -1
            0x876A_C104, // s_and_b32 s106, s4, -1
        ];
        ident(&mut words, 0x0202_1800); // v_cndmask_b32 v1, <identity>, v12, vcc
        let result = if inclusive {
            scan(&mut words, 1, 6);
            1
        } else {
            ident(&mut words, 0x7E06_0200); // v_mov_b32 v3, <identity>
            words.extend([
                0x7E06_02FA, // v_mov_b32 v3, v1 row_shr:1
                0xFF01_1101,
                0xD760_0006, // v_readlane_b32 s6, v1, 15
                0x0001_1F01,
                0xBEFE_03FF, // s_mov_b32 exec_lo, 0x10000
                0x0001_0000,
                0x7E06_0206, // v_mov_b32 v3, s6
                0xBEFE_03C1, // s_mov_b32 exec_lo, -1
            ]);
            scan(&mut words, 3, 7);
            3
        };
        words.extend([
            0xBEFE_0304, // s_mov_b32 exec_lo, s4
            0xE070_1000,
            0x800A_0010 | result << 8, // buffer_store_dword v<result>, v16, s[40:43], 0 offen
            0xBF81_0000,               // s_endpgm
        ]);
        words
    }

    /// What the console's lanes hold after [`Scan::program`], every lane
    /// active: the instructions of the section run on the CPU.
    fn console(&self, input: &[u32; 32], inclusive: bool) -> [u32; 32] {
        let scan = |v: &mut [u32; 32]| {
            for &n in self.steps {
                let old = *v;
                for l in 0..32 {
                    if l % 16 >= n as usize {
                        v[l] = self.apply(old[l - n as usize], old[l]);
                    }
                }
            }
            let lane15 = v[15];
            for x in &mut v[16..] {
                *x = self.apply(lane15, *x);
            }
        };
        let mut v = *input;
        if !inclusive {
            v = [self.ident; 32];
            for l in (1..32).filter(|l| l % 16 != 0) {
                v[l] = input[l - 1];
            }
            v[16] = input[15];
        }
        scan(&mut v);
        v
    }
}

/// Lowers `words` as a wave32 compute program of 32 threads with the table
/// of views at s0: a structured buffer of dwords at entry 0 and a raw UAV
/// at entry 1.
fn build(words: &[u32]) -> Result<HostShader, String> {
    let range = |kind, table_offset_dwords, stride| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride,
        dwords: 4,
        user_data: 0,
        slot: 0,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 32,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![
            range(ResourceType::Srv, 0, 4),
            range(ResourceType::Uav, 8, 0),
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 1,
            threads: [32, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).map_err(|e| e.to_string())?;
    let st = structure::structure(&p).map_err(|e| e.to_string())?;
    let res =
        resource::track(&p, &native.layout, native.stage, Some(&st)).map_err(|e| e.to_string())?;
    let io = io::raise(&p, &native).map_err(|e| e.to_string())?;
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    })
    .map_err(|e| e.to_string())?;
    assert!(
        lowered.wave_ops,
        "the section lowered without a wave operation"
    );
    // Lane order means nothing unless the host's wave is the console's:
    // the module has to require waves of exactly 32 lanes.
    let c = Container::parse(&lowered.dxil).expect("container");
    let info = &c.part(FourCC::PSV0).expect("PSV0").data[4..];
    let lanes = |at: usize| u32::from_le_bytes(info[at..at + 4].try_into().unwrap());
    assert_eq!((lanes(16), lanes(20)), (32, 32), "WaveSize(32)");
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
        eprintln!("dxil.dll: valid");
    } else {
        eprintln!("no dxil.dll; validation skipped");
    }
    Ok(shader)
}

/// Every hardware adapter with SM 6.6 (none with `VICTORY_WARP=1`), then
/// WARP, each with whether it is WARP.
fn hosts() -> Vec<(bool, Host)> {
    let mut out = Vec::new();
    let hardware = !std::env::var("VICTORY_WARP").is_ok_and(|v| v == "1");
    // SAFETY: plain D3D12/DXGI calls with valid out pointers.
    unsafe {
        let factory = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0)).ok();
        let adapters = factory
            .iter()
            .filter(|_| hardware)
            .flat_map(|f| (0..).map_while(|i| f.EnumAdapters1(i).ok()));
        // One GPU can be listed more than once (a virtual display's adapter
        // renders on it): once per model is enough to hear its driver.
        let mut seen = Vec::new();
        for a in adapters {
            let Ok(d) = a.GetDesc1() else {
                continue;
            };
            let model = (d.VendorId, d.DeviceId, d.SubSysId);
            let software = d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0;
            let mut device: Option<ID3D12Device> = None;
            if software
                || seen.contains(&model)
                || D3D12CreateDevice(&a, D3D_FEATURE_LEVEL_12_0, &mut device).is_err()
            {
                continue;
            }
            seen.push(model);
            let Some(device) = device else {
                continue;
            };
            let mut sm = D3D12_FEATURE_DATA_SHADER_MODEL {
                HighestShaderModel: D3D_SHADER_MODEL_6_6,
            };
            let asked = device.CheckFeatureSupport(
                D3D12_FEATURE_SHADER_MODEL,
                (&raw mut sm).cast(),
                size_of_val(&sm) as u32,
            );
            if asked.is_ok() && sm.HighestShaderModel.0 >= D3D_SHADER_MODEL_6_6.0 {
                out.push((false, Host { device }));
            }
        }
    }
    out.extend(Host::new(true).ok().map(|h| (true, h)));
    out
}

/// The name of the adapter of `host`, its driver version, and the smallest
/// and largest wave it runs.
fn adapter(host: &Host) -> (String, String, std::ops::RangeInclusive<u32>) {
    // SAFETY: plain D3D12/DXGI calls with valid out pointers.
    unsafe {
        let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
        let _ = host.device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS1,
            (&raw mut o).cast(),
            size_of_val(&o) as u32,
        );
        let waves = o.WaveLaneCountMin..=o.WaveLaneCountMax;
        let luid = host.device.GetAdapterLuid();
        let found = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            .ok()
            .and_then(|f| {
                (0..)
                    .map_while(|i| f.EnumAdapters1(i).ok())
                    .filter_map(|a| Some((a.GetDesc1().ok()?, a)))
                    .find(|(d, _)| d.AdapterLuid == luid)
            });
        let Some((desc, a)) = found else {
            return ("hardware".to_string(), String::new(), waves);
        };
        let name = String::from_utf16_lossy(&desc.Description);
        let v = a.CheckInterfaceSupport(&IDXGIDevice::IID).unwrap_or(0);
        (
            name.trim_end_matches('\0').to_string(),
            format!(
                "driver {}.{}.{}.{}",
                v >> 48,
                v >> 32 & 0xFFFF,
                v >> 16 & 0xFFFF,
                v & 0xFFFF
            ),
            waves,
        )
    }
}

/// Whether two result dwords agree: equal, or both NaNs of a float scan.
fn same(s: &Scan, a: u32, b: u32) -> bool {
    a == b || s.float && f32::from_bits(a).is_nan() && f32::from_bits(b).is_nan()
}

/// The lanes' inputs, by name: never zero, neither rising nor falling, and
/// lane after lane a new minimum (even lanes) or maximum (odd lanes) but
/// for every fifth lane, which is neither.
fn inputs(float: bool) -> Vec<(&'static str, [u32; 32])> {
    let lanes = |f: &dyn Fn(u32) -> u32| -> [u32; 32] { std::array::from_fn(|l| f(l as u32)) };
    if float {
        let plain = lanes(&|l| {
            let x = (l + 1) as f32;
            let scale = match (l % 5, l % 2) {
                (4, _) => 0.25,
                (_, 0) => -1.5,
                _ => 1.25,
            };
            (scale * x).to_bits()
        });
        let with = |lanes: &[usize], x: u32| {
            let mut v = plain;
            for &l in lanes {
                v[l] = x;
            }
            v
        };
        // The plain terms sum exactly in any order, and a sum that holds
        // 2^60 rounds to it: a prefix made as the sum up to the lane less
        // the lane's own term shows in the lane of the NaN, of 2^60 (0 for
        // the terms below) and of the infinity (a NaN).
        vec![
            ("floats", plain),
            ("floats, NaN in lanes 9 and 20", with(&[9, 20], NAN)),
            ("floats, NaN in lane 0", with(&[0], NAN)),
            ("floats, 2^60 in lane 11", with(&[11], 0x5D80_0000)),
            ("floats, infinity in lane 11", with(&[11], 0x7F80_0000)),
        ]
    } else {
        // Around 0x8000_0000 the order differs between signed and unsigned.
        let around = |mid: u32| {
            lanes(&|l| match (l % 5, l % 2) {
                (4, _) => mid.wrapping_add(l),
                (_, 0) => mid.wrapping_sub(0x0101_0101 * (l + 1)),
                _ => mid.wrapping_add(0x0101_0101 * (l + 1)),
            })
        };
        vec![
            ("integers around 0x80000000", around(0x8000_0000)),
            ("integers around 0", around(0)),
        ]
    }
}

/// Runs the prefix of `s` on every hardware adapter and on WARP, and
/// compares what each lane stored with the console's result.
fn reads_back(s: Scan, inclusive: bool) {
    let form = if inclusive { "inclusive" } else { "exclusive" };
    let what = format!("{form} prefix of {}", s.name);
    let shader = build(&s.program(inclusive)).unwrap_or_else(|e| panic!("{what}: {e}"));
    let sets = inputs(s.float);
    for (_, input) in &sets {
        assert!(input.iter().all(|&x| x != 0));
        // The section run on the CPU is the prefix it stands for.
        if !s.float {
            let console = s.console(input, inclusive);
            let mut acc = s.ident;
            for l in 0..32 {
                let below = acc;
                acc = s.apply(acc, input[l]);
                let want = if inclusive { acc } else { below };
                assert_eq!(console[l], want, "{what}: lane {l}");
            }
        }
    }
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut failures = Vec::new();
    let mut ran = 0;
    for (warp, host) in hosts() {
        let (name, driver, waves) = adapter(&host);
        let label = if warp { "warp" } else { name.as_str() };
        let (least, most) = (waves.start(), waves.end());
        eprintln!("{label}: {driver}, waves of {least} to {most}");
        // WARP runs waves of 4 lanes and refuses one fixed at 32.
        if !waves.contains(&32) {
            eprintln!("{label}: {what}: skipped (no wave of 32 lanes)");
            continue;
        }
        let (root, pso) = match host.compute_pipeline(&shader) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("{label}: {what}: PIPELINE REFUSED ({e})");
                failures.push(format!("{label}: pipeline refused: {e}"));
                continue;
            }
        };
        ran += 1;
        for (set, input) in &sets {
            let want = s.console(input, inclusive);
            // The table starts at heap index 0.
            let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            let views = [
                View::Buffer {
                    data: input,
                    stride: Stride::Structured(4),
                },
                View::RawUav(32),
            ];
            let got = host.dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            );
            // SAFETY: a plain D3D12 call.
            let removed = unsafe { host.device.GetDeviceRemovedReason() };
            let got = got.and_then(|words| match removed {
                Ok(()) => Ok(words[0].clone()),
                Err(e) => Err(format!("device removed: {}", e.message())),
            });
            let got = match got {
                Ok(words) => words,
                Err(e) => {
                    eprintln!("{label}: {what}, {set}: DISPATCH FAILED ({e})");
                    failures.push(format!("{label}: {set}: dispatch failed: {e}"));
                    break;
                }
            };
            let wrong: Vec<usize> = (0..32).filter(|&l| !same(&s, got[l], want[l])).collect();
            // Where every term up to a lane is a NaN the console keeps the
            // NaN (no step of the scan folds a lone term), and the
            // lowering's FMin or FMax of the prefix and the term gives the
            // prefix: the lowering's doing, whatever the driver answers.
            let all_nan = |l: usize| input[..=l].iter().all(|&x| f32::from_bits(x).is_nan());
            let folded = s.float && inclusive && s.opcode != SUM_F32.opcode;
            let by_fold = wrong.iter().filter(|&&l| folded && all_nan(l)).count();
            let unstored = got.iter().filter(|&&x| x == SENTINEL).count();
            match wrong.first() {
                None => eprintln!("{label}: {what}, {set}: EXACT"),
                Some(&l) => {
                    let mut summary = format!(
                        "{} lanes wrong, first lane {l}: got {:#010x}, want {:#010x}",
                        wrong.len(),
                        got[l],
                        want[l],
                    );
                    if unstored > 0 {
                        summary += &format!("; {unstored} lanes stored nothing");
                    }
                    if by_fold > 0 {
                        summary += &format!("; {by_fold} by the fold of an all-NaN prefix");
                    }
                    eprintln!(
                        "{label}: {what}, {set}: WRONG, {summary}\n   in {input:08x?}\n  \
                         got {got:08x?}\n want {want:08x?}"
                    );
                    failures.push(format!("{label}: {set}: {summary}"));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{what}: {failures:#?}");
    // A machine with WARP alone (CI) has nothing to say about a driver.
    if ran == 0 {
        eprintln!("{what}: ran on no device");
    }
}

macro_rules! prefixes {
    ($($scan:ident: $inclusive:ident, $exclusive:ident;)*) => {$(
        #[test]
        fn $inclusive() {
            reads_back($scan, true);
        }

        #[test]
        fn $exclusive() {
            reads_back($scan, false);
        }
    )*};
}

prefixes! {
    // Sums are the kind DXIL documents: the control of the harness.
    SUM_U32: an_inclusive_prefix_sum_of_u32_reads_back, an_exclusive_prefix_sum_of_u32_reads_back;
    SUM_F32: an_inclusive_prefix_sum_of_f32_reads_back, an_exclusive_prefix_sum_of_f32_reads_back;
    MIN_U32: an_inclusive_prefix_min_of_u32_reads_back, an_exclusive_prefix_min_of_u32_reads_back;
    MAX_U32: an_inclusive_prefix_max_of_u32_reads_back, an_exclusive_prefix_max_of_u32_reads_back;
    MIN_I32: an_inclusive_prefix_min_of_i32_reads_back, an_exclusive_prefix_min_of_i32_reads_back;
    MAX_I32: an_inclusive_prefix_max_of_i32_reads_back, an_exclusive_prefix_max_of_i32_reads_back;
    MIN_F32: an_inclusive_prefix_min_of_f32_reads_back, an_exclusive_prefix_min_of_f32_reads_back;
    MAX_F32: an_inclusive_prefix_max_of_f32_reads_back, an_exclusive_prefix_max_of_f32_reads_back;
}
