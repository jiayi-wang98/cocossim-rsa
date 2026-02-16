# Change Logs For RSA Work

## Step 1 - Define RSA knobs and assumptions in code
- Added RSA/CGRA knobs to architecture config in `include/frontends/standard/StandardArch.h`:
  - `reuse_sa_for_vu`
  - `cgra_cluster`
  - `cgra_cfg_ratio`
  - `cgra_default_ii`
  - `cgra_default_pipeline`
- Purpose: make CGRA modeling assumptions explicit and runtime-configurable.

## Step 2 - Add runtime controls (CLI flags)
- Extended parser support for float arch flags in `include/frontends/ArchParser.h`.
- Added new CLI flags in `src/frontends/standard/StandardParser.cc`:
  - `-reuse_sa_for_vu`
  - `-cgra_cluster`
  - `-cgra_cfg_ratio`
  - `-cgra_ii`
  - `-cgra_pipe`
- Added guardrails (minimum values and clamping) in parser.

## Step 3 - Add SA CGRA execution mode and config-overhead model
- Added CGRA op enum and CGRA job type in `include/units/standard/SysArray.h`:
  - `CgraOp` classes include FMA/logic/compare/int16/convert/reduce/exp-approx/normalize.
  - `CgraJob` carries dimensions + timing + memory fields.
- Extended SA state enum with CGRA states:
  - `cgra_config`
  - `cgra_stream`
- Implemented CGRA path in `src/units/standard/SysArray.cc`:
  - `init()` detects `CgraJob` and enters `cgra_config`.
  - Configuration latency modeled as `ceil(cgra_cfg_ratio * sa_sz)`.
  - Streaming cycles modeled from:
    - effective CGRA ALUs per core = `(sa_sz / cgra_cluster)^2`
    - work quanta = `ceil(work_items / n_cgra_alus)`
    - stream cycles = `pipeline_depth + (quanta - 1) * II * op_latency`
  - Final writeback uses existing memory stage machinery.

## Step 4 - Remove VU states in RSA mode
- Updated architecture construction in `src/frontends/standard/StandardArch.cc`:
  - Always instantiate SA cores.
  - Instantiate VU states only when `reuse_sa_for_vu == false`.
- Result: in RSA mode, scheduler runs only SA states (VU removed from active architecture).

## Step 5 - Lower vector-style layers to SA-CGRA jobs in RSA mode
- Added helper lowering functions in `src/frontends/standard/StandardLayers.cc`:
  - `make_activation_jobs(...)`
  - `make_layernorm_jobs(...)`
  - `make_softmax_jobs(...)`
- In RSA mode, these build per-core CGRA job chains on SA cores.
- Kept baseline behavior by preserving existing VecUnit path when RSA mode is off.

## Step 6 - Explicit softmax decomposition for CGRA
- In `src/frontends/standard/StandardLayers.cc`, softmax is decomposed into 4 CGRA stages:
  1. max reduce
  2. exp approximation (includes subtract path proxy)
  3. sum reduce
  4. normalize
- Wired stage dependencies explicitly per core.

## Step 7 - Integration touch-ups
- Updated `MatmulAct`, `ActMatmul`, `Activation`, `LayerNorm`, and `Softmax` to use shared helper lowering.
- Updated `SelfAttention`:
  - WS path now passes actual `n_heads` into softmax builder (removed hardcoded `8`).
  - OS path softmax now uses shared lowering (Vec or CGRA depending on mode).
- VCD state visibility check:
  - Ran a CGRA-only activation workload and confirmed new SA state encodings appear:
    - `0x5` (`cgra_config`)
    - `0x6` (`cgra_stream`)
    - followed by `0x4` write and `0x0` idle.

## Step 8 - Verification
- Build verification:
  - Ran `make -j$(nproc)` in `build/` successfully after RSA changes.
- Baseline regression check:
  - Command:
    - `./perf_model -dram_profile tpuv3 -c 1 -sa_sz 128 -vu_sz 1024 -ws 1 -ws_dbuf 1 -reuse_sa_for_vu 0 -f 1 -i ws_1024.txt -o ws_1024_regcheck.txt`
  - Result:
    - `Cycles 70903` (matches prior WS-dbuf reference behavior).
- RSA smoke A/B (attention + layernorm) with `build/rsa_smoke.txt`:
  - Baseline (`reuse_sa_for_vu=0`):
    - `Cycles 3832`
    - output includes both `SYSTOLIC_ARRAY` and `VECTOR_UNIT` utilization lines.
  - RSA (`reuse_sa_for_vu=1`, `cgra_cluster=4`, `cgra_cfg_ratio=1.0`):
    - `Cycles 4415`
    - output includes only `SYSTOLIC_ARRAY` lines (no VU states instantiated).
- `cgra_cfg_ratio` sanity check with `build/cgra_cfg_test.txt` (`Activation 1024`):
  - `cgra_cfg_ratio=0.0` -> `Cycles 109`
  - `cgra_cfg_ratio=2.0` -> `Cycles 270`
  - Confirms config-overhead ratio directly impacts runtime as intended.

## Step 9 - Initial system-level comparison scaffold
- Added richer reporting to output files in `src/main.cc`:
  - `CGRA_MODE_SWITCH_CYCLES`
  - `LAYER_BREAKDOWN_BEGIN/END` section with, per layer:
    - `ACTIVE`, `WINDOW`, `FIRST`, `LAST`
    - `SA`, `VU`
    - `CGRA_CFG`, `CGRA_STREAM`, `CGRA_WRITE`
- Added layer tagging in jobs (`layer_idx`, `layer_name`) via `src/frontends/standard/StandardLayers.cc` and `include/Job.h`.
- Added per-cycle aggregation in `src/Arch.cc`.
- Validation on `build/rsa_smoke.txt`:
  - baseline report includes non-zero VU layer contribution.
  - RSA report includes non-zero `CGRA_MODE_SWITCH_CYCLES` and CGRA stage counters.
