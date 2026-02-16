# Notes: Changing Workload Precision (`data_type_width`)

## Where to change it

`data_type_width` is defined in:

- `include/global.h`

Current line:

```cpp
const int data_type_width = 2;
```

## What value to use

- `2` for 16-bit data types (for example FP16/BF16-style byte width)
- `1` for 8-bit data types (for example INT8-style byte width)
- `4` for 32-bit data types

## Important limitation

In the current model, changing `data_type_width` mainly changes memory/buffer traffic accounting.  
Compute-cycle formulas are mostly datatype-agnostic, so this is not a full numeric-precision model.

## After editing

Rebuild the simulator:

```bash
cd build
make -j$(nproc)
```

Then run as usual:

```bash
./perf_model -c 4 -sa_sz 128 -vu_sz 1024 -ws 1 -f 0.94 -i layers.txt -o results.txt
```

## Systolic array VCD state mapping

The systolic array state encoding comes from `ExState` in:

- `include/units/standard/SysArray.h`

Mapping:

- `0x0` = `idle`
- `0x1` = `prefetch` (WS-only weight/activation preload)
- `0x2` = `read`
- `0x3` = `shift` (compute/propagate phase)
- `0x4` = `write` (drain/writeback phase)

So in WS mode, seeing `0x1 -> 0x2 -> 0x3` is expected.

State meaning:

- `0x0 (idle)`: no job is running on this systolic-array core.
- `0x1 (prefetch)`: WS-only startup phase; preload initial weights/activations before steady tile processing.
- `0x2 (read)`: read stage for the current tile (memory-read work and read-stage timing).
- `0x3 (shift)`: compute/propagate stage through the array for the tile.
- `0x4 (write)`: drain/writeback stage; write outputs (or partial outputs) and advance to next tile/row or finish.

WS timing clarification (`systolic_fpu_latency=1`, `batch_size=1`):

- `0x2 (read)` stage budget is `M * max(systolic_fpu_latency, batch_size)`.
  For `M=1024`, this is `1024` cycles.
- `0x3 (shift)` stage budget is `sz * max(systolic_fpu_latency, batch_size)`.
  For `sz=128`, this is `128` cycles.

Interpretation:

- `0x2` is the main per-tile streaming/compute window over `M` rows.
- `0x3` is the array propagation/flush window over array depth `sz` (matches skew intuition).
- If you expect `127` skew cycles, this model uses `128` (`sz`) as the stage budget by convention.

Important nuance:

- `0x2` is not a pure "DRAM read-only" stage.
- Memory traffic is modeled via byte/beats counters in each stage; stage completion waits for both cycle budget and memory completion.

WS double-buffer overlap mode:

- Enable with `-ws_dbuf 1` (only meaningful when `-ws 1`).
- In this mode, WS uses an overlap path for steady-state tiles:
  `... -> 0x2(read) -> 0x4(write control tick) -> 0x2(read) -> ...`
- Steady-state `0x3` is removed, but the **final tile keeps tail flush**:
  `0x2 -> 0x3 -> 0x4`.
- Practical effect: tile initiation interval approaches `M` cycles in steady state (plus small control overhead in this model).
- `0x4` on non-final tiles is short control progression; the long drain/writeback `0x4` remains at the end.
- This is an optimistic overlap model; most preload/shift cost is hidden after startup.

Input-readiness note in this model:

- One-tile-ahead preload traffic is charged in the previous tile window.
- If preload is not ready, `SYSTOLIC_ARRAY_0_IDLE_FROM_MEMORY` would assert and stall progression.
- For `Matmul 1024 1024 1024` in the current run, `IDLE_FROM_MEMORY` is asserted only at startup prefetch and final writeback, not before intermediate `0x2` windows.

CLI note:

- `-ws_dbuf` is an architecture flag (`0` or `1`) and can be passed together with `-ws`.

Example:

```bash
# Baseline WS
./perf_model -dram_profile tpuv3 -c 1 -sa_sz 128 -vu_sz 1024 -ws 1 -ws_dbuf 0 -f 1 -i ws_1024.txt -o ws_1024_nodbuf.txt

# WS with double-buffer overlap model
./perf_model -dram_profile tpuv3 -c 1 -sa_sz 128 -vu_sz 1024 -ws 1 -ws_dbuf 1 -f 1 -i ws_1024.txt -o ws_1024_dbuf.txt
```

Observed on `Matmul 1024 1024 1024` (`sa_sz=128`, `c=1`, TPUv3 DRAM profile):

- `ws_dbuf=0` -> `Cycles 86199`
- `ws_dbuf=1` -> `Cycles 70903`

Observed VCD tail (`ws_dbuf=1`):

- Final tile: `0x2 @ t=65235 -> t=66259` (`1024` cycles)
- Tail flush: `0x3 @ t=66259 -> t=66387` (`128` cycles)
- Final drain: `0x4 @ t=66387 -> t=70903` (`4516` cycles to completion)

`IDLE_FROM_MEMORY` windows in this run:

- Startup prefetch wait: `t=128 -> 660`
- Final writeback wait: `t=66388 -> 70903`

## DRAM runtime switch

You can choose DRAM settings at runtime without editing source code.

Global flags:

- `-dram_profile <tpuv3|tpuv4|hbm2_default>`
- `-dram_cfg <path/to/dramsim3_config.ini>`

Built-in DRAM profiles:

- `tpuv3` -> `../dramsim3/configs/HBM2_TPUv3_900GBps_proxy.ini`
- `tpuv4` -> `../dramsim3/configs/HBM2_TPUv4_1228GBps_proxy.ini`
- `hbm2_default` -> `../dramsim3/configs/HBM2_8Gb_x128.ini`

Examples:

```bash
# TPUv3 DRAM proxy
./perf_model -dram_profile tpuv3 -c 1 -sa_sz 128 -vu_sz 1024 -ws 1 -f 1 -i layers.txt -o out.txt

# TPUv4 DRAM proxy
./perf_model -dram_profile tpuv4 -c 1 -sa_sz 128 -vu_sz 512 -ws 1 -f 1 -i layers.txt -o out.txt

# Explicit DRAMSim3 config file
./perf_model -dram_cfg ../dramsim3/configs/HBM2_8Gb_x128.ini -c 1 -sa_sz 128 -vu_sz 1024 -ws 1 -f 1 -i layers.txt -o out.txt
```
