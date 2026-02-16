# Plan For RSA (Reuse Systolic Array As CGRA)

## Goal
Build and evaluate a reconfigurable-array mode that removes Vector Units from the critical path by reusing the systolic array for vector-style work.

## Step-by-step Plan

1. Define modeling assumptions and knobs.
Action: Freeze the architecture/timing assumptions for RSA mode, including `4x4 PE -> 1 CGRA ALU`, supported op classes, per-op `II`, pipeline depth, and mode-switch overhead.
Output: A fixed parameter set so all experiments use the same rules.

2. Add runtime controls for RSA mode.
Action: Add flags such as `-reuse_sa_for_vu`, `-cgra_cluster`, and optional per-op latency/II controls.
Output: Baseline mode and RSA mode can be switched from CLI without code edits.

3. Extend systolic-array execution modes.
Action: Add an SA execution mode for streaming CGRA jobs in `SysArrayState`, while keeping current WS/OS behavior unchanged when RSA is disabled.
Output: SA can execute both matmul tiles and CGRA-style vector/reduction kernels.

4. Introduce CGRA job abstraction.
Action: Add a job type (for example `CgraJob`) carrying dimensions, operation type, streaming parameters, and read/write volume.
Output: A unified way to schedule reduction, elementwise, normalization, and softmax sub-steps on SA.

5. Lower vector layers to CGRA jobs (flag-gated).
Action: In layer lowering, map `Softmax`, `LayerNorm`, and `Activation` to CGRA-job sequences when RSA is enabled; keep legacy VU path for A/B comparison.
Output: Same model input file can run in both old and new execution paths.

6. Model softmax decomposition explicitly.
Action: Represent softmax as sub-steps: row-max reduce, subtract, exp-approx, row-sum reduce, normalize.
Output: Realistic timing breakdown for the most VU-heavy attention stage.

7. Add scheduling and state-trace visibility.
Action: Enforce SA mutual exclusion across matmul/CGRA tasks, add reconfiguration delay states, and expose new states in VCD.
Output: Correct resource contention behavior and debuggable timing traces.

8. Validate with microbenchmarks first.
Action: Run focused tests for elementwise ops, reductions, layernorm, and softmax to verify cycle accounting and stalls.
Output: Confidence that RSA timing model is internally consistent before full workloads.

9. Run end-to-end attention comparisons.
Action: Compare Baseline (SA+VU) vs RSA (SA-reuse) on identical workloads/configs and report total cycles, utilization, stalls, and per-layer latency.
Output: System-level performance delta and bottleneck analysis for the reconfigurable design.

10. Refine hotspots only.
Action: Improve fidelity only where needed (typically softmax/normalization/reduction) after first comparison results.
Output: Fast iteration loop with focused model complexity.

## Success Criteria

1. Baseline behavior is unchanged when RSA mode is disabled.
2. RSA mode runs full target workloads without functional scheduling errors.
3. Reports include apples-to-apples cycle/per-layer comparisons between baseline and RSA modes.
