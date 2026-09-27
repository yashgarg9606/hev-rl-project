# Research reliability and improvement plan

This plan tracks the sequential enhancement requested on 2026-09-27. Changes are
reviewed by a planner, an executor and an independent critic. A passing check is
evidence only for the behavior it exercises, not for publication readiness.

## Preservation and evaluation rules

- Preserve existing PDFs, datasets, checkpoints, figures and result files.
- Write new experiments to separate directories; do not replace historical runs.
- Correct implementation and evaluation defects before interpreting new scores.
- Distinguish simulated degradation from measured battery lifetime.
- Do not tune configurations against the outer held-out battery.
- Report unfavorable results and incomplete checks alongside improvements.

## Baseline

The initial working tree contained only the two untracked base-paper PDFs.
SHA-256 fingerprints of 63 existing research artifacts were captured before edits.

Seventeen selected legacy EMS scripts were executed with the existing Python 3.11
environment: 11 passed and 6 failed. Four failures were package-import errors
(`test_integrated_health`, `test_integrated_powertrain`, `test_motor_power`,
`test_networks`). Two tests retained obsolete assumptions: zero environment reward
and an unimplemented estimated-SOH interface. These are pre-existing failures;
future reports must distinguish them from regressions.

Independent checks using the actual motor maps reproduced stuck estimated-SOH
observations, history-dependent trace lookup, and net battery charging during a
0-to-10-to-0 km/h start/stop pair. These require separate regression tests.

## Ordered work

1. **SOH time propagation and trace lookup.** Make trace lookup independent of
   query order and shared users; propagate elapsed simulation time through the
   powertrain. Verify timestamp crossings, variable steps, reset, rejected steps,
   and physical-mode equivalence. Do not change reward semantics in this slice.
2. **RL action semantics and complete feasibility.** Keep replay, history, actor
   and critic actions normalized. Map to the complete feasible torque interval
   only at the environment boundary. Cover terminal transitions and custom motor
   parameters. Preserve the current flat-limit model rather than silently adding
   a different torque-envelope model.
3. **Energy accounting.** Correct the interval power approximation responsible
   for free launch energy. Check mechanical work, charge/discharge signs, steady
   operation and timestep sensitivity before interpreting energy savings.
4. **Reproducible BMS evaluation.** Add isolated experiments with outer battery
   holdout, purged temporal validation, training-only preprocessing, simple
   baselines, deterministic seeds and explicit provenance. Existing scores remain
   historical evidence and are not directly comparable to a changed protocol.
5. **Matched integration experiments and presentation.** Compare estimates with
   their paired truth; distinguish replayed estimates from online plant inference.
   Use complete dynamic cycles and report energy, modeled aging, violations and
   completed distance. Produce a report from measured outputs, including limits.

Each slice follows: failing baseline -> bounded implementation -> focused tests ->
critic review -> corrections if needed -> regression check -> next slice.

## Verified progress

- **SOH timing completed:** 17 repository regression tests passed under Python
  3.11. The critic independently checked 15 cases with the actual motor maps and
  confirmed exact physical-mode baseline parity for the recorded three-step
  trajectory. The shipped adapter and interface smoke checks also pass.
- **Action contract completed:** 29 combined regression tests passed. The critic
  checked actual Adam updates and 3,102 motor-boundary cases. Replay, actor and
  critic now use normalized actions; execution maps the complete feasible
  interval. Rejected pre-step transitions retain the cycle index, state and clock.
  A legacy assertion expecting the index to advance on rejection was corrected
  and strengthened to compare the complete unchanged physical state.
- **Interval energy completed:** 36 combined regression tests passed, together
  with the end-to-end, full-episode, training-loop and training-stability scripts.
  The critic independently verified kinetic-energy work balance, unchanged
  steady-speed values, radius consistency and rejection of infeasible endpoints
  before state mutation. The actual-map 0-to-10-to-0 km/h pair now consumes
  2.888856 Wh and ends at SOC 0.599842828 from 0.6. Power uses midpoint speed;
  existing road-load and aging formulas remain unchanged. Electrical energy
  remains sensitive to timestep and interpolated efficiency-map sampling.
- **BMS evaluation implementation completed:** 13 distinct protocol tests pass.
  The critic independently verified raw cycle-support separation and trained two
  synthetic hybrid runs with changed held-out labels: fitted preprocessing,
  checkpoint selection, model weights and predictions were identical. The real
  pilot uses all four battery holdouts, the unchanged hybrid architecture, seed
  42, at most 40 epochs and the frozen ridge grid `[0.001, 0.01, 0.1, 1, 10]`.
  Artifacts are isolated in `research_runs/bms_purged_lobo_seed42/`.
- **BMS real-data pilot completed and audited:** all four folds finished. Ridge
  macro RMSE is 0.067535007; the unchanged hybrid pilot is 0.108407011. All 12
  saved fold/model predictors reproduce paired predictions exactly in the tested
  runtime. Negative results and the distinct privileged persistence diagnostic
  are reported, without retuning against held-out scores.
- **Controlled EMS study completed and audited:** 11 new benchmark tests pass.
  All six cycles and all five declared controllers completed (30 runs, 43,930
  intervals), with equal distance per cycle and no constraint violations. The
  energy-grid reference uses 7.13–10.41% less simulated energy than fixed action;
  this is not a trained-RL or real-lifetime claim. Artifacts are isolated in
  `research_runs/ems_controlled_sensitivity_v1/`.
- **Presentation and preservation completed:** `RESEARCH_PROGRESS.md` reports
  measured tables, limitations and ordered next work. The root README now matches
  the implemented model/data contracts and working commands. All 63 protected
  original artifacts retain their SHA-256 hashes; see
  `research_runs/preservation_check.json`. The final checks total 60 distinct unit
  tests plus 21 selected legacy/integration scripts, all passing.
- The existing BMS checkpoint was loaded without alteration and evaluated on all
  556 stored sequences: MAE 0.054061814, RMSE 0.065812629, R2 0.567515135, with
  150,465 parameters. This confirms the historical artifact; these are still
  all-data scores, not independent held-out accuracy.

## Research boundary

Before further model tuning, clarify whether the task is current-SOH estimation
or next-cycle forecasting and whether preceding measured capacity is available.
The separately reported persistence diagnostic shows that this information
contract materially changes the comparison. The completed feature-only pilot
retains its originally declared inputs and protocol.

Inherited aging equations, battery calibration and a thermal model require
separate scientific decisions. Fixing software defects alone does not validate
those assumptions. The potential research question is how SOH estimation errors
and update delays affect a dual-motor EMS; it is not yet an established result.
