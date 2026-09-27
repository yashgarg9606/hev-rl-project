# Full-repository audit remediation

This record follows the full-repository audit on 27 September 2026. It separates
software corrections from scientific questions that the available code and data
cannot settle. Existing datasets, checkpoints, result files, figures and papers
are preserved. No new real-data model training is required for this corrective
pass; small synthetic training fixtures test evaluation isolation.

## Acceptance and review

Work follows a planner–executor–critic loop: reproduce the failure, make a bounded
change, run focused regressions, obtain independent review, and check preserved
behavior. An issue is closed only for its stated contract and tested conditions.
The original experiment manifests remain immutable source/data snapshots.

The independent critic replayed every saved action from the six-cycle, five-arm
EMS benchmark through the corrected core. All **43,930 intervals** reproduced
the saved power, current, SOC, component SOH, elapsed time, reward, executed split,
feasible bounds and termination flags exactly. This protects the recorded
default-run comparisons; it does not validate every parameter configuration or
the physical calibration of the simulator.

## Software corrections

| Finding | Correction and scope | Evidence / status |
|---|---|---|
| A later observation could raise after the plant and episode had advanced. | Validate the cycle's observation demands before starting; restore model-owned physical and episode state if a step raises. Caller-owned external side effects are outside the rollback contract. | Verified with the 0–10–50 km/h failure and injected health, lookup, reward, observation and constraint exceptions in [atomic-step tests](ems_yash/src/environment/test_atomic_step.py). |
| Rated motor efficiency 1 produced a zero lifetime-loss denominator; other accepted values could produce a nonfinite or zero budget. | Reject singular/nonfinite lifetime configurations before stepping, including overflow and underflow of the derived budget. | Independently verified by [atomic-step tests](ems_yash/src/environment/test_atomic_step.py). Default aging equations are unchanged. |
| Roundoff near a singleton feasible split could assign torque to a zero-capacity motor. | Check candidate boundaries with the same motor-limit tolerance used for execution. | Verified for zero/near-zero capacities, both motors, traction/regeneration and adjacent floating-point boundaries. |
| The DDPG constructor silently selected 50-wide fully connected layers while component defaults specified 64. | Default fully connected width is 64; recurrent width remains 50. Both are explicit constructor parameters. `hidden_dim=50` retains the earlier fully connected configuration. | Six DDPG tests pass, including actual optimizer updates and explicit legacy configuration. |
| Legacy episode reports confused normalized preferences with executed torque splits, used a flag's presence as its value, or omitted final-step changes. | Record normalized action separately from executed split; use the violation value; account for executed intervals and final state. | [Legacy-contract regressions](ems_yash/test_legacy_contracts.py) and independent review pass, including rejected steps, final states and variable time steps. |
| Trace generation used the wrong project path, combined cells, assigned unsupported elapsed times and lacked an explicit model contract. | Export one cell with paired targets/IDs, caller-supplied timestamps, isolated model namespaces and a new output file. Distinguish historical checkpoint inference from a held-out frozen predictor and use the saved fold's history length. | [Export regressions](ems_yash/test_bms_trace_export.py) cover model coexistence, metadata/schema guards, timestamp IDs, model selection, saved-fold inference and overwrite refusal. Independent review passed. |
| Historical training/evaluation entrypoints used overlapping splits or overwrote fixed artifacts. | Retire those command-line paths in favor of explicit held-out runners and safe inference; preserve their old outputs as historical evidence. | The command guards direct users to the purged runners; all new archive writers refuse existing files. The historical algorithms are retained as source history, not recommended research entrypoints. |
| Uploaded preprocessing conflated voltage-window coverage and target meanings. | Validate coverage masks and channel order; identify provided unverified labels and discharge-window capacity proxies; preserve old arrays. Canonicalize converter-style cell filenames and record source segment/cycle IDs. | [Spatial-protocol tests](battery-health-cnn-tcn-lstm-main/src/evaluation/test_spatial_protocol.py) verify a partial voltage window, derivative values/order, HNEI proxy metadata, source rows and immutable archive writes. Independent review passed. |
| Uploaded evaluation/plotting callers assumed old NASA identifiers or model outputs; some figure titles asserted unsupported Oxford/CALCE provenance. | Use a four-channel, three-output-compatible held-out protocol; separate target families; require explicit acknowledgment of uncertified labels; label future plots by their actual source. | Seven spatial tests include actual synthetic neural fitting with altered held-out labels and unchanged fitted models/predictions. Mixed schemas are rejected. Plot review verified provenance remains unknown when absent, and existing output directories are refused. |
| Some source-cycle extractors required missing WLTP files. | Accept an explicit source path and fail with an actionable error when it is absent; refuse existing outputs. | [Legacy-contract tests](ems_yash/test_legacy_contracts.py) cover missing sources, invalid values/counts and output preservation. Missing source data are not manufactured. |
| Documentation said raw NASA data were absent and treated different BMS variants as one project. | Describe the newly uploaded raw data, separate model/data contracts and preserve historical result scope. | All root-document local links resolve; the README names explicit safe entrypoints and remaining research boundaries. |

## Scientific and provenance questions that remain open

- **HalfCycle labels:** the historical builder measures capacity over an observed
  discharge window. A partial-window/full-window ratio is not evidence of the
  same percentage of irreversible full-cell capacity loss. Re-labeling a proxy
  does not calibrate it. Comparable reference-capacity measurements and an
  explicit target definition are still required.
- **Provided Kaggle labels:** the supplied CSVs contain labels, but their origin
  and measurement/calibration process are not established by repository code.
  Do not call them synthetic or experimentally validated without additional
  evidence. The supplied multichemistry CSV does not establish Oxford or CALCE
  provenance.
- **Raw HalfCycle conversion:** correcting filename interpretation does not
  independently validate every MATLAB table's conversion to CSV. The available
  CSV columns and segment/cycle identifiers remain the preprocessing input
  contract; full raw-to-CSV measurement lineage requires separate verification.
- **Feature changes and checkpoints:** corrected future preprocessing does not
  retroactively change the features used to train an existing checkpoint.
  Dataset schemas and fitted preprocessing must remain paired with the model.
  Preserved historical predictions are not upgraded to held-out scores.
- **Battery and motor aging:** inherited equations contain calibration and energy
  accounting inconsistencies. Electrical capacity/resistance do not evolve from
  SOH, and temperature is configured rather than solved by a validated thermal
  model. These remain simulation surrogates; this pass does not invent replacement
  coefficients or claim measured lifetime accuracy.
- **Motor maps and integration:** the simulator still uses configured flat torque
  limits and digitized efficiency maps, including extrapolation. Midpoint motor
  speed fixes the launch-work defect but does not remove road-load or electrical
  timestep sensitivity.
- **BMS–EMS alignment:** exporting a timestamped trace enables explicit offline
  replay. It does not establish a causal online estimator, identify NASA cell
  cycles with vehicle seconds, or calibrate cell SOH to the simulated pack.
  Estimated-SOH mode still affects observations, rewards and SOH constraint
  checks; the controlled-bias benchmark separately keeps physical truth.
- **Research evaluation:** four NASA cells, one neural seed and a limited epoch
  budget remain a pilot. The new uploaded evaluation capability needs a declared
  real-data experiment after target/provenance decisions. No new trained EMS
  controller, paper reproduction or publication-readiness claim follows from
  these software corrections.

## Preserved result interpretation

The original [research report](RESEARCH_PROGRESS.md) retains its measured tables:
ridge outperformed the declared hybrid pilot under the same NASA protocol, and
the greedy motor-power grid outperformed fixed normalized action in the six
recorded simulated cycles. The separate previous-measured-SOH diagnostic has
privileged access to past ground-truth labels. None of these claims is expanded
by the current repairs.

The independent review found no remaining implementation blocker in the reviewed
corrections. Final validation passed:

- **89 distinct unit tests:** 60 EMS, 13 original BMS protocol, nine trace-export
  and seven uploaded spatial-protocol tests.
- Eight additional legacy scripts, 18 command-line help checks and 24 retired
  entrypoint guards.
- All 173 Python files compiled in memory; root-document local links resolve and
  `git diff --check` passes.
- The critic's 30-episode replay reproduced all 43,930 saved EMS transitions
  exactly, with maximum checked numerical error zero.
- **All 280 protected pre-existing artifacts retain their original SHA-256
  hashes**, with no mismatches. This includes the uploaded datasets/checkpoints
  and historical figures/results, as well as the original projects' artifacts.

The fresh [verification record](research_runs/remediation_verification_20260927/preservation.json)
contains the artifact fingerprints and validation counts. It is a remediation
record, not a new research experiment. Historical manifests and measured result
tables remain unchanged; no real-data model was retrained for this pass.
