# BTP research progress and evidence

Date: 27 September 2026. This report records the five initial enhancements and
their completed, independently audited experiments. The later full-repository
audit found additional edge-case software defects and issues in the separately
uploaded BMS project. The measured tables below remain snapshots of the original
declared experiments; they are not a claim that every repository path is correct.
The [audit-remediation record](REMEDIATION.md) tracks the subsequent corrections
and their independent checks.
That corrective pass passed 89 unit tests and preserved all 280 protected
pre-existing artifacts. Its exact replay of the 30 saved EMS episodes supports
retaining the tables below under their original stated simulation scope.

The strongest feature-only BMS model in this pilot is ridge: macro RMSE 0.067535,
37.70% below the unchanged hybrid pilot. The EMS one-step grid reference reduces
simulated net electrical energy by 7.13–10.41% versus a fixed normalized action
across six complete driving cycles. These are scoped benchmark findings suitable
for professor review; the scientific limitations below remain material.

## What changed

The repository now has corrected simulation/learning contracts and a separately
auditable BMS evaluation protocol. These are prerequisites for interpreting
research results; they do not establish publication readiness by themselves.

| Area | Observed issue before correction | Implemented correction and evidence |
|---|---|---|
| SOH observations | Simulation did not propagate elapsed time to trace lookup; lookup depended on prior query order. | Elapsed time advances with successful physical steps; timestamp lookup is independent of query order. Variable-step, reset, interleaved-environment and rejected-step tests pass. |
| RL actions | Replay/current-critic and actor/target-critic paths used different action coordinates. | Replay, history and all learning paths use normalized actions; only the environment maps to physical torque split. Real optimizer updates and terminal Bellman targets were checked. |
| Motor feasibility | Mapping only a lower split bound could exceed Motor 1 torque capacity. | Analytic lower and upper bounds use the configured gear ratios and flat torque/speed limits. At 2,500 Nm, a midpoint normalized action now executes sigma 0.710 instead of the infeasible 0.814 produced by the previous lower-bound mapping. |
| Interval energy | A launch from rest used zero starting speed for power, costing no energy before regeneration on braking. | Power uses interval-average speed, and endpoint feasibility is checked before mutation. Road-load-free mechanical work matches kinetic-energy change; the actual-map 0–10–0 km/h pair with one-second transitions now consumes 2.888856 Wh and ends at SOC 0.599842828 from 0.6. |
| Evaluation | Historical all-data scores and overlapping inner splits could not support the intended new comparison. | New battery holdouts, purged chronological validation, train-only scaling, simple baselines, paired predictions and saved provenance provide a defined comparison protocol. |

Implementation: [SOH adapter](ems_yash/src/bms_interface/soh_trace_adapter.py),
[powertrain](ems_yash/src/environment/integrated_powertrain.py),
[motor bounds](ems_yash/src/environment/motor_model.py),
[environment](ems_yash/src/environment/rl_environment.py), and
[DDPG updates](ems_yash/src/agent/ddpg_agent.py).

Steady-speed baseline values were preserved. Transient simulation results are
intentionally different after correcting energy accounting; historical transient
results should not be interpreted as outputs of the corrected model. Network
parameter shapes remain loadable, but this does not validate the learned meaning
of an old critic trained with inconsistent action coordinates.

## BMS: independent battery evaluation

The [new protocol](bms_gtr/src/evaluation/purged_lobo.py) exactly reconstructs the
556 stored sequences from B0005/B0006/B0007/B0018. Each input contains 20 retained
discharge cycles with three derivative channels and 300 voltage-grid points; its
target is the next retained cycle's SOH. SOH is capacity divided by that battery's
first usable capacity, as implemented in
[dataset construction](bms_gtr/scripts/build_dataset.py).

One whole battery is held out in each fold. Within every other battery, the final
20% of sequence rows forms validation and the preceding 20 rows are purged. This
separates both input and target cycle supports across training and validation.
Scaling uses retained training inputs only. Original source record IDs are saved;
they are not converted into invented elapsed times.

| Held-out battery | Training rows | Purged rows | Validation rows | Test rows |
|---|---:|---:|---:|---:|
| B0005 | 265 | 60 | 83 | 148 |
| B0006 | 265 | 60 | 83 | 148 |
| B0007 | 265 | 60 | 83 | 148 |
| B0018 | 294 | 60 | 90 | 112 |

The training-mean baseline, ridge model and unchanged 150,465-parameter hybrid
share the same splits. Ridge uses the predeclared alpha grid
`[0.001, 0.01, 0.1, 1, 10]`, selected on validation MSE. Its centered flattened
standardized inputs are divided by sqrt(feature count); its objective is summed
squared error plus alpha times squared coefficient norm. It is not refitted on
validation rows after selection.

The hybrid pilot uses seed 42, CPU with two threads, batch size 64, Adam learning
rate 0.001, gradient clipping 5, at most 40 epochs, and early-stopping patience 10.
The minimum-validation-MSE state is restored before test prediction. Test labels
do not select hyperparameters or checkpoints. This is one declared pilot, not a
convergence or stochastic-repeatability study.

| Model | Macro MAE | Macro RMSE | Macro R² |
|---|---:|---:|---:|
| Training mean | 0.094312 | 0.108238 | -0.590068 |
| Ridge | 0.058063 | 0.067535 | 0.351526 |
| CNN–TCN–LSTM–attention pilot | 0.087854 | 0.108407 | -0.892245 |

| Held-out battery | Mean RMSE | Ridge RMSE | Hybrid RMSE | Ridge R² | Hybrid R² |
|---|---:|---:|---:|---:|---:|
| B0005 | 0.096555 | 0.057566 | 0.090491 | 0.633606 | 0.094626 |
| B0006 | 0.180872 | 0.105501 | 0.112424 | -0.009558 | -0.146417 |
| B0007 | 0.078572 | 0.044023 | 0.099841 | 0.676915 | -0.661782 |
| B0018 | 0.076953 | 0.063050 | 0.130871 | 0.105142 | -2.855408 |

Ridge has **37.70% lower macro RMSE than the hybrid pilot** under this
same protocol. It is the strongest of the three declared feature-only models.
The hybrid has negative R² on three batteries, and ridge still has negative R² on
B0006. These findings support using a strong simple baseline and investigating
generalization before adding neural complexity. Pooled RMSE is 0.071844 for
ridge and 0.107924 for the hybrid; these are different aggregations from
the macro values above.

A **separate diagnostic with privileged past labels** predicts each target from
the preceding retained discharge's measured SOH. On the same 556 targets it gives
pooled MAE **0.005733**, RMSE **0.009677**, and R² **0.990649**. This assumes a previous
ground-truth capacity measurement from the evaluated cell, which is not an
explicit input to the feature-only models. It is excluded from their primary
ranking and does not establish deployment availability. Its relevance is that
next-cycle SOH is highly persistent: the research must clarify whether previous
capacity is known and whether the intended task is estimation or forecasting.
See the [diagnostic script](bms_gtr/scripts/diagnose_previous_soh.py),
[its information protocol](research_runs/previous_measured_soh_diagnostic/manifest.json),
and [paired diagnostic predictions](research_runs/previous_measured_soh_diagnostic/predictions.csv).

Scores are SOH fractions: RMSE 0.01 means one percentage point. Macro metrics
weight the four batteries equally; pooled metrics weight all test predictions.
Negative R-squared values are retained. The ridge alpha was selected at the lower
edge of the frozen grid in every fold; a broader search is a future separately
declared experiment.

Artifacts: [manifest](research_runs/bms_purged_lobo_seed42/manifest.json),
[paired predictions](research_runs/bms_purged_lobo_seed42/predictions.csv),
[summary](research_runs/bms_purged_lobo_seed42/summary.json), and
[fold-specific models, scalers and histories](research_runs/bms_purged_lobo_seed42/folds/).
Raw NASA files for these four cells are now available in the separately uploaded
project. This completed run still starts from the supplied processed archives.
The raw-data audit matched capacities, SOH labels and source cycle IDs, and found
feature agreement within the checked numerical tolerance. That agreement does
not independently establish measurement quality or vehicle-pack applicability.

## EMS: controlled information and optimization comparisons

The [new runner](ems_yash/run_controlled_benchmark.py) uses every supplied row of
NEDC, UDDS, HWFET, WLTC Class 3b, CLTC-P and FTP75, including their original grades
and timestamps. Each arm starts from the same default plant: SOC 0.6, 72 Ah,
255.5 V nominal OCV and 25 degrees C. The plant, reward and constraint checks use
physical simulated SOH. Only a copy of the controller observation receives bias.

The declared arms are fixed normalized action 0.5; the rule
`a = 0.5 + 0.3 * (1 - observed_SOH)` with correct SOH; the same rule with SOH
underestimated by 0.02 or 0.05; and a one-step electrical-energy reference over
101 normalized float32 actions. Every grid candidate is evaluated through the
production motor/power models. The first minimum is chosen deterministically.
This reference uses the interval demand already available to the environment;
it is not dynamic programming, a global optimum, or a learned RL controller.

Net electrical energy is discharged energy minus recovered energy. Distance uses
interval-average velocity, and RMS current is weighted by elapsed time. A rejected
pre-step transition adds no energy, time or distance; a post-step violation counts
the executed interval but marks the cycle incomplete. Comparisons require full,
equal-distance completion. Physical modeled aging is logged separately for the
battery and both motors.

All **30 of 30** cycle/controller combinations completed their full trajectories,
with equal distance within each cycle and no recorded constraint violations.

Net electrical energy, kWh:

| Cycle | Distance km | Fixed a=0.5 | Grid reference | Grid reduction versus fixed |
|---|---:|---:|---:|---:|
| NEDC | 11.013 | 1.520302 | 1.368810 | 9.96% |
| UDDS | 11.990 | 1.551819 | 1.390328 | 10.41% |
| HWFET | 16.507 | 2.325034 | 2.159332 | 7.13% |
| WLTC_Class3b | 23.266 | 3.435249 | 3.175123 | 7.57% |
| CLTC_P | 14.480 | 1.874014 | 1.694788 | 9.56% |
| FTP75 | 17.769 | 2.330561 | 2.099526 | 9.91% |

SOH-rule sensitivity, net electrical energy in kWh:

| Cycle | Correct SOH | SOH bias −0.02 | SOH bias −0.05 |
|---|---:|---:|---:|
| NEDC | 1.520303 | 1.520161 | 1.519888 |
| UDDS | 1.551813 | 1.551132 | 1.550252 |
| HWFET | 2.325034 | 2.325005 | 2.324807 |
| WLTC_Class3b | 3.435254 | 3.434776 | 3.433991 |
| CLTC_P | 1.874009 | 1.873258 | 1.871976 |
| FTP75 | 2.330551 | 2.329676 | 2.328484 |

The grid reference reduces simulated energy by **7.13–10.41%** relative to the fixed
normalized action. This establishes a comparator for future trained-controller
work, not an RL improvement claim. Lower total energy does not guarantee lower
peak current: on HWFET, grid peak current is 143.931 A versus 142.824 A for the
fixed controller. Component-aging and current metrics are retained in the full
summary rather than converted into unvalidated lifetime-extension claims.

Underestimating SOH by five percentage points slightly *reduces* this rule's net
energy, by approximately 0.010–0.108% across these cycles. Correct information
therefore does not automatically make this particular rule energy-optimal. These
small changes are model-specific and need numerical/physical sensitivity checks;
they are not evidence that an inaccurate estimator is preferable in practice.

Artifacts: [manifest and complete configuration](research_runs/ems_controlled_sensitivity_v1/manifest.json),
[all summary metrics](research_runs/ems_controlled_sensitivity_v1/summary.csv), and
[per-step records](research_runs/ems_controlled_sensitivity_v1/steps/).

## Validation and preservation

The initial selected legacy check set had 11 passes and six failures. After the
corrections, all 17 pass. Four additional end-to-end/full-episode/training-loop/
training-stability scripts also pass. All **60 distinct unit tests** pass: 36
focused EMS tests, 13 BMS protocol tests and 11 controlled-benchmark tests.
The independent critic additionally checked 3,102 motor
boundary cases, mechanical work, steady-speed parity and actual optimizer updates.

The critic's synthetic BMS experiment changed only held-out labels and obtained
identical fitted scalers, model parameters, validation histories, selection
decisions and predictions; only test metrics changed. Saved-model prediction
reproduction and output-overwrite refusal are regression-tested.

The final artifact audit independently recomputed BMS metrics from all 556 paired
predictions and reloaded all 12 saved fold/model predictors. Each reproduced its
saved predictions exactly in the recorded runtime. The EMS audit recomputed all
summary metrics from 43,930 completed intervals, checked state continuity and
original cycle coordinates, and found zero predicted/executed grid-power error.
At the artifact audit, all recorded source, data, map and cycle hashes matched
the files. Subsequent corrective source changes intentionally differ from these
immutable historical manifests; the original datasets and result artifacts are
preserved. A source-hash difference must not be represented as a newly rerun
experiment or repaired by rewriting the old manifest.

All **63 original research artifacts remain byte-for-byte unchanged**:
[before/after SHA-256 verification](research_runs/preservation_check.json).
Original results, datasets, PDFs and checkpoints were preserved; new results have
separate directories. Source-code changes are intentional and tested; no universal
absence-of-regressions claim is implied. `git diff --check` passes.

Reproduction commands and dependency requirements are in [README.md](README.md).
For the separate persistence diagnostic, from the repository root:

```sh
python -B bms_gtr/scripts/diagnose_previous_soh.py \
  --output-dir research_runs/my_previous_soh_diagnostic
```

## What this supports, and what remains

These changes support a professor discussion about a reproducible evaluation
protocol, corrected simulator behavior, useful baselines and the effect of
controller information. They do not yet support a claim of paper-level novelty,
real lifetime extension or reproduction of either base paper's headline results.

The main remaining limitations are grounded in the implementation:

- Only four processed cells and one neural seed are evaluated. Correlated windows
  are not independent experimental units. The six driving profiles also share
  structure, particularly UDDS and FTP75.
- Aging remains an inherited simulation surrogate. Battery nominal capacity and
  resistance are not updated from evolving SOH in
  [the electrical model](ems_yash/src/environment/battery_model.py), and temperature
  is configured rather than evolved by a validated thermal model.
- Motor efficiency is digitized and extrapolated outside the contour hull;
  regeneration reuses the absolute-torque efficiency map. Midpoint speed fixes
  the launch defect but does not make the complete model exact. A 20-second
  0–20–0 km/h diagnostic consumed 10.3323, 10.3205 and 10.2508 Wh at time steps
  1, 0.5 and 0.25 seconds respectively. Small energy differences need numerical
  sensitivity analysis before physical interpretation.
- The observation-bias experiment is not a learned NASA-estimator feedback test.
  Cell-to-vehicle domain mapping and time alignment remain unresolved. In the
  general estimated-SOH environment mode, estimates also enter rewards and SOH
  constraints; the controlled experiment explicitly holds these to physical SOH.
- There is no trained EMS checkpoint. The corrected DDPG needs a declared training
  protocol and held-out cycle comparisons with strong control baselines.

Next tasks, in order: clarify estimation versus next-cycle forecasting and whether
previous measured capacity is available; then repeat the appropriate BMS protocol
across predeclared seeds and simple architecture ablations; validate the newly
available raw data and expand independent cell evaluation; calibrate physical/aging models and
quantify timestep sensitivity; define a justified learned-estimator/plant
alignment; then train and compare EMS controllers with energy, constraint and
health objectives stated in advance.

The original reports remain historical evidence. In particular, the original BMS
checkpoint's all-556-sequence MAE 0.054061814 / RMSE 0.065812629 / R-squared
0.567515135 includes training data. It is not a fair comparator for the new
held-out protocol. The legacy 4.67% fresh-plant/held-estimate mismatch is not a
paired NASA estimation-error measurement.
