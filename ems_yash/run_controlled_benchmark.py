"""Controlled observation sensitivity and an instantaneous power grid reference.

Every arm uses TRUE_PHYSICAL plant health, constraints, and rewards. Synthetic
SOH biases affect only the copied observation passed to the hand-written rule.
This benchmark contains no trained controller or learned BMS estimator.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.feasible_action_mapper import map_action
from src.bms_interface import SOHSource
from src.environment.integrated_powertrain import IntegratedPowertrainParameters, default_motor_map_paths
from src.environment.rl_environment import EnergyManagementEnv


CYCLE_NAMES = ("NEDC", "UDDS", "HWFET", "WLTC_Class3b", "CLTC_P", "FTP75")
GRID_ACTIONS = np.linspace(0., 1., 101, dtype=np.float32)


@dataclass(frozen=True)
class Controller:
    name: str
    kind: str
    soh_bias: float = 0.


CONTROLLERS = (
    Controller("fixed_0p5", "fixed"),
    Controller("health_rule_true", "health_rule"),
    Controller("health_rule_bias_minus_0p02", "health_rule", -.02),
    Controller("health_rule_bias_minus_0p05", "health_rule", -.05),
    Controller("instantaneous_grid_101", "grid"),
)


def load_cycle(path):
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or ()) != {"time_s", "velocity_kmh", "slope_rad"}:
            raise ValueError("Cycle CSV must contain time_s, velocity_kmh, slope_rad")
        return np.asarray([[float(row[name]) for name in ("time_s", "velocity_kmh", "slope_rad")]
                           for row in reader], dtype=np.float64)


def observed_state(state, soh_bias=0.):
    if not np.isfinite(soh_bias):
        raise ValueError("Observation bias must be finite")
    observed = np.asarray(state, dtype=np.float32).copy()
    if observed.shape != (6,) or not np.isfinite(observed).all():
        raise ValueError("Expected a finite six-dimensional observation")
    biased_soh = float(observed[3]) + soh_bias
    if not 0. <= biased_soh <= 1.:
        raise ValueError("Biased observation SOH must remain within [0, 1]")
    observed[3] = np.float32(biased_soh)
    return observed


def transition_context(env):
    index = env.current_index
    if index >= len(env.cycle) - 1:
        raise RuntimeError("No transition remains")
    v0, v1 = float(env.cycle[index, 1]), float(env.cycle[index + 1, 1])
    dt = float(env.cycle[index + 1, 0] - env.cycle[index, 0])
    torque = env.powertrain.vehicle.calculate_wheel_torque(v0, v1, float(env.cycle[index, 2]), dt)
    bounds = env.powertrain.motors.calculate_feasible_sigma_bounds(max(v0, v1), torque)
    return dict(v0=v0, v1=v1, dt=dt, torque=torque, bounds=bounds)


def candidate_motor_power(env, action, context=None):
    """Evaluate actual production components without advancing physical state."""
    context = transition_context(env) if context is None else context
    if context["bounds"] is None:
        return None
    # Match the environment's action conversion before physical mapping.
    normalized = float(np.float32(action))
    sigma = float(map_action(normalized, *context["bounds"]))
    motors = env.powertrain.motors.calculate_operating_point(
        .5 * (context["v0"] + context["v1"]), context["torque"], sigma,
    )
    motor1 = env.powertrain.motor1_power.calculate(motors["motor1_speed_rpm"], motors["motor1_torque_nm"])
    motor2 = env.powertrain.motor2_power.calculate(motors["motor2_speed_rpm"], motors["motor2_torque_nm"])
    return (motor1.electrical_power_w + motor2.electrical_power_w) / 1000.


def grid_action(env, context=None):
    context = transition_context(env) if context is None else context
    if context["bounds"] is None:
        # Let the environment report a normal controlled rejection.
        return np.array([.5], dtype=np.float32), None
    powers = np.asarray([candidate_motor_power(env, action, context) for action in GRID_ACTIONS])
    if not np.isfinite(powers).all():
        raise ValueError("Grid reference produced nonfinite motor power")
    index = int(np.argmin(powers))  # First minimum, including zero-power ties.
    return np.array([GRID_ACTIONS[index]], dtype=np.float32), float(powers[index])


def select_action(env, state, controller, context=None):
    observed = observed_state(state, controller.soh_bias)
    prediction = None
    if controller.kind == "fixed":
        action = np.array([.5], dtype=np.float32)
    elif controller.kind == "health_rule":
        action = np.array([.5 + .3 * (1. - float(observed[3]))], dtype=np.float32)
    elif controller.kind == "grid":
        action, prediction = grid_action(env, context)
    else:
        raise ValueError(f"Unknown controller kind: {controller.kind}")
    return action, observed, prediction


def aggregate_steps(rows):
    """Integrate only intervals whose plant clock actually advanced."""
    completed = [row for row in rows if row["completed_step"]]
    elapsed = sum(row["dt_completed_s"] for row in completed)
    discharge = sum(max(-row["battery_power_kw"], 0.) * row["dt_completed_s"] / 3600. for row in completed)
    recovered = sum(max(row["battery_power_kw"], 0.) * row["dt_completed_s"] / 3600. for row in completed)
    distance = sum((row["velocity_before_kmh"] + row["target_velocity_kmh"]) / 2.
                   * row["dt_completed_s"] / 3600. for row in completed)
    return dict(attempted_transitions=len(rows), completed_transitions=len(completed),
        elapsed_s=elapsed, distance_km=distance, discharge_kwh=discharge,
        recovered_kwh=recovered, net_kwh=discharge - recovered,
        peak_abs_current_a=max((abs(row["battery_current_a"]) for row in completed), default=None),
        rms_current_a=float(np.sqrt(sum(row["battery_current_a"]**2 * row["dt_completed_s"]
            for row in completed) / elapsed)) if elapsed > 0. else None)


def run_episode(cycle, controller, initial_soc=.6, project_root=PROJECT_ROOT):
    started = time.perf_counter()
    env = EnergyManagementEnv(cycle, project_root=project_root,
        powertrain_parameters=IntegratedPowertrainParameters(initial_soc=initial_soc, soh_source=SOHSource.TRUE_PHYSICAL))
    try:
        state, _ = env.reset(seed=42)
        initial_health = env.powertrain.health_model.get_health_state()
        initial_soc = float(env.powertrain.battery.state.soc)
        rows = []
        for attempt in range(len(env.cycle) - 1):
            context = transition_context(env)
            index_before = env.current_index
            before_time = env.powertrain.simulation_time_s
            before_health = env.powertrain.health_model.get_health_state()
            before_soc = float(env.powertrain.battery.state.soc)
            action, observed, prediction = select_action(env, state, controller, context)
            state, reward, terminated, truncated, info = env.step(action)
            after_time = env.powertrain.simulation_time_s
            elapsed = after_time - before_time
            completed = elapsed > 0.
            if completed and not np.isclose(elapsed, context["dt"], rtol=1e-12, atol=1e-12):
                raise RuntimeError("Unexpected partial physical timestep")
            if completed and prediction is not None and not np.isclose(
                prediction, -info["battery_power_kw"], rtol=1e-12, atol=1e-10,
            ):
                raise RuntimeError("Grid prediction does not match executed motor power")
            after_health = env.powertrain.health_model.get_health_state()
            row = dict(attempt=attempt, cycle_index_before=index_before, cycle_index_after=env.current_index,
                completed_step=completed, dt_requested_s=context["dt"], dt_completed_s=elapsed,
                time_before_s=before_time, time_after_s=after_time,
                original_cycle_time_before_s=float(env.cycle[index_before, 0]),
                original_cycle_time_after_s=float(env.cycle[env.current_index, 0]),
                original_cycle_time_target_s=float(env.cycle[index_before + 1, 0]),
                slope_rad=float(env.cycle[index_before, 2]),
                velocity_before_kmh=context["v0"], target_velocity_kmh=context["v1"],
                wheel_torque_nm=context["torque"], normalized_action=float(action[0]),
                sigma_tor=info.get("sigma_tor"), sigma_min=info.get("sigma_min"), sigma_max=info.get("sigma_max"),
                soc_before=before_soc, soc_after=float(env.powertrain.battery.state.soc),
                battery_soh_before=before_health["battery_soh"], observed_battery_soh=float(observed[3]),
                battery_soh_after=after_health["battery_soh"],
                motor1_soh_before=before_health["motor1_soh"], motor1_soh_after=after_health["motor1_soh"],
                motor2_soh_before=before_health["motor2_soh"], motor2_soh_after=after_health["motor2_soh"],
                battery_power_kw=float(info["battery_power_kw"]) if completed else None,
                battery_current_a=float(info["battery_current_a"]) if completed else None,
                predicted_motor_power_kw=prediction if completed else None,
                reward=float(reward), terminated=bool(terminated), truncated=bool(truncated),
                constraint_violation=bool(info.get("constraint_violation", False)),
                constraint_stage=info.get("constraint_stage"), violated_constraints=list(info.get("violated_constraints", ())))
            rows.append(row)
            if terminated or truncated:
                break
        final_health = env.powertrain.health_model.get_health_state()
        summary = aggregate_steps(rows)
        last = rows[-1]
        completed_cycle = summary["completed_transitions"] == len(env.cycle) - 1 and not last["constraint_violation"] and not last["truncated"]
        reason = "cycle_complete" if completed_cycle else "constraint_violation" if last["constraint_violation"] else "truncated" if last["truncated"] else "incomplete"
        errors = [abs(row["predicted_motor_power_kw"] + row["battery_power_kw"]) for row in rows
                  if row["completed_step"] and row["predicted_motor_power_kw"] is not None]
        summary.update(controller=controller.name, controller_kind=controller.kind, soh_bias=controller.soh_bias,
            status="completed" if completed_cycle else "incomplete", completed_cycle=completed_cycle,
            completion_reason=reason, expected_transitions=len(env.cycle) - 1,
            initial_soc=initial_soc, final_soc=float(env.powertrain.battery.state.soc),
            initial_battery_soh=initial_health["battery_soh"], final_battery_soh=final_health["battery_soh"],
            initial_motor1_soh=initial_health["motor1_soh"], final_motor1_soh=final_health["motor1_soh"],
            initial_motor2_soh=initial_health["motor2_soh"], final_motor2_soh=final_health["motor2_soh"],
            battery_soh_loss=initial_health["battery_soh"] - final_health["battery_soh"],
            motor1_soh_loss=initial_health["motor1_soh"] - final_health["motor1_soh"],
            motor2_soh_loss=initial_health["motor2_soh"] - final_health["motor2_soh"],
            total_reward=sum(row["reward"] for row in rows),
            grid_max_prediction_error_kw=max(errors) if errors else None,
            constraint_stage=last["constraint_stage"], violated_constraints=last["violated_constraints"],
            runtime_s=time.perf_counter() - started)
        return rows, summary
    finally:
        env.close()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                          for key, value in row.items()} for row in rows)


def physical_configuration(project_root, initial_soc):
    """Serialize the actual initial components used for every fresh episode."""
    env = EnergyManagementEnv(np.array([[0., 0., 0.], [1., 0., 0.]]), project_root=project_root,
        powertrain_parameters=IntegratedPowertrainParameters(initial_soc=initial_soc, soh_source=SOHSource.TRUE_PHYSICAL))
    try:
        plant = env.powertrain
        parameters = asdict(plant.parameters)
        parameters["soh_source"] = plant.parameters.soh_source.value
        temperature = plant.parameters.battery_temperature_c
        return dict(powertrain=parameters, vehicle=asdict(plant.vehicle.params),
            motors=asdict(plant.motors.params), battery_electrical=asdict(plant.battery.parameters),
            battery_initial_state=asdict(plant.battery.state),
            initial_rc_parameters=dict(
                charge=plant.battery.electrical_parameters(current_a=1., temperature_c=temperature),
                discharge=plant.battery.electrical_parameters(current_a=-1., temperature_c=temperature)),
            battery_health=asdict(plant.health_model.battery_parameters),
            motor1_health=asdict(plant.health_model.motor1_parameters),
            motor2_health=asdict(plant.health_model.motor2_parameters),
            initial_health=plant.health_model.get_health_state(),
            constraints=asdict(env.constraint_checker.parameters), reward=asdict(env.reward_model.parameters))
    finally:
        env.close()


def run_benchmark(output_dir, cycles=None, controllers=CONTROLLERS, initial_soc=.6,
                  project_root=PROJECT_ROOT, progress=None):
    project_root, output_dir = Path(project_root), Path(output_dir)
    cycles = cycles if cycles is not None else {name: project_root / "data/raw" / f"{name}_raw.csv" for name in CYCLE_NAMES}
    if not cycles or not controllers or len({arm.name for arm in controllers}) != len(controllers):
        raise ValueError("Select nonempty cycles and unique controllers")
    for name in (*cycles, *(arm.name for arm in controllers)):
        if not name or any(not (character.isalnum() or character in "_-") for character in name):
            raise ValueError("Cycle and controller names must be simple identifiers")
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "steps").mkdir()
    manifest_path = output_dir / "manifest.json"
    manifest = dict(status="running", started_utc=datetime.now(timezone.utc).isoformat(),
        scope="synthetic observation sensitivity and an instantaneous grid reference; no trained policy or learned BMS",
        controllers=[asdict(arm) for arm in controllers], cycles=list(cycles),
        config=dict(soh_source=SOHSource.TRUE_PHYSICAL.value, initial_soc=initial_soc, seed=42,
            rule_base=.5, rule_sensitivity=.3, observation_dtype="float32", bias_application="additive copied observation only; out-of-range SOH rejected",
            grid_actions=GRID_ACTIONS.tolist(), grid_objective="minimum signed instantaneous motor electrical kW",
            grid_tie_break="first argmin", grid_execution_power_atol_kw=1e-10, grid_execution_power_rtol=1e-12,
            energy_units="kWh", soh_loss_units="fraction", incomplete_runs_comparable=False),
        versions=dict(python=platform.python_version(), numpy=np.__version__), episodes={})
    write_json(manifest_path, manifest)
    summaries = []
    started = time.perf_counter()
    try:
        import gymnasium
        import pandas
        import scipy
        import torch
        manifest["versions"].update(gymnasium=gymnasium.__version__, pandas=pandas.__version__,
                                    scipy=scipy.__version__, torch=torch.__version__)
        manifest["physical_configuration"] = physical_configuration(project_root, initial_soc)
        source_paths = sorted(set(project_root.glob("src/**/*.py"))
                              | set(project_root.glob("test_controlled_benchmark.py")) | {Path(__file__).resolve()})
        manifest["source_sha256"] = {str(path.relative_to(project_root)): file_sha256(path) for path in source_paths}
        manifest["map_sha256"] = {str(path.relative_to(project_root)): file_sha256(path) for path in default_motor_map_paths(project_root)}
        manifest["cycle_sha256"] = {name: dict(path=str(Path(path).resolve()), sha256=file_sha256(path)) for name, path in cycles.items()}
        manifest["episodes"] = {f"{cycle}__{arm.name}": dict(status="pending") for cycle in cycles for arm in controllers}
        write_json(manifest_path, manifest)
        for cycle_name, path in cycles.items():
            cycle = load_cycle(path)
            for controller in controllers:
                key = f"{cycle_name}__{controller.name}"
                manifest["episodes"][key]["status"] = "running"
                write_json(manifest_path, manifest)
                rows, summary = run_episode(cycle, controller, initial_soc, project_root)
                summary = dict(cycle=cycle_name, **summary)
                write_csv(output_dir / "steps" / f"{key}.csv", rows)
                summaries.append(summary)
                write_json(output_dir / "summary.json", summaries)
                write_csv(output_dir / "summary.csv", summaries)
                manifest["episodes"][key] = dict(status=summary["status"], completion_reason=summary["completion_reason"])
                write_json(manifest_path, manifest)
                if progress:
                    progress(f"{key}: status={summary['status']} net_kwh={summary['net_kwh']:.9g}")
        manifest.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat(), runtime_s=time.perf_counter() - started)
        write_json(manifest_path, manifest)
        return summaries
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}", runtime_s=time.perf_counter() - started)
        for episode in manifest["episodes"].values():
            if episode["status"] == "running":
                episode["status"] = "failed"
        write_json(manifest_path, manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="Must not already exist")
    parser.add_argument("--cycles", nargs="+", choices=CYCLE_NAMES, default=list(CYCLE_NAMES))
    args = parser.parse_args()
    cycles = {name: PROJECT_ROOT / "data/raw" / f"{name}_raw.csv" for name in args.cycles}
    run_benchmark(args.output_dir, cycles, progress=lambda message: print(message, flush=True))


if __name__ == "__main__":
    main()
