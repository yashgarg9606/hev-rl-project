"""
Phase 3B Experiment: Compare TRUE_PHYSICAL vs BMS_ESTIMATED modes.

This diagnostic runs the deterministic rule through EnergyManagementEnv.
It compares physical-model health with a separate source-battery trace; those
are distinct trajectories, not automatically matched estimation targets.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.environment.integrated_powertrain import (
    IntegratedPowertrainParameters,
)
from src.environment.rl_environment import EnergyManagementEnv
from src.bms_interface.health_interface import SOHSource
from src.bms_interface.soh_trace_adapter import SOHTraceAdapter
from src.agent.health_aware_policy import HealthAwareDeterministicPolicy

def load_driving_cycle():
    """Load a simple test driving cycle."""
    # Explicit synthetic pattern with stops, acceleration and cruise.
    timestep_s = 1.0
    cycle = []

    # Start from rest
    for t in range(10):
        cycle.append([t * timestep_s, 0.0, 0.0])

    # Accelerate
    for t in range(10, 30):
        vel = (t - 10) * 2.0
        cycle.append([t * timestep_s, vel, 0.0])

    # Cruise
    for t in range(30, 60):
        cycle.append([t * timestep_s, 40.0, 0.0])

    # Accelerate again
    for t in range(60, 80):
        vel = 40.0 + (t - 60) * 1.5
        cycle.append([t * timestep_s, vel, 0.0])

    # High-speed cruise
    for t in range(80, 100):
        cycle.append([t * timestep_s, 70.0, 0.0])

    return np.array(cycle)

def run_experiment(soh_source: SOHSource, soh_adapter, cycle, episode_length=100):
    """Run actual consecutive cycle transitions through the normalized-action environment.

    State trajectories include the initial point and every returned next state.
    Action/reward entries at the initial point are NaN/zero respectively.
    A trace's paired target describes the source battery, not the physical plant.
    """
    if episode_length < 1:
        raise ValueError("episode_length must be positive")
    params = IntegratedPowertrainParameters(soh_source=soh_source, soh_trace_adapter=soh_adapter)
    env = EnergyManagementEnv(cycle, project_root=project_root, powertrain_parameters=params)
    policy = HealthAwareDeterministicPolicy()
    state, _ = env.reset()
    keys = ('velocity', 'battery_soh_ems', 'battery_soh_true', 'trace_true_soh',
            'motor1_soh', 'motor2_soh', 'normalized_action', 'sigma_tor', 'soc', 'reward', 'time_s')
    trajectory = {key: [] for key in keys}

    def record(action=np.nan, sigma=np.nan, reward=0.):
        physical = env.powertrain.health_model.get_health_state()
        target = (soh_adapter.get_true_soh_at_time(env.simulation_time_s)
                  if soh_adapter is not None and hasattr(soh_adapter, 'get_true_soh_at_time') else np.nan)
        values = (float(state[0]), float(state[3]), physical['battery_soh'], target,
                  physical['motor1_soh'], physical['motor2_soh'], action, sigma,
                  float(env.powertrain.battery.state.soc), reward, env.simulation_time_s)
        for key, value in zip(keys, values):
            trajectory[key].append(value)

    record()
    info = {}
    terminated = truncated = False
    print(f"Running up to {min(episode_length, len(cycle) - 1)} transitions...")
    try:
        for _ in range(min(episode_length, len(cycle) - 1)):
            action = policy.select_action(state)
            time_before = env.simulation_time_s
            state, reward, terminated, truncated, info = env.step(action)
            sigma = info.get('sigma_tor') if env.simulation_time_s > time_before else None
            record(float(action[0]), float(sigma) if sigma is not None else np.nan, reward)
            if terminated or truncated:
                break
        result = {key: np.asarray(values) for key, values in trajectory.items()}
        result['completed_cycle'] = (env.current_index == len(cycle) - 1
                                     and not bool(info.get('constraint_violation', False)) and not truncated)
        result['constraint_violation'] = bool(info.get('constraint_violation', False))
        result['completed_transitions'] = env.current_index
        return result
    finally:
        env.close()

def print_results(traj, mode_name):
    """Print experiment results."""
    print(f"\n{mode_name} Results:")
    print(f"  Steps completed:       {traj['completed_transitions']}")
    print(f"  Initial battery SOH:   {traj['battery_soh_true'][0]:.9f}")
    print(f"  Final battery SOH:     {traj['battery_soh_true'][-1]:.9f}")
    print(f"  Battery degradation:   {traj['battery_soh_true'][0] - traj['battery_soh_true'][-1]:.9f}")
    print(f"  Mean sigma_tor:        {np.nanmean(traj['sigma_tor']):.6f}")
    print(f"  Total reward:          {np.sum(traj['reward']):.3f}")
    print(f"  Final motor1 SOH:      {traj['motor1_soh'][-1]:.9f}")
    print(f"  Final motor2 SOH:      {traj['motor2_soh'][-1]:.9f}")
    print(f"  Initial SOC:           {traj['soc'][0]:.6f}")
    print(f"  Final SOC:             {traj['soc'][-1]:.6f}")
    print(f"  SOC change:            {traj['soc'][-1] - traj['soc'][0]:+.6f}")

def main():
    print("\n" + "="*70)
    print("PHASE 3B EXPERIMENT: TRUE_PHYSICAL vs BMS_ESTIMATED")
    print("="*70)

    # Load BMS trace
    trace_path = project_root / 'data/bms_soh_trace.npz'
    soh_adapter = SOHTraceAdapter(trace_path)
    print(f"\nLoaded: {soh_adapter}")

    # Load driving cycle
    cycle = load_driving_cycle()
    print(f"Loaded driving cycle: {len(cycle)} timesteps")

    # Experiment A: TRUE_PHYSICAL
    print("\n" + "-"*70)
    print("EXPERIMENT A: TRUE_PHYSICAL MODE")
    print("-"*70)

    true_traj = run_experiment(SOHSource.TRUE_PHYSICAL, soh_adapter, cycle)
    print_results(true_traj, "TRUE_PHYSICAL")

    # Experiment B: BMS_ESTIMATED
    print("\n" + "-"*70)
    print("EXPERIMENT B: BMS_ESTIMATED MODE")
    print("-"*70)

    ai_traj = run_experiment(SOHSource.BMS_ESTIMATED, soh_adapter, cycle)
    print_results(ai_traj, "BMS_ESTIMATED")

    # SOH estimation metrics
    soh_error = ai_traj['battery_soh_ems'] - ai_traj['trace_true_soh']
    soh_mae = np.mean(np.abs(soh_error))
    soh_rmse = np.sqrt(np.mean(soh_error**2))

    print(f"\nMatched source-trace SOH error:")
    print(f"  Mean EMS-visible SOH (AI): {np.mean(ai_traj['battery_soh_ems']):.9f}")
    print(f"  Mean paired source SOH:             {np.mean(ai_traj['trace_true_soh']):.9f}")
    print(f"  MAE:                       {soh_mae:.9f}")
    print(f"  RMSE:                      {soh_rmse:.9f}")
    print(f"  Max absolute error:        {np.max(np.abs(soh_error)):.9f}")

    # Comparison
    print("\n" + "="*70)
    print("TRUE vs AI COMPARISON")
    print("="*70)

    common = min(len(ai_traj['sigma_tor']), len(true_traj['sigma_tor']))
    sigma_difference = ai_traj['sigma_tor'][1:common] - true_traj['sigma_tor'][1:common]
    sigma_diff_rms = np.sqrt(np.nanmean(sigma_difference**2))
    sigma_diff_max = np.nanmax(np.abs(sigma_difference))

    print(f"\nEMS Decision Differences:")
    print(f"  Mean sigma_tor (TRUE):  {np.nanmean(true_traj['sigma_tor']):.6f}")
    print(f"  Mean sigma_tor (AI):    {np.nanmean(ai_traj['sigma_tor']):.6f}")
    print(f"  Difference:             {np.nanmean(ai_traj['sigma_tor']) - np.nanmean(true_traj['sigma_tor']):+.6f}")
    print(f"  RMS difference:         {sigma_diff_rms:.6f}")
    print(f"  Max difference:         {sigma_diff_max:.6f}")

    print(f"\nBattery Degradation:")
    true_deg = true_traj['battery_soh_true'][0] - true_traj['battery_soh_true'][-1]
    ai_deg = ai_traj['battery_soh_true'][0] - ai_traj['battery_soh_true'][-1]
    print(f"  TRUE mode:  {true_deg:.9f}")
    print(f"  AI mode:    {ai_deg:.9f}")
    print(f"  Difference: {ai_deg - true_deg:+.9f}")

    print(f"\nMotor Degradation:")
    true_m1_deg = true_traj['motor1_soh'][0] - true_traj['motor1_soh'][-1]
    true_m2_deg = true_traj['motor2_soh'][0] - true_traj['motor2_soh'][-1]
    ai_m1_deg = ai_traj['motor1_soh'][0] - ai_traj['motor1_soh'][-1]
    ai_m2_deg = ai_traj['motor2_soh'][0] - ai_traj['motor2_soh'][-1]
    print(f"  Motor 1 (TRUE):  {true_m1_deg:.9f}")
    print(f"  Motor 1 (AI):    {ai_m1_deg:.9f}")
    print(f"  Motor 1 Diff:    {ai_m1_deg - true_m1_deg:+.9f}")
    print(f"  Motor 2 (TRUE):  {true_m2_deg:.9f}")
    print(f"  Motor 2 (AI):    {ai_m2_deg:.9f}")
    print(f"  Motor 2 Diff:    {ai_m2_deg - true_m2_deg:+.9f}")

    print(f"\nSOC Change:")
    true_soc_change = true_traj['soc'][-1] - true_traj['soc'][0]
    ai_soc_change = ai_traj['soc'][-1] - ai_traj['soc'][0]
    print(f"  TRUE mode:  {true_soc_change:+.6f}")
    print(f"  AI mode:    {ai_soc_change:+.6f}")
    print(f"  Difference: {ai_soc_change - true_soc_change:+.6f}")

    print(f"\nReward:")
    print(f"  TRUE mode:  {np.sum(true_traj['reward']):.3f}")
    print(f"  AI mode:    {np.sum(ai_traj['reward']):.3f}")
    print(f"  Difference: {np.sum(ai_traj['reward']) - np.sum(true_traj['reward']):+.3f}")

    print("\n" + "="*70)
    print("PHASE 3B EXPERIMENT COMPLETE")
    print("="*70)
    print(f"\nCompleted cycles: TRUE={true_traj['completed_cycle']}, AI={ai_traj['completed_cycle']}")
    print(f"Observed torque-split difference: RMS = {sigma_diff_rms:.6f}")
    print("Physical degradation follows actual executed actions in each mode.")
    print(f"Matched source-trace error: MAE = {soh_mae:.6f}, RMSE = {soh_rmse:.6f}")

if __name__ == '__main__':
    main()
