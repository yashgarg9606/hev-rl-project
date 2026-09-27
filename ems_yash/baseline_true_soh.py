"""
Phase 3A — TRUE-SOH Baseline Experiment

Establishes a reproducible baseline for TRUE_PHYSICAL SOH mode.

Records:
- Timestep data (velocity, torque, SOC, SOH, power, etc.)
- Final aggregate metrics
- Deterministic configuration

This baseline will be compared against BMS_ESTIMATED mode in Phase 3B.
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.environment.rl_environment import EnergyManagementEnv
from src.bms_interface.soh_source import SOHSource


def build_wltp_class3_excerpt():
    """
    Load a deterministic excerpt from WLTP Class 3 driving cycle.

    Use the first 100 samples (99 intervals) of the canonical WLTC Class3b file.
    """
    wltp_file = project_root / "data" / "raw" / "WLTC_Class3b_raw.csv"

    if not wltp_file.exists():
        raise FileNotFoundError(f"Required canonical driving cycle is missing: {wltp_file}")

    # Load WLTP data
    data = pd.read_csv(wltp_file)

    # Extract first 100 seconds
    cycle_data = data.iloc[:100].copy()

    # Format as [time, velocity, slope]
    cycle = np.column_stack([
        cycle_data['time_s'].values,
        cycle_data['velocity_kmh'].values,
        cycle_data['slope_rad'].values,
    ])

    return cycle.astype(np.float64)


def build_synthetic_cycle():
    """
    Explicit synthetic fixture retained for callers; no automatic substitution.

    30 timesteps with gentle velocity profile to avoid motor constraint violations.
    """
    timesteps = np.arange(0, 30, 1, dtype=np.float64)

    # Gentle velocity ramp: 20-50 km/h
    # Avoid extreme accelerations that violate motor constraints
    velocity = 20.0 + 30.0 * (timesteps / 29.0)

    # Flat terrain
    slope = np.zeros_like(timesteps)

    cycle = np.column_stack([timesteps, velocity, slope])
    return cycle


def run_baseline_experiment(seed=42, use_random_actions=True):
    """
    Run TRUE-SOH baseline experiment.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    use_random_actions : bool
        If True, use random actions. If False, use fixed mid-point action (0.5).
        Note: Trained DDPG checkpoint is not available, so we cannot use
        a trained policy.

    Returns
    -------
    DataFrame with timestep data and dict with aggregate metrics.
    """

    # Set random seed
    np.random.seed(seed)

    print("=" * 80)
    print("PHASE 3A — TRUE-SOH BASELINE EXPERIMENT")
    print("=" * 80)
    print()

    # Load driving cycle
    print("Loading driving cycle...")
    cycle = build_wltp_class3_excerpt()
    cycle_name = "WLTC Class 3b (first 100 samples)"

    print(f"Cycle: {cycle_name}")
    print(f"Duration: {len(cycle)} timesteps")
    print()

    # Create environment with TRUE_PHYSICAL mode
    print("Creating EnergyManagementEnv with TRUE_PHYSICAL SOH source...")
    env = EnergyManagementEnv(cycle=cycle)

    # Verify interface source
    assert env.powertrain.health_interface.soh_source == SOHSource.TRUE_PHYSICAL
    print(f"BMS interface source: {env.powertrain.health_interface.soh_source.name}")
    print()

    # Configuration
    print("Experiment Configuration:")
    print(f"  Random seed: {seed}")
    print(f"  Action source: {'random [0, 1]' if use_random_actions else 'fixed (0.5)'}")
    print(f"  Initial SOC: 0.6")
    print(f"  Initial battery SOH: 1.0")
    print(f"  Initial motor 1 SOH: 1.0")
    print(f"  Initial motor 2 SOH: 1.0")
    print(f"  State dimension: {env.state_dim}")
    print(f"  Action dimension: {env.action_dim}")
    print()

    # Reset environment
    print("Resetting environment...")
    state, _ = env.reset()
    print(f"Initial state: {state}")
    print()

    # Data collection
    timestep_data = []

    print("Running episode...")
    step_count = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        # Get action
        if use_random_actions:
            action = np.random.uniform(0.0, 1.0, size=(1,)).astype(np.float32)
        else:
            action = np.array([0.5], dtype=np.float32)

        # Get current state before step
        velocity_kmh = state[0]
        wheel_torque_nm = state[1]
        soc = float(env.powertrain.battery.state.soc)
        battery_soh_state = state[3]
        motor1_soh_state = state[4]
        motor2_soh_state = state[5]

        # Get health from interface (before step)
        health = env.powertrain.health_interface.get_health_state(env.simulation_time_s)
        true_battery_soh = env.powertrain.health_interface.get_true_battery_soh()
        ems_battery_soh = health['battery_soh']
        motor1_soh = health['motor1_soh']
        motor2_soh = health['motor2_soh']

        # Step environment
        time_before = env.simulation_time_s
        next_state, reward, terminated, truncated, info = env.step(action)
        completed_dt = env.simulation_time_s - time_before

        # Get sigma_tor from info if available
        sigma_tor = info.get('sigma_tor') if completed_dt > 0. else None

        # Get power/current/voltage from info dict
        battery_power_kw = info.get('battery_power_kw', np.nan)
        battery_current_a = info.get('battery_current_a', np.nan)
        battery_voltage = info.get('battery_terminal_voltage_v', np.nan)

        # Record timestep data
        timestep_data.append({
            'timestep': step_count,
            'velocity_kmh': velocity_kmh,
            'wheel_torque_nm': wheel_torque_nm,
            'action': float(action[0]),
            'normalized_action': float(action[0]),
            'sigma_tor': sigma_tor,
            'completed_dt_s': completed_dt,
            'constraint_violation': bool(info.get('constraint_violation', False)),
            'soc': soc,
            'true_battery_soh': float(true_battery_soh),
            'ems_battery_soh': float(ems_battery_soh),
            'state_battery_soh': battery_soh_state,
            'motor1_soh': float(motor1_soh),
            'motor2_soh': float(motor2_soh),
            'state_motor1_soh': motor1_soh_state,
            'state_motor2_soh': motor2_soh_state,
            'battery_power_kw': battery_power_kw,
            'battery_current_a': battery_current_a,
            'battery_voltage_v': battery_voltage,
            'reward': reward,
        })

        # Update state
        state = next_state
        step_count += 1

        if step_count % 10 == 0:
            print(f"  Step {step_count}: SOC={soc:.4f}, Battery SOH={true_battery_soh:.9f}, Reward={reward:.2f}")

    print()
    print(f"Episode completed: {step_count} steps")
    print(f"Terminated: {terminated}, Truncated: {truncated}")
    print()

    # Convert to DataFrame
    df = pd.DataFrame(timestep_data)
    completed = df[df['completed_dt_s'] > 0.]
    total_duration = completed['completed_dt_s'].sum()

    # Calculate aggregate metrics
    metrics = {
        'num_steps': step_count,
        'initial_soc': df['soc'].iloc[0],
        'final_soc': float(env.powertrain.battery.state.soc),
        'initial_battery_soh': df['true_battery_soh'].iloc[0],
        'final_battery_soh': env.powertrain.health_model.battery_state.soh,
        'initial_motor1_soh': df['motor1_soh'].iloc[0],
        'final_motor1_soh': env.powertrain.health_model.motor1_state.soh,
        'initial_motor2_soh': df['motor2_soh'].iloc[0],
        'final_motor2_soh': env.powertrain.health_model.motor2_state.soh,
        'total_battery_energy_kwh': -(df['battery_power_kw'] * df['completed_dt_s']).sum() / 3600.,
        'mean_battery_power_kw': float((completed['battery_power_kw'] * completed['completed_dt_s']).sum() / total_duration) if total_duration else np.nan,
        'rms_battery_current_a': float(np.sqrt((completed['battery_current_a'] ** 2 * completed['completed_dt_s']).sum() / total_duration)) if total_duration else np.nan,
        'mean_sigma_tor': df['sigma_tor'].mean(),
        'total_reward': df['reward'].sum(),
        'mean_reward': df['reward'].mean(),
        'cycle_name': cycle_name,
        'seed': seed,
        'action_source': 'random' if use_random_actions else 'fixed',
        'constraint_violation': bool(info.get('constraint_violation', False)),
        'completed_cycle': env.current_index == len(env.cycle) - 1 and not bool(info.get('constraint_violation', False)) and not truncated,
    }

    return df, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True, help='New result directory; must not exist')
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"Output already exists: {args.output_dir}")
    # Run experiment with fixed action for reproducibility and feasibility
    df, metrics = run_baseline_experiment(seed=42, use_random_actions=False)

    # Print metrics
    print("=" * 80)
    print("BASELINE METRICS")
    print("=" * 80)
    print()
    print(f"Cycle: {metrics['cycle_name']}")
    print(f"Random seed: {metrics['seed']}")
    print(f"Action source: {metrics['action_source']}")
    print()
    print(f"Number of simulation steps: {metrics['num_steps']}")
    print()
    print("SOC:")
    print(f"  Initial: {metrics['initial_soc']:.6f}")
    print(f"  Final:   {metrics['final_soc']:.6f}")
    print(f"  Change:  {metrics['final_soc'] - metrics['initial_soc']:.6f}")
    print()
    print("Battery SOH:")
    print(f"  Initial: {metrics['initial_battery_soh']:.12f}")
    print(f"  Final:   {metrics['final_battery_soh']:.12f}")
    print(f"  Change:  {metrics['final_battery_soh'] - metrics['initial_battery_soh']:.12e}")
    print()
    print("Motor 1 SOH:")
    print(f"  Initial: {metrics['initial_motor1_soh']:.12f}")
    print(f"  Final:   {metrics['final_motor1_soh']:.12f}")
    print(f"  Change:  {metrics['final_motor1_soh'] - metrics['initial_motor1_soh']:.12e}")
    print()
    print("Motor 2 SOH:")
    print(f"  Initial: {metrics['initial_motor2_soh']:.12f}")
    print(f"  Final:   {metrics['final_motor2_soh']:.12f}")
    print(f"  Change:  {metrics['final_motor2_soh'] - metrics['initial_motor2_soh']:.12e}")
    print()
    print("Energy and Power:")
    print(f"  Total battery energy: {metrics['total_battery_energy_kwh']:.6f} kWh")
    print(f"  Mean battery power:   {metrics['mean_battery_power_kw']:.4f} kW")
    print(f"  RMS battery current:  {metrics['rms_battery_current_a']:.4f} A")
    print()
    print("Torque Split:")
    print(f"  Mean sigma_tor: {metrics['mean_sigma_tor']:.4f}")
    print()
    print("Reward:")
    print(f"  Total reward: {metrics['total_reward']:.2f}")
    print(f"  Mean reward:  {metrics['mean_reward']:.2f}")
    print()

    # Save data
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)

    csv_path = output_dir / "baseline_true_soh_timesteps.csv"
    df.to_csv(csv_path, index=False)
    print(f"Timestep data saved to: {csv_path}")

    metrics_path = output_dir / "baseline_true_soh_metrics.txt"
    with open(metrics_path, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    print(f"Aggregate metrics saved to: {metrics_path}")
    print()

    print("=" * 80)
    print("BASELINE EXPERIMENT COMPLETE")
    print("=" * 80)
    print()
    print("This baseline establishes TRUE_PHYSICAL SOH behavior.")
    print("In Phase 3B, BMS_ESTIMATED mode will be compared against this baseline.")


if __name__ == "__main__":
    main()
