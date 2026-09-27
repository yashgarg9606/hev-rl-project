"""
Phase 3B — TRUE vs AI SOH Comparison Experiment

This experiment compares EMS behavior under two SOH information modes:

1. TRUE_PHYSICAL: EMS receives true battery SOH from physical degradation model
2. BMS_ESTIMATED: EMS receives AI-estimated battery SOH from BMS trace

IMPORTANT: This experiment does NOT reproduce the trained DDPG-GRU-SA policy.
It uses a health-aware deterministic policy to isolate the effect of SOH
estimation error on EMS decisions.

The experiment validates:
- BMS_ESTIMATED mode successfully provides AI-estimated SOH to EMS
- Physical battery SOH continues to evolve independently of AI estimates
- True and estimated SOH remain properly separated (no data leakage)
- Health-aware policy responds to SOH differences between modes
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.environment.rl_environment import EnergyManagementEnv
from src.agent.health_aware_policy import HealthAwareDeterministicPolicy
from src.bms_interface.soh_source import SOHSource
from src.bms_interface.soh_trace_adapter import SOHTraceAdapter
from src.environment.integrated_powertrain import IntegratedPowertrainParameters


def load_udds_cycle():
    """Load UDDS driving cycle."""
    import pandas as pd

    cycle_path = project_root / "data" / "raw" / "UDDS_raw.csv"

    if not cycle_path.exists():
        raise FileNotFoundError(f"UDDS cycle not found: {cycle_path}")

    # Load CSV
    df = pd.read_csv(cycle_path)

    # Expected columns: time_s, velocity_kmh, slope_rad (or velocity_mph)
    # Convert to expected format: [time_s, velocity_kmh, slope_rad]

    if 'velocity_mph' in df.columns:
        # Convert mph to km/h
        velocity_kmh = df['velocity_mph'].values * 1.60934
    elif 'velocity_kmh' in df.columns:
        velocity_kmh = df['velocity_kmh'].values
    elif 'speed_mph' in df.columns:
        velocity_kmh = df['speed_mph'].values * 1.60934
    elif 'speed_kmh' in df.columns:
        velocity_kmh = df['speed_kmh'].values
    else:
        # Try second column as velocity
        velocity_kmh = df.iloc[:, 1].values * 1.60934

    # Time column
    if 'time_s' in df.columns:
        time_s = df['time_s'].values
    elif 'time' in df.columns:
        time_s = df['time'].values
    else:
        # First column as time
        time_s = df.iloc[:, 0].values

    # Slope (assume flat if not present)
    if 'slope_rad' in df.columns:
        slope_rad = df['slope_rad'].values
    else:
        slope_rad = np.zeros_like(time_s)

    # Construct cycle array
    cycle = np.column_stack([time_s, velocity_kmh, slope_rad])

    print(f"Loaded UDDS cycle: {len(cycle)} timesteps, {cycle[-1, 0]:.1f} seconds")

    return cycle


def run_episode(env, policy, mode_name):
    """
    Run one episode and collect metrics.

    Parameters
    ----------
    env : EnergyManagementEnv
        Gymnasium environment
    policy : HealthAwareDeterministicPolicy
        Deterministic health-aware policy
    mode_name : str
        Name of SOH mode for logging

    Returns
    -------
    dict
        Episode metrics
    """

    print(f"\n{'='*70}")
    print(f"RUNNING EPISODE: {mode_name}")
    print(f"{'='*70}\n")

    state, info = env.reset()

    episode_reward = 0.0
    timesteps = 0

    # Track SOH throughout episode
    soh_history = {
        'timestep': [],
        'time_s': [],
        'ems_battery_soh': [],  # SOH exposed to EMS
        'true_battery_soh': [],  # True physical SOH
        'motor1_soh': [],
        'motor2_soh': [],
        'sigma_tor': [],
        'normalized_action': [],
        'expected_ems_soh': [],
    }

    # Initial state
    soh_history['timestep'].append(0)
    soh_history['time_s'].append(float(info['time_s']))
    soh_history['ems_battery_soh'].append(float(state[3]))
    soh_history['true_battery_soh'].append(
        env.powertrain.health_interface.get_true_battery_soh()
    )
    soh_history['motor1_soh'].append(float(state[4]))
    soh_history['motor2_soh'].append(float(state[5]))
    soh_history['sigma_tor'].append(np.nan)
    soh_history['normalized_action'].append(np.nan)
    soh_history['expected_ems_soh'].append(env.powertrain.health_interface.get_health_state(env.simulation_time_s)['battery_soh'])

    terminated = False
    truncated = False

    while not (terminated or truncated):
        # Policy selects action based on EMS-visible state
        action = policy.select_action(state)

        # Environment step
        time_before = env.simulation_time_s
        next_state, reward, terminated, truncated, step_info = env.step(action)

        episode_reward += reward
        timesteps += 1

        # Record SOH values
        soh_history['timestep'].append(timesteps)
        soh_history['time_s'].append(step_info['time_s'])
        soh_history['ems_battery_soh'].append(float(next_state[3]))
        soh_history['true_battery_soh'].append(
            env.powertrain.health_interface.get_true_battery_soh()
        )
        soh_history['motor1_soh'].append(float(next_state[4]))
        soh_history['motor2_soh'].append(float(next_state[5]))
        sigma = step_info.get('sigma_tor') if env.simulation_time_s > time_before else None
        soh_history['sigma_tor'].append(float(sigma) if sigma is not None else np.nan)
        soh_history['normalized_action'].append(float(action[0]))
        soh_history['expected_ems_soh'].append(env.powertrain.health_interface.get_health_state(env.simulation_time_s)['battery_soh'])

        state = next_state

        # Progress indicator
        if timesteps % 100 == 0:
            print(f"  Step {timesteps:4d} | "
                  f"Reward: {episode_reward:8.2f} | "
                  f"EMS SOH: {state[3]:.6f} | "
                  f"True SOH: {soh_history['true_battery_soh'][-1]:.6f} | "
                  f"SOC: {state[2]:.4f}")

    # Convert to arrays
    for key in soh_history:
        soh_history[key] = np.array(soh_history[key])

    # Final metrics
    final_soc = float(state[2])
    final_ems_soh = float(state[3])
    final_true_soh = soh_history['true_battery_soh'][-1]

    print(f"\n{'='*70}")
    print(f"EPISODE COMPLETE: {mode_name}")
    print(f"{'='*70}")
    print(f"  Timesteps: {timesteps}")
    print(f"  Total Reward: {episode_reward:.2f}")
    print(f"  Final SOC: {final_soc:.6f}")
    print(f"  Final EMS Battery SOH: {final_ems_soh:.6f}")
    print(f"  Final True Battery SOH: {final_true_soh:.6f}")

    if bool(step_info.get('constraint_violation', False)):
        print(f"\n  [TERMINATED] Constraint violation: {step_info['violated_constraints']}")

    return {
        'mode': mode_name,
        'timesteps': timesteps,
        'episode_reward': episode_reward,
        'final_soc': final_soc,
        'final_ems_soh': final_ems_soh,
        'final_true_soh': final_true_soh,
        'soh_history': soh_history,
        'constraint_violation': bool(step_info.get('constraint_violation', False)),
        'completed_cycle': env.current_index == len(env.cycle) - 1 and not bool(step_info.get('constraint_violation', False)) and not truncated,
    }


def compare_results(true_results, ai_results):
    """
    Compare TRUE_PHYSICAL vs BMS_ESTIMATED results.

    Parameters
    ----------
    true_results : dict
        Results from TRUE_PHYSICAL mode
    ai_results : dict
        Results from BMS_ESTIMATED mode
    """

    print(f"\n{'='*70}")
    print("COMPARISON: TRUE_PHYSICAL vs BMS_ESTIMATED")
    print(f"{'='*70}\n")

    # 1. Validate SOH separation
    print("1. SOH SEPARATION VALIDATION")
    print("-" * 70)

    true_ems_soh = true_results['soh_history']['ems_battery_soh']
    true_physical_soh = true_results['soh_history']['true_battery_soh']

    ai_ems_soh = ai_results['soh_history']['ems_battery_soh']
    ai_physical_soh = ai_results['soh_history']['true_battery_soh']

    # In TRUE mode: EMS SOH should equal true physical SOH
    true_mode_match = np.allclose(true_ems_soh, true_physical_soh, rtol=0., atol=np.finfo(np.float32).eps)
    print(f"  TRUE mode: EMS SOH == Physical SOH? {true_mode_match}")

    if not true_mode_match:
        max_diff = np.max(np.abs(true_ems_soh - true_physical_soh))
        print(f"    [WARNING] Max difference: {max_diff:.9f}")

    ai_mode_match = np.allclose(ai_ems_soh, ai_results['soh_history']['expected_ems_soh'],
                                rtol=0., atol=np.finfo(np.float32).eps)
    print(f"  AI observation matches configured SOH interface? {ai_mode_match}")
    print(f"  Mean absolute trace/physical-model discrepancy: {np.mean(np.abs(ai_ems_soh - ai_physical_soh)):.6f}")
    print("  This discrepancy is not matched-target BMS estimation error.")
    print("  Different actions may legitimately produce different physical degradation.")

    print()

    # 2. Episode metrics comparison
    print("2. EPISODE METRICS")
    print("-" * 70)
    print(f"  {'Metric':<25} {'TRUE':<15} {'AI':<15} {'Difference':<15}")
    print("-" * 70)

    metrics = [
        ('Timesteps', 'timesteps', ''),
        ('Episode Reward', 'episode_reward', '.2f'),
        ('Final SOC', 'final_soc', '.6f'),
        ('Final EMS SOH', 'final_ems_soh', '.6f'),
        ('Final True SOH', 'final_true_soh', '.6f'),
    ]

    for label, key, fmt in metrics:
        true_val = true_results[key]
        ai_val = ai_results[key]

        if fmt:
            diff = ai_val - true_val
            print(f"  {label:<25} {true_val:<15{fmt}} {ai_val:<15{fmt}} {diff:<15{fmt}}")
        else:
            print(f"  {label:<25} {true_val:<15} {ai_val:<15} {ai_val - true_val:<15}")

    print()

    # 3. Policy response analysis
    print("3. POLICY RESPONSE TO SOH DIFFERENCES")
    print("-" * 70)

    true_sigma = true_results['soh_history']['sigma_tor'][1:]  # Skip NaN at index 0
    ai_sigma = ai_results['soh_history']['sigma_tor'][1:]

    for mode, values in (("TRUE", true_sigma), ("AI", ai_sigma)):
        values = values[np.isfinite(values)]
        print(f"  {mode} executed sigma range: " + (f"[{values.min():.4f}, {values.max():.4f}]" if len(values) else "no feasible split"))
    common_steps = min(len(ai_sigma), len(true_sigma))
    differences = ai_sigma[:common_steps] - true_sigma[:common_steps]
    differences = differences[np.isfinite(differences)]
    print(f"  Mean sigma_tor difference over {len(differences)} mapped common attempts: "
          + (f"{differences.mean():.6f}" if len(differences) else "unavailable"))

    print()

    # 4. Validation status
    print("4. VALIDATION STATUS")
    print("-" * 70)

    checks = [
        ("TRUE mode: EMS uses physical SOH", true_mode_match),
        ("AI mode: observation matches SOH interface", ai_mode_match),
        ("Both episodes completed", true_results['completed_cycle'] and ai_results['completed_cycle']),
    ]

    all_passed = all(check[1] for check in checks)

    for check_name, passed in checks:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {check_name}")

    print()

    if all_passed:
        print("[SUCCESS] SOH interface and episode-completion checks passed.")
        print("  Physical degradation follows the actions executed in each mode.")
    else:
        print("[FAILURE] Validation checks failed. Review integration logic.")

    print()


def main():
    """Run TRUE vs AI comparison experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, default=project_root / 'data/bms_soh_trace.npz')
    parser.add_argument('--output-dir', type=Path, required=True, help='New result directory; must not exist')
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"Output already exists: {args.output_dir}")

    print("=" * 70)
    print("PHASE 3B — TRUE vs AI SOH COMPARISON EXPERIMENT")
    print("=" * 70)
    print()
    print("This experiment compares EMS behavior under two SOH information modes:")
    print("  1. TRUE_PHYSICAL: EMS receives true battery SOH")
    print("  2. BMS_ESTIMATED: EMS receives AI-estimated battery SOH")
    print()
    print("Policy: Health-aware deterministic torque allocation")
    print("  normalized a = 0.50 + 0.10 * (1.0 - battery_soh)")
    print("  The environment maps a into the feasible physical split interval.")
    print()

    # Load driving cycle
    cycle = load_udds_cycle()

    # Retain this diagnostic's declared response coefficient.
    policy = HealthAwareDeterministicPolicy(
        sigma_base=0.50,
        soh_sensitivity=0.10,
    )

    print(f"\nPolicy: {policy}")
    print()

    # Load BMS trace for AI mode
    trace_file = args.trace

    if not trace_file.exists():
        raise FileNotFoundError(
            f"BMS trace not found: {trace_file}\n"
            "Use generate_bms_trace.py --help to create a trace, then supply --trace."
        )

    trace_adapter = SOHTraceAdapter(trace_file)
    print(f"Loaded BMS trace: {trace_adapter}")
    print()

    # -------------------------------------------------------------------------
    # Experiment 1: TRUE_PHYSICAL mode
    # -------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("EXPERIMENT 1: TRUE_PHYSICAL MODE")
    print("=" * 70)
    print("EMS receives true physical battery SOH from degradation model")
    print()

    true_params = IntegratedPowertrainParameters(
        soh_source=SOHSource.TRUE_PHYSICAL,
        soh_trace_adapter=None,
    )

    env_true = EnergyManagementEnv(
        cycle=cycle,
        project_root=project_root,
        powertrain_parameters=true_params,
    )

    true_results = run_episode(env_true, policy, "TRUE_PHYSICAL")

    # -------------------------------------------------------------------------
    # Experiment 2: BMS_ESTIMATED mode
    # -------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("EXPERIMENT 2: BMS_ESTIMATED MODE")
    print("=" * 70)
    print("EMS receives AI-estimated battery SOH from BMS trace")
    print()

    ai_params = IntegratedPowertrainParameters(
        soh_source=SOHSource.BMS_ESTIMATED,
        soh_trace_adapter=trace_adapter,
    )

    env_ai = EnergyManagementEnv(
        cycle=cycle,
        project_root=project_root,
        powertrain_parameters=ai_params,
    )

    ai_results = run_episode(env_ai, policy, "BMS_ESTIMATED")

    # -------------------------------------------------------------------------
    # Compare results
    # -------------------------------------------------------------------------

    compare_results(true_results, ai_results)

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)

    output_file = output_dir / "true_vs_ai_comparison.npz"

    np.savez_compressed(
        output_file,
        true_mode=true_results['mode'],
        true_timesteps=true_results['timesteps'],
        true_episode_reward=true_results['episode_reward'],
        true_final_soc=true_results['final_soc'],
        true_final_ems_soh=true_results['final_ems_soh'],
        true_final_true_soh=true_results['final_true_soh'],
        true_soh_history_timestep=true_results['soh_history']['timestep'],
        true_soh_history_time_s=true_results['soh_history']['time_s'],
        true_soh_history_ems_battery_soh=true_results['soh_history']['ems_battery_soh'],
        true_soh_history_true_battery_soh=true_results['soh_history']['true_battery_soh'],
        true_soh_history_motor1_soh=true_results['soh_history']['motor1_soh'],
        true_soh_history_motor2_soh=true_results['soh_history']['motor2_soh'],
        true_soh_history_sigma_tor=true_results['soh_history']['sigma_tor'],
        true_soh_history_normalized_action=true_results['soh_history']['normalized_action'],

        ai_mode=ai_results['mode'],
        ai_timesteps=ai_results['timesteps'],
        ai_episode_reward=ai_results['episode_reward'],
        ai_final_soc=ai_results['final_soc'],
        ai_final_ems_soh=ai_results['final_ems_soh'],
        ai_final_true_soh=ai_results['final_true_soh'],
        ai_soh_history_timestep=ai_results['soh_history']['timestep'],
        ai_soh_history_time_s=ai_results['soh_history']['time_s'],
        ai_soh_history_ems_battery_soh=ai_results['soh_history']['ems_battery_soh'],
        ai_soh_history_true_battery_soh=ai_results['soh_history']['true_battery_soh'],
        ai_soh_history_motor1_soh=ai_results['soh_history']['motor1_soh'],
        ai_soh_history_motor2_soh=ai_results['soh_history']['motor2_soh'],
        ai_soh_history_sigma_tor=ai_results['soh_history']['sigma_tor'],
        ai_soh_history_normalized_action=ai_results['soh_history']['normalized_action'],
    )

    print(f"Results saved to: {output_file}")
    print()

    print("=" * 70)
    print("PHASE 3B EXPERIMENT COMPLETE")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
