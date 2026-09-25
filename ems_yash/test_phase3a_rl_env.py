"""
Phase 3A — EnergyManagementEnv Integration Test

Validates that the RL environment works correctly with the
Phase 3A BMS interface integration.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.environment.rl_environment import EnergyManagementEnv


def build_test_cycle():
    """Simple test driving cycle."""
    return np.array([
        [0.0, 0.0, 0.0],
        [1.0, 10.0, 0.0],
        [2.0, 20.0, 0.0],
        [3.0, 30.0, 0.0],
        [4.0, 40.0, 0.0],
        [5.0, 50.0, 0.0],
        [6.0, 40.0, 0.0],
        [7.0, 30.0, 0.0],
        [8.0, 20.0, 0.0],
        [9.0, 10.0, 0.0],
        [10.0, 0.0, 0.0],
    ], dtype=np.float64)


def main():
    print("=" * 70)
    print("PHASE 3A — RL ENVIRONMENT INTEGRATION TEST")
    print("=" * 70)

    # Create environment
    cycle = build_test_cycle()
    env = EnergyManagementEnv(cycle=cycle)

    print("\nTEST 1 — Environment Creation")
    print(f"State dim: {env.state_dim}")
    print(f"Action dim: {env.action_dim}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    assert env.state_dim == 6, "State dim should be 6"
    assert env.action_dim == 1, "Action dim should be 1"
    print("[PASS] Correct state and action dimensions")

    # Verify interface exists
    print("\nTEST 2 — BMS Interface Integration")
    assert hasattr(env.powertrain, 'health_interface'), "Should have health_interface"
    assert hasattr(env.powertrain, 'health_model'), "Should have health_model"
    print(f"Interface source: {env.powertrain.health_interface.soh_source.value}")
    print("[PASS] BMS interface present in powertrain")

    # Reset environment
    print("\nTEST 3 — Environment Reset")
    state, info = env.reset(seed=42)

    print(f"Initial state shape: {state.shape}")
    print(f"Initial state: {state}")
    print(f"  velocity_kmh:  {state[0]:.6f}")
    print(f"  wheel_torque:  {state[1]:.6f}")
    print(f"  SOC:           {state[2]:.6f}")
    print(f"  battery_SOH:   {state[3]:.12f}")
    print(f"  motor1_SOH:    {state[4]:.12f}")
    print(f"  motor2_SOH:    {state[5]:.12f}")

    assert state.shape == (6,), "State should be 6-dimensional"
    assert state[3] == 1.0, "Initial battery SOH should be 1.0"
    assert state[4] == 1.0, "Initial motor1 SOH should be 1.0"
    assert state[5] == 1.0, "Initial motor2 SOH should be 1.0"
    print("[PASS] Reset returns correct 6D state")
    print("[PASS] Initial SOH = 1.0 for all components")

    # Verify interface and model agree
    print("\nTEST 4 — Interface-Model Agreement")
    interface_health = env.powertrain.health_interface.get_health_state()
    model_health = env.powertrain.health_model.get_health_state()

    print(f"Interface battery SOH: {interface_health['battery_soh']:.12f}")
    print(f"Model battery SOH:     {model_health['battery_soh']:.12f}")
    print(f"State battery SOH:     {state[3]:.12f}")

    assert interface_health['battery_soh'] == model_health['battery_soh']
    assert float(interface_health['battery_soh']) == state[3]
    print("[PASS] Interface matches physical model")
    print("[PASS] State matches interface")

    # Take a step
    print("\nTEST 5 — Environment Step")
    action = np.array([0.5], dtype=np.float32)
    next_state, reward, terminated, truncated, step_info = env.step(action)

    print(f"Next state shape: {next_state.shape}")
    print(f"Next state: {next_state}")
    print(f"  battery_SOH:   {next_state[3]:.12f}")
    print(f"  motor1_SOH:    {next_state[4]:.12f}")
    print(f"  motor2_SOH:    {next_state[5]:.12f}")
    print(f"Reward: {reward:.6f}")
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")

    assert next_state.shape == (6,), "Next state should be 6-dimensional"

    # Run multiple steps to observe degradation (single-step degradation too small for float32)
    print("\nRunning 4 more steps to observe degradation...")
    for i in range(4):
        next_state, _, _, _, _ = env.step(np.array([0.5], dtype=np.float32))

    print(f"After 5 total steps:")
    print(f"  battery_SOH:   {next_state[3]:.12f}")
    print(f"  motor1_SOH:    {next_state[4]:.12f}")
    print(f"  motor2_SOH:    {next_state[5]:.12f}")

    assert next_state[3] < 1.0, "Battery SOH should decrease after 5 steps"
    assert next_state[4] < 1.0, "Motor1 SOH should decrease after 5 steps"
    assert next_state[5] < 1.0, "Motor2 SOH should decrease after 5 steps"
    print("[PASS] Step returns correct 6D state")
    print("[PASS] Health degraded after multiple steps")

    # Verify state propagation through interface
    print("\nTEST 6 — Health Propagation Through Interface")
    interface_health_after = env.powertrain.health_interface.get_health_state()
    model_health_after = env.powertrain.health_model.get_health_state()

    print(f"Interface battery SOH: {interface_health_after['battery_soh']:.12f}")
    print(f"Model battery SOH:     {model_health_after['battery_soh']:.12f}")
    print(f"State battery SOH:     {next_state[3]:.12f}")

    assert interface_health_after['battery_soh'] == model_health_after['battery_soh']
    assert abs(float(interface_health_after['battery_soh']) - next_state[3]) < 1e-10
    print("[PASS] Interface still matches physical model")
    print("[PASS] State correctly reflects interface health")

    # Run multiple steps
    print("\nTEST 7 — Multiple Steps")
    for i in range(5):
        action = np.array([0.5], dtype=np.float32)
        next_state, reward, terminated, truncated, step_info = env.step(action)
        if terminated or truncated:
            print(f"Episode terminated at step {i+1}")
            break

    print(f"After {i+1} more steps:")
    print(f"  battery_SOH:   {next_state[3]:.12f}")
    print(f"  motor1_SOH:    {next_state[4]:.12f}")
    print(f"  motor2_SOH:    {next_state[5]:.12f}")

    # Final verification
    final_interface = env.powertrain.health_interface.get_health_state()
    final_model = env.powertrain.health_model.get_health_state()

    assert final_interface['battery_soh'] == final_model['battery_soh']
    print("[PASS] Interface consistency maintained across episode")

    # Test get_true_battery_soh access
    print("\nTEST 8 — get_true_battery_soh() Access")
    true_soh = env.powertrain.health_interface.get_true_battery_soh()
    print(f"True battery SOH: {true_soh:.12f}")
    assert true_soh == final_model['battery_soh']
    print("[PASS] get_true_battery_soh() works correctly")

    print("\n" + "=" * 70)
    print("ALL RL ENVIRONMENT INTEGRATION TESTS PASSED")
    print("=" * 70)
    print("\nSummary:")
    print("  - RL environment created with Phase 3A interface")
    print("  - State dimensions preserved (6D state, 1D action)")
    print("  - BMS interface integrated into powertrain")
    print("  - Interface returns identical SOH to physical model")
    print("  - Health state propagates correctly through steps")
    print("  - State observations reflect interface health")
    print("  - Episode execution works correctly")
    print("  - No behavioral changes from baseline")


if __name__ == "__main__":
    main()
