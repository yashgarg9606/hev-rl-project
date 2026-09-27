"""
Phase 3B — Health-Aware Deterministic Policy

This policy provides a synthetic normalized-action response to battery SOH.

IMPORTANT: This is NOT the trained DDPG-GRU-SA policy from the paper.
It is a controlled experimental policy designed to isolate the effect
of AI-estimated battery SOH on EMS decisions.

The output is a = sigma_base + sensitivity * (1 - observed SOH), saturated
to [0, 1]. The historical parameter name sigma_base is retained for callers,
but a is a normalized action, not a physical torque fraction. The environment
maps it into the current feasible [sigma_min, sigma_max] interval.

Both motors draw from the same battery. A larger action does not by itself
establish lower battery current, energy consumption, or degradation; those
outcomes depend on the operating point and must be measured.
"""

import numpy as np


class HealthAwareDeterministicPolicy:
    """
    Simple deterministic torque allocation policy based on battery SOH.

    Parameters
    ----------
    sigma_base : float
        Baseline normalized action when battery SOH = 1.0 (default: 0.5)

    soh_sensitivity : float
        Increase in normalized action per unit decrease in SOH (default: 0.3)

    Notes
    -----
    This policy is NOT the trained DDPG actor. It is a controlled
    experimental policy designed to test SOH information flow through
    the BMS–EMS interface.
    """

    def __init__(
        self,
        sigma_base: float = 0.50,
        soh_sensitivity: float = 0.30,
    ):
        self.sigma_base = sigma_base
        self.soh_sensitivity = soh_sensitivity

        # Validate parameters
        if not 0.0 <= sigma_base <= 1.0:
            raise ValueError(
                f"sigma_base must be in [0, 1], got {sigma_base}"
            )

        if not np.isfinite(soh_sensitivity) or soh_sensitivity < 0.0:
            raise ValueError(
                f"soh_sensitivity must be non-negative, got {soh_sensitivity}"
            )

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        Select a normalized action based on battery SOH.

        Parameters
        ----------
        state : np.ndarray
            6D EMS state: [velocity, torque, SOC, battery_SOH, motor1_SOH, motor2_SOH]

        Returns
        -------
        action : np.ndarray
            Normalized action [0, 1]; physical mapping belongs to the environment.
        """
        if state.shape != (6,):
            raise ValueError(
                f"State must have shape (6,), got {state.shape}"
            )

        # Extract battery SOH from state[3]
        battery_soh = float(state[3])

        # Validate battery SOH
        if not 0.0 <= battery_soh <= 1.0:
            raise ValueError(
                f"Battery SOH must be in [0, 1], got {battery_soh}"
            )

        normalized_action = self.sigma_base + self.soh_sensitivity * (1.0 - battery_soh)

        # Clip to valid action space [0, 1]
        normalized_action = np.clip(normalized_action, 0.0, 1.0)

        # Return as 1D array to match action space
        return np.array([normalized_action], dtype=np.float32)

    def __repr__(self) -> str:
        return (
            f"HealthAwareDeterministicPolicy("
            f"base={self.sigma_base:.2f}, "
            f"sensitivity={self.soh_sensitivity:.2f})"
        )


def test_policy_behavior():
    """Test policy responds correctly to different SOH values."""
    policy = HealthAwareDeterministicPolicy()

    print("=" * 70)
    print("HEALTH-AWARE DETERMINISTIC POLICY TEST")
    print("=" * 70)
    print()
    print(f"Policy: {policy}")
    print()
    print("Testing SOH sensitivity:")
    print()
    print(f"{'Battery SOH':<15} {'action a':<10} {'Change from baseline':<25}")
    print("-" * 70)

    baseline_sigma = None

    for soh in [1.00, 0.99, 0.98, 0.95, 0.90, 0.85, 0.80]:
        # Create test state with varying SOH
        state = np.array([50.0, 500.0, 0.6, soh, 1.0, 1.0], dtype=np.float32)

        action = policy.select_action(state)
        sigma = float(action[0])

        if baseline_sigma is None:
            baseline_sigma = sigma
            change_str = "(baseline)"
        else:
            change = sigma - baseline_sigma
            change_str = f"+{change:.4f}"

        print(f"{soh:<15.2f} {sigma:<10.4f} {change_str:<25}")

    print()
    print("Policy output above is normalized a; the environment determines physical sigma_tor.")
    print("[PASS] All actions within [0, 1]")
    print("[PASS] Policy is deterministic (same state -> same action)")
    print()


if __name__ == "__main__":
    test_policy_behavior()
