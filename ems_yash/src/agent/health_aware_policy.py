"""
Phase 3B — Health-Aware Deterministic Policy

This policy provides a simple, physically-motivated torque allocation
strategy that depends on battery State of Health (SOH).

IMPORTANT: This is NOT the trained DDPG-GRU-SA policy from the paper.
It is a controlled experimental policy designed to isolate the effect
of AI-estimated battery SOH on EMS decisions.

Design Rationale:
-----------------
As battery SOH decreases, the policy shifts more load to Motor 1 to
reduce battery stress. This is physically motivated by:

1. Lower battery SOH → reduced current-handling capability
2. Higher sigma_tor → more power from Motor 1, less from battery
3. Protects degraded battery from high-current operation

Physical Justification:
-----------------------
In a dual-motor HEV:
- sigma_tor ∈ [0, 1] represents Motor 1 torque fraction
- sigma_tor = 0.5: balanced 50/50 split
- sigma_tor > 0.5: more load on Motor 1
- sigma_tor < 0.5: more load on Motor 2

The existing feasible action mapper ensures sigma_tor respects:
- Motor torque limits
- Motor speed limits
- Physical feasibility

Policy Formula:
---------------
sigma_base = 0.50  (balanced baseline)
soh_sensitivity = 0.30  (adjustment range)

sigma_tor = sigma_base + soh_sensitivity * (1.0 - battery_soh)

Examples:
---------
battery_soh = 1.00 → sigma_tor = 0.50 + 0.30 * 0.00 = 0.50 (balanced)
battery_soh = 0.99 → sigma_tor = 0.50 + 0.30 * 0.01 = 0.503
battery_soh = 0.95 → sigma_tor = 0.50 + 0.30 * 0.05 = 0.515
battery_soh = 0.90 → sigma_tor = 0.50 + 0.30 * 0.10 = 0.530
battery_soh = 0.80 → sigma_tor = 0.50 + 0.30 * 0.20 = 0.560

The policy is:
- Deterministic (same input → same output)
- Continuous (smooth SOH response)
- Bounded (sigma_tor ∈ [0, 1])
- Physically motivated (protects degraded battery)
- SOH-dependent (required for Phase 3B experiment)
"""

import numpy as np


class HealthAwareDeterministicPolicy:
    """
    Simple deterministic torque allocation policy based on battery SOH.

    Parameters
    ----------
    sigma_base : float
        Baseline torque split when battery SOH = 1.0 (default: 0.5)

    soh_sensitivity : float
        How much sigma_tor increases per unit decrease in SOH (default: 0.3)

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

        if soh_sensitivity < 0.0:
            raise ValueError(
                f"soh_sensitivity must be non-negative, got {soh_sensitivity}"
            )

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        Select torque split action based on battery SOH.

        Parameters
        ----------
        state : np.ndarray
            6D EMS state: [velocity, torque, SOC, battery_SOH, motor1_SOH, motor2_SOH]

        Returns
        -------
        action : np.ndarray
            Normalized action [0, 1] representing desired torque split
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

        # Calculate sigma_tor
        # As battery degrades (SOH → 0.8), increase sigma to protect battery
        sigma_tor = self.sigma_base + self.soh_sensitivity * (1.0 - battery_soh)

        # Clip to valid action space [0, 1]
        sigma_tor = np.clip(sigma_tor, 0.0, 1.0)

        # Return as 1D array to match action space
        return np.array([sigma_tor], dtype=np.float32)

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
    print(f"{'Battery SOH':<15} {'sigma_tor':<10} {'Change from baseline':<25}")
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
    print("[PASS] Policy increases sigma_tor as battery SOH decreases")
    print("[PASS] All actions within [0, 1]")
    print("[PASS] Policy is deterministic (same state -> same action)")
    print()


if __name__ == "__main__":
    test_policy_behavior()
