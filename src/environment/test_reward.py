"""
Phase 3D.4 — Reward function validation.

Validates the implementation of Wu et al. (2024), Eq. (25).
"""

from __future__ import annotations

import numpy as np

from src.environment.reward import RewardParameters, WuReward


def main() -> None:
    print("=" * 70)
    print("PHASE 3D.4 — REWARD FUNCTION VALIDATION")
    print("=" * 70)

    reward_model = WuReward()

    # ---------------------------------------------------------------
    # TEST 1 — Paper coefficients
    # ---------------------------------------------------------------

    print("\nTEST 1 — Reward coefficients")

    p = reward_model.parameters

    print(f"K1 = {p.k1}")
    print(f"K2 = {p.k2}")
    print(f"K3 = {p.k3}")
    print(f"K4 = {p.k4}")
    print(f"K5 = {p.k5}")

    assert p.k1 == 1.0
    assert p.k2 == 100.0
    assert p.k3 == 1000.0
    assert p.k4 == 1000.0
    assert p.k5 == 0.001

    print("✓ Reward coefficients match Wu et al. (2024)")

    # ---------------------------------------------------------------
    # TEST 2 — Zero-change transition
    # ---------------------------------------------------------------

    print("\nTEST 2 — Zero-change transition")

    state = np.array(
        [50.0, 104.14, 0.60, 1.00, 1.00, 1.00],
        dtype=np.float32,
    )

    next_state = state.copy()

    reward = reward_model.compute(state, next_state)

    expected = 1.0 / 0.001

    print(f"Reward:   {reward:.6f}")
    print(f"Expected: {expected:.6f}")

    assert np.isclose(reward, expected)

    print("✓ Zero-change reward is correct")

    # ---------------------------------------------------------------
    # TEST 3 — Battery SOC decrease only
    # ---------------------------------------------------------------

    print("\nTEST 3 — SOC decrease only")

    state = np.array(
        [50.0, 104.14, 0.6000, 1.0, 1.0, 1.0],
        dtype=np.float64,
    )

    next_state = np.array(
        [50.0, 104.14, 0.5999, 1.0, 1.0, 1.0],
        dtype=np.float64,
    )

    delta_soc = 0.6000 - 0.5999

    expected_denominator = (
        1.0 * delta_soc
        + 0.001
    )

    expected_reward = 1.0 / expected_denominator

    reward = reward_model.compute(state, next_state)

    print(f"ΔSOC:      {delta_soc:.10f}")
    print(f"Denominator: {expected_denominator:.10f}")
    print(f"Reward:      {reward:.6f}")

    assert np.isclose(reward, expected_reward)

    print("✓ SOC contribution is correct")

    # ---------------------------------------------------------------
    # TEST 4 — Battery SOH contribution
    # ---------------------------------------------------------------

    print("\nTEST 4 — Battery SOH degradation")

    state = np.array(
        [50.0, 104.14, 0.60, 1.0000, 1.0, 1.0],
        dtype=np.float64,
    )

    next_state = np.array(
        [50.0, 104.14, 0.60, 0.9999, 1.0, 1.0],
        dtype=np.float64,
    )

    delta_soh = 1.0000 - 0.9999

    expected_denominator = (
        100.0 * delta_soh
        + 0.001
    )

    expected_reward = 1.0 / expected_denominator

    reward = reward_model.compute(state, next_state)

    print(f"ΔSOH:        {delta_soh:.10f}")
    print(f"Denominator: {expected_denominator:.10f}")
    print(f"Reward:      {reward:.6f}")

    assert np.isclose(reward, expected_reward)

    print("✓ Battery SOH contribution is correct")

    # ---------------------------------------------------------------
    # TEST 5 — Motor SOH weighting
    # ---------------------------------------------------------------

    print("\nTEST 5 — Motor SOH weighting")

    state = np.array(
        [50.0, 104.14, 0.60, 1.0, 1.0, 1.0],
        dtype=np.float64,
    )

    next_state = np.array(
        [50.0, 104.14, 0.60, 1.0, 0.9999, 1.0],
        dtype=np.float64,
    )

    delta_sohm1 = 1.0 - 0.9999

    expected_denominator = (
        1000.0 * delta_sohm1
        + 0.001
    )

    expected_reward = 1.0 / expected_denominator

    reward = reward_model.compute(state, next_state)

    print(f"ΔSOHM1:      {delta_sohm1:.10f}")
    print(f"Denominator: {expected_denominator:.10f}")
    print(f"Reward:      {reward:.6f}")

    assert np.isclose(reward, expected_reward)

    print("✓ Motor SOH weighting is correct")

    # ---------------------------------------------------------------
    # TEST 6 — All four degradation terms
    # ---------------------------------------------------------------

    print("\nTEST 6 — Combined reward")

    state = np.array(
        [50.0, 104.14, 0.60, 1.0000, 1.0000, 1.0000],
        dtype=np.float64,
    )

    next_state = np.array(
        [50.0, 104.14, 0.5999, 0.9999, 0.9999, 0.9999],
        dtype=np.float64,
    )

    delta_soc = 0.60 - 0.5999
    delta_soh = 1.0 - 0.9999
    delta_sohm1 = 1.0 - 0.9999
    delta_sohm2 = 1.0 - 0.9999

    expected_denominator = (
        1.0 * delta_soc
        + 100.0 * delta_soh
        + 1000.0 * delta_sohm1
        + 1000.0 * delta_sohm2
        + 0.001
    )

    expected_reward = 1.0 / expected_denominator

    reward = reward_model.compute(state, next_state)

    print(f"ΔSOC:       {delta_soc:.10f}")
    print(f"ΔSOH:       {delta_soh:.10f}")
    print(f"ΔSOHM1:     {delta_sohm1:.10f}")
    print(f"ΔSOHM2:     {delta_sohm2:.10f}")
    print(f"Denominator: {expected_denominator:.10f}")
    print(f"Reward:      {reward:.6f}")

    assert np.isclose(reward, expected_reward)

    print("✓ Combined reward is correct")

    # ---------------------------------------------------------------
    # TEST 7 — Charging direction
    # ---------------------------------------------------------------

    print("\nTEST 7 — Signed SOC change")

    state = np.array(
        [50.0, -100.0, 0.60, 1.0, 1.0, 1.0],
        dtype=np.float64,
    )

    next_state = np.array(
        [50.0, -100.0, 0.6001, 1.0, 1.0, 1.0],
        dtype=np.float64,
    )

    reward = reward_model.compute(state, next_state)

    delta_soc = 0.60 - 0.6001
    expected_denominator = delta_soc + 0.001
    expected_reward = 1.0 / expected_denominator

    print(f"ΔSOC:        {delta_soc:.10f}")
    print(f"Denominator: {expected_denominator:.10f}")
    print(f"Reward:      {reward:.6f}")

    assert np.isclose(reward, expected_reward)

    print("✓ Signed SOC convention preserved")

    # ---------------------------------------------------------------
    # TEST 8 — Invalid state dimensions
    # ---------------------------------------------------------------

    print("\nTEST 8 — Invalid state dimensions")

    try:
        reward_model.compute(
            [0.0, 0.0, 0.6, 1.0, 1.0],
            [0.0, 0.0, 0.6, 1.0, 1.0],
        )
    except ValueError:
        print("✓ Invalid state dimension rejected")
    else:
        raise AssertionError(
            "Invalid state dimension was accepted."
        )

    # ---------------------------------------------------------------
    # TEST 9 — Invalid health values
    # ---------------------------------------------------------------

    print("\nTEST 9 — Invalid health values")

    invalid_state = np.array(
        [50.0, 104.14, 1.2, 1.0, 1.0, 1.0],
        dtype=np.float64,
    )

    valid_next_state = np.array(
        [50.0, 104.14, 1.0, 1.0, 1.0, 1.0],
        dtype=np.float64,
    )

    try:
        reward_model.compute(
            invalid_state,
            valid_next_state,
        )
    except ValueError:
        print("✓ Invalid SOC rejected")
    else:
        raise AssertionError(
            "Invalid SOC value was accepted."
        )

    # ---------------------------------------------------------------
    # FINAL
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("✓ PHASE 3D.4 REWARD FUNCTION VALIDATION PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
    