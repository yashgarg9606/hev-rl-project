"""
Phase 3D.4 — Wu et al. (2024) reward function.

Reference:
C. Wu et al., "Health-awareness energy management strategy for
battery electric vehicles based on self-attention deep reinforcement
learning," Journal of Power Sources 623 (2024) 235463.

Paper reward equation (25):

    Reward = 1 / (
        K1 * ΔSOC
        + K2 * ΔSOH
        + K3 * ΔSOHM1
        + K4 * ΔSOHM2
        + K5
    )

with:

    K1 = 1
    K2 = 100
    K3 = 1000
    K4 = 1000
    K5 = 0.001

Implementation convention:
    ΔSOC  = SOC_t  - SOC_{t+1}
    ΔSOH  = SOH_t  - SOH_{t+1}
    ΔSOHM1 = SOHM1_t - SOHM1_{t+1}
    ΔSOHM2 = SOHM2_t - SOHM2_{t+1}

The paper's equation uses Δ notation but does not explicitly define
these deltas as absolute values in the surrounding text. Therefore
this implementation uses signed decreases rather than silently
introducing abs().
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardParameters:
    """Reward coefficients from Wu et al. (2024), Eq. (25)."""

    k1: float = 1.0
    k2: float = 100.0
    k3: float = 1000.0
    k4: float = 1000.0
    k5: float = 0.001


class WuReward:
    """
    Implementation of the reward function from Wu et al. (2024).

    The reward is computed from the state transition:

        state_t -> state_{t+1}

    using the SOC and SOH components of the state.
    """

    def __init__(
        self,
        parameters: RewardParameters | None = None,
    ) -> None:
        self.parameters = (
            parameters
            if parameters is not None
            else RewardParameters()
        )

    @staticmethod
    def _validate_health_values(
        soc: float,
        soh: float,
        sohm1: float,
        sohm2: float,
    ) -> None:
        values = {
            "SOC": soc,
            "SOH": soh,
            "SOHM1": sohm1,
            "SOHM2": sohm2,
        }

        for name, value in values.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must be within [0, 1], got {value}."
                )

    def compute(
        self,
        state: list[float] | tuple[float, ...],
        next_state: list[float] | tuple[float, ...],
    ) -> float:
        """
        Compute the Wu et al. reward for one environment transition.

        State ordering:

            [velocity, demanded_torque, SOC, SOH, SOHM1, SOHM2]

        Returns:
            Scalar reward.
        """

        if len(state) != 6:
            raise ValueError(
                f"state must have length 6, got {len(state)}."
            )

        if len(next_state) != 6:
            raise ValueError(
                f"next_state must have length 6, got {len(next_state)}."
            )

        soc_t = float(state[2])
        soh_t = float(state[3])
        sohm1_t = float(state[4])
        sohm2_t = float(state[5])

        soc_next = float(next_state[2])
        soh_next = float(next_state[3])
        sohm1_next = float(next_state[4])
        sohm2_next = float(next_state[5])

        self._validate_health_values(
            soc_t,
            soh_t,
            sohm1_t,
            sohm2_t,
        )

        self._validate_health_values(
            soc_next,
            soh_next,
            sohm1_next,
            sohm2_next,
        )

        # Signed decreases.
        delta_soc = soc_t - soc_next
        delta_soh = soh_t - soh_next
        delta_sohm1 = sohm1_t - sohm1_next
        delta_sohm2 = sohm2_t - sohm2_next

        p = self.parameters

        denominator = (
            p.k1 * delta_soc
            + p.k2 * delta_soh
            + p.k3 * delta_sohm1
            + p.k4 * delta_sohm2
            + p.k5
        )

        if denominator == 0.0:
            raise ZeroDivisionError(
                "Reward denominator is exactly zero."
            )

        return float(1.0 / denominator)