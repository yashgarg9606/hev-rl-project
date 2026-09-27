"""
Phase 3D.6 — Physical constraint handling.

Reference:
C. Wu et al., "Health-awareness energy management strategy for
battery electric vehicles based on self-attention deep reinforcement
learning," Journal of Power Sources 623 (2024) 235463.

The paper specifies the following boundary conditions:

    0.2 <= SOC <= 0.9

    0.8 <= battery SOH <= 1.0
    0.8 <= motor 1 SOH <= 1.0
    0.8 <= motor 2 SOH <= 1.0

    |motor 1 speed| <= 12000 rpm
    |motor 2 speed| <= 12000 rpm

    |motor 1 torque| <= 180 Nm
    |motor 2 torque| <= 155 Nm

Important:
    The paper specifies these boundary conditions but does not
    explicitly specify a numerical reward penalty or action-clipping
    mechanism for violations.

Therefore this module only evaluates constraint satisfaction.
It does not invent a penalty, clipping rule, or recovery action.
"""

from __future__ import annotations

from dataclasses import dataclass
from .motor_model import within_motor_limit


@dataclass(frozen=True)
class ConstraintParameters:
    """Physical boundary conditions from Wu et al. (2024)."""

    soc_min: float = 0.2
    soc_max: float = 0.9

    battery_soh_min: float = 0.8
    battery_soh_max: float = 1.0

    motor1_soh_min: float = 0.8
    motor1_soh_max: float = 1.0

    motor2_soh_min: float = 0.8
    motor2_soh_max: float = 1.0

    motor1_max_speed_rpm: float = 12000.0
    motor2_max_speed_rpm: float = 12000.0

    motor1_max_torque_nm: float = 180.0
    motor2_max_torque_nm: float = 155.0


@dataclass(frozen=True)
class ConstraintResult:
    """Result of evaluating all physical constraints."""

    soc_valid: bool

    battery_soh_valid: bool
    motor1_soh_valid: bool
    motor2_soh_valid: bool

    motor1_speed_valid: bool
    motor2_speed_valid: bool

    motor1_torque_valid: bool
    motor2_torque_valid: bool

    overall_valid: bool

    violated_constraints: tuple[str, ...]


class PhysicalConstraintChecker:
    """
    Evaluate the physical boundary conditions defined by Wu et al.
    """

    def __init__(
        self,
        parameters: ConstraintParameters | None = None,
    ) -> None:

        self.parameters = (
            parameters
            if parameters is not None
            else ConstraintParameters()
        )

    def check(
        self,
        *,
        soc: float,
        battery_soh: float,
        motor1_soh: float,
        motor2_soh: float,
        motor1_speed_rpm: float,
        motor2_speed_rpm: float,
        motor1_torque_nm: float,
        motor2_torque_nm: float,
    ) -> ConstraintResult:
        """
        Evaluate all physical constraints.

        All speed and torque limits are evaluated using absolute
        magnitudes, matching Eq. (26) of Wu et al. (2024).
        """

        p = self.parameters

        # --------------------------------------------------------------
        # SOC
        # --------------------------------------------------------------

        soc_valid = (
            p.soc_min <= soc <= p.soc_max
        )

        # --------------------------------------------------------------
        # Battery / motor SOH
        # --------------------------------------------------------------

        battery_soh_valid = (
            p.battery_soh_min
            <= battery_soh
            <= p.battery_soh_max
        )

        motor1_soh_valid = (
            p.motor1_soh_min
            <= motor1_soh
            <= p.motor1_soh_max
        )

        motor2_soh_valid = (
            p.motor2_soh_min
            <= motor2_soh
            <= p.motor2_soh_max
        )

        # --------------------------------------------------------------
        # Motor speed
        # --------------------------------------------------------------

        motor1_speed_valid = (
            within_motor_limit(motor1_speed_rpm, p.motor1_max_speed_rpm)
        )

        motor2_speed_valid = (
            within_motor_limit(motor2_speed_rpm, p.motor2_max_speed_rpm)
        )

        # --------------------------------------------------------------
        # Motor torque
        # --------------------------------------------------------------

        motor1_torque_valid = (
            within_motor_limit(motor1_torque_nm, p.motor1_max_torque_nm)
        )

        motor2_torque_valid = (
            within_motor_limit(motor2_torque_nm, p.motor2_max_torque_nm)
        )

        # --------------------------------------------------------------
        # Collect violations
        # --------------------------------------------------------------

        violations: list[str] = []

        if not soc_valid:
            violations.append("soc")

        if not battery_soh_valid:
            violations.append("battery_soh")

        if not motor1_soh_valid:
            violations.append("motor1_soh")

        if not motor2_soh_valid:
            violations.append("motor2_soh")

        if not motor1_speed_valid:
            violations.append("motor1_speed")

        if not motor2_speed_valid:
            violations.append("motor2_speed")

        if not motor1_torque_valid:
            violations.append("motor1_torque")

        if not motor2_torque_valid:
            violations.append("motor2_torque")

        overall_valid = len(violations) == 0

        return ConstraintResult(
            soc_valid=soc_valid,
            battery_soh_valid=battery_soh_valid,
            motor1_soh_valid=motor1_soh_valid,
            motor2_soh_valid=motor2_soh_valid,
            motor1_speed_valid=motor1_speed_valid,
            motor2_speed_valid=motor2_speed_valid,
            motor1_torque_valid=motor1_torque_valid,
            motor2_torque_valid=motor2_torque_valid,
            overall_valid=overall_valid,
            violated_constraints=tuple(violations),
        )
