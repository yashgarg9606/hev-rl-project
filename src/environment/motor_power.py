"""
Motor electrical-power model for the dual-motor BEV plant.

The model converts motor torque and speed into mechanical and
electrical power using the reconstructed motor efficiency map.

Sign convention
---------------
Positive torque  -> motoring
Negative torque  -> regenerative braking

Positive electrical power:
    battery/electrical system -> motor

Negative electrical power:
    motor -> battery/electrical system

The published Wu et al. (2024) efficiency contours do not provide
a separate negative-torque efficiency map. Therefore, regenerative
operation uses the efficiency obtained from the positive-torque map
at the corresponding absolute torque magnitude.

This regenerative-efficiency treatment is an implementation
assumption and is not claimed as a direct equation from the paper.
"""

from dataclasses import dataclass

from motor_efficiency_map import MotorEfficiencyMap


@dataclass
class MotorPowerResult:
    """Calculated motor power quantities."""

    speed_rpm: float
    torque_nm: float
    angular_speed_rad_s: float
    mechanical_power_w: float
    mechanical_power_kw: float
    efficiency_percent: float
    electrical_power_w: float
    electrical_power_kw: float


class MotorPowerModel:
    """Convert motor torque/speed into electrical power."""

    def __init__(self, efficiency_map: MotorEfficiencyMap):
        self.efficiency_map = efficiency_map

    @staticmethod
    def rpm_to_rad_s(speed_rpm: float) -> float:
        """Convert rotational speed from rpm to rad/s."""

        if speed_rpm < 0:
            raise ValueError(
                "Motor speed cannot be negative."
            )

        return speed_rpm * 2.0 * 3.141592653589793 / 60.0

    def calculate(
        self,
        speed_rpm: float,
        torque_nm: float,
    ) -> MotorPowerResult:
        """
        Calculate mechanical and electrical motor power.

        Positive torque:
            P_elec = P_mech / eta

        Negative torque:
            P_elec = P_mech * eta

        The efficiency map is evaluated at |torque| during
        regenerative operation.
        """

        if speed_rpm < 0:
            raise ValueError(
                "Motor speed cannot be negative."
            )

        angular_speed = self.rpm_to_rad_s(speed_rpm)

        mechanical_power_w = torque_nm * angular_speed

        mechanical_power_kw = (
            mechanical_power_w / 1000.0
        )

        # The published efficiency map contains positive torque.
        map_torque = abs(torque_nm)

        # Zero torque means zero power. No efficiency lookup
        # is physically required in this special case.
        if torque_nm == 0.0:
            efficiency_percent = 0.0
            electrical_power_w = 0.0

        else:
            efficiency_percent = (
                self.efficiency_map.get_efficiency(
                    speed_rpm,
                    map_torque,
                )
            )

            efficiency = (
                efficiency_percent / 100.0
            )

            if torque_nm > 0.0:
                # Motoring:
                # electrical input > mechanical output
                electrical_power_w = (
                    mechanical_power_w / efficiency
                )

            else:
                # Regeneration:
                # mechanical power is negative and only
                # eta fraction is returned electrically.
                electrical_power_w = (
                    mechanical_power_w * efficiency
                )

        electrical_power_kw = (
            electrical_power_w / 1000.0
        )

        return MotorPowerResult(
            speed_rpm=speed_rpm,
            torque_nm=torque_nm,
            angular_speed_rad_s=angular_speed,
            mechanical_power_w=mechanical_power_w,
            mechanical_power_kw=mechanical_power_kw,
            efficiency_percent=efficiency_percent,
            electrical_power_w=electrical_power_w,
            electrical_power_kw=electrical_power_kw,
        )