"""
Dual-motor model for the BEV plant.

Based on:
C. Wu et al., Journal of Power Sources 623 (2024) 235463.

The paper models Motor 1 and Motor 2 as quasi-static PMSMs.
This module currently implements:

    - torque distribution
    - wheel-to-motor torque conversion
    - motor speed calculation
    - torque/speed constraints
    - operating-point information

Efficiency maps and motor aging are added in the next stage.
"""

from dataclasses import dataclass
import math


def within_motor_limit(value: float, limit: float) -> bool:
    """Compare a motor magnitude with only floating-point roundoff tolerance."""
    magnitude = abs(value)
    return math.isfinite(magnitude) and (
        magnitude <= limit
        or math.isclose(magnitude, limit, rel_tol=1e-12, abs_tol=1e-12)
    )


@dataclass
class MotorParameters:
    """Parameters of one drive motor."""

    max_torque_nm: float
    max_speed_rpm: float
    rated_power_kw: float
    rated_efficiency: float
    gear_ratio: float


@dataclass
class DualMotorParameters:
    """Parameters of the complete dual-motor BEV."""

    motor1: MotorParameters
    motor2: MotorParameters

    @classmethod
    def from_wu_paper(cls):
        """Create parameters reported in Wu et al. (2024)."""

        motor1 = MotorParameters(
            max_torque_nm=180.0,
            max_speed_rpm=12000.0,
            rated_power_kw=28.0,
            rated_efficiency=0.98,
            gear_ratio=11.0,
        )

        motor2 = MotorParameters(
            max_torque_nm=155.0,
            max_speed_rpm=12000.0,
            rated_power_kw=24.0,
            rated_efficiency=0.98,
            gear_ratio=6.0,
        )

        return cls(
            motor1=motor1,
            motor2=motor2,
        )


class DualMotorModel:
    """
    Dual-motor BEV powertrain model.

    Motor 1:
        Front axle
        Gear ratio = 11
        Maximum torque = 180 Nm
        Maximum speed = 12000 rpm

    Motor 2:
        Rear axle
        Gear ratio = 6
        Maximum torque = 155 Nm
        Maximum speed = 12000 rpm
    """

    def __init__(
        self,
        params: DualMotorParameters | None = None,
        wheel_radius_m: float = 0.325,
    ):
        self.params = params or DualMotorParameters.from_wu_paper()
        self.wheel_radius_m = wheel_radius_m
        if not math.isfinite(wheel_radius_m) or wheel_radius_m <= 0:
            raise ValueError("Wheel radius must be finite and positive.")
        for motor in (self.params.motor1, self.params.motor2):
            if not math.isfinite(motor.gear_ratio) or motor.gear_ratio <= 0:
                raise ValueError("Motor gear ratios must be finite and positive.")
            if any(not math.isfinite(limit) or limit < 0 for limit in
                   (motor.max_torque_nm, motor.max_speed_rpm)):
                raise ValueError("Motor torque/speed limits must be finite and nonnegative.")

    def calculate_feasible_sigma_bounds(
        self,
        velocity_kmh: float,
        wheel_torque_nm: float,
    ) -> tuple[float, float] | None:
        """Return the full interval for the configured flat torque/speed limits.

        Both traction and regeneration obey |sigma * Td / k1| <= T1max
        and |(1-sigma) * Td / k2| <= T2max. None denotes an empty interval;
        the wheel-driven shafts must satisfy speed limits even at zero torque.
        """
        if not math.isfinite(velocity_kmh) or not math.isfinite(wheel_torque_nm):
            raise ValueError("Velocity and wheel torque must be finite.")
        motor1, motor2 = self.params.motor1, self.params.motor2
        for motor in (motor1, motor2):
            speed = self.calculate_motor_speed(velocity_kmh, motor.gear_ratio)
            if not within_motor_limit(speed, motor.max_speed_rpm):
                return None

        demand = abs(wheel_torque_nm)
        if demand == 0:
            return 0.0, 1.0
        capacity1 = motor1.max_torque_nm * motor1.gear_ratio
        capacity2 = motor2.max_torque_nm * motor2.gear_ratio
        if not within_motor_limit(demand, capacity1 + capacity2):
            return None
        lower = max(0.0, 1.0 - capacity2 / demand)
        upper = min(1.0, capacity1 / demand)
        if lower > upper:
            # At total capacity the two independently computed endpoints can
            # differ by roundoff. Use their common numerical boundary.
            if not math.isclose(lower, upper, rel_tol=1e-12, abs_tol=1e-12):
                return None
            lower = upper = (lower + upper) / 2.0
        return lower, upper

    @staticmethod
    def vehicle_speed_to_wheel_speed_rpm(
        velocity_kmh: float,
        wheel_radius_m: float,
    ) -> float:
        """
        Convert vehicle speed to wheel rotational speed.

        N_wheel = v / R * 60 / (2*pi)
        """

        velocity_mps = velocity_kmh / 3.6

        if wheel_radius_m <= 0:
            raise ValueError("Wheel radius must be positive.")

        return (
            velocity_mps
            / wheel_radius_m
            * 60.0
            / (2.0 * math.pi)
        )

    def calculate_motor_speed(
        self,
        velocity_kmh: float,
        gear_ratio: float,
    ) -> float:
        """
        Calculate motor speed from vehicle speed.

        N_motor = N_wheel * gear_ratio
        """

        wheel_speed_rpm = self.vehicle_speed_to_wheel_speed_rpm(
            velocity_kmh,
            self.wheel_radius_m,
        )

        return wheel_speed_rpm * gear_ratio

    def distribute_torque(
        self,
        wheel_torque_nm: float,
        sigma_tor: float,
    ) -> dict:
        """
        Distribute required wheel torque between the two motors.

        sigma_tor = fraction of wheel torque assigned to Motor 1.

        Motor 1 wheel contribution:
            T_w1 = sigma * T_d

        Motor 2 wheel contribution:
            T_w2 = (1-sigma) * T_d

        Motor torque:
            T_m1 = T_w1 / k1
            T_m2 = T_w2 / k2
        """

        if not 0.0 <= sigma_tor <= 1.0:
            raise ValueError(
                "sigma_tor must be between 0 and 1."
            )

        motor1 = self.params.motor1
        motor2 = self.params.motor2

        wheel_torque_1 = sigma_tor * wheel_torque_nm
        wheel_torque_2 = (
            (1.0 - sigma_tor)
            * wheel_torque_nm
        )

        motor1_torque = (
            wheel_torque_1
            / motor1.gear_ratio
        )

        motor2_torque = (
            wheel_torque_2
            / motor2.gear_ratio
        )

        return {
            "wheel_torque_motor1_nm": wheel_torque_1,
            "wheel_torque_motor2_nm": wheel_torque_2,
            "motor1_torque_nm": motor1_torque,
            "motor2_torque_nm": motor2_torque,
        }

    def calculate_operating_point(
        self,
        velocity_kmh: float,
        wheel_torque_nm: float,
        sigma_tor: float,
    ) -> dict:
        """
        Calculate the complete motor operating point.
        """

        torque = self.distribute_torque(
            wheel_torque_nm=wheel_torque_nm,
            sigma_tor=sigma_tor,
        )

        motor1_speed = self.calculate_motor_speed(
            velocity_kmh,
            self.params.motor1.gear_ratio,
        )

        motor2_speed = self.calculate_motor_speed(
            velocity_kmh,
            self.params.motor2.gear_ratio,
        )

        motor1_torque = torque["motor1_torque_nm"]
        motor2_torque = torque["motor2_torque_nm"]

        motor1_feasible = (
            within_motor_limit(motor1_torque, self.params.motor1.max_torque_nm)
            and within_motor_limit(motor1_speed, self.params.motor1.max_speed_rpm)
        )

        motor2_feasible = (
            within_motor_limit(motor2_torque, self.params.motor2.max_torque_nm)
            and within_motor_limit(motor2_speed, self.params.motor2.max_speed_rpm)
        )

        return {
            **torque,
            "motor1_speed_rpm": motor1_speed,
            "motor2_speed_rpm": motor2_speed,
            "motor1_feasible": motor1_feasible,
            "motor2_feasible": motor2_feasible,
            "overall_feasible": (
                motor1_feasible
                and motor2_feasible
            ),
        }
