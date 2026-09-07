"""
Integrated vehicle + dual-motor + battery powertrain model.

Phase 2D
--------

Composes the validated Phase 2A, 2B and 2C components:

    driving-cycle point
        -> vehicle longitudinal dynamics
        -> wheel torque
        -> dual-motor torque split
        -> motor operating points
        -> motor efficiency maps
        -> motor electrical power
        -> battery electrical power
        -> battery current
        -> battery terminal voltage
        -> battery state update

This module is intentionally an integration layer. It does not duplicate
the underlying vehicle, motor, efficiency-map, or battery equations.

Current conventions
-------------------
Vehicle:
    positive wheel torque -> traction
    negative wheel torque -> braking / regeneration

Motor:
    positive electrical power -> battery supplies motor
    negative electrical power -> motor regenerates to battery

Battery (Wu convention):
    positive battery power -> charging
    negative battery power -> discharging

    positive current -> charging
    negative current -> discharging
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from battery_model import (
    BatteryElectricalModel,
    BatteryElectricalParameters,
    BatteryState,
)
from motor_efficiency_map import MotorEfficiencyMap
from motor_model import DualMotorModel, DualMotorParameters
from motor_power import MotorPowerModel
from vehicle_dynamics import VehicleDynamics, VehicleParameters


@dataclass
class IntegratedPowertrainParameters:
    """Configuration for the complete Phase 2D plant."""

    wheel_radius_m: float = 0.325

    battery_capacity_ah: float = 72.0
    battery_ocv_v: float = 255.5
    battery_ohmic_resistance_ohm: float = 0.0031

    initial_soc: float = 0.6
    initial_polarization_voltage_1_v: float = 0.0
    initial_polarization_voltage_2_v: float = 0.0

    battery_temperature_c: float = 25.0


@dataclass
class PowertrainStepResult:
    """Complete result from one integrated simulation step."""

    # Driving / vehicle
    velocity_kmh: float
    target_velocity_kmh: float
    slope_rad: float
    acceleration_mps2: float
    wheel_torque_nm: float

    # Motor operating points
    motor1_speed_rpm: float
    motor1_torque_nm: float
    motor1_efficiency_percent: float
    motor1_electrical_power_kw: float

    motor2_speed_rpm: float
    motor2_torque_nm: float
    motor2_efficiency_percent: float
    motor2_electrical_power_kw: float

    # Battery
    motor_total_electrical_power_kw: float
    battery_power_kw: float
    battery_current_a: float
    battery_terminal_voltage_v: float

    # Battery state after the step
    soc: float
    polarization_voltage_1_v: float
    polarization_voltage_2_v: float

    # Feasibility
    motor1_feasible: bool
    motor2_feasible: bool
    overall_feasible: bool


class IntegratedPowertrain:
    """
    Complete physical plant for Phase 2D.

    The environment/agent layer will later call this model once per
    simulation timestep.
    """

    def __init__(
        self,
        motor1_map_path: str | Path,
        motor2_map_path: str | Path,
        vehicle_parameters: VehicleParameters | None = None,
        motor_parameters: DualMotorParameters | None = None,
        parameters: IntegratedPowertrainParameters | None = None,
    ):
        self.parameters = parameters or IntegratedPowertrainParameters()

        self.vehicle = VehicleDynamics(
            params=vehicle_parameters
        )

        self.motors = DualMotorModel(
            params=motor_parameters,
            wheel_radius_m=self.parameters.wheel_radius_m,
        )

        self.motor1_power = MotorPowerModel(
            MotorEfficiencyMap(motor1_map_path)
        )

        self.motor2_power = MotorPowerModel(
            MotorEfficiencyMap(motor2_map_path)
        )

        self.battery = BatteryElectricalModel(
            parameters=BatteryElectricalParameters(
                nominal_capacity_ah=self.parameters.battery_capacity_ah,
                nominal_ocv_v=self.parameters.battery_ocv_v,
                ohmic_resistance_ohm=(
                    self.parameters.battery_ohmic_resistance_ohm
                ),
            ),
            initial_state=BatteryState(
                soc=self.parameters.initial_soc,
                polarization_voltage_1_v=(
                    self.parameters.initial_polarization_voltage_1_v
                ),
                polarization_voltage_2_v=(
                    self.parameters.initial_polarization_voltage_2_v
                ),
            ),
        )

    def step(
        self,
        velocity_kmh: float,
        target_velocity_kmh: float,
        sigma_tor: float,
        slope_rad: float = 0.0,
        dt_s: float = 1.0,
        battery_temperature_c: float | None = None,
    ) -> PowertrainStepResult:
        """
        Advance the complete physical plant by one timestep.

        Parameters
        ----------
        velocity_kmh:
            Current vehicle velocity.

        target_velocity_kmh:
            Velocity at the end of the timestep.

        sigma_tor:
            Fraction of wheel torque assigned to Motor 1.

        slope_rad:
            Road slope in radians.

        dt_s:
            Simulation timestep.

        battery_temperature_c:
            Battery temperature used by the electrical
            parameterization. Defaults to configured temperature.
        """

        if dt_s <= 0:
            raise ValueError("dt_s must be positive.")

        if battery_temperature_c is None:
            battery_temperature_c = (
                self.parameters.battery_temperature_c
            )

        # --------------------------------------------------------------
        # 1. Vehicle longitudinal dynamics
        # --------------------------------------------------------------

        vehicle_result = self.vehicle.calculate_required_force(
            velocity_kmh=velocity_kmh,
            target_velocity_kmh=target_velocity_kmh,
            slope_rad=slope_rad,
            dt=dt_s,
        )

        wheel_torque = self.vehicle.calculate_wheel_torque(
            velocity_kmh=velocity_kmh,
            target_velocity_kmh=target_velocity_kmh,
            slope_rad=slope_rad,
            dt=dt_s,
        )

        # --------------------------------------------------------------
        # 2. Dual-motor operating point
        # --------------------------------------------------------------

        motor_result = self.motors.calculate_operating_point(
            velocity_kmh=velocity_kmh,
            wheel_torque_nm=wheel_torque,
            sigma_tor=sigma_tor,
        )

        if not motor_result["overall_feasible"]:
            raise ValueError(
                "Requested motor operating point is infeasible: "
                f"M1 feasible={motor_result['motor1_feasible']}, "
                f"M2 feasible={motor_result['motor2_feasible']}"
            )

        # --------------------------------------------------------------
        # 3. Motor electrical power
        # --------------------------------------------------------------

        motor1_result = self.motor1_power.calculate(
            speed_rpm=motor_result["motor1_speed_rpm"],
            torque_nm=motor_result["motor1_torque_nm"],
        )

        motor2_result = self.motor2_power.calculate(
            speed_rpm=motor_result["motor2_speed_rpm"],
            torque_nm=motor_result["motor2_torque_nm"],
        )

        total_motor_electrical_power_w = (
            motor1_result.electrical_power_w
            + motor2_result.electrical_power_w
        )

        # Motor convention:
        #   +P = battery -> motor
        #   -P = motor -> battery
        #
        # Wu battery convention:
        #   +P_batt = charging
        #   -P_batt = discharging
        #
        # Therefore:
        battery_power_w = -total_motor_electrical_power_w

        # --------------------------------------------------------------
        # 4. Battery power -> current
        # --------------------------------------------------------------

        battery_current_a = (
            self.battery.current_from_battery_power(
                battery_power_w=battery_power_w,
                temperature_c=battery_temperature_c,
            )
        )

        # --------------------------------------------------------------
        # 5. Battery terminal voltage BEFORE state update
        # --------------------------------------------------------------

        battery_voltage_v = self.battery.terminal_voltage(
            current_a=battery_current_a,
            temperature_c=battery_temperature_c,
        )

        # --------------------------------------------------------------
        # 6. Battery state update
        # --------------------------------------------------------------

        battery_state = self.battery.update(
            current_a=battery_current_a,
            temperature_c=battery_temperature_c,
            dt_s=dt_s,
        )

        return PowertrainStepResult(
            velocity_kmh=velocity_kmh,
            target_velocity_kmh=target_velocity_kmh,
            slope_rad=slope_rad,
            acceleration_mps2=vehicle_result[
                "acceleration_mps2"
            ],
            wheel_torque_nm=wheel_torque,

            motor1_speed_rpm=motor_result[
                "motor1_speed_rpm"
            ],
            motor1_torque_nm=motor_result[
                "motor1_torque_nm"
            ],
            motor1_efficiency_percent=(
                motor1_result.efficiency_percent
            ),
            motor1_electrical_power_kw=(
                motor1_result.electrical_power_kw
            ),

            motor2_speed_rpm=motor_result[
                "motor2_speed_rpm"
            ],
            motor2_torque_nm=motor_result[
                "motor2_torque_nm"
            ],
            motor2_efficiency_percent=(
                motor2_result.efficiency_percent
            ),
            motor2_electrical_power_kw=(
                motor2_result.electrical_power_kw
            ),

            motor_total_electrical_power_kw=(
                total_motor_electrical_power_w / 1000.0
            ),
            battery_power_kw=battery_power_w / 1000.0,
            battery_current_a=battery_current_a,
            battery_terminal_voltage_v=battery_voltage_v,

            soc=battery_state.soc,
            polarization_voltage_1_v=(
                battery_state.polarization_voltage_1_v
            ),
            polarization_voltage_2_v=(
                battery_state.polarization_voltage_2_v
            ),

            motor1_feasible=motor_result[
                "motor1_feasible"
            ],
            motor2_feasible=motor_result[
                "motor2_feasible"
            ],
            overall_feasible=motor_result[
                "overall_feasible"
            ],
        )


def default_motor_map_paths(
    project_root: str | Path,
) -> tuple[Path, Path]:
    """Return the project's processed Motor 1/2 map paths."""

    project_root = Path(project_root)

    map_dir = (
        project_root
        / "data"
        / "processed"
        / "motor_maps"
    )

    return (
        map_dir / "motor1_efficiency_contours.csv",
        map_dir / "motor2_efficiency_contours.csv",
    )
