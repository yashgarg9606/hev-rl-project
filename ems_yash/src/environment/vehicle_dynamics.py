"""
Vehicle longitudinal dynamics for the dual-motor BEV plant.

Based on the vehicle model described in:
Wu et al., Journal of Power Sources 623 (2024) 235463.

The model calculates:
    - vehicle acceleration
    - rolling resistance
    - aerodynamic drag
    - grade force
    - inertial force
    - total tractive force
    - required wheel torque
"""

from dataclasses import dataclass
import math


@dataclass
class VehicleParameters:
    """Physical parameters of the BEV."""

    mass_kg: float = 1635.0
    rotational_mass_factor: float = 1.04

    frontal_area_m2: float = 2.16
    drag_coefficient: float = 0.25
    rolling_resistance: float = 0.016

    wheel_radius_m: float = 0.325
    gravity_mps2: float = 9.81

    air_density_kgpm3: float = 1.225


class VehicleDynamics:
    """
    Longitudinal vehicle dynamics model.

    All internal velocity calculations use m/s.
    Driving-cycle inputs may be supplied in km/h.
    """

    def __init__(self, params: VehicleParameters | None = None):
        self.params = params or VehicleParameters()

    @staticmethod
    def kmh_to_mps(velocity_kmh: float) -> float:
        """Convert velocity from km/h to m/s."""
        return velocity_kmh / 3.6

    @staticmethod
    def mps_to_kmh(velocity_mps: float) -> float:
        """Convert velocity from m/s to km/h."""
        return velocity_mps * 3.6

    def calculate_acceleration(
        self,
        velocity_kmh: float,
        target_velocity_kmh: float,
        dt: float = 1.0,
    ) -> float:
        """
        Calculate required acceleration.

        a = (v_target(t+1) - v(t)) / dt
        """

        v_current = self.kmh_to_mps(velocity_kmh)
        v_target = self.kmh_to_mps(target_velocity_kmh)

        return (v_target - v_current) / dt

    def rolling_force(self, velocity_mps: float) -> float:
        """Calculate rolling resistance force."""

        p = self.params

        return (
            p.mass_kg
            * p.gravity_mps2
            * p.rolling_resistance
        )

    def aerodynamic_force(self, velocity_mps: float) -> float:
        """Calculate aerodynamic drag force."""

        p = self.params

        return (
            0.5
            * p.air_density_kgpm3
            * p.drag_coefficient
            * p.frontal_area_m2
            * velocity_mps**2
        )

    def grade_force(self, slope_rad: float) -> float:
        """Calculate road grade force."""

        p = self.params

        return (
            p.mass_kg
            * p.gravity_mps2
            * math.sin(slope_rad)
        )

    def inertial_force(self, acceleration_mps2: float) -> float:
        """Calculate force required for vehicle acceleration."""

        p = self.params

        return (
            p.rotational_mass_factor
            * p.mass_kg
            * acceleration_mps2
        )

    def calculate_required_force(
        self,
        velocity_kmh: float,
        target_velocity_kmh: float,
        slope_rad: float = 0.0,
        dt: float = 1.0,
    ) -> dict:
        """
        Calculate all longitudinal force components.

        Returns a dictionary containing acceleration and
        individual force components.
        """

        velocity_mps = self.kmh_to_mps(velocity_kmh)

        acceleration = self.calculate_acceleration(
            velocity_kmh,
            target_velocity_kmh,
            dt,
        )

        f_inertia = self.inertial_force(acceleration)
        f_grade = self.grade_force(slope_rad)
        f_rolling = self.rolling_force(velocity_mps)
        f_aero = self.aerodynamic_force(velocity_mps)

        total_force = (
            f_inertia
            + f_grade
            + f_rolling
            + f_aero
        )

        return {
            "velocity_mps": velocity_mps,
            "acceleration_mps2": acceleration,
            "force_inertia_n": f_inertia,
            "force_grade_n": f_grade,
            "force_rolling_n": f_rolling,
            "force_aero_n": f_aero,
            "force_total_n": total_force,
        }

    def calculate_wheel_torque(
        self,
        velocity_kmh: float,
        target_velocity_kmh: float,
        slope_rad: float = 0.0,
        dt: float = 1.0,
    ) -> float:
        """
        Calculate required wheel torque.

        T_d = F_total * R
        """

        forces = self.calculate_required_force(
            velocity_kmh=velocity_kmh,
            target_velocity_kmh=target_velocity_kmh,
            slope_rad=slope_rad,
            dt=dt,
        )

        return (
            forces["force_total_n"]
            * self.params.wheel_radius_m
        )