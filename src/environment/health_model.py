"""
Phase 2E — Battery and Motor Health / Degradation Model

Source:
    Wu et al., Journal of Power Sources 623 (2024), 235463.

Battery:
    Energy-throughput / Arrhenius aging model.
    Equations (5)-(9).

Motor:
    Cumulative Loss Ratio (CLR) model.
    Equations (10)-(13).

Important:
    No degradation coefficients are invented here.

The battery B(c) values are taken from the cited
energy-throughput aging source used by Wu et al.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ----------------------------------------------------------------------
# Battery aging parameters
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class BatteryHealthParameters:
    """
    Parameters for the energy-throughput battery aging model.
    """

    nominal_capacity_ah: float = 72.0

    gas_constant_j_mol_k: float = 8.31

    power_law_exponent: float = 0.55

    # Explicitly identified tabulated B(c) values.
    #
    # C-rate [C] : B(c)
    # 0.5        : 31630
    # 2.0        : 21681
    # 10.0       : 12934
    c_rate_table: tuple[float, ...] = (
        0.5,
        2.0,
        10.0,
    )

    b_table: tuple[float, ...] = (
        31630.0,
        21681.0,
        12934.0,
    )

    # Wu et al. Eq. (6):
    # Ea(c) = 31700 - 370.3*c
    activation_energy_intercept_j_mol: float = 31700.0
    activation_energy_c_coefficient: float = 370.3

    # Paper constraint:
    # EOL occurs at 20% capacity loss.
    eol_capacity_loss_percent: float = 20.0


@dataclass
class BatteryHealthState:
    """
    Dynamic battery health state.
    """

    soh: float = 1.0

    cumulative_ah_throughput: float = 0.0


# ----------------------------------------------------------------------
# Motor aging parameters
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class MotorHealthParameters:
    """
    Parameters for the motor cumulative-loss-ratio model.
    """

    rated_power_kw: float
    rated_efficiency: float

    life_hours: float = 30_000.0


@dataclass
class MotorHealthState:
    """
    Dynamic motor health state.
    """

    soh: float = 1.0

    cumulative_energy_loss_kwh: float = 0.0


# ----------------------------------------------------------------------
# Health model
# ----------------------------------------------------------------------

class HealthDegradationModel:
    """
    Battery + dual-motor health model.

    Battery:
        Uses C-rate and internal battery temperature to determine
        the EOL cycle count N(c, Ta), then updates SOH.

    Motors:
        Uses cumulative energy loss to calculate CLR and SOH.

    The health model is intentionally independent of the physical
    powertrain. This allows independent validation before integration.
    """

    def __init__(
        self,
        battery_parameters: BatteryHealthParameters | None = None,
        motor1_parameters: MotorHealthParameters | None = None,
        motor2_parameters: MotorHealthParameters | None = None,
        battery_state: BatteryHealthState | None = None,
        motor1_state: MotorHealthState | None = None,
        motor2_state: MotorHealthState | None = None,
    ):
        self.battery_parameters = (
            battery_parameters or BatteryHealthParameters()
        )

        self.motor1_parameters = motor1_parameters or MotorHealthParameters(
            rated_power_kw=28.0,
            rated_efficiency=0.98,
        )

        self.motor2_parameters = motor2_parameters or MotorHealthParameters(
            rated_power_kw=24.0,
            rated_efficiency=0.98,
        )

        self.battery_state = (
            battery_state or BatteryHealthState()
        )

        self.motor1_state = (
            motor1_state or MotorHealthState()
        )

        self.motor2_state = (
            motor2_state or MotorHealthState()
        )

        self._validate_parameters()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_parameters(self) -> None:

        bp = self.battery_parameters

        if bp.nominal_capacity_ah <= 0:
            raise ValueError(
                "Battery nominal capacity must be positive."
            )

        if bp.gas_constant_j_mol_k <= 0:
            raise ValueError(
                "Gas constant must be positive."
            )

        if bp.power_law_exponent <= 0:
            raise ValueError(
                "Power-law exponent must be positive."
            )

        if len(bp.c_rate_table) != len(bp.b_table):
            raise ValueError(
                "C-rate and B(c) tables must have equal length."
            )

        if any(
            b <= 0
            for b in bp.b_table
        ):
            raise ValueError(
                "All B(c) values must be positive."
            )

        for motor_parameters in (
            self.motor1_parameters,
            self.motor2_parameters,
        ):
            if motor_parameters.rated_power_kw <= 0:
                raise ValueError(
                    "Motor rated power must be positive."
                )

            if not (
                0 < motor_parameters.rated_efficiency <= 1
            ):
                raise ValueError(
                    "Motor rated efficiency must be in (0, 1]."
                )

            if motor_parameters.life_hours <= 0:
                raise ValueError(
                    "Motor life must be positive."
                )

    # ------------------------------------------------------------------
    # Battery aging
    # ------------------------------------------------------------------

    def c_rate(
        self,
        current_a: float,
    ) -> float:
        """
        Calculate absolute battery C-rate.

        c = |I| / Qbat
        """

        return abs(current_a) / (
            self.battery_parameters.nominal_capacity_ah
        )

    def b_factor(
        self,
        c_rate: float,
    ) -> float:
        """
        Obtain B(c) by linear interpolation of the
        explicitly available tabulated values.

        For C-rates below/above the tabulated range,
        the nearest tabulated value is used.

        This boundary treatment is an implementation
        convention; no new coefficient is introduced.
        """

        c_values = np.asarray(
            self.battery_parameters.c_rate_table,
            dtype=float,
        )

        b_values = np.asarray(
            self.battery_parameters.b_table,
            dtype=float,
        )

        return float(
            np.interp(
                c_rate,
                c_values,
                b_values,
            )
        )

    def activation_energy(
        self,
        c_rate: float,
    ) -> float:
        """
        Wu et al. Eq. (6):

            Ea(c) = 31700 - 370.3*c
        """

        return (
            self.battery_parameters.activation_energy_intercept_j_mol
            - self.battery_parameters.activation_energy_c_coefficient
            * c_rate
        )

    def end_of_life_amp_hours(
        self,
        c_rate: float,
        temperature_c: float,
    ) -> float:
        """
        Wu et al. Eq. (7):

            Ah(c,Ta) =
                [
                    20 /
                    (B(c) *
                     exp(-Ea(c)/(Rg*(Ta+273.15))))
                ]^(1/z)

        Returns the amp-hour throughput corresponding
        to the 20% capacity-loss EOL definition.
        """

        temperature_k = temperature_c + 273.15

        if temperature_k <= 0:
            raise ValueError(
                "Absolute battery temperature must be positive."
            )

        b_value = self.b_factor(c_rate)

        ea = self.activation_energy(c_rate)

        exponent = -ea / (
            self.battery_parameters.gas_constant_j_mol_k
            * temperature_k
        )

        denominator = b_value * np.exp(exponent)

        ah_eol = (
            self.battery_parameters.eol_capacity_loss_percent
            / denominator
        ) ** (
            1.0
            / self.battery_parameters.power_law_exponent
        )

        return float(ah_eol)

    def eol_cycles(
        self,
        c_rate: float,
        temperature_c: float,
    ) -> float:
        """
        Wu et al. Eq. (8):

            N(c,Ta) = 3600 * Ah(c,Ta) / Qbat
        """

        ah_eol = self.end_of_life_amp_hours(
            c_rate=c_rate,
            temperature_c=temperature_c,
        )

        return (
            3600.0
            * ah_eol
            / self.battery_parameters.nominal_capacity_ah
        )

    def battery_soh_change(
        self,
        current_a: float,
        temperature_c: float,
        dt_s: float,
    ) -> float:
        """
        Wu et al. Eq. (9):

            ΔSOH =
                - |I(t)| Δt /
                  [2 N(c,Ta) Qbat]

        The simulation timestep is supplied in seconds,
        consistent with the discrete-time implementation.
        """

        if dt_s <= 0:
            raise ValueError(
                "dt_s must be positive."
            )

        if current_a == 0.0:
            return 0.0

        c_rate = self.c_rate(current_a)

        n_eol = self.eol_cycles(
            c_rate=c_rate,
            temperature_c=temperature_c,
        )

        delta_soh = -(
            abs(current_a)
            * dt_s
            / (
                2.0
                * n_eol
                * self.battery_parameters.nominal_capacity_ah
            )
        )

        return float(delta_soh)

    def update_battery(
        self,
        current_a: float,
        temperature_c: float,
        dt_s: float,
    ) -> float:
        """
        Update battery SOH and cumulative Ah throughput.
        """

        delta_soh = self.battery_soh_change(
            current_a=current_a,
            temperature_c=temperature_c,
            dt_s=dt_s,
        )

        self.battery_state.soh += delta_soh

        # Cumulative absolute Ah throughput.
        self.battery_state.cumulative_ah_throughput += (
            abs(current_a)
            * dt_s
            / 3600.0
        )

        # Numerical protection.
        self.battery_state.soh = float(
            np.clip(
                self.battery_state.soh,
                0.0,
                1.0,
            )
        )

        return delta_soh

    # ------------------------------------------------------------------
    # Motor aging
    # ------------------------------------------------------------------

    @staticmethod
    def motor_lifetime_energy_loss_kwh(
        parameters: MotorHealthParameters,
    ) -> float:
        """
        Wu et al. Eq. (10):

            Wlosses =
                ∫ (
                    1/ηrated - 1
                ) Prated dt

        With constant rated operation:

            Wlosses =
                (1/ηrated - 1)
                Prated
                tlife
        """

        return (
            (
                1.0 / parameters.rated_efficiency
                - 1.0
            )
            * parameters.rated_power_kw
            * parameters.life_hours
        )

    @staticmethod
    def motor_step_energy_loss_kwh(
        mechanical_power_kw: float,
        efficiency: float,
        dt_s: float,
    ) -> float:
        """
        Wu et al. Eq. (11):

            Wloss_i =
                ∫ Pout (1 - η) dt

        Absolute mechanical power is used for the
        degradation throughput so that regenerative
        operation also contributes positively to motor
        wear.

        This absolute-value treatment for regenerative
        operation is an explicit implementation convention.
        """

        if dt_s <= 0:
            raise ValueError(
                "dt_s must be positive."
            )

        if mechanical_power_kw == 0.0:
            return 0.0

        if not (
            0 < efficiency <= 1
        ):
            raise ValueError(
                "Motor efficiency must be in (0,1]."
            )

        return (
            abs(mechanical_power_kw)
            * (1.0 - efficiency)
            * dt_s
            / 3600.0
        )

    def motor_soh_change(
        self,
        motor_parameters: MotorHealthParameters,
        motor_state: MotorHealthState,
        mechanical_power_kw: float,
        efficiency: float,
        dt_s: float,
    ) -> float:
        """
        Calculate the SOH change caused by one timestep.
        """

        step_loss = self.motor_step_energy_loss_kwh(
            mechanical_power_kw=mechanical_power_kw,
            efficiency=efficiency,
            dt_s=dt_s,
        )

        lifetime_loss = (
            self.motor_lifetime_energy_loss_kwh(
                motor_parameters
            )
        )

        delta_soh = -step_loss / lifetime_loss

        return float(delta_soh)

    def update_motor1(
        self,
        mechanical_power_kw: float,
        efficiency: float,
        dt_s: float,
    ) -> float:

        delta_soh = self.motor_soh_change(
            motor_parameters=self.motor1_parameters,
            motor_state=self.motor1_state,
            mechanical_power_kw=mechanical_power_kw,
            efficiency=efficiency,
            dt_s=dt_s,
        )

        step_loss = self.motor_step_energy_loss_kwh(
            mechanical_power_kw=mechanical_power_kw,
            efficiency=efficiency,
            dt_s=dt_s,
        )

        self.motor1_state.soh += delta_soh

        self.motor1_state.cumulative_energy_loss_kwh += (
            step_loss
        )

        self.motor1_state.soh = float(
            np.clip(
                self.motor1_state.soh,
                0.0,
                1.0,
            )
        )

        return delta_soh

    def update_motor2(
        self,
        mechanical_power_kw: float,
        efficiency: float,
        dt_s: float,
    ) -> float:

        delta_soh = self.motor_soh_change(
            motor_parameters=self.motor2_parameters,
            motor_state=self.motor2_state,
            mechanical_power_kw=mechanical_power_kw,
            efficiency=efficiency,
            dt_s=dt_s,
        )

        step_loss = self.motor_step_energy_loss_kwh(
            mechanical_power_kw=mechanical_power_kw,
            efficiency=efficiency,
            dt_s=dt_s,
        )

        self.motor2_state.soh += delta_soh

        self.motor2_state.cumulative_energy_loss_kwh += (
            step_loss
        )

        self.motor2_state.soh = float(
            np.clip(
                self.motor2_state.soh,
                0.0,
                1.0,
            )
        )

        return delta_soh

    # ------------------------------------------------------------------
    # Convenience reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Reset all health states to fresh-component condition.
        """

        self.battery_state = BatteryHealthState()
        self.motor1_state = MotorHealthState()
        self.motor2_state = MotorHealthState()

    # ------------------------------------------------------------------
    # Health summary
    # ------------------------------------------------------------------

    def get_health_state(self) -> dict[str, float]:
        return {
            "battery_soh": self.battery_state.soh,
            "motor1_soh": self.motor1_state.soh,
            "motor2_soh": self.motor2_state.soh,
        }