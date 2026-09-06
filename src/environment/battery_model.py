"""
Battery electrical model for the dual-motor BEV plant.

Model structure based on:

C. Wu et al.,
"Health-awareness energy management strategy for battery electric
vehicles based on self-attention deep reinforcement learning,"
Journal of Power Sources 623 (2024) 235463.

The paper describes a two-RC battery electrical model:

    dSOC/dt = I / (3600 * Qbat)

    dVb1/dt =
        -Vb1/(Rb1*C b1) + I/Cb1

    dVb2/dt =
        -Vb2/(Rb2*C b2) + I/Cb2

    Vt = Voc(SOC) + Vb1 + Vb2 + Rs*I

IMPORTANT
---------
The 2024 Wu paper refers to Ref. [28] for specific battery
parameters. The exact numerical parameter set is not currently
available to us.

Therefore:

    - No undocumented numerical battery parameters are used here.
    - Voc(SOC) is supplied as a callable function.
    - Electrical parameters must be explicitly supplied.
    - This module does not silently create a "typical" battery.

Current sign convention
-----------------------
The Wu model uses:

    I > 0  -> charging  -> SOC increases
    I < 0  -> discharging -> SOC decreases

Battery power under this convention:

    P_batt > 0 -> charging
    P_batt < 0 -> discharging
"""

from dataclasses import dataclass
from typing import Callable


@dataclass
class BatteryElectricalParameters:
    """
    Electrical parameters for the two-RC battery model.

    Parameters
    ----------
    nominal_capacity_ah:
        Battery nominal capacity in Ah.

    ohmic_resistance_ohm:
        Series/ohmic resistance Rs in ohms.

    rc1_resistance_ohm:
        First polarization resistance Rb1 in ohms.

    rc1_capacitance_f:
        First polarization capacitance Cb1 in farads.

    rc2_resistance_ohm:
        Second polarization resistance Rb2 in ohms.

    rc2_capacitance_f:
        Second polarization capacitance Cb2 in farads.
    """

    nominal_capacity_ah: float

    ohmic_resistance_ohm: float

    rc1_resistance_ohm: float
    rc1_capacitance_f: float

    rc2_resistance_ohm: float
    rc2_capacitance_f: float


@dataclass
class BatteryState:
    """Dynamic electrical state of the battery."""

    soc: float

    polarization_voltage_1_v: float

    polarization_voltage_2_v: float


class BatteryElectricalModel:
    """
    Second-order RC battery electrical model.

    The OCV-SOC relationship is injected as a function so that
    the model itself does not contain undocumented assumptions.
    """

    def __init__(
        self,
        parameters: BatteryElectricalParameters,
        initial_state: BatteryState,
        ocv_function: Callable[[float], float],
    ):
        self.parameters = parameters
        self.state = initial_state
        self.ocv_function = ocv_function

        self._validate_parameters()
        self._validate_state()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_parameters(self):
        p = self.parameters

        if p.nominal_capacity_ah <= 0:
            raise ValueError(
                "Nominal battery capacity must be positive."
            )

        if p.ohmic_resistance_ohm < 0:
            raise ValueError(
                "Ohmic resistance cannot be negative."
            )

        if p.rc1_resistance_ohm <= 0:
            raise ValueError(
                "RC1 resistance must be positive."
            )

        if p.rc1_capacitance_f <= 0:
            raise ValueError(
                "RC1 capacitance must be positive."
            )

        if p.rc2_resistance_ohm <= 0:
            raise ValueError(
                "RC2 resistance must be positive."
            )

        if p.rc2_capacitance_f <= 0:
            raise ValueError(
                "RC2 capacitance must be positive."
            )

        if not callable(self.ocv_function):
            raise TypeError(
                "ocv_function must be callable."
            )

    def _validate_state(self):
        if not 0.0 <= self.state.soc <= 1.0:
            raise ValueError(
                "Initial SOC must be between 0 and 1."
            )

    # ------------------------------------------------------------------
    # OCV
    # ------------------------------------------------------------------

    def open_circuit_voltage(
        self,
        soc: float,
    ) -> float:
        """
        Return open-circuit voltage for a given SOC.

        The numerical SOC-OCV relationship is deliberately supplied
        externally because the exact relationship used by Wu et al.
        is not available in the accessible source material.
        """

        if not 0.0 <= soc <= 1.0:
            raise ValueError(
                f"SOC must be between 0 and 1. Received: {soc}"
            )

        voltage = float(self.ocv_function(soc))

        if voltage <= 0:
            raise ValueError(
                "OCV must be positive."
            )

        return voltage

    # ------------------------------------------------------------------
    # Terminal voltage
    # ------------------------------------------------------------------

    def calculate_terminal_voltage(
        self,
        current_a: float,
        soc: float | None = None,
    ) -> float:
        """
        Calculate battery terminal voltage.

        Model equation:

            Vt = Voc + Vb1 + Vb2 + Rs*I

        Sign convention:

            I > 0 -> charging
            I < 0 -> discharging
        """

        if soc is None:
            soc = self.state.soc

        voc = self.open_circuit_voltage(soc)

        voltage = (
            voc
            + self.state.polarization_voltage_1_v
            + self.state.polarization_voltage_2_v
            + self.parameters.ohmic_resistance_ohm
            * current_a
        )

        if voltage <= 0:
            raise ValueError(
                "Calculated terminal voltage is non-positive."
            )

        return voltage

    # ------------------------------------------------------------------
    # Battery power
    # ------------------------------------------------------------------

    def calculate_battery_power(
        self,
        current_a: float,
    ) -> float:
        """
        Calculate battery electrical power using:

            P = Vt * I

        Under the Wu current convention:

            P > 0 -> charging
            P < 0 -> discharging
        """

        terminal_voltage = self.calculate_terminal_voltage(
            current_a=current_a
        )

        return terminal_voltage * current_a

    # ------------------------------------------------------------------
    # State update
    # ------------------------------------------------------------------

    def update(
        self,
        current_a: float,
        dt_s: float,
    ) -> BatteryState:
        """
        Advance the battery electrical states by one time step.

        SOC:

            dSOC/dt = I / (3600 * Qbat)

        RC branch 1:

            dVb1/dt =
                -Vb1/(Rb1*Cb1) + I/Cb1

        RC branch 2:

            dVb2/dt =
                -Vb2/(Rb2*Cb2) + I/Cb2
        """

        if dt_s <= 0:
            raise ValueError(
                "Time step must be positive."
            )

        p = self.parameters
        s = self.state

        # SOC dynamics
        dsoc_dt = (
            current_a
            / (3600.0 * p.nominal_capacity_ah)
        )

        # Polarization branch 1
        dvb1_dt = (
            -s.polarization_voltage_1_v
            / (
                p.rc1_resistance_ohm
                * p.rc1_capacitance_f
            )
            + current_a
            / p.rc1_capacitance_f
        )

        # Polarization branch 2
        dvb2_dt = (
            -s.polarization_voltage_2_v
            / (
                p.rc2_resistance_ohm
                * p.rc2_capacitance_f
            )
            + current_a
            / p.rc2_capacitance_f
        )

        # Euler integration
        new_soc = (
            s.soc
            + dsoc_dt * dt_s
        )

        new_vb1 = (
            s.polarization_voltage_1_v
            + dvb1_dt * dt_s
        )

        new_vb2 = (
            s.polarization_voltage_2_v
            + dvb2_dt * dt_s
        )

        # Do not silently clip SOC.
        # The environment will later handle SOC constraints.
        self.state = BatteryState(
            soc=new_soc,
            polarization_voltage_1_v=new_vb1,
            polarization_voltage_2_v=new_vb2,
        )

        return self.state