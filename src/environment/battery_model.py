"""
Research-grade 2-RC battery electrical model for the Wu et al. BEV EMS lineage.

Primary model lineage:
    Wu et al., Journal of Power Sources 623 (2024) 235463
        -> Ref. [28]
    Wu et al., Applied Energy 376 (2024) 124306
        -> Ref. [25] for time-varying electrical parameters
    Lin et al., Journal of Power Sources 257 (2014) 1-11
        -> A123 26650 LiFePO4/graphite parameterization

The numerical functional coefficients for R1/R2/C1/C2 are taken from the
publicly available predecessor parameterization by Perez et al. (ASME 2012),
which uses the same A123 26650 electro-thermal model family and publishes
the coefficients explicitly. The 2014 Lin paper confirms the same functional
forms and the same thermal parameter set, but does not print the electrical
fit coefficients in its main text.

IMPORTANT PACK-SCALE CAVEAT
---------------------------
Wu et al. (Applied Energy 2024) explicitly gives:
    Qbat = 72 Ah
    VOC  = 255.5 V
    Rs   = 0.0031 ohm

The target JPS 2024 paper separately reports Ebat = 128 kWh. The papers do
not provide enough information to reconcile that energy value with the
72-Ah/255.5-V battery specification. This implementation therefore uses the
72-Ah/255.5-V values as the directly published battery-model baseline and
records the 128-kWh discrepancy as a provenance caveat.

CURRENT SIGN CONVENTION
-----------------------
Wu model:
    I > 0 : charging  -> SOC increases
    I < 0 : discharging -> SOC decreases

The Perez/Lin parameterization labels:
    I >= 0 : discharge
    I < 0  : charge

Therefore the direction branch is intentionally reversed when evaluating
R1/R2/C1/C2:
    Wu charging current  -> Lin/Perez charge branch
    Wu discharging current -> Lin/Perez discharge branch

TEMPERATURE VALIDITY
--------------------
The published coefficient tables used here were identified over the
15-45 degC range. The 2014 Lin paper also shows 5 degC data in Figs. 10-13,
but the 2012 published coefficient tables do not include a validated
5 degC coefficient set. This implementation therefore raises an error
outside 15-45 degC instead of silently extrapolating.

OCV
---
The Applied Energy 2024 paper gives a battery voltage of 255.5 V but does
not publish a numerical Voc(SOC) lookup table. Until a pack-level OCV-SOC
curve is recovered, this model uses 255.5 V as a constant baseline OCV.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import math


Direction = Literal["charge", "discharge"]


@dataclass(frozen=True)
class BatteryElectricalParameters:
    nominal_capacity_ah: float = 72.0
    nominal_ocv_v: float = 255.5
    ohmic_resistance_ohm: float = 0.0031
    min_parameter_temperature_c: float = 15.0
    max_parameter_temperature_c: float = 45.0


@dataclass
class BatteryState:
    soc: float
    polarization_voltage_1_v: float = 0.0
    polarization_voltage_2_v: float = 0.0


class BatteryElectricalModel:
    """Wu-sign-convention 2-RC battery model."""

    def __init__(
        self,
        parameters: BatteryElectricalParameters | None = None,
        initial_state: BatteryState | None = None,
    ):
        self.parameters = parameters or BatteryElectricalParameters()
        self.state = initial_state or BatteryState(soc=0.6)
        self._validate_parameters()
        self._validate_state()

    def _validate_parameters(self) -> None:
        p = self.parameters
        if p.nominal_capacity_ah <= 0:
            raise ValueError("Nominal capacity must be positive.")
        if p.nominal_ocv_v <= 0:
            raise ValueError("Nominal OCV must be positive.")
        if p.ohmic_resistance_ohm < 0:
            raise ValueError("Ohmic resistance cannot be negative.")
        if p.min_parameter_temperature_c >= p.max_parameter_temperature_c:
            raise ValueError("Invalid temperature range.")

    def _validate_state(self) -> None:
        if not 0.0 <= self.state.soc <= 1.0:
            raise ValueError("SOC must be between 0 and 1.")

    @staticmethod
    def _direction_from_current(current_a: float) -> Direction:
        # Wu: +I = charge, -I = discharge.
        # Perez/Lin: +I = discharge, -I = charge.
        return "charge" if current_a > 0.0 else "discharge"

    def _validate_temperature(self, temperature_c: float) -> None:
        p = self.parameters
        if not (
            p.min_parameter_temperature_c
            <= temperature_c
            <= p.max_parameter_temperature_c
        ):
            raise ValueError(
                f"Electrical parameterization is validated for "
                f"{p.min_parameter_temperature_c:.1f} to "
                f"{p.max_parameter_temperature_c:.1f} degC. "
                f"Received {temperature_c:.2f} degC."
            )

    @staticmethod
    def _r1(soc: float, temperature_c: float, direction: Direction) -> float:
        # Perez et al. (ASME 2012), Table 3.
        if direction == "discharge":
            r0, r1, r2 = 7.1135e-4, -4.3865e-4, 2.3788e-4
            tref, tshift = 347.4707, -79.5816
        else:
            r0, r1, r2 = 0.0016, -0.0032, 0.0045
            tref, tshift = 159.2819, -41.4548

        return (r0 + r1 * soc + r2 * soc * soc) * math.exp(
            tref / (temperature_c - tshift)
        )

    @staticmethod
    def _r2(soc: float, temperature_c: float, direction: Direction) -> float:
        # Perez et al. (ASME 2012), Table 4.
        if direction == "discharge":
            r0, r1, r2, tref = 0.0288, -0.073, 0.0605, 16.6712
        else:
            r0, r1, r2, tref = 0.0113, -0.027, 0.0339, 17.0224

        return (r0 + r1 * soc + r2 * soc * soc) * math.exp(
            tref / temperature_c
        )

    @staticmethod
    def _c1(soc: float, temperature_c: float, direction: Direction) -> float:
        # Perez et al. (ASME 2012), Table 5.
        if direction == "discharge":
            c0, c1, c2 = 335.4518, 3.1712e3, -1.3214e3
            c3, c4, c5 = 53.2138, -65.4786, 44.3761
        else:
            c0, c1, c2 = 523.215, 6.4171e3, -7.5555e3
            c3, c4, c5 = 50.7107, -131.2298, 162.4688

        return (
            c0
            + c1 * soc
            + c2 * soc * soc
            + (c3 + c4 * soc + c5 * soc * soc) * temperature_c
        )

    @staticmethod
    def _c2(soc: float, temperature_c: float, direction: Direction) -> float:
        # Perez et al. (ASME 2012), Table 6.
        if direction == "discharge":
            c0, c1, c2 = 3.1887e4, -1.1593e5, 1.0493e5
            c3, c4, c5 = 60.3114, 1.0175e4, -9.5924e3
        else:
            c0, c1, c2 = 6.2449e4, -1.055e5, 4.4432e4
            c3, c4, c5 = 198.9753, 7.5621e3, -6.9365e3

        return (
            c0
            + c1 * soc
            + c2 * soc * soc
            + (c3 + c4 * soc + c5 * soc * soc) * temperature_c
        )

    def electrical_parameters(
        self,
        temperature_c: float,
        current_a: float,
        soc: float | None = None,
    ) -> dict[str, float | str]:
        self._validate_temperature(temperature_c)

        if soc is None:
            soc = self.state.soc
        if not 0.0 <= soc <= 1.0:
            raise ValueError("SOC must be between 0 and 1.")

        direction = self._direction_from_current(current_a)

        r1 = self._r1(soc, temperature_c, direction)
        r2 = self._r2(soc, temperature_c, direction)
        c1 = self._c1(soc, temperature_c, direction)
        c2 = self._c2(soc, temperature_c, direction)

        if min(r1, r2, c1, c2) <= 0:
            raise ValueError(
                "Parameterization produced a non-positive electrical parameter."
            )

        return {
            "direction": direction,
            "r1_ohm": r1,
            "r2_ohm": r2,
            "c1_f": c1,
            "c2_f": c2,
        }

    def open_circuit_voltage(self, soc: float | None = None) -> float:
        if soc is None:
            soc = self.state.soc
        if not 0.0 <= soc <= 1.0:
            raise ValueError("SOC must be between 0 and 1.")
        return self.parameters.nominal_ocv_v

    def terminal_voltage(
        self,
        current_a: float,
        temperature_c: float,
        soc: float | None = None,
    ) -> float:
        if soc is None:
            soc = self.state.soc

        voc = self.open_circuit_voltage(soc)
        v = (
            voc
            + self.state.polarization_voltage_1_v
            + self.state.polarization_voltage_2_v
            + self.parameters.ohmic_resistance_ohm * current_a
        )

        if v <= 0:
            raise ValueError("Terminal voltage is non-positive.")

        return v

    def battery_power(
        self,
        current_a: float,
        temperature_c: float,
    ) -> float:
        """Battery electrical power under Wu convention.

        Positive = charging, negative = discharging.
        """
        return self.terminal_voltage(current_a, temperature_c) * current_a

    def current_from_battery_power(
        self,
        battery_power_w: float,
        temperature_c: float,
    ) -> float:
        """Solve P_batt = V_t * I for current.

        P_batt > 0 = charging.
        P_batt < 0 = discharging.
        """
        self._validate_temperature(temperature_c)

        rs = self.parameters.ohmic_resistance_ohm
        vbase = (
            self.open_circuit_voltage()
            + self.state.polarization_voltage_1_v
            + self.state.polarization_voltage_2_v
        )

        if rs == 0.0:
            if vbase <= 0:
                raise ValueError("Base voltage must be positive.")
            return battery_power_w / vbase

        discriminant = vbase * vbase + 4.0 * rs * battery_power_w
        if discriminant < 0.0:
            raise ValueError(
                "Requested battery power is outside the physical solution "
                "of the terminal-voltage/current equation."
            )

        sqrt_disc = math.sqrt(discriminant)

        # Root closest to P/Vbase is the physically continuous branch.
        root1 = (-vbase + sqrt_disc) / (2.0 * rs)
        root2 = (-vbase - sqrt_disc) / (2.0 * rs)

        candidates = [root1, root2]
        return min(candidates, key=lambda x: abs(x - battery_power_w / vbase))

    def update(
        self,
        current_a: float,
        temperature_c: float,
        dt_s: float,
    ) -> BatteryState:
        if dt_s <= 0:
            raise ValueError("Time step must be positive.")

        params = self.electrical_parameters(
            temperature_c=temperature_c,
            current_a=current_a,
        )

        p = self.parameters
        s = self.state

        dsoc_dt = current_a / (3600.0 * p.nominal_capacity_ah)

        dv1_dt = (
            -s.polarization_voltage_1_v
            / (params["r1_ohm"] * params["c1_f"])
            + current_a / params["c1_f"]
        )

        dv2_dt = (
            -s.polarization_voltage_2_v
            / (params["r2_ohm"] * params["c2_f"])
            + current_a / params["c2_f"]
        )

        new_soc = s.soc + dsoc_dt * dt_s
        new_v1 = s.polarization_voltage_1_v + dv1_dt * dt_s
        new_v2 = s.polarization_voltage_2_v + dv2_dt * dt_s

        if not 0.0 <= new_soc <= 1.0:
            raise ValueError(
                f"Battery update would move SOC outside [0, 1]: {new_soc:.6f}"
            )

        self.state = BatteryState(
            soc=new_soc,
            polarization_voltage_1_v=new_v1,
            polarization_voltage_2_v=new_v2,
        )
        return self.state


