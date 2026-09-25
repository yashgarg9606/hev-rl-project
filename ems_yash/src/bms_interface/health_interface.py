"""
Phase 3A — BMS-EMS Health Interface

Interface between the physical health degradation model and the EMS.

This interface does NOT implement battery degradation.
It only provides a source-selection layer that routes health data
from the physical model to the EMS.

Phase 3A: Only TRUE_PHYSICAL is implemented.
Future:   BMS_ESTIMATED will provide AI-estimated battery SOH.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from .soh_source import SOHSource

if TYPE_CHECKING:
    from ..environment.health_model import HealthDegradationModel
    from .soh_trace_adapter import SOHTraceAdapter


class BMSHealthInterface:
    """
    Interface between health degradation model and EMS.

    Provides battery SOH to the EMS, sourced from either:
        - True physical model (Phase 3A baseline)
        - AI-estimated SOH (future Phase 3B experiments)

    Important:
        This interface does NOT modify or replace the physical
        battery SOH. It only controls which SOH value is exposed
        to the EMS for decision-making.

        The true physical SOH always exists and continues to evolve
        according to the energy-throughput degradation model.

    Parameters
    ----------
    health_model : HealthDegradationModel
        The physical health degradation model.

    soh_source : SOHSource
        Source of battery SOH for the EMS.
        Defaults to TRUE_PHYSICAL.

    soh_trace_adapter : SOHTraceAdapter, optional
        BMS SOH trace adapter for BMS_ESTIMATED mode.
        Required when soh_source is BMS_ESTIMATED.
    """

    def __init__(
        self,
        health_model: HealthDegradationModel,
        soh_source: SOHSource = SOHSource.TRUE_PHYSICAL,
        soh_trace_adapter: Optional[SOHTraceAdapter] = None,
    ):
        self.health_model = health_model
        self.soh_source = soh_source
        self.soh_trace_adapter = soh_trace_adapter

        # Validate BMS_ESTIMATED configuration
        if soh_source == SOHSource.BMS_ESTIMATED and soh_trace_adapter is None:
            raise ValueError(
                "BMS_ESTIMATED mode requires soh_trace_adapter"
            )

    def get_health_state(self, simulation_time: float = 0.0) -> dict[str, float]:
        """
        Return health state for EMS consumption.

        Parameters
        ----------
        simulation_time : float
            Current simulation time in seconds (required for BMS_ESTIMATED mode)

        Returns
        -------
        dict
            Health state dictionary with keys:
                - battery_soh: Battery SOH (true or estimated based on source)
                - motor1_soh: Motor 1 SOH (always true physical)
                - motor2_soh: Motor 2 SOH (always true physical)

        Notes
        -----
        Motor SOH values are always from the true physical model.
        Only battery SOH can be sourced from BMS estimation.
        """

        # Always get true physical health state
        true_state = self.health_model.get_health_state()

        if self.soh_source == SOHSource.TRUE_PHYSICAL:
            # Phase 3A: Return true physical battery SOH
            return true_state

        elif self.soh_source == SOHSource.BMS_ESTIMATED:
            # Phase 3B: Return AI-estimated battery SOH from BMS trace
            estimated_battery_soh = self.soh_trace_adapter.get_soh_at_time(simulation_time)

            return {
                "battery_soh": estimated_battery_soh,
                "motor1_soh": true_state["motor1_soh"],
                "motor2_soh": true_state["motor2_soh"],
            }

        else:
            raise ValueError(f"Unknown SOH source: {self.soh_source}")

    def get_true_battery_soh(self) -> float:
        """
        Always return the true physical battery SOH.

        This method bypasses the source selection and always returns
        the physical battery SOH from the degradation model.

        Used for validation and comparison experiments where we need
        to track both the true SOH and the EMS-consumed SOH.

        Returns
        -------
        float
            True physical battery State of Health [0, 1].
        """

        return self.health_model.get_health_state()["battery_soh"]

    def get_true_motor_soh(self) -> tuple[float, float]:
        """
        Return true physical motor SOH values.

        Returns
        -------
        tuple[float, float]
            (motor1_soh, motor2_soh) from physical degradation model.
        """

        health = self.health_model.get_health_state()
        return health["motor1_soh"], health["motor2_soh"]
