"""
Phase 3A — SOH Source Configuration

Defines the source of battery State of Health (SOH) data
supplied to the EMS.
"""

from enum import Enum


class SOHSource(Enum):
    """
    Source of battery SOH data for the EMS.

    TRUE_PHYSICAL:
        Use the true physical battery SOH from the energy-throughput
        battery degradation model.

    BMS_ESTIMATED:
        Use AI-estimated battery SOH from the BMS CNN-TCN-LSTM-Attention
        model. (Not implemented in Phase 3A)
    """

    TRUE_PHYSICAL = "true_physical"
    BMS_ESTIMATED = "bms_estimated"
