"""
Phase 3A — BMS-EMS Health Interface

Provides a communication layer between the physical health degradation
model and the EMS, enabling selection between:
    - TRUE_PHYSICAL: Use true physical battery SOH (Phase 3A)
    - BMS_ESTIMATED: Use AI-estimated battery SOH (future phases)
"""

from .soh_source import SOHSource
from .health_interface import BMSHealthInterface

__all__ = [
    "SOHSource",
    "BMSHealthInterface",
]
