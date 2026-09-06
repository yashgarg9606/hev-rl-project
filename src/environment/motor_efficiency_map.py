"""
Motor efficiency-map lookup for the dual-motor BEV plant.

The numerical contour data are reconstructed from Wu et al. (2024),
Fig. 1(d)/(e). They are image-derived contour coordinates rather than
original numerical map arrays supplied by the paper.

The lookup uses scattered 2-D interpolation because the source figure
provides iso-efficiency contours rather than a rectangular efficiency grid.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.interpolate import LinearNDInterpolator



class MotorEfficiencyMap:
    """2-D motor efficiency lookup from digitized contour data."""

    def __init__(self, contour_csv: str | Path):
        self.contour_csv = Path(contour_csv)

        if not self.contour_csv.exists():
            raise FileNotFoundError(
                f"Motor efficiency map not found: {self.contour_csv}"
            )

        data = pd.read_csv(self.contour_csv)

        required = {
            "speed_rpm",
            "torque_nm",
            "efficiency_percent",
        }

        missing = required - set(data.columns)

        if missing:
            raise ValueError(
                f"Missing required columns: {sorted(missing)}"
            )

        data = data.dropna(
            subset=list(required)
        ).copy()

        self.data = data

        points = data[
            ["speed_rpm", "torque_nm"]
        ].to_numpy(dtype=float)

        values = data[
            "efficiency_percent"
        ].to_numpy(dtype=float)

        # Piecewise-linear interpolation inside the convex hull
        # of the digitized contour points.
        self._linear = LinearNDInterpolator(
            points,
            values,
            fill_value=np.nan,
        )

    def get_efficiency(
        self,
        speed_rpm: float,
        torque_nm: float,
    ) -> float:
        """
        Return motor efficiency in percent.

        Positive torque corresponds to motoring operation.
        Negative torque is intentionally rejected because the
        published map contains the positive-torque region only.
        """

        if speed_rpm < 0:
            raise ValueError(
                "Motor speed cannot be negative."
            )

        if torque_nm < 0:
            raise ValueError(
                "Negative torque is not represented by the "
                "published efficiency map."
            )

        point = np.array(
            [[speed_rpm, torque_nm]],
            dtype=float,
        )

        efficiency = float(
            self._linear(point)[0]
        )

        if np.isnan(efficiency):
            raise ValueError(
                "Operating point is outside the reconstructed "
                "motor efficiency map."
            )

        return float(
            np.clip(
                efficiency,
                51.0,
                95.0,
            )
        )

    def get_efficiency_fraction(
        self,
        speed_rpm: float,
        torque_nm: float,
    ) -> float:
        """Return efficiency as a fraction between 0 and 1."""

        return (
            self.get_efficiency(
                speed_rpm,
                torque_nm,
            )
            / 100.0
        )