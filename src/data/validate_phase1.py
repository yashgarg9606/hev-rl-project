from pathlib import Path
import pandas as pd
import numpy as np


RAW = Path("data/raw")
PROCESSED = Path("data/processed")


FILES = {
    "WLTC Class 3b": RAW / "WLTC_Class3b_raw.csv",
    "NEDC": RAW / "NEDC_raw.csv",
    "UDDS": RAW / "UDDS_raw.csv",
    "HWFET": RAW / "HWFET_raw.csv",
    "CLTC-P": RAW / "CLTC_P_raw.csv",
    "FTP75": RAW / "FTP75_raw.csv",
    "Actual Driving Cycle": RAW / "Actual_Driving_Cycle_raw.csv",

    "Mixed Training": PROCESSED / "Mixed_Training_Cycle.csv",
    "Mixed Test": PROCESSED / "Mixed_Test_Cycle.csv",
}


EXPECTED_ROWS = {
    "WLTC Class 3b": 1801,
    "NEDC": 1180,
    "UDDS": 1370,
    "HWFET": 766,
    "CLTC-P": 1800,
    "FTP75": 1875,

    # Figure-derived reconstruction.
    # We expect approximately 3450 samples, but do not make
    # the audit fail if the digitization produces a slightly
    # different endpoint.
    "Actual Driving Cycle": None,

    "Mixed Training": 3001,
    "Mixed Test": 5808,
}


def check_cycle(name, path):
    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    if not path.exists():
        print("❌ FILE NOT FOUND:", path)
        return False

    df = pd.read_csv(path)

    passed = True

    # ---------------------------------------------------------
    # Columns
    # ---------------------------------------------------------
    expected_columns = [
        "time_s",
        "velocity_kmh",
        "slope_rad",
    ]

    if list(df.columns) != expected_columns:
        print("❌ Columns:", list(df.columns))
        passed = False
    else:
        print("✅ Columns correct")

    # ---------------------------------------------------------
    # Row count
    # ---------------------------------------------------------
    expected = EXPECTED_ROWS[name]

    if expected is not None:
        if len(df) == expected:
            print(f"✅ Rows: {len(df)}")
        else:
            print(
                f"❌ Rows: {len(df)} "
                f"(expected {expected})"
            )
            passed = False
    else:
        print(f"ℹ️ Rows: {len(df)} (digitized reconstruction)")

    # ---------------------------------------------------------
    # Missing values
    # ---------------------------------------------------------
    missing = int(df.isna().sum().sum())

    if missing == 0:
        print("✅ No missing values")
    else:
        print(f"❌ Missing values: {missing}")
        passed = False

    # ---------------------------------------------------------
    # Numeric values
    # ---------------------------------------------------------
    numeric_ok = all(
        pd.api.types.is_numeric_dtype(df[c])
        for c in expected_columns
        if c in df.columns
    )

    if numeric_ok:
        print("✅ Numeric columns")
    else:
        print("❌ Non-numeric values")
        passed = False

    # ---------------------------------------------------------
    # Time monotonicity
    # ---------------------------------------------------------
    if "time_s" in df.columns:

        dt = df["time_s"].diff().dropna()

        if len(dt) > 0 and np.all(dt > 0):
            print("✅ Time strictly increasing")
        else:
            print("❌ Time is not strictly increasing")
            passed = False

        if len(dt) > 0:
            unique_dt = np.unique(np.round(dt, 6))

            if np.allclose(dt, 1.0):
                print("✅ Sampling interval = 1 s")
            else:
                print("⚠️ Sampling interval:", unique_dt[:10])

    # ---------------------------------------------------------
    # Velocity
    # ---------------------------------------------------------
    if "velocity_kmh" in df.columns:

        vmin = df["velocity_kmh"].min()
        vmax = df["velocity_kmh"].max()

        print(f"Velocity range: {vmin:.3f} → {vmax:.3f} km/h")

        if vmin >= -1e-6:
            print("✅ No negative velocity")
        else:
            print("❌ Negative velocity detected")
            passed = False

        if np.isfinite(df["velocity_kmh"]).all():
            print("✅ Velocity finite")
        else:
            print("❌ Non-finite velocity")
            passed = False

    # ---------------------------------------------------------
    # Slope
    # ---------------------------------------------------------
    if "slope_rad" in df.columns:

        slope_unique = df["slope_rad"].unique()

        if np.allclose(df["slope_rad"], 0.0):
            print("✅ Slope = 0 rad")
        else:
            print(
                "⚠️ Non-zero slope detected:",
                slope_unique[:10]
            )

    # ---------------------------------------------------------
    # Distance
    # ---------------------------------------------------------
    if len(df) >= 2:

        distance_km = np.trapezoid(
            df["velocity_kmh"] / 3.6,
            df["time_s"]
        ) / 1000

        print(f"Distance: {distance_km:.3f} km")

    return passed


def main():

    print()
    print("=" * 70)
    print("PHASE 1 — DATASET INTEGRITY AUDIT")
    print("=" * 70)

    results = {}

    for name, path in FILES.items():
        results[name] = check_cycle(name, path)

    print()
    print("=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)

    for name, passed in results.items():

        if passed:
            print(f"✅ {name}")
        else:
            print(f"❌ {name}")

    print()

    if all(results.values()):
        print("🎉 ALL AUTOMATED CHECKS PASSED")
    else:
        print("⚠️ SOME CHECKS FAILED — REVIEW ABOVE")

    # ---------------------------------------------------------
    # Actual-cycle endpoint inspection
    # ---------------------------------------------------------
    actual = RAW / "Actual_Driving_Cycle_raw.csv"

    if actual.exists():

        df = pd.read_csv(actual)

        print()
        print("=" * 70)
        print("ACTUAL CYCLE ENDPOINT INSPECTION")
        print("=" * 70)

        print(
            f"Initial velocity: "
            f"{df.velocity_kmh.iloc[0]:.6f} km/h"
        )

        print(
            f"Final velocity:   "
            f"{df.velocity_kmh.iloc[-1]:.6f} km/h"
        )

        print()
        print(
            "The published Fig. 4(c) visually begins and ends "
            "approximately at 0 km/h."
        )

        if abs(df.velocity_kmh.iloc[0]) > 1.0:
            print(
                "⚠️ Initial point differs noticeably from 0 km/h."
            )
        else:
            print("✅ Initial point is approximately 0 km/h.")

        if abs(df.velocity_kmh.iloc[-1]) > 1.0:
            print(
                "⚠️ Final point differs noticeably from 0 km/h."
            )
        else:
            print("✅ Final point is approximately 0 km/h.")


if __name__ == "__main__":
    main()