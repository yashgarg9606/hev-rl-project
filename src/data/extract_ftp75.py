from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_FILE = PROJECT_ROOT / "data" / "raw" / "FTP75_raw.txt"
OUTPUT_FILE = PROJECT_ROOT / "data" / "raw" / "FTP75_raw.csv"


print("Loading FTP75...")

df_raw = pd.read_csv(
    SOURCE_FILE,
    sep=r"\s+",
    skiprows=2,
    header=None,
    names=["time_s", "velocity_mph"],
    engine="python",
)

df_raw["time_s"] = pd.to_numeric(
    df_raw["time_s"],
    errors="coerce"
)

df_raw["velocity_mph"] = pd.to_numeric(
    df_raw["velocity_mph"],
    errors="coerce"
)

df_raw = df_raw.dropna().reset_index(drop=True)


# mph → km/h
time_s = df_raw["time_s"].to_numpy(dtype=float)

velocity_kmh = (
    df_raw["velocity_mph"].to_numpy(dtype=float)
    * 1.609344
)

# Standard cycle has no grade information
slope_rad = np.zeros(len(df_raw), dtype=float)


df = pd.DataFrame({
    "time_s": time_s,
    "velocity_kmh": velocity_kmh,
    "slope_rad": slope_rad,
})


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------
print("\n--- FTP75 Validation ---")

assert len(df) == 1875

assert df["time_s"].iloc[0] == 0
assert df["time_s"].iloc[-1] == 1874

dt = np.diff(df["time_s"].to_numpy())

assert np.allclose(dt, 1.0)

assert df["velocity_kmh"].isna().sum() == 0
assert df["slope_rad"].isna().sum() == 0

assert df["velocity_kmh"].min() >= 0

assert np.allclose(df["slope_rad"], 0.0)

print(f"✓ Number of samples: {len(df)}")
print(f"✓ Start time: {df['time_s'].iloc[0]:.1f} s")
print(f"✓ End timestamp: {df['time_s'].iloc[-1]:.1f} s")
print(f"✓ Time step: {dt.min():.1f}–{dt.max():.1f} s")
print(f"✓ Minimum velocity: {df['velocity_kmh'].min():.3f} km/h")
print(f"✓ Maximum velocity: {df['velocity_kmh'].max():.3f} km/h")
print("✓ Slope: 0 rad")
print("✓ No missing values")


# ---------------------------------------------------------
# Distance
# ---------------------------------------------------------
velocity_ms = df["velocity_kmh"].to_numpy() / 3.6

distance_km = np.trapezoid(
    velocity_ms,
    df["time_s"].to_numpy()
) / 1000

print(f"✓ Distance: {distance_km:.3f} km")
print("✓ Validation successful!")


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------
OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved to:")
print(OUTPUT_FILE)