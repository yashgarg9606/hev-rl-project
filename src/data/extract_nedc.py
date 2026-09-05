from pathlib import Path
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_FILE = PROJECT_ROOT / "wltp" / "wltp" / "cycles" / "V_nedc.txt"
OUTPUT_FILE = PROJECT_ROOT / "data" / "raw" / "NEDC_raw.csv"


# ---------------------------------------------------------
# Load NEDC velocity data
# ---------------------------------------------------------
print("Loading NEDC...")

velocity_kmh = np.array(
    [float(x) for x in SOURCE_FILE.read_text().split()],
    dtype=float
)

print(f"Raw velocity samples: {len(velocity_kmh)}")


# ---------------------------------------------------------
# Create time axis
# ---------------------------------------------------------
time_s = np.arange(len(velocity_kmh), dtype=float)

# Standard drive cycle has no road-grade information
slope_rad = np.zeros(len(velocity_kmh), dtype=float)


# ---------------------------------------------------------
# Create dataframe
# ---------------------------------------------------------
df = pd.DataFrame({
    "time_s": time_s,
    "velocity_kmh": velocity_kmh,
    "slope_rad": slope_rad,
})


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------
print("\n--- NEDC Validation ---")

assert len(df) == 1180
assert df["time_s"].iloc[0] == 0.0
assert df["time_s"].iloc[-1] == 1179.0

dt = np.diff(df["time_s"].to_numpy())
assert np.allclose(dt, 1.0)

assert df["velocity_kmh"].isna().sum() == 0
assert df["slope_rad"].isna().sum() == 0

assert df["velocity_kmh"].min() >= 0
assert df["velocity_kmh"].max() <= 120

assert np.allclose(df["slope_rad"], 0.0)

print(f"✓ Number of samples: {len(df)}")
print(f"✓ Start time: {df['time_s'].iloc[0]:.1f} s")
print(f"✓ End timestamp: {df['time_s'].iloc[-1]:.1f} s")
print(f"✓ Time step: {dt.min():.1f}–{dt.max():.1f} s")
print(f"✓ Minimum velocity: {df['velocity_kmh'].min():.2f} km/h")
print(f"✓ Maximum velocity: {df['velocity_kmh'].max():.2f} km/h")
print(f"✓ Slope: 0 rad")
print("✓ No missing values")
print("✓ Validation successful!")


# ---------------------------------------------------------
# Distance calculation
# ---------------------------------------------------------
velocity_ms = df["velocity_kmh"].to_numpy() / 3.6
distance_km = np.trapezoid(
    velocity_ms,
    df["time_s"].to_numpy()
) / 1000

print(f"✓ Distance: {distance_km:.3f} km")


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT_FILE, index=False)

print("\nSaved to:")
print(OUTPUT_FILE)