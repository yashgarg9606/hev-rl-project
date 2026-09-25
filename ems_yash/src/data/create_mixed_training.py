from pathlib import Path
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

WLTC_FILE = PROJECT_ROOT / "data" / "raw" / "WLTC_Class3b_raw.csv"
NEDC_FILE = PROJECT_ROOT / "data" / "raw" / "NEDC_raw.csv"

OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "Mixed_Training_Cycle.csv"


# ---------------------------------------------------------
# Load raw cycles
# ---------------------------------------------------------
print("Loading WLTC Class 3b...")
wltc = pd.read_csv(WLTC_FILE)

print("Loading NEDC...")
nedc = pd.read_csv(NEDC_FILE)


# ---------------------------------------------------------
# Validate source files
# ---------------------------------------------------------
assert len(wltc) == 1801
assert len(nedc) == 1180

assert np.allclose(
    np.diff(wltc["time_s"].to_numpy()),
    1.0
)

assert np.allclose(
    np.diff(nedc["time_s"].to_numpy()),
    1.0
)

print(f"WLTC samples: {len(wltc)}")
print(f"NEDC active samples: {len(nedc)}")


# ---------------------------------------------------------
# WLTC section
#
# Keep original velocity/slope values.
# Time remains 0 ... 1800 s.
# ---------------------------------------------------------
wltc_section = pd.DataFrame({
    "time_s": np.arange(0, 1801, dtype=float),
    "velocity_kmh": wltc["velocity_kmh"].to_numpy(),
    "slope_rad": wltc["slope_rad"].to_numpy(),
})


# ---------------------------------------------------------
# 20-second initial NEDC pre-test idle
#
# According to the specified NEDC convention:
# 1801 ... 1820 s = 20 seconds at 0 km/h.
# ---------------------------------------------------------
idle_section = pd.DataFrame({
    "time_s": np.arange(1801, 1821, dtype=float),
    "velocity_kmh": np.zeros(20),
    "slope_rad": np.zeros(20),
})


# ---------------------------------------------------------
# Active NEDC section
#
# The NEDC raw file contains 1180 active samples.
# First sample is placed at t = 1821 s.
#
# Therefore:
#   first sample = 1821 s
#   last sample  = 3000 s
# ---------------------------------------------------------
nedc_section = pd.DataFrame({
    "time_s": np.arange(1821, 3001, dtype=float),
    "velocity_kmh": nedc["velocity_kmh"].to_numpy(),
    "slope_rad": nedc["slope_rad"].to_numpy(),
})


# ---------------------------------------------------------
# Concatenate
# ---------------------------------------------------------
mixed = pd.concat(
    [
        wltc_section,
        idle_section,
        nedc_section,
    ],
    ignore_index=True,
)


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------
print("\n--- Mixed Training Cycle Validation ---")

# Expected total:
# 1801 WLTC samples
# + 20 idle samples
# + 1180 NEDC samples
# = 3001 samples
assert len(mixed) == 3001

# Time axis
assert mixed["time_s"].iloc[0] == 0.0
assert mixed["time_s"].iloc[-1] == 3000.0

dt = np.diff(mixed["time_s"].to_numpy())

assert np.allclose(dt, 1.0)

# No missing values
assert mixed.isna().sum().sum() == 0

# Velocity limits
assert mixed["velocity_kmh"].min() >= 0
assert mixed["velocity_kmh"].max() <= 131.3

# Flat road
assert np.allclose(mixed["slope_rad"], 0.0)

# Check the 20-second idle region
idle = mixed[
    (mixed["time_s"] >= 1801)
    & (mixed["time_s"] <= 1820)
]

assert len(idle) == 20
assert np.allclose(idle["velocity_kmh"], 0.0)
assert np.allclose(idle["slope_rad"], 0.0)

# Check boundaries
assert mixed.loc[
    mixed["time_s"] == 1800,
    "velocity_kmh"
].iloc[0] == wltc["velocity_kmh"].iloc[-1]

assert mixed.loc[
    mixed["time_s"] == 1801,
    "velocity_kmh"
].iloc[0] == 0.0

assert mixed.loc[
    mixed["time_s"] == 1820,
    "velocity_kmh"
].iloc[0] == 0.0

assert mixed.loc[
    mixed["time_s"] == 1821,
    "velocity_kmh"
].iloc[0] == nedc["velocity_kmh"].iloc[0]

# ---------------------------------------------------------
# Distance
# ---------------------------------------------------------
velocity_ms = mixed["velocity_kmh"].to_numpy() / 3.6
time_s = mixed["time_s"].to_numpy()

distance_km = np.trapezoid(
    velocity_ms,
    time_s
) / 1000


# ---------------------------------------------------------
# Print summary
# ---------------------------------------------------------
print(f"✓ Total samples: {len(mixed)}")
print(f"✓ Start time: {mixed['time_s'].iloc[0]:.1f} s")
print(f"✓ End time: {mixed['time_s'].iloc[-1]:.1f} s")
print(f"✓ Duration: {mixed['time_s'].iloc[-1]:.1f} s")
print(f"✓ Time step: {dt.min():.1f}–{dt.max():.1f} s")
print(f"✓ Maximum velocity: {mixed['velocity_kmh'].max():.2f} km/h")
print(f"✓ Distance: {distance_km:.3f} km")
print("✓ 20-second NEDC pre-test idle verified")
print("✓ No missing values")
print("✓ Slope = 0 rad throughout")
print("✓ Validation successful!")


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------
OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

mixed.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved to:")
print(OUTPUT_FILE)