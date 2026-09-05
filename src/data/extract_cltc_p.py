import pandas as pd
import numpy as np
from pathlib import Path


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "CLTC_P_source.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "raw" / "CLTC_P_raw.csv"


# ---------------------------------------------------------
# Load source data
# ---------------------------------------------------------
df = pd.read_csv(INPUT_FILE)

print("Original columns:", list(df.columns))
print("Original shape:", df.shape)


# ---------------------------------------------------------
# Validate source
# ---------------------------------------------------------
required_columns = ["Time_s", "Speed_km_h"]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")


if len(df) != 1800:
    raise ValueError(
        f"Expected 1800 samples, found {len(df)}"
    )


# ---------------------------------------------------------
# Convert to canonical project format
# ---------------------------------------------------------
output = pd.DataFrame()

# Original source is 1..1800 s.
# Shift to 0..1799 s for our project convention.
output["time_s"] = np.arange(len(df), dtype=float)

output["velocity_kmh"] = df["Speed_km_h"].astype(float)

# Standard cycle has no road-grade information.
output["slope_rad"] = 0.0


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

# Time
assert len(output) == 1800
assert output["time_s"].iloc[0] == 0.0
assert output["time_s"].iloc[-1] == 1799.0

time_diff = output["time_s"].diff().dropna()

assert np.allclose(time_diff, 1.0)


# Velocity
assert output["velocity_kmh"].min() >= 0
assert output["velocity_kmh"].max() <= 120

assert not output.isna().any().any()


# Slope
assert np.all(output["slope_rad"] == 0.0)


# Distance
distance_km = (
    output["velocity_kmh"].sum() / 3600.0
)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------
output.to_csv(
    OUTPUT_FILE,
    index=False,
    float_format="%.6f"
)


# ---------------------------------------------------------
# Report
# ---------------------------------------------------------
print("\nCLTC-P validation successful!")
print("--------------------------------")
print(f"Samples:       {len(output)}")
print(f"Start time:    {output['time_s'].iloc[0]:.1f} s")
print(f"End time:      {output['time_s'].iloc[-1]:.1f} s")
print(f"Time step:     {time_diff.iloc[0]:.1f} s")
print(f"Min velocity:  {output['velocity_kmh'].min():.1f} km/h")
print(f"Max velocity:  {output['velocity_kmh'].max():.1f} km/h")
print(f"Distance:      {distance_km:.3f} km")
print(f"Output:        {OUTPUT_FILE}")