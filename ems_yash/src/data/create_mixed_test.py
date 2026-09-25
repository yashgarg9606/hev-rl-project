import pandas as pd
from pathlib import Path


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "Mixed_Test_Cycle.csv"
)


# ---------------------------------------------------------
# Input files
# ---------------------------------------------------------
cycles = [
    ("UDDS", DATA_DIR / "UDDS_raw.csv"),
    ("HWFET", DATA_DIR / "HWFET_raw.csv"),
    ("CLTC-P", DATA_DIR / "CLTC_P_raw.csv"),
    ("FTP75", DATA_DIR / "FTP75_raw.csv"),
]


# ---------------------------------------------------------
# Load and combine
# ---------------------------------------------------------
combined = []

global_time = 0

for i, (name, filepath) in enumerate(cycles):

    df = pd.read_csv(filepath)

    required = [
        "time_s",
        "velocity_kmh",
        "slope_rad"
    ]

    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"{name}: missing column '{col}'"
            )

    # -----------------------------------------------------
    # For every cycle after the first:
    # remove its first sample to avoid duplicating
    # the boundary point.
    # -----------------------------------------------------
    if i > 0:
        df = df.iloc[1:].reset_index(drop=True)

    # Assign new global time
    df["time_s"] = range(
        global_time,
        global_time + len(df)
    )

    combined.append(df)

    global_time += len(df)


# ---------------------------------------------------------
# Create final dataframe
# ---------------------------------------------------------
mixed = pd.concat(
    combined,
    ignore_index=True
)


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

# Required columns
assert list(mixed.columns) == [
    "time_s",
    "velocity_kmh",
    "slope_rad"
]

# No missing values
assert not mixed.isna().any().any()

# Time should increase by exactly 1 second
time_diff = mixed["time_s"].diff().dropna()

assert (time_diff == 1).all()

# Velocity limits
assert mixed["velocity_kmh"].min() >= 0
assert mixed["velocity_kmh"].max() <= 120

# Standard cycles have zero road grade
assert (mixed["slope_rad"] == 0).all()


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------
mixed.to_csv(
    OUTPUT_FILE,
    index=False,
    float_format="%.6f"
)


# ---------------------------------------------------------
# Report
# ---------------------------------------------------------
print("\nMixed test cycle created successfully!")
print("--------------------------------------")

print(f"Total samples:  {len(mixed)}")
print(f"Start time:     {mixed['time_s'].iloc[0]:.0f} s")
print(f"End time:       {mixed['time_s'].iloc[-1]:.0f} s")
print(f"Time step:      {time_diff.iloc[0]:.0f} s")
print(f"Max velocity:   {mixed['velocity_kmh'].max():.2f} km/h")

distance_km = mixed["velocity_kmh"].sum() / 3600

print(f"Distance:       {distance_km:.3f} km")

print(f"\nSaved to:")
print(OUTPUT_FILE)