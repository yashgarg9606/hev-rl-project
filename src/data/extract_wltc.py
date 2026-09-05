from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

SOURCE_FILE = Path("wltp/wltp/cycles/V_class3b.txt")
OUTPUT_FILE = Path("data/raw/WLTC_Class3b_raw.csv")

DT = 1.0  # WLTC velocity trace is sampled at 1 Hz


# ---------------------------------------------------------
# Read WLTC Class 3b
# ---------------------------------------------------------

def load_wltc_class3b():
    """Load the JRC WLTC Class 3b velocity trace."""

    values = SOURCE_FILE.read_text().split()

    velocity_kmh = [float(value) for value in values]

    return velocity_kmh


# ---------------------------------------------------------
# Create DataFrame
# ---------------------------------------------------------

def create_dataframe(velocity_kmh):

    time_s = [
        i * DT
        for i in range(len(velocity_kmh))
    ]

    df = pd.DataFrame({
        "time_s": time_s,
        "velocity_kmh": velocity_kmh,
        "slope_rad": 0.0
    })

    return df


# ---------------------------------------------------------
# Validate
# ---------------------------------------------------------

def validate_wltc(df):

    print("\n--- WLTC Class 3b Validation ---")

    # Number of samples
    assert len(df) == 1801, (
        f"Expected 1801 samples, got {len(df)}"
    )

    # Time
    assert df["time_s"].iloc[0] == 0.0

    assert df["time_s"].iloc[-1] == 1800.0

    # Sampling interval
    dt = df["time_s"].diff().dropna()

    assert (dt == DT).all(), (
        "Time interval is not consistently 1 second"
    )

    # Velocity
    assert df["velocity_kmh"].min() >= 0

    assert df["velocity_kmh"].max() <= 140

    # Slope
    assert (df["slope_rad"] == 0).all()

    # Missing values
    assert not df.isna().any().any()

    print("✓ Number of samples:", len(df))
    print("✓ Start time:", df["time_s"].iloc[0], "s")
    print("✓ End time:", df["time_s"].iloc[-1], "s")
    print("✓ Time step:", DT, "s")
    print("✓ Minimum velocity:", df["velocity_kmh"].min(), "km/h")
    print("✓ Maximum velocity:", df["velocity_kmh"].max(), "km/h")
    print("✓ Slope:", "0 rad")
    print("✓ No missing values")
    print("✓ Validation successful!")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("Loading WLTC Class 3b...")

    velocity = load_wltc_class3b()

    print("Raw velocity samples:", len(velocity))

    df = create_dataframe(velocity)

    validate_wltc(df)

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


if __name__ == "__main__":
    main()
