from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


INPUT = Path("data/raw/Actual_Driving_Cycle_raw.csv")
OUTPUT = Path("results/figures/Actual_Driving_Cycle.png")


# Load data
df = pd.read_csv(INPUT)

# Create output directory
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

# Plot
plt.figure(figsize=(12, 4.5))

plt.plot(
    df["time_s"],
    df["velocity_kmh"],
    linewidth=1.2
)

plt.xlabel("Time (s)")
plt.ylabel("Velocity (km/h)")
plt.title("Actual Driving Cycle — Figure-derived Reconstruction")

plt.xlim(
    df["time_s"].min(),
    df["time_s"].max()
)

plt.ylim(
    0,
    max(100, df["velocity_kmh"].max() + 5)
)

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT,
    dpi=250,
    bbox_inches="tight"
)

plt.close()

print("Actual driving cycle plot generated successfully.")
print("Saved to:", OUTPUT)