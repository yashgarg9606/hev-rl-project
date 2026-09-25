import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "CLTC_P_raw.csv"
OUTPUT_FILE = PROJECT_ROOT / "results" / "figures" / "CLTC_P.png"


# Load data
df = pd.read_csv(INPUT_FILE)


# Plot
plt.figure(figsize=(12, 5))

plt.plot(
    df["time_s"],
    df["velocity_kmh"],
    linewidth=1.2
)

plt.xlabel("Time (s)")
plt.ylabel("Velocity (km/h)")
plt.title("CLTC-P Driving Cycle")

plt.grid(True, alpha=0.3)
plt.tight_layout()


# Save
plt.savefig(
    OUTPUT_FILE,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print(f"Plot saved to: {OUTPUT_FILE}")