import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "Mixed_Test_Cycle.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "Mixed_Test_Cycle.png"
)


df = pd.read_csv(INPUT_FILE)


plt.figure(figsize=(14, 5))

plt.plot(
    df["time_s"],
    df["velocity_kmh"],
    linewidth=1.0
)

plt.xlabel("Time (s)")
plt.ylabel("Velocity (km/h)")
plt.title("Mixed Test Driving Cycle")

plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print(f"Plot saved to: {OUTPUT_FILE}")