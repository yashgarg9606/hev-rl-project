from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "UDDS_raw.csv"
OUTPUT_FILE = PROJECT_ROOT / "results" / "figures" / "UDDS.png"


df = pd.read_csv(INPUT_FILE)

plt.figure(figsize=(14, 5))

plt.plot(
    df["time_s"],
    df["velocity_kmh"],
    linewidth=1
)

plt.xlabel("Time (s)")
plt.ylabel("Velocity (km/h)")
plt.title("UDDS — Urban Dynamometer Driving Schedule")

plt.grid(True, alpha=0.3)
plt.tight_layout()

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(
    OUTPUT_FILE,
    dpi=200
)

plt.close()

print(f"Saved plot to:\n{OUTPUT_FILE}")