from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "Mixed_Training_Cycle.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "Mixed_Training_Cycle.png"
)


# Load data
df = pd.read_csv(INPUT_FILE)


# Plot
plt.figure(figsize=(14, 5))

plt.plot(
    df["time_s"],
    df["velocity_kmh"],
    linewidth=1
)

plt.xlabel("Time (s)")
plt.ylabel("Velocity (km/h)")
plt.title("Mixed Training Cycle — WLTC Class 3b + NEDC")

plt.grid(True, alpha=0.3)
plt.tight_layout()


# Save only — no plt.show()
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_FILE, dpi=200)

plt.close()

print(f"Saved plot to:\n{OUTPUT_FILE}")