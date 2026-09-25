import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv(
    "data/raw/WLTC_Class3b_raw.csv"
)

plt.figure(figsize=(14, 5))

plt.plot(
    df["time_s"],
    df["velocity_kmh"]
)

plt.xlabel("Time (s)")
plt.ylabel("Velocity (km/h)")
plt.title("WLTC Class 3b — Complete 4-Phase Cycle")

plt.grid(True)
plt.tight_layout()

plt.savefig(
    "results/figures/WLTC_Class3b.png",
    dpi=300
)

