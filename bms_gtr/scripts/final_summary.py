from pathlib import Path
import numpy as np

RESULTS=Path("results/experiments")
OUTPUT=Path("results/final_summary")

OUTPUT.mkdir(parents=True,exist_ok=True)

cv=np.load(RESULTS/"cross_validation_results.npz")
baseline=np.load(RESULTS/"baseline_lstm_results.npz")
ablation=np.load(RESULTS/"ablation_results.npz")

batteries=cv["batteries"]

hybrid_mae=cv["mae"]
hybrid_rmse=cv["rmse"]
hybrid_r2=cv["r2"]

lstm_mae=baseline["mae"]
lstm_rmse=baseline["rmse"]
lstm_r2=baseline["r2"]

ablation_models=ablation["models"]
ablation_mae=ablation["mae"]
ablation_rmse=ablation["rmse"]
ablation_r2=ablation["r2"]

print("="*90)
print("FINAL PROJECT RESULTS SUMMARY")
print("="*90)

print("\n1. HYBRID MODEL - LEAVE-ONE-BATTERY-OUT CROSS-VALIDATION")
print("-"*90)
print(f"{'Battery':<12}{'MAE':<14}{'RMSE':<14}{'R2':<14}")

for i,battery in enumerate(batteries):
    print(
        f"{battery:<12}"
        f"{hybrid_mae[i]:<14.6f}"
        f"{hybrid_rmse[i]:<14.6f}"
        f"{hybrid_r2[i]:<14.6f}"
    )

print("-"*90)
print(
    f"{'Average':<12}"
    f"{np.mean(hybrid_mae):<14.6f}"
    f"{np.mean(hybrid_rmse):<14.6f}"
    f"{np.mean(hybrid_r2):<14.6f}"
)

print("\n2. LSTM BASELINE")
print("-"*90)
print(f"{'Battery':<12}{'MAE':<14}{'RMSE':<14}{'R2':<14}")

for i,battery in enumerate(batteries):
    print(
        f"{battery:<12}"
        f"{lstm_mae[i]:<14.6f}"
        f"{lstm_rmse[i]:<14.6f}"
        f"{lstm_r2[i]:<14.6f}"
    )

print("-"*90)
print(
    f"{'Average':<12}"
    f"{np.mean(lstm_mae):<14.6f}"
    f"{np.mean(lstm_rmse):<14.6f}"
    f"{np.mean(lstm_r2):<14.6f}"
)

print("\n3. ABLATION STUDY")
print("-"*90)
print(f"{'Model':<24}{'MAE':<14}{'RMSE':<14}{'R2':<14}")

for i,model in enumerate(ablation_models):
    model_name=str(model)
    print(
        f"{model_name:<24}"
        f"{ablation_mae[i]:<14.6f}"
        f"{ablation_rmse[i]:<14.6f}"
        f"{ablation_r2[i]:<14.6f}"
    )

print("="*90)

summary_file=OUTPUT/"final_results.txt"

with open(summary_file,"w") as f:
    f.write("FINAL PROJECT RESULTS SUMMARY\n")
    f.write("="*90+"\n\n")

    f.write("HYBRID MODEL - LEAVE-ONE-BATTERY-OUT CROSS-VALIDATION\n")
    f.write("-"*90+"\n")
    f.write(f"{'Battery':<12}{'MAE':<14}{'RMSE':<14}{'R2':<14}\n")

    for i,battery in enumerate(batteries):
        f.write(
            f"{battery:<12}"
            f"{hybrid_mae[i]:<14.6f}"
            f"{hybrid_rmse[i]:<14.6f}"
            f"{hybrid_r2[i]:<14.6f}\n"
        )

    f.write("\nLSTM BASELINE\n")
    f.write("-"*90+"\n")

    for i,battery in enumerate(batteries):
        f.write(
            f"{battery:<12}"
            f"{lstm_mae[i]:<14.6f}"
            f"{lstm_rmse[i]:<14.6f}"
            f"{lstm_r2[i]:<14.6f}\n"
        )

    f.write("\nABLATION STUDY\n")
    f.write("-"*90+"\n")

    for i,model in enumerate(ablation_models):
        f.write(
            f"{str(model):<24}"
            f"{ablation_mae[i]:<14.6f}"
            f"{ablation_rmse[i]:<14.6f}"
            f"{ablation_r2[i]:<14.6f}\n"
        )

np.savez(
    OUTPUT/"final_results.npz",
    batteries=batteries,
    hybrid_mae=hybrid_mae,
    hybrid_rmse=hybrid_rmse,
    hybrid_r2=hybrid_r2,
    lstm_mae=lstm_mae,
    lstm_rmse=lstm_rmse,
    lstm_r2=lstm_r2,
    ablation_models=ablation_models,
    ablation_mae=ablation_mae,
    ablation_rmse=ablation_rmse,
    ablation_r2=ablation_r2
)

print("\nSaved:")
print("results/final_summary/final_results.txt")
print("results/final_summary/final_results.npz")