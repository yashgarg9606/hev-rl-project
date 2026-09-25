from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

RESULTS=Path("results/experiments")
PLOTS=Path("results/plots/model_comparison")
PLOTS.mkdir(parents=True,exist_ok=True)

BATTERIES=["B0005","B0006","B0007","B0018"]


def load_results():
    hybrid=np.load(
        RESULTS/"cross_validation_results.npz"
    )

    lstm=np.load(
        RESULTS/"baseline_lstm_results.npz"
    )

    return hybrid,lstm


def print_comparison(hybrid,lstm):
    hybrid_mae=hybrid["mae"]
    hybrid_rmse=hybrid["rmse"]
    hybrid_r2=hybrid["r2"]

    lstm_mae=lstm["mae"]
    lstm_rmse=lstm["rmse"]
    lstm_r2=lstm["r2"]

    print("\n========================================")
    print("MODEL COMPARISON")
    print("========================================")

    print(
        f"{'Battery':<12}"
        f"{'Hybrid MAE':<14}"
        f"{'LSTM MAE':<14}"
        f"{'Hybrid RMSE':<15}"
        f"{'LSTM RMSE':<14}"
        f"{'Hybrid R2':<14}"
        f"{'LSTM R2':<12}"
    )

    for i,battery in enumerate(BATTERIES):
        print(
            f"{battery:<12}"
            f"{hybrid_mae[i]:<14.6f}"
            f"{lstm_mae[i]:<14.6f}"
            f"{hybrid_rmse[i]:<15.6f}"
            f"{lstm_rmse[i]:<14.6f}"
            f"{hybrid_r2[i]:<14.6f}"
            f"{lstm_r2[i]:<12.6f}"
        )

    print("----------------------------------------")

    print(
        f"{'Average':<12}"
        f"{np.mean(hybrid_mae):<14.6f}"
        f"{np.mean(lstm_mae):<14.6f}"
        f"{np.mean(hybrid_rmse):<15.6f}"
        f"{np.mean(lstm_rmse):<14.6f}"
        f"{np.mean(hybrid_r2):<14.6f}"
        f"{np.mean(lstm_r2):<12.6f}"
    )

    print("========================================")


def plot_metric(hybrid,lstm,metric_name,filename):
    hybrid_values=hybrid[metric_name.lower()]
    lstm_values=lstm[metric_name.lower()]

    x=np.arange(len(BATTERIES))
    width=0.35

    plt.figure(figsize=(10,6))

    plt.bar(
        x-width/2,
        hybrid_values,
        width,
        label="CNN-TCN-LSTM-Attention"
    )

    plt.bar(
        x+width/2,
        lstm_values,
        width,
        label="LSTM Baseline"
    )

    plt.xticks(x,BATTERIES)
    plt.xlabel("Test Battery")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} Comparison")
    plt.legend()
    plt.grid(axis="y",alpha=0.3)

    plt.tight_layout()

    path=PLOTS/filename

    plt.savefig(path,dpi=300)
    plt.close()

    print("Saved:",path)


def plot_average_metrics(hybrid,lstm):
    metrics=["MAE","RMSE","R2"]

    hybrid_values=[
        np.mean(hybrid["mae"]),
        np.mean(hybrid["rmse"]),
        np.mean(hybrid["r2"])
    ]

    lstm_values=[
        np.mean(lstm["mae"]),
        np.mean(lstm["rmse"]),
        np.mean(lstm["r2"])
    ]

    x=np.arange(len(metrics))
    width=0.35

    plt.figure(figsize=(9,6))

    plt.bar(
        x-width/2,
        hybrid_values,
        width,
        label="CNN-TCN-LSTM-Attention"
    )

    plt.bar(
        x+width/2,
        lstm_values,
        width,
        label="LSTM Baseline"
    )

    plt.xticks(x,metrics)
    plt.ylabel("Score")
    plt.title("Average Model Performance")
    plt.legend()
    plt.grid(axis="y",alpha=0.3)

    plt.tight_layout()

    path=PLOTS/"average_model_performance.png"

    plt.savefig(path,dpi=300)
    plt.close()

    print("Saved:",path)


def save_comparison(hybrid,lstm):
    output=PLOTS/"model_comparison.npz"

    np.savez(
        output,
        batteries=np.array(BATTERIES),
        hybrid_mae=hybrid["mae"],
        hybrid_rmse=hybrid["rmse"],
        hybrid_r2=hybrid["r2"],
        lstm_mae=lstm["mae"],
        lstm_rmse=lstm["rmse"],
        lstm_r2=lstm["r2"]
    )

    print("Saved:",output)


def main():
    print("Loading experiment results...")

    hybrid,lstm=load_results()

    print_comparison(
        hybrid,
        lstm
    )

    print("\nGenerating comparison plots...")

    plot_metric(
        hybrid,
        lstm,
        "MAE",
        "mae_comparison.png"
    )

    plot_metric(
        hybrid,
        lstm,
        "RMSE",
        "rmse_comparison.png"
    )

    plot_metric(
        hybrid,
        lstm,
        "R2",
        "r2_comparison.png"
    )

    plot_average_metrics(
        hybrid,
        lstm
    )

    save_comparison(
        hybrid,
        lstm
    )

    print("\n========================================")
    print("MODEL COMPARISON COMPLETE")
    print("========================================")
    print("Results:",RESULTS)
    print("Plots:",PLOTS)


if __name__=="__main__":
    main()