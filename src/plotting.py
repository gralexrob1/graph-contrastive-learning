import matplotlib.pyplot as plt
import seaborn as sns


def cross_augmentation_plot(df, title=None, savefig=None):

    grouped_data = df.groupby(["Aug1", "Aug2"])["MicroF1"].agg(["mean", "std"]).reset_index()

    pivot_data_mean = grouped_data.pivot_table(index="Aug1", columns="Aug2", values="mean")
    pivot_data_std = grouped_data.pivot_table(index="Aug1", columns="Aug2", values="std")

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        pivot_data_mean,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=pivot_data_mean.values.mean(),
    )

    for i in range(len(pivot_data_mean.index)):
        for j in range(len(pivot_data_mean.columns)):
            _ = pivot_data_mean.index[i]
            _ = pivot_data_mean.columns[j]
            mean_val = pivot_data_mean.values[i][j]
            std_val = pivot_data_std.values[i][j]
            plt.text(
                j + 0.5,
                i + 0.5,
                f"{mean_val:.2f} +/- {std_val:.2f}",
                ha="center",
                va="center",
                bbox=dict(facecolor="white", alpha=1.0, edgecolor="none"),
            )
    plt.title(title)
    plt.xlabel("Augmentation 2")
    plt.ylabel("Augmentation 1")
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()
