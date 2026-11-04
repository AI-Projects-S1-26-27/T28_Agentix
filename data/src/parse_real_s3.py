"""Parse WESAD Subject S3 and generate a participant comparison plot."""
import pandas as pd
import matplotlib.pyplot as plt

if __package__:
    from .parse_real_wesad_full import parse_subject
else:
    from parse_real_wesad_full import parse_subject


def generate_comparison_plot():
    if parse_subject("S3") is None:
        raise FileNotFoundError("Raw WESAD recording for S3 was not found.")

    df_s2 = pd.read_csv("data/processed/wesad_s2_real.csv")
    df_s3 = pd.read_csv("data/processed/wesad_s3_real.csv")
    s2_baseline = df_s2.loc[df_s2["stress_label"] == 1, "heart_rate_bpm"]
    s3_baseline = df_s3.loc[df_s3["stress_label"] == 1, "heart_rate_bpm"]

    plt.figure(figsize=(7, 5))
    plt.boxplot([s2_baseline, s3_baseline])
    plt.xticks([1, 2], ["Subject S2", "Subject S3"])
    plt.axhline(
        100.0,
        color="red",
        linestyle="--",
        label="Illustrative reference (100 bpm; not diagnostic)",
    )
    plt.title("WESAD Baseline Heart Rate by Participant", fontsize=11)
    plt.ylabel("Heart Rate (bpm)")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("data/eda/plots/real_wesad_multi_subject_variance.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    generate_comparison_plot()
