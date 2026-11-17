"""Parse WESAD Subjects S4 and S5 and generate a participant comparison plot."""
import pandas as pd
import matplotlib.pyplot as plt

if __package__:
    from .parse_real_wesad_full import parse_subject
else:
    from parse_real_wesad_full import parse_subject


def generate_comparison_plot():
    for participant in ("S4", "S5"):
        if parse_subject(participant) is None:
            raise FileNotFoundError(
                f"Raw WESAD recording for {participant} was not found."
            )

    participants = ("s2", "s3", "s4", "s5")
    dataframes = [
        pd.read_csv(f"data/processed/wesad_{participant}_real.csv")
        for participant in participants
    ]
    baseline_heart_rates = [
        frame.loc[frame["stress_label"] == 1, "heart_rate_bpm"]
        for frame in dataframes
    ]

    plt.figure(figsize=(9, 5))
    plt.boxplot(baseline_heart_rates)
    plt.xticks(
        [1, 2, 3, 4],
        ["Subject S2", "Subject S3", "Subject S4", "Subject S5"],
    )
    plt.axhline(
        100.0,
        color="red",
        linestyle="--",
        label="Illustrative reference (100 bpm; not diagnostic)",
    )
    plt.title("WESAD Baseline Heart Rate by Participant", fontsize=11)
    plt.ylabel("Heart Rate (bpm)")
    plt.xlabel("WESAD participant")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("data/eda/plots/real_wesad_multi_subject_variance.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    generate_comparison_plot()
