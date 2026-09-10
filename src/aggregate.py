"""Agrege plusieurs seeds : moyenne +/- ecart-type par (config, severite).

Les runs sont regroupes en retirant le suffixe _s<k> du nom (student_resnet20_kd_s1 -> student_resnet20_kd).

    python -m src.aggregate --results results.csv results_augmix.csv --out figures
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .style import legend, style_for

METRICS = [("acc", "Accuracy"), ("ece", "ECE"), ("agreement", "Agreement avec le teacher")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", nargs="+", default=["results.csv"])
    p.add_argument("--out", default="figures")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)

    df = pd.concat([pd.read_csv(f) for f in a.results], ignore_index=True)
    df["config"] = df.run.str.replace(r"_s\d+$", "", regex=True)
    df["seed"] = df.run.str.extract(r"_s(\d+)$")[0].fillna("0").astype(int)

    # moyenne sur les corruptions par (config, seed, severite), puis stats sur les seeds
    per_seed = df.groupby(["config", "seed", "severity"]).mean(numeric_only=True).reset_index()
    stats = per_seed.groupby(["config", "severity"]).agg(
        n=("seed", "count"),
        **{f"{m}_mean": (m, "mean") for m, _ in METRICS},
        **{f"{m}_std": (m, "std") for m, _ in METRICS},
    ).reset_index()
    stats.to_csv(out / "seeds_by_severity.csv", index=False)

    # resume clean / corrompu
    per_seed["split"] = per_seed.severity.map(lambda s: "clean" if s == 0 else "corrupted")
    summ = per_seed.groupby(["config", "seed", "split"]).mean(numeric_only=True).reset_index()
    summ = summ.groupby(["config", "split"]).agg(
        n=("seed", "count"),
        **{f"{m}_mean": (m, "mean") for m, _ in METRICS},
        **{f"{m}_std": (m, "std") for m, _ in METRICS},
    ).round(4)
    summ.to_csv(out / "seeds_summary.csv")
    pd.set_option("display.width", 200)
    print(summ.to_string())

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (m, title) in zip(axes, METRICS):
        for cfg, g in stats.groupby("config"):
            ax.errorbar(g.severity, g[f"{m}_mean"], yerr=g[f"{m}_std"].fillna(0),
                        capsize=3, label=f"{cfg} (n={int(g.n.max())})", **style_for(cfg))
        ax.set_title(f"{title} : moyenne +/- ecart-type sur les seeds")
        ax.set_xlabel("severite (0 = propre)")
        ax.grid(alpha=0.3)
    legend(axes[0], fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "seeds_by_severity.png", dpi=150)
    print("->", out / "seeds_by_severity.png")


if __name__ == "__main__":
    main()
