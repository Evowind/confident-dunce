"""Courbes par severite (moyenne sur les corruptions) + tableau recapitulatif.

    python -m src.plot --results results.csv --out figures
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .style import hatch_for, legend, style_for

METRICS = [("acc", "Accuracy (plus haut = mieux)"),
           ("ece", "ECE (plus bas = mieux)"),
           ("nll", "NLL (plus bas = mieux)"),
           ("agreement", "Agreement top-1 avec le teacher"),
           ("kl_ts", "KL(teacher || student)"),
           ("agreement_on_teacher_errors", "Agreement sur les erreurs du teacher")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results.csv")
    p.add_argument("--out", default="figures")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)

    df = pd.read_csv(a.results)
    by_sev = df.groupby(["run", "severity"]).mean(numeric_only=True).reset_index()

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (m, title) in zip(axes.flat, METRICS):
        if m not in by_sev:
            ax.set_visible(False)
            continue
        for run, g in by_sev.groupby("run"):
            ax.plot(g["severity"], g[m], label=run, **style_for(run))
        ax.set_title(title)
        ax.set_xlabel("severite (0 = propre)")
        ax.grid(alpha=0.3)
    legend(axes.flat[0])
    fig.tight_layout()
    fig.savefig(out / "by_severity.png", dpi=150)

    # Par corruption, severite moyenne : ou la KD aide-t-elle le plus ?
    corrupted = df[df.severity > 0]
    if len(corrupted):
        by_corr = corrupted.groupby(["run", "corruption"])["acc"].mean().unstack(0)
        ax = by_corr.plot.barh(figsize=(8, 10), color=[style_for(c)["color"] for c in by_corr.columns], edgecolor="black", linewidth=0.5)
        for patches, c in zip(ax.containers, by_corr.columns):
            for p in patches:
                p.set_hatch(hatch_for(c))
        legend(ax)
        plt.title("Accuracy par corruption (moyenne des severites)")
        plt.tight_layout()
        plt.savefig(out / "by_corruption.png", dpi=150)
    else:
        print("Pas de lignes corrompues : by_corruption.png non genere.")

    summary = df.assign(split=df.severity.map(lambda s: "clean" if s == 0 else "corrupted")) \
                .groupby(["run", "split"]).mean(numeric_only=True) \
                .drop(columns="severity").round(4)
    summary.to_csv(out / "summary.csv")
    print(summary.to_string())
    print("->", out)


if __name__ == "__main__":
    main()
