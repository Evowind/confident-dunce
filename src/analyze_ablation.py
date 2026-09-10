"""Ablation KD Hinton (H2) : temperature et poids CE vs accuracy / ECE / agreement, par severite.

    python -m src.analyze_ablation --results results.csv results_ablation_a.csv results_ablation_b.csv --out figures
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .style import legend, style_for

ORDER = [("student_resnet20_scratch", "scratch", None),
         ("student_resnet20_kd_T1", "KD T=1 ce=0.1", None),
         ("student_resnet20_kd_T2", "KD T=2 ce=0.1", None),
         ("student_resnet20_kd", "KD T=4 ce=0.1 (ref)", None),
         ("student_resnet20_kd_T8", "KD T=8 ce=0.1", None),
         ("student_resnet20_kd_ce05", "KD T=4 ce=0.5", None),
         ("student_resnet20_dkd", "DKD (ref)", None),
         ("teacher_wrn40_2", "teacher", None)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", nargs="+", required=True)
    p.add_argument("--out", default="figures")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    pd.set_option("display.width", 220)

    df = pd.concat([pd.read_csv(f) for f in a.results], ignore_index=True)
    df = df[df.run.isin([r for r, _, _ in ORDER])]
    label = {r: l for r, l, _ in ORDER}
    df["label"] = df.run.map(label)

    rows = []
    for r, l, _ in ORDER:
        d = df[df.run == r]
        if d.empty:
            continue
        c, k = d[d.severity == 0], d[d.severity > 0]
        rows.append({"config": l, "acc_clean": c.acc.mean(), "acc_corr": k.acc.mean(),
                     "ece_clean": c.ece.mean(), "ece_corr": k.ece.mean(),
                     "nll_corr": k.nll.mean(), "agree_clean": c.agreement.mean(),
                     "agree_corr": k.agreement.mean(),
                     "agree_err_corr": k.agreement_on_teacher_errors.mean()})
    tab = pd.DataFrame(rows).round(4)
    print("=== Ablation KD Hinton ===")
    print(tab.to_string(index=False))
    tab.to_csv(out / "ablation_table.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, m in zip(axes, ["acc", "ece", "agreement"]):
        for r, l, _ in ORDER:
            d = df[df.run == r]
            if d.empty or (m == "agreement" and r == "teacher_wrn40_2"):
                continue
            ax.plot(d.groupby("severity")[m].mean(), label=l, **style_for(r))
        ax.set_title(m)
        ax.set_xlabel("severite (0 = propre)")
        ax.grid(alpha=0.3)
    legend(axes[1], fontsize=7)
    fig.suptitle("Ablation KD Hinton : temperature et poids CE")
    fig.tight_layout()
    fig.savefig(out / "ablation_by_severity.png", dpi=150)
    print("->", out / "ablation_by_severity.png")


if __name__ == "__main__":
    main()
