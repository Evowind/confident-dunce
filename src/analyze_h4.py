"""H4 : un teacher robuste (AugMix) transmet-il sa robustesse ?  +  H1 : le gain du student
suit-il l'avantage du teacher, corruption par corruption ?

    python -m src.analyze_h4 --standard results.csv results_seeds_a.csv results_seeds_b.csv \
                             --augmix results_augmix.csv --out figures
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import legend, style_for


def load(files):
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["config"] = df.run.str.replace(r"_s\d+$", "", regex=True)
    return df


def by_sev(df, cfg, metric="acc"):
    return df[df.config == cfg].groupby("severity")[metric].mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--standard", nargs="+", required=True)
    p.add_argument("--augmix", nargs="+", required=True)
    p.add_argument("--out", default="figures")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    pd.set_option("display.width", 220)

    std = load(a.standard)
    aug = load(a.augmix)
    scratch = std[std.config == "student_resnet20_scratch"]

    groups = {
        "teacher standard": (std, "teacher_wrn40_2", "student_resnet20_kd", "student_resnet20_dkd"),
        "teacher AugMix":   (aug, "teacher_wrn40_2_augmix", "student_resnet20_kd_augmixT", "student_resnet20_dkd_augmixT"),
    }

    # ---- Tableau H4 : accuracy propre / corrompue / robustesse relative
    rows = []
    for gname, (df, t, kd, dkd) in groups.items():
        for label, cfg, d in [("teacher", t, df), ("student KD", kd, df), ("student DKD", dkd, df)]:
            s = by_sev(d, cfg)
            rows.append({"groupe": gname, "modele": label, "clean": s[0], "corrupted": s[1:].mean(),
                         "sev5": s[5], "rel_robust": s[1:].mean() / s[0],
                         "ece_clean": by_sev(d, cfg, "ece")[0], "ece_corr": by_sev(d, cfg, "ece")[1:].mean(),
                         "agree_clean": by_sev(d, cfg, "agreement")[0],
                         "agree_corr": by_sev(d, cfg, "agreement")[1:].mean()})
    s = by_sev(scratch, "student_resnet20_scratch")
    rows.append({"groupe": "-", "modele": "student scratch (moy. seeds)", "clean": s[0],
                 "corrupted": s[1:].mean(), "sev5": s[5], "rel_robust": s[1:].mean() / s[0],
                 "ece_clean": by_sev(scratch, "student_resnet20_scratch", "ece")[0],
                 "ece_corr": by_sev(scratch, "student_resnet20_scratch", "ece")[1:].mean(),
                 "agree_clean": np.nan, "agree_corr": np.nan})
    tab = pd.DataFrame(rows).round(4)
    print("=== H4 : teacher standard vs teacher AugMix ===")
    print(tab.to_string(index=False))
    tab.to_csv(out / "h4_table.csv", index=False)

    # ---- Transfert de robustesse : de combien le teacher AugMix bat le standard, et le student ?
    t_gap = by_sev(aug, "teacher_wrn40_2_augmix") - by_sev(std, "teacher_wrn40_2")
    kd_gap = by_sev(aug, "student_resnet20_kd_augmixT") - by_sev(std, "student_resnet20_kd")
    dkd_gap = by_sev(aug, "student_resnet20_dkd_augmixT") - by_sev(std, "student_resnet20_dkd")
    gap = pd.DataFrame({"teacher AugMix - standard": t_gap, "student KD (AugMix T) - KD (std T)": kd_gap,
                        "student DKD (AugMix T) - DKD (std T)": dkd_gap}).round(4)
    print("\n=== Gain de robustesse du au teacher AugMix, par severite (points d'accuracy) ===")
    print(gap.to_string())
    gap.to_csv(out / "h4_transfer_by_severity.csv")

    # ---- H1 : scatter avantage du teacher vs gain du student, par corruption
    sc_corr = scratch[scratch.severity > 0].groupby("corruption").acc.mean()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, (gname, (df, t, kd, dkd)) in zip(axes, groups.items()):
        d = df[df.severity > 0]
        t_adv = d[d.config == t].groupby("corruption").acc.mean() - sc_corr
        for cfg, lab in [(kd, "KD"), (dkd, "DKD")]:
            s_gain = d[d.config == cfg].groupby("corruption").acc.mean() - sc_corr
            r = np.corrcoef(t_adv.values, s_gain.values)[0, 1]
            st = style_for(cfg, hollow=False)
            ax.scatter(t_adv * 100, s_gain * 100, label=f"{lab} (r = {r:.2f})", color=st["color"], marker=st["marker"], s=55)
            for corr in t_adv.index:
                ax.annotate(corr.replace("_", "\n"), (t_adv[corr] * 100, s_gain[corr] * 100),
                            fontsize=5, alpha=0.7)
        lim = max(abs(ax.get_xlim()[1]), 1)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("avantage du teacher sur scratch (pts)")
        ax.set_title(gname)
        ax.grid(alpha=0.3)
        legend(ax, handlelength=1.5)
    axes[0].set_ylabel("gain du student sur scratch (pts)")
    fig.suptitle("H1 : le gain du student suit-il l'avantage du teacher, corruption par corruption ?")
    fig.tight_layout()
    fig.savefig(out / "h1_teacher_advantage_vs_student_gain.png", dpi=150)

    # ---- Courbes H4
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, m in zip(axes, ["acc", "ece", "agreement"]):
        for gname, (df, t, kd, dkd) in groups.items():
            for cfg, lab in [(t, "teacher"), (kd, "KD"), (dkd, "DKD")]:
                # les variantes AugMix : marqueur creux + tirets (style_for les detecte dans le nom)
                ax.plot(by_sev(df, cfg, m), label=f"{lab} / {gname}", **style_for(cfg))
        if m != "agreement":
            ax.plot(by_sev(scratch, "student_resnet20_scratch", m), label="scratch", **style_for("scratch"))
        ax.set_title(m)
        ax.set_xlabel("severite")
        ax.grid(alpha=0.3)
    legend(axes[0], fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "h4_by_severity.png", dpi=150)
    print("->", out)


if __name__ == "__main__":
    main()
