"""(1) Hypothese de capacite : le terme non-cible seul (NCKD) transmet-il la robustesse selon la
taille du student ?  (2) NCKD seul depuis un teacher AugMix : robustesse + calibration ?

    python -m src.analyze_capacity --results results.csv results_dkd_ablation_r20.csv results_pairs_wrn.csv \
        results_pairs_res.csv results_dkd_ablation_r8x4.csv results_capacity.csv results_nckd_augmix.csv \
        results_augmix.csv results_augmix_seeds_a.csv results_augmix_seeds_b.csv --out figures
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .models import build_model, count_params
from .style import legend, style_for

STUDENTS = ["resnet20", "resnet32", "wrn_16_2", "resnet8x4"]
TEACHER_OF = {"resnet20": "wrn_40_2", "resnet32": "wrn_40_2", "wrn_16_2": "wrn_40_2", "resnet8x4": "resnet32x4"}


def summarize(df, run_prefix):
    """Moyenne sur seeds et corruptions pour un run (prefixe sans suffixe de seed)."""
    d = df[df.config == run_prefix]
    if d.empty:
        return None
    c, k = d[d.severity == 0], d[d.severity > 0]
    return dict(acc_clean=c.acc.mean(), acc_corr=k.acc.mean(), ece_clean=c.ece.mean(), ece_corr=k.ece.mean(),
                n_seeds=d.run.nunique())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", nargs="+", required=True)
    p.add_argument("--out", default="figures")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    pd.set_option("display.width", 220)

    df = pd.concat([pd.read_csv(f) for f in a.results], ignore_index=True).drop_duplicates(["run", "corruption", "severity"])
    df["config"] = df.run.str.replace(r"_s\d+$", "", regex=True)

    # ---------- (1) capacite
    rows = []
    for s in STUDENTS:
        params = count_params(build_model(s)) / 1e6
        sc = summarize(df, f"student_{s}_scratch")
        if sc is None:
            continue
        for variant, label in [("dkd_nckd", "NCKD seul"), ("dkd", "DKD complet"), ("kd", "KD Hinton T=4")]:
            v = summarize(df, f"student_{s}_{variant}")
            if v is None:
                continue
            rows.append({"student": s, "params_M": round(params, 2), "teacher": TEACHER_OF[s], "variante": label,
                         "d_acc_clean": (v["acc_clean"] - sc["acc_clean"]) * 100,
                         "d_acc_corr": (v["acc_corr"] - sc["acc_corr"]) * 100,
                         "d_ece_corr": (v["ece_corr"] - sc["ece_corr"]) * 100, "seeds": v["n_seeds"]})
    cap = pd.DataFrame(rows).round(2)
    print("=== (1) Hypothese de capacite : ecarts a scratch (points) ===")
    print(cap.sort_values(["variante", "params_M"]).to_string(index=False))
    cap.to_csv(out / "capacity_table.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, key in [("NCKD seul", "dkd_nckd"), ("DKD complet", "dkd"), ("KD Hinton T=4", "kd")]:
        g = cap[cap.variante == label].sort_values("params_M")
        ax.plot(g.params_M, g.d_acc_corr, label=label, **style_for(key))
        for _, r in g.iterrows():
            ax.annotate(r.student, (r.params_M, r.d_acc_corr), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("parametres du student (M, log)")
    ax.set_ylabel("gain d'accuracy sous corruption vs scratch (pts)")
    ax.set_title("Robustesse transmise selon la capacite du student")
    ax.grid(alpha=0.3)
    legend(ax)
    fig.tight_layout()
    fig.savefig(out / "capacity_nckd.png", dpi=150)

    # ---------- (2) NCKD seul depuis AugMix
    rows = []
    for s in ["resnet20", "wrn_16_2"]:
        for cfg, label in [(f"student_{s}_scratch", "scratch"), (f"student_{s}_dkd", "DKD / T std"),
                           (f"student_{s}_dkd_nckd", "NCKD / T std"), (f"student_{s}_dkd_augmixT", "DKD / T AugMix"),
                           (f"student_{s}_dkd_nckd_augmixT", "NCKD / T AugMix")]:
            v = summarize(df, cfg)
            if v is not None:
                rows.append({"student": s, "modele": label, **{k: round(x, 4) for k, x in v.items()}})
    aug = pd.DataFrame(rows)
    print("\n=== (2) NCKD seul depuis le teacher AugMix ===")
    print(aug.to_string(index=False))
    aug.to_csv(out / "nckd_augmix_table.csv", index=False)
    print("->", out)


if __name__ == "__main__":
    main()
