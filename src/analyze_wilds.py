"""Bilan d'un jeu WILDS (Camelyon17 ou iWildCam) : accuracy, macro-F1, ECE, confiance, fidelite,
par split, avec moyenne +/- ecart-type sur les seeds (suffixe _s<k> ignore).

    python -m src.analyze_wilds --runs runs_iwc --prefix iwc --out figures/iwildcam
    python -m src.analyze_wilds --runs runs_cam --prefix cam --out figures/camelyon
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .style import legend, style_for

SPLITS = [("id_val", "connu"), ("val", "OOD val"), ("test", "OOD test")]
ORDER = ["teacher_resnet50", "teacher_resnet50_strong", "student_mnv3s_scratch",
         "student_mnv3s_kd", "student_mnv3s_dkd", "student_mnv3s_dkd_nckd",
         "student_mnv3s_kd_strongT", "student_mnv3s_dkd_strongT", "student_mnv3s_dkd_nckd_strongT",
         "student_mnv3s_scratch_aug", "student_mnv3s_kd_strongT_aug", "student_mnv3s_dkd_strongT_aug",
         "student_mnv3s_feat_std", "student_mnv3s_feat_strongT", "student_mnv3s_feat_strongT_aug"]
COLORS = {"teacher_resnet50": "tab:red", "teacher_resnet50_strong": "darkred", "student_mnv3s_scratch": "tab:green",
          "student_mnv3s_kd": "#f16913", "student_mnv3s_dkd": "tab:blue", "student_mnv3s_dkd_nckd": "#6baed6",
          "student_mnv3s_kd_strongT": "#fdae6b", "student_mnv3s_dkd_strongT": "navy", "student_mnv3s_dkd_nckd_strongT": "#9ecae1",
          "student_mnv3s_scratch_aug": "darkgreen", "student_mnv3s_kd_strongT_aug": "#a63603", "student_mnv3s_dkd_strongT_aug": "#08306b",
          "student_mnv3s_feat_std": "tab:purple", "student_mnv3s_feat_strongT": "#6a3d9a", "student_mnv3s_feat_strongT_aug": "#cab2d6"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True)
    p.add_argument("--prefix", required=True, help="cam ou iwc")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 260)

    rows = []
    for f in Path(a.runs).glob("*/final.json"):
        name = f.parent.name
        if not name.startswith(a.prefix + "_"):
            continue
        cfg = re.sub(r"_s\d+$", "", name)[len(a.prefix) + 1:]
        d = json.load(open(f))
        r = {"config": cfg, "run": name, "params_M": d["params"] / 1e6}
        for s, _ in SPLITS:
            for m in ("acc", "macro_f1", "ece", "mean_conf", "agreement"):
                r[f"{m}_{s}"] = d[s].get(m)
        rows.append(r)
    df = pd.DataFrame(rows)
    g = df.groupby("config")
    agg = g.agg(n=("run", "count"), **{c: (c, "mean") for c in df.columns if c not in ("config", "run")},
                **{c + "_std": (c, "std") for c in df.columns if c.startswith(("acc_", "macro_f1_", "ece_", "mean_conf_"))})
    known = [c for c in ORDER if c in agg.index]
    # configs hors ORDER (ex. suffixe _v2) : gardees, triees par leur base dans ORDER
    extra = sorted([c for c in agg.index if c not in ORDER],
                   key=lambda c: next((i for i, o in enumerate(ORDER) if c.startswith(o)), len(ORDER)))
    agg = agg.reindex(known + extra)
    agg.to_csv(out / f"{a.prefix}_table.csv")

    def fmt(m, s):
        col = f"{m}_{s}"
        return agg.apply(lambda r: f"{100*r[col]:5.1f}" + (f" +/-{100*r[col+'_std']:4.1f}" if r["n"] > 1 and pd.notna(r[col + "_std"]) else "        "), axis=1)

    for m, title in [("acc", "accuracy"), ("macro_f1", "macro-F1"), ("ece", "ECE"), ("mean_conf", "confiance moyenne")]:
        if agg[f"{m}_test"].notna().any():
            t = pd.DataFrame({"n": agg.n, **{lab: fmt(m, s) for s, lab in SPLITS}})
            print(f"=== {title} (x100) ===")
            print(t.to_string())
            print()
    ag = agg[[f"agreement_{s}" for s, _ in SPLITS]].dropna(how="all").round(3)
    if len(ag):
        print("=== fidelite au teacher ===")
        print(ag.to_string())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x = list(range(len(SPLITS)))
    # beaucoup de series : couleur + marqueur + trait par methode, marqueur creux pour teacher fort / images augmentees
    for cfg, r in agg.iterrows():
        for ax, m in zip(axes, ["acc", "macro_f1", "ece"]):
            ax.plot(x, [r[f"{m}_{s}"] for s, _ in SPLITS], label=f"{cfg} (n={int(r.n)})", **style_for(cfg))
    for ax, t in zip(axes, ["accuracy", "macro-F1", "ECE"]):
        ax.set_xticks(x)
        ax.set_xticklabels([lab for _, lab in SPLITS])
        ax.set_title(t)
        ax.grid(alpha=0.3)
    legend(axes[0], fontsize=6, ncol=2)
    fig.suptitle(f"{a.prefix} : decalage reel, moyenne sur seeds")
    fig.tight_layout()
    fig.savefig(out / f"{a.prefix}_splits.png", dpi=150)
    print("->", out)


if __name__ == "__main__":
    main()
