"""Le seuil de confiance (KD T=1 garde sa confiance, T=4 prend celle du teacher) tient-il sur
plusieurs paires teacher -> student ?

    python -m src.analyze_pairs --pair wrn_40_2:wrn_16_2 results_pairs_wrn.csv \
                                --pair resnet32x4:resnet8x4 results_pairs_res.csv \
                                --pair wrn_40_2:resnet20 results.csv results_ablation_b.csv --out figures

Chaque --pair : "teacher_model:student_model" suivi des CSV qui contiennent ses runs.
Les runs attendus : teacher_<T>, student_<S>_scratch, student_<S>_kd_T1, student_<S>_kd, student_<S>_dkd
(pour resnet20 les noms historiques student_resnet20_* sont acceptes).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import re

from .style import legend, style_for

ROLES = [("scratch", "scratch", None), ("kd_T1", "KD T=1", None),
         ("kd", "KD T=4", None), ("dkd", "DKD", None),
         ("dkd_nckd", "DKD NCKD seul", None), ("dkd_tckd", "DKD TCKD seul", None),
         ("teacher", "teacher", None)]
KNOWN = {r for r, _, _ in ROLES}


def role_of(run, teacher, student):
    """Nom de run -> role. Les suffixes de seed (_s1, _s2) sont ignores : les seeds sont moyennes."""
    run = re.sub(r"_s\d+$", "", run)
    if run.startswith("teacher_"):
        return "teacher" if run == f"teacher_{teacher}" or run == f"teacher_{teacher.replace('_', '', 1)}" else None
    prefix = f"student_{student}_"
    if not run.startswith(prefix):
        return None
    rest = run[len(prefix):]
    return rest if rest in KNOWN else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pair", nargs="+", action="append", required=True)
    p.add_argument("--out", default="figures")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    pd.set_option("display.width", 220)

    rows, curves = [], {}
    for spec in a.pair:
        teacher, student = spec[0].split(":")
        df = pd.concat([pd.read_csv(f) for f in spec[1:]], ignore_index=True)
        df["role"] = df.run.map(lambda r: role_of(r, teacher, student))
        df = df[df.role.notna()]
        if "mean_conf" not in df:
            df["mean_conf"] = float("nan")
        for role, label, _ in ROLES:
            d = df[df.role == role]
            if d.empty:
                continue
            c, k = d[d.severity == 0], d[d.severity > 0]
            rows.append({"pair": f"{teacher} -> {student}", "model": label,
                         "acc_clean": c.acc.mean(), "acc_corr": k.acc.mean(),
                         "ece_clean": c.ece.mean(), "ece_corr": k.ece.mean(),
                         "conf_clean": c.mean_conf.mean(), "conf_corr": k.mean_conf.mean(),
                         "agree_corr": k.agreement.mean() if "agreement" in k else float("nan")})
            curves[(spec[0], role)] = d.groupby("severity")[["acc", "ece"]].mean()
    tab = pd.DataFrame(rows).round(4)
    print("=== Seuil de confiance sur plusieurs paires ===")
    print(tab.to_string(index=False))
    tab.to_csv(out / "pairs_table.csv", index=False)

    # delta vs scratch, par paire : la signature "T=1 ~ scratch, T=4 ~ teacher" doit se repeter
    print("\n=== Ecarts a scratch (points) ===")
    for pair, g in tab.groupby("pair", sort=False):
        s = g[g.model == "scratch"].iloc[0]
        d = g.assign(d_acc_corr=(g.acc_corr - s.acc_corr) * 100, d_ece_corr=(g.ece_corr - s.ece_corr) * 100,
                     d_conf_clean=(g.conf_clean - s.conf_clean) * 100)[["model", "d_acc_corr", "d_ece_corr", "d_conf_clean"]]
        print(f"-- {pair}")
        print(d.round(1).to_string(index=False))

    pairs = list(dict.fromkeys(s[0] for s in a.pair))
    fig, axes = plt.subplots(2, len(pairs), figsize=(5 * len(pairs), 8), squeeze=False)
    for j, pair in enumerate(pairs):
        for i, m in enumerate(["acc", "ece"]):
            ax = axes[i, j]
            for role, label, _ in ROLES:
                if (pair, role) in curves:
                    ax.plot(curves[(pair, role)][m], label=label, **style_for(role))
            ax.set_title(f"{pair.replace(':', ' -> ')} : {m}")
            ax.set_xlabel("severite")
            ax.grid(alpha=0.3)
            if i == 0 and j == 0:
                legend(ax, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "pairs_by_severity.png", dpi=150)
    print("->", out / "pairs_by_severity.png")


if __name__ == "__main__":
    main()
