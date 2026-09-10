"""Bilan Camelyon17 : les mecanismes trouves sur CIFAR tiennent-ils sur un decalage reel (hopital) ?

    python -m src.analyze_camelyon --runs runs_cam --out figures/camelyon
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ORDER = [("cam_teacher_resnet50", "teacher ResNet-50", "tab:red"),
         ("cam_teacher_resnet50_strong", "teacher ResNet-50 aug. forte", "darkred"),
         ("cam_student_mnv3s_scratch", "student seul", "tab:green"),
         ("cam_student_mnv3s_kd", "KD Hinton / T std", "#f16913"),
         ("cam_student_mnv3s_dkd", "DKD / T std", "tab:blue"),
         ("cam_student_mnv3s_dkd_nckd", "NCKD seul / T std", "#6baed6"),
         ("cam_student_mnv3s_kd_strongT", "KD Hinton / T fort", "#fdae6b"),
         ("cam_student_mnv3s_dkd_strongT", "DKD / T fort", "navy"),
         ("cam_student_mnv3s_dkd_nckd_strongT", "NCKD seul / T fort", "#9ecae1")]
SPLITS = [("id_val", "hopitaux connus"), ("val", "hopital 1 (OOD)"), ("test", "hopital 2 (OOD)")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs_cam")
    p.add_argument("--out", default="figures/camelyon")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 240)

    rows = []
    for run, label, _ in ORDER:
        f = Path(a.runs) / run / "final.json"
        if not f.exists():
            continue
        d = json.load(open(f))
        r = {"modele": label, "params_M": round(d["params"] / 1e6, 2)}
        for s, _ in SPLITS:
            r[f"acc_{s}"] = d[s]["acc"]
            r[f"ece_{s}"] = d[s]["ece"]
            r[f"conf_{s}"] = d[s].get("mean_conf")
            r[f"agree_{s}"] = d[s].get("agreement")
        r["ood_gap"] = d["id_val"]["acc"] - (d["val"]["acc"] + d["test"]["acc"]) / 2
        rows.append(r)
    df = pd.DataFrame(rows)
    print("=== Camelyon17 : accuracy ===")
    print(df[["modele", "params_M"] + [f"acc_{s}" for s, _ in SPLITS] + ["ood_gap"]].round(4).to_string(index=False))
    print("\n=== Camelyon17 : ECE / confiance ===")
    print(df[["modele"] + [f"ece_{s}" for s, _ in SPLITS] + [f"conf_{s}" for s, _ in SPLITS]].round(4).to_string(index=False))
    print("\n=== fidelite au teacher (agreement) ===")
    print(df[["modele"] + [f"agree_{s}" for s, _ in SPLITS]].round(4).to_string(index=False))
    df.to_csv(out / "camelyon_table.csv", index=False)

    sc = df[df.modele == "student seul"]
    if len(sc):
        sc = sc.iloc[0]
        print("\n=== ecarts au student seul (points) ===")
        d = df[df.modele.str.startswith(("KD", "DKD", "NCKD"))].copy()
        for s, _ in SPLITS:
            d[f"d_acc_{s}"] = (d[f"acc_{s}"] - sc[f"acc_{s}"]) * 100
            d[f"d_ece_{s}"] = (d[f"ece_{s}"] - sc[f"ece_{s}"]) * 100
        print(d[["modele"] + [c for c in d if c.startswith("d_")]].round(2).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = range(len(SPLITS))
    for run, label, col in ORDER:
        r = df[df.modele == label]
        if r.empty:
            continue
        r = r.iloc[0]
        ls = "--" if "teacher" in label else "-"
        axes[0].plot(x, [r[f"acc_{s}"] for s, _ in SPLITS], ls, marker="o", color=col, label=label)
        axes[1].plot(x, [r[f"ece_{s}"] for s, _ in SPLITS], ls, marker="o", color=col, label=label)
    for ax, t in zip(axes, ["Accuracy", "ECE"]):
        ax.set_xticks(list(x))
        ax.set_xticklabels([n for _, n in SPLITS])
        ax.set_title(t)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("Camelyon17 : decalage d'hopital reel")
    fig.tight_layout()
    fig.savefig(out / "camelyon_splits.png", dpi=150)
    print("->", out / "camelyon_splits.png")


if __name__ == "__main__":
    main()
