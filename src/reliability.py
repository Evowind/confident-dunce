"""Diagrammes de fiabilite : dans quel sens un modele est-il mal calibre (sur- ou sous-confiance) ?

    python -m src.reliability --runs student_resnet20_scratch student_resnet20_kd student_resnet20_dkd \
                              teacher_wrn40_2 --severity 3 --out figures

Une colonne par run, une ligne pour le jeu propre et une pour la severite choisie (moyenne des
19 corruptions). Barres = accuracy par bin de confiance, diagonale = calibration parfaite.
Au-dessus de la diagonale : sous-confiance ; en dessous : sur-confiance.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .data import CORRUPTIONS, cifar100_loaders, cifar100c_loader
from .metrics import collect_logits, ece
from .models import load_checkpoint

ROOT = Path(__file__).resolve().parent.parent


def reliability(logits, labels, n_bins=15):
    p = F.softmax(logits, 1)
    conf, pred = p.max(1)
    correct = (pred == labels).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    accs, confs, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        counts.append(m.sum().item())
        accs.append(correct[m].mean().item() if m.any() else np.nan)
        confs.append(conf[m].mean().item() if m.any() else np.nan)
    return np.array(accs), np.array(confs), np.array(counts), edges.numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--severity", type=int, default=3)
    p.add_argument("--out", default="figures")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)

    clean_loader = cifar100_loaders(128, num_workers=4)[2]
    models = {r: load_checkpoint(ROOT / "runs" / r / "best.pt") for r in a.runs}

    # logits propres + logits concatenes sur toutes les corruptions a la severite choisie
    data = {}
    for r, m in models.items():
        lc, yc = collect_logits(m, clean_loader)
        data[(r, "clean")] = (lc, yc)
    for c in CORRUPTIONS:
        loader = cifar100c_loader(c, a.severity)
        for r, m in models.items():
            l, y = collect_logits(m, loader)
            key = (r, f"sev{a.severity}")
            if key in data:
                data[key] = (torch.cat([data[key][0], l]), torch.cat([data[key][1], y]))
            else:
                data[key] = (l, y)

    rows = ["clean", f"sev{a.severity}"]
    fig, axes = plt.subplots(len(rows), len(a.runs), figsize=(3.6 * len(a.runs), 3.6 * len(rows)), squeeze=False)
    summary = []
    for j, r in enumerate(a.runs):
        for i, split in enumerate(rows):
            logits, labels = data[(r, split)]
            accs, confs, counts, edges = reliability(logits, labels)
            ax = axes[i, j]
            w = edges[1] - edges[0]
            ax.bar(edges[:-1], np.nan_to_num(accs), width=w, align="edge", alpha=0.8, edgecolor="k")
            ax.plot([0, 1], [0, 1], "r--", lw=1)
            e = ece(logits, labels)
            mean_conf = F.softmax(logits, 1).max(1)[0].mean().item()
            acc = (logits.argmax(1) == labels).float().mean().item()
            ax.set_title(f"{r.replace('student_resnet20_', '')} / {split}\nECE {e:.3f}  conf {mean_conf:.3f}  acc {acc:.3f}", fontsize=8)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            if i == len(rows) - 1:
                ax.set_xlabel("confiance")
            if j == 0:
                ax.set_ylabel("accuracy")
            summary.append({"run": r, "split": split, "ece": e, "mean_conf": mean_conf, "acc": acc,
                            "overconfidence": mean_conf - acc})
    fig.tight_layout()
    fig.savefig(out / f"reliability_sev{a.severity}.png", dpi=150)

    import pandas as pd
    pd.set_option("display.width", 200)
    df = pd.DataFrame(summary).round(4)
    print(df.to_string(index=False))
    df.to_csv(out / f"reliability_sev{a.severity}.csv", index=False)
    print("->", out / f"reliability_sev{a.severity}.png")


if __name__ == "__main__":
    main()
