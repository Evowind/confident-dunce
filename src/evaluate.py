"""Evalue tous les runs (runs/*/best.pt) sur CIFAR-100 propre et CIFAR-100-C.

    python -m src.evaluate --runs runs --teacher runs/teacher_wrn40_2/best.pt --out results.csv

Sortie : un CSV long, une ligne par (run, corruption, severite). 'clean' a severite 0.
Le teacher est evalue une seule fois par jeu de donnees, ses logits servent aux metriques
de fidelite (agreement, KL) de chaque student.
"""
import argparse
import csv
from pathlib import Path

from tqdm import tqdm

from .data import CORRUPTIONS, SEVERITIES, cifar100_loaders, cifar100c_loader
from .metrics import all_metrics, collect_logits
from .models import load_checkpoint

FIELDS = ["run", "model", "kd_method", "corruption", "severity",
          "acc", "ece", "nll", "brier", "mean_conf", "agreement", "kl_ts", "agreement_on_teacher_errors"]


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--teacher", default=None, help="checkpoint teacher pour agreement / KL")
    p.add_argument("--out", default="results.csv")
    p.add_argument("--corruptions", nargs="*", default=CORRUPTIONS)
    p.add_argument("--severities", nargs="*", type=int, default=SEVERITIES)
    p.add_argument("--only", nargs="*", default=None, help="noms de runs a evaluer")
    return p.parse_args()


def main():
    args = parse()
    device = "cuda"
    runs = {}
    for d in sorted(Path(args.runs).iterdir()):
        ck = d / "best.pt"
        if ck.exists() and (args.only is None or d.name in args.only):
            m = load_checkpoint(ck, device)
            cfg = __import__("torch").load(ck, map_location="cpu", weights_only=False)["config"]
            runs[d.name] = (m, cfg["model"], cfg.get("kd", {}).get("method", "none"))
    print(f"{len(runs)} runs :", ", ".join(runs))
    teacher = load_checkpoint(args.teacher, device) if args.teacher else None

    datasets = [("clean", 0, cifar100_loaders(128, num_workers=4)[2])]
    datasets += [(c, s, None) for c in args.corruptions for s in args.severities]

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for corr, sev, loader in tqdm(datasets, ncols=100):
            if loader is None:
                loader = cifar100c_loader(corr, sev)
            t_logits = None
            if teacher is not None:
                t_logits, _ = collect_logits(teacher, loader)
            for name, (model, arch, method) in runs.items():
                s_logits, labels = collect_logits(model, loader)
                m = all_metrics(s_logits, labels, t_logits)
                w.writerow({"run": name, "model": arch, "kd_method": method,
                            "corruption": corr, "severity": sev, **m})
            f.flush()
    print("->", args.out)


if __name__ == "__main__":
    main()
