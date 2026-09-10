"""Entrainement d'un teacher (kd.method: none) ou d'un student (kd.method: kd | dkd).

    python -m src.train --config configs/teacher_wrn40_2.yaml
    python -m src.train --config configs/student_resnet20_kd.yaml --epochs 2 --name smoke
"""
import argparse
import csv
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from .data import cifar100_loaders
from .losses import total_loss
from .metrics import all_metrics, collect_logits
from .models import build_model, count_params, load_checkpoint

ROOT = Path(__file__).resolve().parent.parent


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--name", default=None, help="remplace cfg['name']")
    p.add_argument("--epochs", type=int, default=None, help="remplace cfg['epochs'] (smoke test)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()


def main():
    args = parse()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    if args.name:
        cfg["name"] = args.name
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.seed is not None:
        cfg["seed"] = args.seed
    cfg.setdefault("seed", 0)
    cfg.setdefault("kd", {"method": "none"})
    set_seed(cfg["seed"])
    device = "cuda"

    run_dir = ROOT / "runs" / cfg["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(run_dir / "config.yaml", "w", encoding="utf-8"))

    train_loader, val_loader, test_loader = cifar100_loaders(
        cfg["batch_size"], augmix=cfg.get("augmix", False),
        seed=cfg.get("split_seed", 0),  # split train/val fixe ; cfg['seed'] ne change que init et ordre
        num_workers=args.workers)

    model = build_model(cfg["model"]).to(device).to(memory_format=torch.channels_last)
    print(f"[{cfg['name']}] {cfg['model']} : {count_params(model)/1e6:.2f} M params")

    teacher = None
    if cfg["kd"]["method"] != "none":
        tpath = ROOT / cfg["teacher_ckpt"]
        teacher = load_checkpoint(tpath, device)
        for p in teacher.parameters():
            p.requires_grad_(False)
        t_logits, t_labels = collect_logits(teacher, test_loader)
        print(f"  teacher {tpath.name} : test acc {all_metrics(t_logits, t_labels)['acc']:.4f}")

    opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9,
                          weight_decay=cfg["weight_decay"], nesterov=True)
    sched_cfg = cfg.get("schedule", {"type": "multistep", "milestones": [150, 180, 210], "gamma": 0.1})
    if sched_cfg["type"] == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"])
    else:
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, sched_cfg["milestones"], sched_cfg["gamma"])

    log_path = run_dir / "log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "lr", "train_loss", "train_acc", "val_acc", "val_ece", "time_s"])

    best_val, t0 = -1.0, time.time()
    for epoch in range(cfg["epochs"]):
        model.train()
        tot_loss, tot_correct, tot_n = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"ep {epoch+1}/{cfg['epochs']}", leave=False, ncols=100)
        for x, y in pbar:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s_logits = model(x)
                t_logits = None
                if teacher is not None:
                    with torch.no_grad():
                        t_logits = teacher(x)
            loss, parts = total_loss(cfg, s_logits.float(),
                                     None if t_logits is None else t_logits.float(), y, epoch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot_loss += loss.item() * y.size(0)
            tot_correct += (s_logits.argmax(1) == y).sum().item()
            tot_n += y.size(0)
            pbar.set_postfix({k: f"{v:.3f}" for k, v in parts.items()})
        sched.step()

        v_logits, v_labels = collect_logits(model, val_loader)
        vm = all_metrics(v_logits, v_labels)
        row = [epoch + 1, opt.param_groups[0]["lr"], tot_loss / tot_n, tot_correct / tot_n,
               vm["acc"], vm["ece"], time.time() - t0]
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        print(f"ep {epoch+1:3d} | loss {row[2]:.3f} | train {row[3]:.4f} | "
              f"val {vm['acc']:.4f} ece {vm['ece']:.3f} | {row[6]/60:.1f} min")

        ckpt = {"model": cfg["model"], "num_classes": 100, "state_dict": model.state_dict(),
                "epoch": epoch + 1, "val_acc": vm["acc"], "config": cfg}
        torch.save(ckpt, run_dir / "last.pt")
        if vm["acc"] > best_val:
            best_val = vm["acc"]
            shutil.copyfile(run_dir / "last.pt", run_dir / "best.pt")

    # Evaluation finale sur le test propre, avec le meilleur checkpoint (selectionne sur val).
    best = load_checkpoint(run_dir / "best.pt", device)
    s_logits, labels = collect_logits(best, test_loader)
    t_logits = None if teacher is None else collect_logits(teacher, test_loader)[0]
    final = all_metrics(s_logits, labels, t_logits)
    final.update({"best_val_acc": best_val, "params": count_params(model),
                  "train_minutes": (time.time() - t0) / 60})
    json.dump(final, open(run_dir / "final_test.json", "w"), indent=2)
    print("TEST:", json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
