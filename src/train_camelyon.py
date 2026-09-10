"""Entrainement sur Camelyon17 (teacher ou student distille), modeles timm.

    python -m src.train_camelyon --config configs/cam_teacher_resnet50.yaml
    python -m src.train_camelyon --config configs/cam_student_mnv3s_dkd_nckd.yaml --epochs 1 --name smoke
    python -m src.train_camelyon --config configs/cam_student_mnv3s_kd.yaml --eval-only   # best.pt existant

Selection du meilleur checkpoint sur id_val (hopitaux connus) a chaque epoch. Les hopitaux OOD
(val = hopital 1, test = hopital 2) ne sont evalues qu'une fois, a la fin, avec le meilleur
checkpoint : jamais regardes pendant l'entrainement.
"""
import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import timm
import torch
import yaml
from tqdm import tqdm

from .camelyon import SPLITS, camelyon_loaders
from .losses import total_loss
from .metrics import all_metrics, collect_logits
from .train import set_seed

ROOT = Path(__file__).resolve().parent.parent


def build(name: str, pretrained: bool):
    return timm.create_model(name, pretrained=pretrained, num_classes=2)


def load_cam_checkpoint(path, device="cuda"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    m = build(ckpt["model"], pretrained=False)
    m.load_state_dict(ckpt["state_dict"])
    return m.to(device).to(memory_format=torch.channels_last).eval()


def final_eval(run_dir, cfg, teacher, n_params, train_minutes, workers, device="cuda"):
    """Evalue best.pt sur id_val, val (hopital 1) et test (hopital 2). Un loader a la fois,
    workers liberes entre deux splits."""
    bm = load_cam_checkpoint(run_dir / "best.pt", device)
    final = {}
    for s in SPLITS[1:]:
        loader = camelyon_loaders(cfg["batch_size"], splits=[s], eval_workers=min(workers, 3))[s]
        lg, lb = collect_logits(bm, loader)
        tl = None if teacher is None else collect_logits(teacher, loader)[0]
        final[s] = all_metrics(lg, lb, tl)
        del loader
    final["params"] = n_params
    final["train_minutes"] = train_minutes
    json.dump(final, open(run_dir / "final.json", "w"), indent=2)
    print("FINAL:", json.dumps({s: {k: round(v, 4) for k, v in final[s].items()} for s in SPLITS[1:]}, indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--name", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--eval-only", action="store_true", help="saute l'entrainement, evalue best.pt existant")
    args = p.parse_args()
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

    run_dir = ROOT / "runs_cam" / cfg["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    if not args.eval_only:
        yaml.safe_dump(cfg, open(run_dir / "config.yaml", "w", encoding="utf-8"))

    model = build(cfg["model"], cfg.get("pretrained", True)).to(device).to(memory_format=torch.channels_last)
    n_params = sum(q.numel() for q in model.parameters())

    teacher = None
    if cfg["kd"]["method"] != "none":
        teacher = load_cam_checkpoint(ROOT / cfg["teacher_ckpt"], device)
        for q in teacher.parameters():
            q.requires_grad_(False)

    if args.eval_only:
        assert (run_dir / "best.pt").exists(), f"pas de best.pt dans {run_dir}"
        log = list(csv.reader(open(run_dir / "log.csv")))
        train_minutes = float(log[-1][-1]) / 60 if len(log) > 1 else 0.0
        print(f"[{cfg['name']}] eval-only, best.pt de l'epoch "
              f"{torch.load(run_dir / 'best.pt', map_location='cpu', weights_only=False)['epoch']}")
        final_eval(run_dir, cfg, teacher, n_params, train_minutes, args.workers, device)
        return

    loaders = camelyon_loaders(cfg["batch_size"], strong_aug=cfg.get("strong_aug", False),
                               train_subset=cfg.get("train_subset"), seed=cfg["seed"],
                               num_workers=args.workers, splits=["train", "id_val"])
    print(f"[{cfg['name']}] {cfg['model']} : {n_params/1e6:.2f} M params, "
          f"{len(loaders['train'].dataset)} patches d'entrainement")

    # Distillation de features : projecteur lineaire student -> dimension du teacher, entraine avec le student.
    use_feat = cfg["kd"]["method"] == "feat"
    projector = None
    if use_feat:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 96, 96, device=device).to(memory_format=torch.channels_last)
            s_dim = model.forward_head(model.forward_features(dummy), pre_logits=True).shape[1]
            t_dim = teacher.forward_head(teacher.forward_features(dummy), pre_logits=True).shape[1]
        projector = torch.nn.Linear(s_dim, t_dim).to(device)
        print(f"  features : student {s_dim} -> projecteur -> teacher {t_dim}")
    params = list(model.parameters()) + (list(projector.parameters()) if projector is not None else [])
    opt = torch.optim.SGD(params, lr=cfg["lr"], momentum=0.9,
                          weight_decay=cfg["weight_decay"], nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"])

    log_path = run_dir / "log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "lr", "train_loss", "train_acc", "id_val_acc", "id_val_ece", "time_s"])

    best, t0 = -1.0, time.time()
    for epoch in range(cfg["epochs"]):
        model.train()
        tot_loss, tot_correct, tot_n = 0.0, 0, 0
        pbar = tqdm(loaders["train"], desc=f"ep {epoch+1}/{cfg['epochs']}", leave=False, ncols=100)
        for x, y in pbar:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s_feat = t_feat = t_logits = None
                if use_feat:
                    s_pre = model.forward_head(model.forward_features(x), pre_logits=True)
                    s_logits = model.get_classifier()(s_pre)
                    s_feat = projector(s_pre.float())
                    with torch.no_grad():
                        t_pre = teacher.forward_head(teacher.forward_features(x), pre_logits=True)
                        t_logits = teacher.get_classifier()(t_pre)
                        t_feat = t_pre.float()
                else:
                    s_logits = model(x)
                    if teacher is not None:
                        with torch.no_grad():
                            t_logits = teacher(x)
            loss, parts = total_loss(cfg, s_logits.float(),
                                     None if t_logits is None else t_logits.float(), y, epoch,
                                     s_feat=s_feat, t_feat=t_feat)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot_loss += loss.item() * y.size(0)
            tot_correct += (s_logits.argmax(1) == y).sum().item()
            tot_n += y.size(0)
            pbar.set_postfix({k: f"{v:.3f}" for k, v in parts.items()})
        sched.step()

        lg, lb = collect_logits(model, loaders["id_val"])
        ev = all_metrics(lg, lb)
        row = [epoch + 1, opt.param_groups[0]["lr"], tot_loss / tot_n, tot_correct / tot_n,
               ev["acc"], ev["ece"], time.time() - t0]
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        print(f"ep {epoch+1:2d} | loss {row[2]:.3f} | train {row[3]:.4f} | "
              f"id_val {ev['acc']:.4f} ece {ev['ece']:.3f} | {row[-1]/60:.1f} min")

        ckpt = {"model": cfg["model"], "state_dict": model.state_dict(), "epoch": epoch + 1,
                "id_val_acc": ev["acc"], "config": cfg}
        torch.save(ckpt, run_dir / "last.pt")
        if ev["acc"] > best:
            best = ev["acc"]
            shutil.copyfile(run_dir / "last.pt", run_dir / "best.pt")

    train_minutes = (time.time() - t0) / 60
    del loaders   # libere les workers persistants avant l'evaluation OOD
    final_eval(run_dir, cfg, teacher, n_params, train_minutes, args.workers, device)


if __name__ == "__main__":
    main()
