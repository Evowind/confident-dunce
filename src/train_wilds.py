"""Entrainement sur un jeu WILDS (camelyon17 ou iwildcam), teacher ou student distille, modeles timm.
Generalise train_camelyon.py : choix du jeu par cfg['dataset'], macro-F1 en plus (iWildCam).

    python -m src.train_wilds --config configs/iwc_teacher_resnet50.yaml
    python -m src.train_wilds --config configs/iwc_student_mnv3s_dkd_nckd.yaml --eval-only

Selection du meilleur checkpoint sur id_val a chaque epoch ; les splits OOD (val, test) ne sont
evalues qu'a la fin, avec le meilleur checkpoint.
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
from sklearn.metrics import f1_score
from tqdm import tqdm

from .losses import total_loss
from .metrics import all_metrics, collect_logits
from .train import set_seed

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ["train", "id_val", "val", "test"]


def dataset_api(name):
    if name == "camelyon17":
        from .camelyon import camelyon_loaders
        return camelyon_loaders, 2, 96, "runs_cam"
    if name == "iwildcam":
        from .iwildcam import iwildcam_loaders
        return iwildcam_loaders, 182, None, "runs_iwc"
    raise ValueError(name)


def build(name, pretrained, n_classes):
    return timm.create_model(name, pretrained=pretrained, num_classes=n_classes)


def load_wilds_checkpoint(path, device="cuda"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    n_classes = ckpt.get("n_classes")
    if n_classes is None:   # checkpoints de train_camelyon.py : deduit du biais du classifieur
        n_classes = [v for v in ckpt["state_dict"].values() if v.ndim == 1][-1].shape[0]
    m = build(ckpt["model"], False, n_classes)
    m.load_state_dict(ckpt["state_dict"])
    return m.to(device).to(memory_format=torch.channels_last).eval()


def metrics_plus(logits, labels, t_logits=None):
    m = all_metrics(logits, labels, t_logits)
    m["macro_f1"] = float(f1_score(labels.numpy(), logits.argmax(1).numpy(), average="macro"))
    return m


def final_eval(run_dir, cfg, make_loaders, teacher, n_params, train_minutes, workers, device="cuda"):
    bm = load_wilds_checkpoint(run_dir / "best.pt", device)
    final = {}
    for s in SPLITS[1:]:
        loader = make_loaders(cfg["batch_size"], splits=[s], eval_workers=min(workers, 3),
                              **({"size": cfg["size"]} if "size" in cfg else {}))[s]
        lg, lb = collect_logits(bm, loader)
        tl = None if teacher is None else collect_logits(teacher, loader)[0]
        final[s] = metrics_plus(lg, lb, tl)
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
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--eval-only", action="store_true")
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
    cfg.setdefault("dataset", "camelyon17")
    set_seed(cfg["seed"])
    device = "cuda"

    make_loaders, n_classes, fixed_size, runs_dirname = dataset_api(cfg["dataset"])
    size = cfg.get("size", fixed_size)
    size_kw = {} if fixed_size is not None else {"size": size}
    run_dir = ROOT / runs_dirname / cfg["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    if not args.eval_only:
        yaml.safe_dump(cfg, open(run_dir / "config.yaml", "w", encoding="utf-8"))

    model = build(cfg["model"], cfg.get("pretrained", True), n_classes).to(device).to(memory_format=torch.channels_last)
    n_params = sum(q.numel() for q in model.parameters())

    teacher = None
    if cfg["kd"]["method"] != "none":
        teacher = load_wilds_checkpoint(ROOT / cfg["teacher_ckpt"], device)
        for q in teacher.parameters():
            q.requires_grad_(False)

    if args.eval_only:
        assert (run_dir / "best.pt").exists(), f"pas de best.pt dans {run_dir}"
        log = list(csv.reader(open(run_dir / "log.csv")))
        train_minutes = float(log[-1][-1]) / 60 if len(log) > 1 else 0.0
        print(f"[{cfg['name']}] eval-only")
        final_eval(run_dir, {**cfg, **size_kw}, make_loaders, teacher, n_params, train_minutes, args.workers, device)
        return

    loaders = make_loaders(cfg["batch_size"], strong_aug=cfg.get("strong_aug", False),
                           train_subset=cfg.get("train_subset"), seed=cfg["seed"],
                           num_workers=args.workers, splits=["train", "id_val"], **size_kw)
    print(f"[{cfg['name']}] {cfg['dataset']} / {cfg['model']} : {n_params/1e6:.2f} M params, "
          f"{len(loaders['train'].dataset)} images d'entrainement, {n_classes} classes")

    use_feat = cfg["kd"]["method"] == "feat"
    projector = None
    if use_feat:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, size, size, device=device).to(memory_format=torch.channels_last)
            s_dim = model.forward_head(model.forward_features(dummy), pre_logits=True).shape[1]
            t_dim = teacher.forward_head(teacher.forward_features(dummy), pre_logits=True).shape[1]
        projector = torch.nn.Linear(s_dim, t_dim).to(device)
        print(f"  features : student {s_dim} -> projecteur -> teacher {t_dim}")
    params = list(model.parameters()) + (list(projector.parameters()) if projector is not None else [])
    opt = torch.optim.SGD(params, lr=cfg["lr"], momentum=0.9, weight_decay=cfg["weight_decay"], nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"])

    log_path = run_dir / "log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "lr", "train_loss", "train_acc", "id_val_acc", "id_val_f1", "id_val_ece", "time_s"])

    # Montee lineaire du terme KD par iteration sur warmup_epochs (les logits initiaux de la tete
    # timm sont grands : sans warmup, DKD a T=4 a fait diverger un run Camelyon). + ecretage du gradient.
    warm_iters = cfg["kd"].get("warmup_epochs", 0) * len(loaders["train"])
    clip = cfg.get("grad_clip", 5.0)

    best, t0 = -1.0, time.time()
    for epoch in range(cfg["epochs"]):
        model.train()
        tot_loss, tot_correct, tot_n = 0.0, 0, 0
        pbar = tqdm(loaders["train"], desc=f"ep {epoch+1}/{cfg['epochs']}", leave=False, ncols=100)
        for it, (x, y) in enumerate(pbar):
            step = epoch * len(loaders["train"]) + it + 1
            ramp = min(1.0, step / warm_iters) if warm_iters > 0 else 1.0
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
            loss, parts = total_loss(cfg, s_logits.float(), None if t_logits is None else t_logits.float(),
                                     y, epoch, s_feat=s_feat, t_feat=t_feat, ramp=ramp)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(params, clip)
            opt.step()
            tot_loss += loss.item() * y.size(0)
            tot_correct += (s_logits.argmax(1) == y).sum().item()
            tot_n += y.size(0)
            pbar.set_postfix({k: f"{v:.3f}" for k, v in parts.items()})
        sched.step()

        lg, lb = collect_logits(model, loaders["id_val"])
        ev = metrics_plus(lg, lb)
        sel = ev["macro_f1"] if cfg["dataset"] == "iwildcam" else ev["acc"]   # critere de selection WILDS
        row = [epoch + 1, opt.param_groups[0]["lr"], tot_loss / tot_n, tot_correct / tot_n,
               ev["acc"], ev["macro_f1"], ev["ece"], time.time() - t0]
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        print(f"ep {epoch+1:2d} | loss {row[2]:.3f} | train {row[3]:.4f} | "
              f"id_val acc {ev['acc']:.4f} f1 {ev['macro_f1']:.4f} ece {ev['ece']:.3f} | {row[-1]/60:.1f} min")

        ckpt = {"model": cfg["model"], "n_classes": n_classes, "state_dict": model.state_dict(),
                "epoch": epoch + 1, "id_val_sel": sel, "config": cfg}
        torch.save(ckpt, run_dir / "last.pt")
        if sel > best:
            best = sel
            shutil.copyfile(run_dir / "last.pt", run_dir / "best.pt")

    train_minutes = (time.time() - t0) / 60
    del loaders
    final_eval(run_dir, {**cfg, **size_kw}, make_loaders, teacher, n_params, train_minutes, args.workers, device)


if __name__ == "__main__":
    main()
