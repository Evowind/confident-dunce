"""Planches portfolio pour confident-dunce, rendues depuis les CSV du depot (aucun chiffre saisi a la main).

    .venv/Scripts/python.exe portfolio/make_figures.py [--out <dossier>]

Sortie par defaut : ../netlify-portfolio/public/images/confident-dunce/
  01-confidence-copied.webp      hero, 1920x1080 exactement
  02-target-vs-non-target.webp   le mecanisme (ablation DKD, deux paires)
  03-real-shift.webp             iWildCam (3 seeds) et Camelyon17 v2 (3 seeds)

Style : docs/figure-style.md du portfolio (palette, IBM Plex Sans / JetBrains Mono, rien sous 16 px,
titre = un resultat, pied de provenance obligatoire, WebP q82 1920 px, <= 400 Ko).
"""
import argparse
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
FONTS = Path(__file__).resolve().parent / "fonts"

# ---- palette (docs/figure-style.md, copiee telle quelle)
BG, PANEL, RAISED = "#12141c", "#191c26", "#1f2331"
BORDER, BORDER_STRONG = "#262a38", "#363c4e"
TEXT, BRIGHT, MUTED, FAINT = "#e8e6df", "#f4f2ec", "#8b90a3", "#4a4f63"
AMBER, BLUE, GREEN, PINK, RED = "#e8a33d", "#6c9ce0", "#7fb88a", "#d98ba3", "#e0716b"

W, DPI = 1920, 100
SANS, MONO = "IBM Plex Sans", "JetBrains Mono"


def ensure_fonts():
    for p in FONTS.glob("*.ttf"):
        font_manager.fontManager.addfont(str(p))
    names = {f.name for f in font_manager.fontManager.ttflist}
    missing = [n for n in (SANS, MONO) if n not in names]
    if missing:
        sys.exit(f"polices manquantes : {missing} (attendues dans {FONTS})")
    plt.rcParams.update({"font.family": SANS, "text.color": TEXT, "axes.edgecolor": BORDER,
                         "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.facecolor": PANEL, "figure.facecolor": BG, "savefig.facecolor": BG})


def pt(px):
    """taille en px du fichier exporte -> points matplotlib a 100 dpi"""
    return px * 72 / DPI


def new_plate(height):
    fig = plt.figure(figsize=(W / DPI, height / DPI), dpi=DPI)
    fig.patch.set_facecolor(BG)
    return fig, height


def text(fig, H, x, y, s, size, color=TEXT, mono=False, weight="normal", ha="left", va="baseline", max_w=None, zorder=6):
    """x, y en px depuis le coin haut-gauche. max_w : retrecit jusqu'a tenir (plancher 36 px)."""
    fam = MONO if mono else SANS
    t = fig.text(x / W, 1 - y / H, s, fontsize=pt(size), color=color, family=fam, weight=weight, ha=ha, va=va, zorder=zorder)
    if max_w:
        r = fig.canvas.get_renderer()
        while t.get_window_extent(r).width > max_w and size > 36:
            size -= 2
            t.set_fontsize(pt(size))
    return t


def footer(fig, H, s):
    text(fig, H, 96, H - 42, s, 17, FAINT, mono=True)


def axes_px(fig, H, x, y, w, h):
    ax = fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_color(BORDER)
    ax.tick_params(colors=MUTED, labelsize=pt(19), length=0, pad=8)
    ax.grid(True, color=BORDER, lw=1)
    ax.set_axisbelow(True)
    return ax


def fade(fig, H, y0, y1):
    """degrade transparent -> BG entre y0 et y1 (px), sur toute la largeur"""
    ax = fig.add_axes([0, 1 - y1 / H, 1, (y1 - y0) / H], zorder=5)
    ax.axis("off")
    g = np.linspace(0, 1, 64).reshape(-1, 1)
    rgba = np.zeros((64, 1, 4))
    rgba[..., :3] = matplotlib.colors.to_rgb(BG)
    rgba[..., 3] = g
    ax.imshow(rgba, aspect="auto", extent=(0, 1, 0, 1), interpolation="bilinear")


def save(fig, path, hero=False):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, facecolor=BG)
    plt.close(fig)
    im = Image.open(buf).convert("RGB")
    if hero:
        assert im.size == (1920, 1080), im.size
    assert im.size[0] == 1920 and im.size[1] <= 1400, im.size
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, "WEBP", quality=82, method=6)
    kb = path.stat().st_size / 1024
    assert kb <= 400, f"{path.name}: {kb:.0f} Ko > 400"
    print(f"{path.name}: {im.size[0]}x{im.size[1]}, {kb:.0f} Ko")


# ----------------------------------------------------------------- donnees
def cifar_by_severity(files, runs):
    df = pd.concat([pd.read_csv(ROOT / f) for f in files], ignore_index=True)
    df["config"] = df.run.str.replace(r"_s\d+$", "", regex=True)
    df = df[df.config.isin(runs)]
    per_seed = df.groupby(["config", "run", "severity"])[["acc", "ece"]].mean().reset_index()
    return per_seed.groupby(["config", "severity"])[["acc", "ece"]].agg(["mean", "std", "count"])


def cifar_corrupted_delta(files, runs, base):
    df = pd.concat([pd.read_csv(ROOT / f) for f in files], ignore_index=True)
    df["config"] = df.run.str.replace(r"_s\d+$", "", regex=True)
    k = df[df.severity > 0].groupby("config")[["acc", "ece"]].mean()
    return {r: (100 * (k.loc[r, "acc"] - k.loc[base, "acc"]), 100 * (k.loc[r, "ece"] - k.loc[base, "ece"])) for r in runs}


# ----------------------------------------------------------------- planches
def plate_hero(out):
    files = ["results.csv", "results_seeds_a.csv", "results_seeds_b.csv", "results_dkd_ablation_r20.csv"]
    # couleur + marqueur + trait pour chaque serie : lisible en niveaux de gris et pour les daltonismes
    runs = {"student_resnet20_scratch": ("student alone", TEXT, "--", "s", 9),
            "student_resnet20_kd": ("Hinton KD", BLUE, "-", "o", 9),
            "student_resnet20_dkd_nckd": ("non-target term only", AMBER, "-", "D", 8),
            "teacher_wrn40_2": ("teacher WRN-40-2", MUTED, ":", "^", 10)}
    d = cifar_by_severity(files, list(runs))
    fig, H = new_plate(1080)
    for i, (m, lab) in enumerate([("acc", "accuracy"), ("ece", "expected calibration error")]):
        ax = axes_px(fig, H, 96 + i * 944, 48, 784, 540)
        for r, (name, col, ls, mk, ms) in runs.items():
            g = d.loc[r]
            ax.plot(g.index, g[(m, "mean")], ls, color=col, lw=3.5, label=name, marker=mk, ms=ms,
                    markerfacecolor=BG if r.startswith("teacher") else col, markeredgewidth=2.5)
            if g[(m, "count")].max() > 1:
                ax.fill_between(g.index, g[(m, "mean")] - g[(m, "std")].fillna(0), g[(m, "mean")] + g[(m, "std")].fillna(0), color=col, alpha=0.12, lw=0)
        ax.set_xlabel("corruption severity (0 = clean)", fontsize=pt(21), color=MUTED)
        ax.set_ylabel(lab, fontsize=pt(21), color=MUTED)
        ax.set_xticks(range(6))
        if i == 0:
            ax.legend(fontsize=pt(20), frameon=False, labelcolor=TEXT, loc="upper right", handlelength=4.5, numpoints=1)
    fade(fig, H, 660, 760)
    text(fig, H, 96, 724, "CIFAR-100-C  ·  19 corruptions × 5 severities  ·  mean ± sd over 3 seeds", 17, MUTED, mono=True, zorder=8)
    text(fig, H, W - 96, 724, "WRN-40-2 → ResNet-20", 17, MUTED, mono=True, ha="right", zorder=8)
    text(fig, H, 96, 806, "The student inherits the teacher's confidence, not its accuracy", 54, BRIGHT, weight="semibold", max_w=W - 192)
    text(fig, H, 96, 848, "What knowledge distillation transmits to a small model when the data shift: robustness, calibration or just confidence.", 24, MUTED)
    text(fig, H, 96, 882, "Hinton KD matches the student alone on accuracy but doubles its calibration error; the non-target term alone keeps it calibrated.", 24, MUTED)
    kd = d.loc["student_resnet20_kd"]; nc = d.loc["student_resnet20_dkd_nckd"]; sc = d.loc["student_resnet20_scratch"]
    ece_kd = kd[("ece", "mean")].loc[1:].mean(); ece_nc = nc[("ece", "mean")].loc[1:].mean(); ece_sc = sc[("ece", "mean")].loc[1:].mean()
    stats = [("86", "valid runs · 3 datasets · 4 pairs"),
             (f"{ece_kd:.2f} → {ece_nc:.2f}", "ECE, corrupted: Hinton KD → non-target only"),
             (f"{ece_sc:.2f}", "ECE, corrupted: student alone"),
             ("0", "non-target term with two classes")]
    for i, (v, l) in enumerate(stats):
        text(fig, H, 96 + i * 440, 942, v, 36, BRIGHT, mono=True)
        text(fig, H, 96 + i * 440, 978, l, 19, MUTED)
    footer(fig, H, "results.csv, results_seeds_a/b.csv, results_dkd_ablation_r20.csv (repository); non-target-only curve: 1 seed; CIFAR-100 / CIFAR-100-C test sets")
    save(fig, out / "01-confidence-copied.webp", hero=True)


def plate_mechanism(out):
    pairs = [("WRN-40-2 → ResNet-20  (0.28 M)", ["results.csv", "results_dkd_ablation_r20.csv"], "student_resnet20_scratch",
              {"student_resnet20_dkd_tckd": "target term only", "student_resnet20_dkd_nckd": "non-target term only", "student_resnet20_dkd": "both (DKD)"}),
             ("ResNet-32x4 → ResNet-8x4  (1.25 M)", ["results_pairs_res.csv", "results_dkd_ablation_r8x4.csv"], "student_resnet8x4_scratch",
              {"student_resnet8x4_dkd_tckd": "target term only", "student_resnet8x4_dkd_nckd": "non-target term only", "student_resnet8x4_dkd": "both (DKD)"})]
    cols = {"target term only": BLUE, "non-target term only": AMBER, "both (DKD)": PINK}
    fig, H = new_plate(1160)
    text(fig, H, 96, 64, "MECHANISM  ·  DECOUPLED KD ABLATION", 18, MUTED, mono=True)
    text(fig, H, 96, 128, "The target term carries the confidence, the non-target term carries the robustness", 52, BRIGHT, weight="semibold", max_w=W - 192)
    text(fig, H, 96, 172, "Change versus the student trained alone, averaged over the 95 corrupted test sets. Teacher's KL splits exactly into these two terms (Zhao et al., 2022).", 24, MUTED)
    for j, (title, files, base, runs) in enumerate(pairs):
        deltas = cifar_corrupted_delta(files, list(runs), base)
        for i, (idx, lab, good_up) in enumerate([(0, "accuracy under corruption (points)", True), (1, "calibration error under corruption (points)", False)]):
            ax = axes_px(fig, H, 96 + j * 944, 250 + i * 400, 784, 320)
            names = list(runs.values()); vals = [deltas[r][idx] for r in runs]
            bars = ax.bar(range(3), vals, color=[cols[n] for n in names], width=0.62)
            ax.axhline(0, color=MUTED, lw=1.5, ls="--")
            ax.set_xticks(range(3)); ax.set_xticklabels(names, fontsize=pt(20), color=TEXT)
            ax.set_ylabel(lab, fontsize=pt(20), color=MUTED)
            lo, hi = min(vals + [0]), max(vals + [0]); span = hi - lo
            ax.set_ylim(lo - 0.35 * span - 0.5, hi + 0.35 * span + 0.5)
            for b, v in zip(bars, vals):
                col = GREEN if (v > 0) == good_up else RED
                ax.text(b.get_x() + b.get_width() / 2, v + (0.04 if v >= 0 else -0.04) * span * (1 if v >= 0 else 1) + (0.25 if v >= 0 else -0.25),
                        f"{v:+.1f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=pt(21), color=col, family=MONO)
            if i == 0:
                ax.set_title(title, fontsize=pt(24), color=TEXT, loc="left", pad=14, weight="semibold")
    footer(fig, H, "results.csv, results_dkd_ablation_r20.csv, results_pairs_res.csv, results_dkd_ablation_r8x4.csv; ablations: 1 seed; green = better than alone, red = worse")
    save(fig, out / "02-target-vs-non-target.webp")


def plate_real(out):
    iwc = pd.read_csv(ROOT / "figures/iwildcam/iwc_table.csv").set_index("config")
    cam = pd.read_csv(ROOT / "figures/camelyon/cam_table.csv").set_index("config")
    fig, H = new_plate(1220)
    text(fig, H, 96, 64, "REAL DISTRIBUTION SHIFT  ·  WILDS", 18, MUTED, mono=True)
    text(fig, H, 96, 128, "With two classes the non-target term vanishes and only confidence is transmitted", 52, BRIGHT, weight="semibold", max_w=W - 192)
    text(fig, H, 96, 172, "Left: 182 species, 48 camera traps never seen in training. Right: tumour patches from a hospital never seen in training. Mean ± sd over 3 seeds.", 24, MUTED)

    def bars(ax, rows, metric, ylabel, teacher_row=None, ylim=None, label_side="right"):
        names = [n for n, _, _ in rows]; vals = [100 * float(rows_src.loc[r, f"{metric}_test"]) for _, r, rows_src in rows]
        errs = [100 * float(rows_src.loc[r, f"{metric}_test_std"]) if not np.isnan(rows_src.loc[r, f"{metric}_test_std"]) else 0 for _, r, rows_src in rows]
        cols_ = [c for _, _, c in [(n, r, col) for (n, r, _), col in zip(rows, colors)]]
        b = ax.bar(range(len(rows)), vals, yerr=errs, color=colors, width=0.62, error_kw=dict(ecolor=TEXT, lw=2, capsize=6))
        for bi, v in zip(b, vals):
            ax.text(bi.get_x() + bi.get_width() / 2, v + (ylim[1] - ylim[0]) * 0.03 + max(errs), f"{v:.1f}", ha="center", fontsize=pt(20), color=TEXT, family=MONO)
        if teacher_row is not None:
            tv = 100 * float(teacher_row[f"{metric}_test"])
            ax.axhline(tv, color=MUTED, lw=1.5, ls="--")
            xa, ha = (0.985, "right") if label_side == "right" else (0.015, "left")
            ax.text(xa, 0.94, f"teacher (dashed)  {tv:.1f}", transform=ax.transAxes, ha=ha, va="top",
                    fontsize=pt(18), color=MUTED, family=MONO)
        ax.set_xticks(range(len(rows))); ax.set_xticklabels(names, fontsize=pt(19), color=TEXT)
        ax.set_ylabel(ylabel, fontsize=pt(20), color=MUTED); ax.set_ylim(*ylim)

    # iWildCam
    colors = [MUTED, BLUE, AMBER]
    rows = [("student alone", "student_mnv3s_scratch", iwc), ("Hinton KD", "student_mnv3s_kd", iwc), ("non-target only", "student_mnv3s_dkd_nckd", iwc)]
    ax = axes_px(fig, H, 96, 250, 784, 330); bars(ax, rows, "acc", "accuracy on unseen cameras (%)", iwc.loc["teacher_resnet50"], (0, 100))
    ax.set_title("iWildCam, MobileNetV3-Small from ResNet-50", fontsize=pt(24), color=TEXT, loc="left", pad=14, weight="semibold")
    ax = axes_px(fig, H, 96, 660, 784, 330); bars(ax, rows, "ece", "calibration error on unseen cameras (%)", iwc.loc["teacher_resnet50"], (0, 45))
    # Camelyon v2
    colors = [MUTED, BLUE, BLUE, MUTED, BLUE]
    rows = [("alone", "student_mnv3s_scratch_v2", cam), ("KD, std T", "student_mnv3s_kd_v2", cam), ("KD, robust T", "student_mnv3s_kd_strongT_v2", cam),
            ("alone + aug", "student_mnv3s_scratch_aug_v2", cam), ("KD + aug", "student_mnv3s_kd_strongT_aug_v2", cam)]
    ax = axes_px(fig, H, 1040, 250, 784, 330); bars(ax, rows, "acc", "accuracy on the unseen hospital (%)", cam.loc["teacher_resnet50"], (0, 110), label_side="left")
    ax.set_title("Camelyon17, binary: the non-target term is identically zero", fontsize=pt(24), color=TEXT, loc="left", pad=14, weight="semibold")
    ax = axes_px(fig, H, 1040, 660, 784, 330); bars(ax, rows, "ece", "calibration error on the unseen hospital (%)", cam.loc["teacher_resnet50"], (0, 50))
    text(fig, H, 1040, 1060, "The student alone overfits the known hospitals; KD passes on the teacher's calibration;", 20, MUTED)
    text(fig, H, 1040, 1090, "augmenting the student's own data dominates everything else.", 20, MUTED)
    text(fig, H, 96, 1060, "Same accuracy for both distillations, half the calibration error", 20, MUTED)
    text(fig, H, 96, 1090, "with the non-target term alone.", 20, MUTED)
    footer(fig, H, "figures/iwildcam/iwc_table.csv, figures/camelyon/cam_table.csv (repository); 3 seeds per student, 1 per teacher; WILDS official OOD test splits")
    save(fig, out / "03-real-shift.webp")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT.parent / "netlify-portfolio" / "public" / "images" / "confident-dunce"))
    a = p.parse_args()
    out = Path(a.out)
    ensure_fonts()
    plate_hero(out)
    plate_mechanism(out)
    plate_real(out)
    print("->", out)


if __name__ == "__main__":
    main()
