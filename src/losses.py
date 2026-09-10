"""Pertes de distillation. Toutes retournent un scalaire moyenne sur le batch."""
import torch
import torch.nn.functional as F


def kd_hinton(s_logits, t_logits, T: float):
    """KL(p_teacher^T || p_student^T) * T^2  (Hinton et al. 2015)."""
    log_ps = F.log_softmax(s_logits / T, dim=1)
    pt = F.softmax(t_logits / T, dim=1)
    return F.kl_div(log_ps, pt, reduction="batchmean") * (T * T)


def _cat_mask(p, gt_mask, other_mask):
    p_gt = (p * gt_mask).sum(1, keepdim=True)
    p_other = (p * other_mask).sum(1, keepdim=True)
    return torch.cat([p_gt, p_other], dim=1)


def dkd(s_logits, t_logits, target, alpha: float, beta: float, T: float):
    """Decoupled Knowledge Distillation (Zhao et al., CVPR 2022).
    TCKD : binaire cible / non-cible. NCKD : distribution parmi les non-cibles."""
    gt_mask = F.one_hot(target, s_logits.size(1)).bool()
    other_mask = ~gt_mask
    ps = _cat_mask(F.softmax(s_logits / T, 1), gt_mask, other_mask)
    pt = _cat_mask(F.softmax(t_logits / T, 1), gt_mask, other_mask)
    tckd = F.kl_div(torch.log(ps.clamp_min(1e-8)), pt, reduction="batchmean") * (T * T)

    big = 1000.0 * gt_mask
    log_ps2 = F.log_softmax(s_logits / T - big, 1)
    pt2 = F.softmax(t_logits / T - big, 1)
    nckd = F.kl_div(log_ps2, pt2, reduction="batchmean") * (T * T)
    return alpha * tckd + beta * nckd


def feat_loss(s_feat, t_feat):
    """Distillation de features (style FitNets / hint) sur la representation penultieme :
    distance cosinus entre les features projetees du student et celles du teacher.
    s_feat doit deja etre projete a la dimension du teacher. Independant des logits, donc
    utilisable en binaire ou la KD par logits ne porte que la confiance."""
    return (1.0 - F.cosine_similarity(s_feat, t_feat, dim=1)).mean()


def total_loss(cfg: dict, s_logits, t_logits, target, epoch: int, s_feat=None, t_feat=None, ramp=None):
    """Combine CE et terme KD selon cfg['kd']. Renvoie (loss, dict de composantes).
    ramp : facteur de montee du terme KD dans [0, 1]. Si None, calcule par epoch entiere
    (comportement historique : warmup_epochs=1 vaut alors 1.0 des l'epoch 0, donc aucun warmup)."""
    kd_cfg = cfg["kd"]
    ce = F.cross_entropy(s_logits, target)
    method = kd_cfg["method"]
    if method == "none" or (t_logits is None and t_feat is None):
        return ce, {"ce": ce.item()}

    if ramp is None:
        warm = kd_cfg.get("warmup_epochs", 0)
        ramp = min(1.0, (epoch + 1) / warm) if warm > 0 else 1.0
    T = kd_cfg.get("temperature", 1.0)   # sans objet pour "feat"
    if method == "kd":
        kd = kd_hinton(s_logits, t_logits, T)
        loss = kd_cfg["ce_weight"] * ce + ramp * kd_cfg["kd_weight"] * kd
    elif method == "dkd":
        kd = dkd(s_logits, t_logits, target, kd_cfg["alpha"], kd_cfg["beta"], T)
        loss = kd_cfg["ce_weight"] * ce + ramp * kd
    elif method == "feat":
        kd = feat_loss(s_feat, t_feat)
        loss = kd_cfg["ce_weight"] * ce + ramp * kd_cfg["feat_weight"] * kd
    else:
        raise ValueError(method)
    return loss, {"ce": ce.item(), "kd": kd.item()}
