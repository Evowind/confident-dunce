"""Metriques : accuracy, calibration (ECE, NLL, Brier), fidelite teacher/student."""
import torch
import torch.nn.functional as F


@torch.no_grad()
def collect_logits(model, loader, device="cuda"):
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
        logits.append(out.float().cpu())
        labels.append(y)
    return torch.cat(logits), torch.cat(labels)


def accuracy(logits, labels):
    return (logits.argmax(1) == labels).float().mean().item()


def nll(logits, labels):
    return F.cross_entropy(logits, labels).item()


def brier(logits, labels):
    p = F.softmax(logits, 1)
    onehot = F.one_hot(labels, p.size(1)).float()
    return ((p - onehot) ** 2).sum(1).mean().item()


def ece(logits, labels, n_bins: int = 15):
    """Expected Calibration Error (Guo et al. 2017), bins de confiance egaux."""
    p = F.softmax(logits, 1)
    conf, pred = p.max(1)
    correct = (pred == labels).float()
    bins = torch.linspace(0, 1, n_bins + 1)
    e = torch.zeros(())
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.float().mean() * (conf[m].mean() - correct[m].mean()).abs()
    return e.item()


def agreement(s_logits, t_logits):
    """Fidelite top-1 : fraction d'exemples ou student et teacher predisent la meme classe."""
    return (s_logits.argmax(1) == t_logits.argmax(1)).float().mean().item()


def kl_teacher_student(s_logits, t_logits, T: float = 1.0):
    """KL(p_teacher || p_student) a temperature T. Fidelite continue (Stanton et al. 2021)."""
    return F.kl_div(F.log_softmax(s_logits / T, 1), F.softmax(t_logits / T, 1),
                    reduction="batchmean").item()


def all_metrics(s_logits, labels, t_logits=None):
    m = {
        "acc": accuracy(s_logits, labels),
        "ece": ece(s_logits, labels),
        "nll": nll(s_logits, labels),
        "brier": brier(s_logits, labels),
        "mean_conf": F.softmax(s_logits, 1).max(1)[0].mean().item(),
    }
    if t_logits is not None:
        m["agreement"] = agreement(s_logits, t_logits)
        m["kl_ts"] = kl_teacher_student(s_logits, t_logits)
        # Sur les exemples ou le teacher se trompe : le student copie-t-il ses erreurs ?
        wrong = t_logits.argmax(1) != labels
        m["agreement_on_teacher_errors"] = (
            agreement(s_logits[wrong], t_logits[wrong]) if wrong.any() else float("nan"))
    return m
