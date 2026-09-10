"""CIFAR-100 (propre) et CIFAR-100-C (19 corruptions x 5 severites)."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MEAN = (0.5071, 0.4865, 0.4409)
STD = (0.2673, 0.2564, 0.2762)

# 15 corruptions standard de Hendrycks & Dietterich + 4 "extra".
CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
    "speckle_noise", "gaussian_blur", "spatter", "saturate",
]
SEVERITIES = [1, 2, 3, 4, 5]


def train_transform(augmix: bool = False):
    ops = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    if augmix:
        ops.append(transforms.AugMix())
    ops += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(ops)


def test_transform():
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def cifar100_loaders(batch_size: int, augmix: bool = False, val_size: int = 5000,
                     seed: int = 0, num_workers: int = 8):
    """Retourne (train, val, test). La validation est prelevee sur le train :
    la selection de modele ne touche jamais au test."""
    full_train = datasets.CIFAR100(DATA_DIR, train=True, download=True,
                                   transform=train_transform(augmix))
    full_val = datasets.CIFAR100(DATA_DIR, train=True, download=False,
                                 transform=test_transform())
    test = datasets.CIFAR100(DATA_DIR, train=False, download=True,
                             transform=test_transform())

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(full_train), generator=g).tolist()
    val_idx, train_idx = perm[:val_size], perm[val_size:]

    common = dict(num_workers=num_workers, pin_memory=True,
                  persistent_workers=num_workers > 0)
    train_loader = DataLoader(Subset(full_train, train_idx), batch_size=batch_size,
                              shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(Subset(full_val, val_idx), batch_size=1024, shuffle=False, **common)
    test_loader = DataLoader(test, batch_size=1024, shuffle=False, **common)
    return train_loader, val_loader, test_loader


class CIFAR100C(Dataset):
    """Une corruption a une severite donnee. Fichiers .npy de shape (50000,32,32,3) :
    les 10000 premieres images = severite 1, les 10000 suivantes = severite 2, etc."""

    def __init__(self, corruption: str, severity: int, root: Path = DATA_DIR / "CIFAR-100-C"):
        assert corruption in CORRUPTIONS, corruption
        assert severity in SEVERITIES, severity
        root = Path(root)
        if not (root / f"{corruption}.npy").exists():
            raise FileNotFoundError(
                f"{root / corruption}.npy introuvable. Lancer scripts/download_cifar100c.py")
        lo, hi = (severity - 1) * 10000, severity * 10000
        self.x = np.load(root / f"{corruption}.npy", mmap_mode="r")[lo:hi]
        self.y = np.load(root / "labels.npy")[lo:hi].astype(np.int64)
        self.tf = test_transform()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        img = np.ascontiguousarray(self.x[i])
        return self.tf(transforms.functional.to_pil_image(img)), int(self.y[i])


def cifar100c_loader(corruption: str, severity: int, batch_size: int = 1024, num_workers: int = 4):
    return DataLoader(CIFAR100C(corruption, severity), batch_size=batch_size,
                      shuffle=False, num_workers=num_workers, pin_memory=True)
