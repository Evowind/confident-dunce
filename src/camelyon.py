"""Camelyon17 (WILDS) : patches 96x96 de ganglions lymphatiques, tumeur / normal, 5 hopitaux.

Splits WILDS officiels :
  train : hopitaux 0, 3, 4 (302 k patches)
  val (id) : memes hopitaux, patches non vus
  val (ood) : hopital 1  (34 k)
  test (ood) : hopital 2 (85 k)   <- le decalage de distribution reel (scanner, coloration)

Le decalage vient de la coloration et du scanner, pas d'une corruption synthetique.
"""
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from wilds import get_dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MEAN = (0.485, 0.456, 0.406)   # ImageNet : les teachers timm sont pre-entraines dessus
STD = (0.229, 0.224, 0.225)
SPLITS = ["train", "id_val", "val", "test"]     # WILDS : val = hopital 1 (ood), test = hopital 2 (ood)


def train_transform(strong: bool = False):
    ops = [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
    if strong:   # analogue d'AugMix cote CIFAR : perturbation de couleur / coloration
        ops += [transforms.ColorJitter(0.3, 0.3, 0.3, 0.1), transforms.RandomGrayscale(0.1),
                transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3)]
    ops += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(ops)


def test_transform():
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class _Wrap(torch.utils.data.Dataset):
    """WILDS renvoie (x, y, metadata) ; on ne garde que (x, y) pour reutiliser metrics.py."""
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x, y, _ = self.ds[i]
        return x, int(y)


def camelyon_loaders(batch_size: int, strong_aug: bool = False, train_subset: int | None = None,
                     seed: int = 0, num_workers: int = 8, splits=SPLITS, eval_workers: int = 3):
    """Retourne dict split -> DataLoader pour les splits demandes.

    Memoire (Windows, spawn) : chaque worker est un processus complet (~600 Mo avec torch + timm).
    Seuls train et id_val, utilises a chaque epoch, gardent des workers persistants. val / test
    (OOD) sont evalues une fois a la fin avec peu de workers, liberes ensuite.
    train_subset limite le nombre de patches d'entrainement (None = les 302 k)."""
    full = get_dataset(dataset="camelyon17", download=False, root_dir=str(DATA_DIR))
    loaders = {}
    for split in splits:
        is_train = split == "train"
        tf = train_transform(strong_aug) if is_train else test_transform()
        ds = _Wrap(full.get_subset(split, transform=tf))
        if is_train and train_subset:
            g = torch.Generator().manual_seed(seed)
            idx = torch.randperm(len(ds), generator=g)[:train_subset].tolist()
            ds = Subset(ds, idx)
        nw = num_workers if is_train else eval_workers
        loaders[split] = DataLoader(ds, batch_size=batch_size if is_train else 512,
                                    shuffle=is_train, drop_last=is_train,
                                    num_workers=nw, pin_memory=True,
                                    persistent_workers=(split in ("train", "id_val") and nw > 0))
    return loaders
