"""iWildCam 2020 (WILDS) : photos de pieges photographiques, 182 especes, decalage = cameras
jamais vues (lieu, vegetation, jour / nuit, capteur).

Splits WILDS officiels :
  train    : 129 k images, 243 cameras
  id_val / id_test : memes cameras, images non vues
  val      : 14 k images, 32 cameras jamais vues  (OOD)
  test     : 42 k images, 48 cameras jamais vues  (OOD)   <- le decalage reel

Metrique officielle WILDS : macro-F1 (classes tres desequilibrees, 'vide' = ~30 %). On rapporte
accuracy, macro-F1, ECE. Multi-classes : le terme non-cible de DKD existe ici, contrairement
a Camelyon17 (binaire).
"""
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from wilds import get_dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
SPLITS = ["train", "id_val", "val", "test"]
N_CLASSES = 182


def train_transform(size: int, strong: bool = False):
    ops = [transforms.Resize((size, size)), transforms.RandomHorizontalFlip()]
    if strong:   # analogue d'AugMix pour un decalage de camera : couleur, eclairage, nettete
        ops += [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1), transforms.RandomGrayscale(0.2),
                transforms.RandomApply([transforms.GaussianBlur(5)], p=0.3)]
    ops += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(ops)


def test_transform(size: int):
    return transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                               transforms.Normalize(MEAN, STD)])


class _Wrap(torch.utils.data.Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x, y, _ = self.ds[i]
        return x, int(y)


def iwildcam_loaders(batch_size: int, size: int = 224, strong_aug: bool = False,
                     train_subset: int | None = None, seed: int = 0, num_workers: int = 4,
                     splits=SPLITS, eval_workers: int = 3):
    """dict split -> DataLoader. Memes regles memoire que camelyon.py : workers persistants pour
    train et id_val seulement. Les JPEG sont plus lourds a decoder que les patches 96x96 :
    rester a 4 workers d'entrainement."""
    full = get_dataset(dataset="iwildcam", download=False, root_dir=str(DATA_DIR))
    loaders = {}
    for split in splits:
        is_train = split == "train"
        tf = train_transform(size, strong_aug) if is_train else test_transform(size)
        ds = _Wrap(full.get_subset(split, transform=tf))
        if is_train and train_subset:
            g = torch.Generator().manual_seed(seed)
            idx = torch.randperm(len(ds), generator=g)[:train_subset].tolist()
            ds = Subset(ds, idx)
        nw = num_workers if is_train else eval_workers
        loaders[split] = DataLoader(ds, batch_size=batch_size if is_train else 256,
                                    shuffle=is_train, drop_last=is_train,
                                    num_workers=nw, pin_memory=True,
                                    persistent_workers=(split in ("train", "id_val") and nw > 0))
    return loaders
