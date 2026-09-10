"""Telecharge CIFAR-100-C (Hendrycks & Dietterich, ~2.9 Go) depuis Zenodo dans data/CIFAR-100-C/."""
import sys
import tarfile
import urllib.request
from pathlib import Path

URL = "https://zenodo.org/record/3555552/files/CIFAR-100-C.tar?download=1"
DATA = Path(__file__).resolve().parent.parent / "data"
TAR = DATA / "CIFAR-100-C.tar"


def progress(count, block, total):
    done = count * block
    pct = 100 * done / total if total > 0 else 0
    sys.stdout.write(f"\r{done/1e9:.2f} / {total/1e9:.2f} Go ({pct:.1f} %)")
    sys.stdout.flush()


def main():
    DATA.mkdir(exist_ok=True)
    if (DATA / "CIFAR-100-C" / "labels.npy").exists():
        print("Deja present :", DATA / "CIFAR-100-C")
        return
    if not TAR.exists():
        print("Telechargement", URL)
        urllib.request.urlretrieve(URL, TAR, reporthook=progress)
        print()
    print("Extraction...")
    with tarfile.open(TAR) as t:
        t.extractall(DATA)
    TAR.unlink()
    print("OK :", DATA / "CIFAR-100-C")


if __name__ == "__main__":
    main()
