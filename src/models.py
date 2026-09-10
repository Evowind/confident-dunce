"""Architectures CIFAR classiques de la litterature KD (CRD, DKD) :
ResNet-8/20/32/56/110 (et variantes x4) et WideResNet-d-k.
forward(x, return_feat=True) renvoie aussi la representation penultieme (pour CRD en phase 2)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------- ResNet CIFAR
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNetCIFAR(nn.Module):
    def __init__(self, depth, num_classes=100, width=16):
        super().__init__()
        assert (depth - 2) % 6 == 0, "depth doit etre 6n+2"
        n = (depth - 2) // 6
        self.in_planes = width
        self.conv1 = nn.Conv2d(3, width, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.layer1 = self._make(width, n, 1)
        self.layer2 = self._make(width * 2, n, 2)
        self.layer3 = self._make(width * 4, n, 2)
        self.fc = nn.Linear(width * 4, num_classes)
        self.feat_dim = width * 4
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def _make(self, planes, n, stride):
        layers = []
        for s in [stride] + [1] * (n - 1):
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x, return_feat=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer3(self.layer2(self.layer1(out)))
        feat = F.adaptive_avg_pool2d(out, 1).flatten(1)
        logits = self.fc(feat)
        return (logits, feat) if return_feat else logits


# ------------------------------------------------------------------- WideResNet
class WideBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, 1, 1, bias=False)
        self.drop = drop
        self.equal = in_planes == out_planes
        self.shortcut = None if self.equal else nn.Conv2d(in_planes, out_planes, 1, stride, bias=False)

    def forward(self, x):
        o = F.relu(self.bn1(x))
        if not self.equal:
            x = o
        out = self.conv1(o)
        out = F.relu(self.bn2(out))
        if self.drop > 0:
            out = F.dropout(out, self.drop, self.training)
        out = self.conv2(out)
        return out + (x if self.equal else self.shortcut(x))


class WideResNet(nn.Module):
    def __init__(self, depth, widen, num_classes=100, drop=0.0):
        super().__init__()
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        w = [16, 16 * widen, 32 * widen, 64 * widen]
        self.conv1 = nn.Conv2d(3, w[0], 3, 1, 1, bias=False)
        self.block1 = self._make(w[0], w[1], n, 1, drop)
        self.block2 = self._make(w[1], w[2], n, 2, drop)
        self.block3 = self._make(w[2], w[3], n, 2, drop)
        self.bn = nn.BatchNorm2d(w[3])
        self.fc = nn.Linear(w[3], num_classes)
        self.feat_dim = w[3]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    @staticmethod
    def _make(i, o, n, stride, drop):
        return nn.Sequential(*[WideBlock(i if k == 0 else o, o, stride if k == 0 else 1, drop)
                               for k in range(n)])

    def forward(self, x, return_feat=False):
        out = self.block3(self.block2(self.block1(self.conv1(x))))
        out = F.relu(self.bn(out))
        feat = F.adaptive_avg_pool2d(out, 1).flatten(1)
        logits = self.fc(feat)
        return (logits, feat) if return_feat else logits


# --------------------------------------------------------------------- registre
def build_model(name: str, num_classes: int = 100) -> nn.Module:
    """resnet20, resnet56, resnet8x4 (x4 = 4 fois plus de canaux), wrn_40_2, wrn_16_2..."""
    name = name.lower()
    if name.startswith("resnet"):
        body = name[len("resnet"):]
        if "x" in body:
            depth, mult = body.split("x")
            return ResNetCIFAR(int(depth), num_classes, width=16 * int(mult))
        return ResNetCIFAR(int(body), num_classes)
    if name.startswith("wrn_"):
        _, depth, widen = name.split("_")
        return WideResNet(int(depth), int(widen), num_classes)
    raise ValueError(f"Modele inconnu : {name}")


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def load_checkpoint(path, device="cuda"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["model"], ckpt.get("num_classes", 100))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).to(memory_format=torch.channels_last).eval()
