"""UNet with an ImageNet-pretrained ResNet18 encoder (binary lesion segmentation, 1 logit channel)."""
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class _Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResNetUNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        enc = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu)          # 1/2, 64
        self.pool = enc.maxpool                                          # 1/4
        self.l1, self.l2, self.l3, self.l4 = enc.layer1, enc.layer2, enc.layer3, enc.layer4  # 64,128,256,512
        self.up4 = _Up(512, 256, 256)
        self.up3 = _Up(256, 128, 128)
        self.up2 = _Up(128, 64, 64)
        self.up1 = _Up(64, 64, 32)
        self.up0 = _Up(32, 0, 16)
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.l1(self.pool(s0))
        s2 = self.l2(s1)
        s3 = self.l3(s2)
        s4 = self.l4(s3)
        d = self.up4(s4, s3)
        d = self.up3(d, s2)
        d = self.up2(d, s1)
        d = self.up1(d, s0)
        d = self.up0(d)
        return self.head(d)


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, eps=1e-6) -> torch.Tensor:
    """Per-image Dice of binary masks [B, 1, H, W] -> [B]."""
    dims = (1, 2, 3)
    inter = (pred * target).sum(dims)
    return (2 * inter + eps) / (pred.sum(dims) + target.sum(dims) + eps)


def jaccard(pred: torch.Tensor, target: torch.Tensor, eps=1e-6) -> torch.Tensor:
    dims = (1, 2, 3)
    inter = (pred * target).sum(dims)
    union = pred.sum(dims) + target.sum(dims) - inter
    return (inter + eps) / (union + eps)
