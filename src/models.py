"""Model definitions."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, input_size=28):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        # Two 2x2 pools shrink each spatial dim by 4x.
        feat = input_size // 4
        self.fc1 = nn.Linear(64 * feat * feat, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class LeNetSigmoid(nn.Module):
    """The LeNet variant from the Deep Leakage from Gradients paper (Zhu 2019)."""

    def __init__(self, in_channels=1, num_classes=10, input_size=28):
        super().__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, 12, kernel_size=5, padding=2, stride=2), act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=2, stride=2), act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=2, stride=1), act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=2, stride=1), act(),
        )
        with torch.no_grad():
            feat = self.body(torch.zeros(1, in_channels, input_size, input_size)).numel()
        self.fc = nn.Linear(feat, num_classes)

    def forward(self, x):
        x = self.body(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def dlg_init(model, low=-0.5, high=0.5):
    """Uniform weight init used in the DLG paper."""
    for m in model.modules():
        if hasattr(m, "weight") and m.weight is not None:
            m.weight.data.uniform_(low, high)
        if hasattr(m, "bias") and m.bias is not None:
            m.bias.data.uniform_(low, high)
    return model


def build_model(cfg_model):
    """Construct a model from the resolved config's `model` section."""
    name = cfg_model.get("name", "smallcnn").lower()
    kwargs = dict(
        in_channels=cfg_model["in_channels"],
        num_classes=cfg_model["num_classes"],
        input_size=cfg_model["input_size"],
    )
    if name == "smallcnn":
        return SmallCNN(**kwargs)
    if name in ("lenet", "lenetsigmoid", "dlg"):
        return LeNetSigmoid(**kwargs)
    raise ValueError(f"Unknown model: {name}")
