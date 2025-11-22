import os
from typing import Dict

import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from torch import nn
from torchvision.models import vgg16, resnet18, resnet50

from tool import get_dataloaders, load_checkpoint, get_device


# CIFAR-10 类别名称（横纵坐标标签）
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def build_model(model_name: str) -> nn.Module:
    if model_name == "vgg16":
        net = vgg16()
        net.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        net.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )
        return net

    if model_name == "resnet18":
        net = resnet18()
        net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        net.maxpool = nn.Identity()
        net.fc = nn.Linear(512, 10)
        return net

    if model_name == "resnet50":
        net = resnet50()
        net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        net.maxpool = nn.Identity()
        net.fc = nn.Linear(2048, 10)
        return net

    if model_name == "resnet50-pretrained":
        net = resnet50()
        net.fc = nn.Linear(2048, 10)
        return net

    raise ValueError(f"Unknown model name: {model_name}")


@torch.no_grad()
def confusion_matrix(
    net: nn.Module,
    val_loader,
    device: str = "cuda",
    num_classes: int = 10
) -> torch.Tensor:
    net.eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = net(images)
        preds = outputs.argmax(dim=1)

        for t, p in zip(labels.view(-1), preds.view(-1)):
            cm[int(t), int(p)] += 1

    return cm


def plot_and_save_cm(
    cm: torch.Tensor,
    model_name: str,
    classes=CIFAR10_CLASSES,
    save_dir: str = "./figs"
):
    os.makedirs(save_dir, exist_ok=True)

    cm_np = cm.numpy()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_np, cmap="Blues", norm=LogNorm())
    plt.colorbar(im, ax=ax)

    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    tick_marks = range(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)

    # 在格子里写上数字
    thresh = cm_np.max() / 2.0
    for i in range(cm_np.shape[0]):
        for j in range(cm_np.shape[1]):
            ax.text(
                j, i, int(cm_np[i, j]),
                ha="center", va="center",
                color="white" if cm_np[i, j] > thresh else "black",
                fontsize=8,
            )

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"confusion_{model_name}.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"[{model_name}] Confusion matrix saved to: {save_path}")


def main():
    device = get_device()
    print("Using device:", device)

    # 四个 best 模型的 ckpt 路径 & dataloader 配置
    model_info: Dict[str, Dict] = {
        "vgg16": {
            "ckpt": "./best/vgg16_SGD_lr=0.1_momentum=0.9_weight_decay=5e-4_MultiStepLR_50,100_LabelSmoothing=0.1_2025-11-18_23-23-19best.ckpt",
            "normalize_imagenet": False,
            "resize_to_224": False,
        },
        "resnet18": {
            "ckpt": "./best/resnet18_SGD_lr=0.1_momentum=0.9_weight_decay=5e-4_MultiStepLR_100,150_LabelSmoothing=0.1_2025-11-18_22-44-44best.ckpt",
            "normalize_imagenet": False,
            "resize_to_224": False,
        },
        "resnet50": {
            "ckpt": "./best/resnet50_SGD_lr=0.1_momentum=0.9_weight_decay=5e-4_CosineAnnealingLR_Tmax=300_LabelSmoothing=0.1_2025-11-21_15-35-37best.ckpt",
            "normalize_imagenet": False,
            "resize_to_224": False,
        },
        "resnet50-pretrained": {
            "ckpt": "./best/resnet50-pretrained_train=l3+l4+fc_SGD_lr=0.01+0.1_momentum=0.9_weight_decay=5e-4_2025-11-21_20-56-21_best.ckpt",
            "normalize_imagenet": True,
            "resize_to_224": True,
        },
    }

    for model_name, info in model_info.items():
        print(f"\n=== Processing {model_name} ===")
        ckpt_path = info["ckpt"]

        if not os.path.exists(ckpt_path):
            print(f"  [WARN] ckpt not found: {ckpt_path}")
            continue

        net = build_model(model_name)
        net.to(device)

        load_checkpoint(
            ckpt_path,
            net,
            optimizer=None,
            map_location=device,
            scheduler=None,
        )
        print("  checkpoint loaded.")

        _, val_loader = get_dataloaders(
            normalize_imagenet=info["normalize_imagenet"],
            resize_to_224=info["resize_to_224"],
        )

        cm = confusion_matrix(net, val_loader, device, num_classes=10)
        print("  confusion matrix:\n", cm)

        plot_and_save_cm(cm, model_name)


if __name__ == "__main__":
    main()
