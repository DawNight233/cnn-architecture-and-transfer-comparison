from typing import Tuple, Any
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2023, 0.1994, 0.2010)

def get_dataloaders(batch_size: int = 256, num_workers: int = 8, normalize_imagenet: bool = False) -> Tuple[DataLoader, DataLoader]:
    if normalize_imagenet:
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    else:
        mean, std = CIFAR_MEAN, CIFAR_STD

    train_trans = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_trans)
    val_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=val_trans)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers>0)
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers>0)
    )
    return train_loader, val_loader

def accuracy(output: torch.Tensor, label: torch.Tensor):
    with torch.no_grad():
        preds = output.argmax(dim=1)
        correct = (preds == label).sum()
        acc = correct.float() / label.size(0)
        return acc # 返回Tensor

@torch.no_grad()
def evaluate(
        net: nn.Module,
        val_loader: DataLoader,
        device="cuda"
):
    net.eval()
    running_loss = 0.0
    running_acc = 0.0
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        acc = accuracy(outputs, labels)
        running_loss += loss.item()
        running_acc += acc.item()
    val_loss = running_loss / len(val_loader)
    val_acc = running_acc / len(val_loader)
    return val_loss, val_acc

def train_epoch(
        net: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device="cuda"
):
    net.train()
    running_loss = 0.0
    running_acc = 0.0
    for i, (features, labels) in enumerate(train_loader):
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = net(features)
        loss = nn.CrossEntropyLoss()(output, labels)
        loss.backward()
        optimizer.step()
        acc = accuracy(output, labels)
        running_acc += acc.item()
        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)
    train_acc = running_acc / len(train_loader)
    return train_loss, train_acc

def save_checkpoint(
        path: str,
        net: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        best_loss: float,
        scheduler: Any = None
):
    checkpoint = {
        "net": net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_loss": best_loss
    }
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()
    torch.save(checkpoint, path)

def load_checkpoint(
        path: str,
        net: nn.Module,
        optimizer: torch.optim.Optimizer,
        map_location="cpu",
        scheduler: Any = None
):
    ckpt = torch.load(path, map_location=map_location)
    net.load_state_dict(ckpt["net"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["epoch"], ckpt["best_loss"]

def train(
        net: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
        ckpt_dir,
        notes: str,
        scheduler: Any = None,
        ckpt_every_epochs=5,
        device="cuda"
):
    net.to(device)
    best_loss = float('inf')
    epoch = 0
    run_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(log_dir=f"runs/{net.name}/{notes}_{run_time}")
    try:
        for epoch in range(num_epochs):
            train_loss, train_acc = train_epoch(net, train_loader, optimizer, device)
            val_loss, val_acc = evaluate(net, val_loader, device)

            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Accuracy/train", train_acc, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)

            current_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("LR", current_lr, epoch)

            print(f'Epoch{epoch},train_loss={train_loss:.4f},train_acc={train_acc:.2f},val_loss={val_loss:.4f},val_acc={val_acc:.2f}')

            if epoch % ckpt_every_epochs == 0:
                save_checkpoint(f"{ckpt_dir}/{net.name}_epoch{epoch}.ckpt", net, optimizer, epoch, best_loss, scheduler)
            if best_loss > val_loss:
                best_loss = val_loss
                save_checkpoint(f"{ckpt_dir}/{net.name}_{notes}_{run_time}best.ckpt", net, optimizer, epoch, best_loss, scheduler)
                print(f"  -> New BEST saved ({best_loss:.4f})")

            if scheduler is not None:
                scheduler.step()

    except KeyboardInterrupt:
        print("\nSaving LAST checkpoint...")
        save_checkpoint(f'{ckpt_dir}/{net.name}_last.ckpt', net, optimizer, epoch, best_loss, scheduler)
        print(f"Stopped at epoch {epoch}. last checkpoint saved.")
