import torch
from torchvision.models import resnet50, ResNet50_Weights
from torch import nn

from tool import get_dataloaders, train, load_checkpoint, get_device

def main():
    net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    net.name = 'resnet50-pretrained'
    net.fc = nn.Linear(2048, 10)
    for param in net.conv1.parameters():
        param.requires_grad = False
    for param in net.bn1.parameters():
        param.requires_grad = False
    for param in net.layer1.parameters():
        param.requires_grad = False
    for param in net.layer2.parameters():
        param.requires_grad = False

    params_backbone = list(net.layer3.parameters()) + list(net.layer4.parameters())
    params_fc = list(net.fc.parameters())

    device = get_device()
    net.to(device)

    train_loader, val_loader = get_dataloaders(normalize_imagenet=True, resize_to_224=True)
    optimizer = torch.optim.SGD(
        [
            {"params": params_backbone, "lr": 0.01},
            {"params": params_fc, "lr": 0.1},
        ],
        momentum=0.9,
        weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[100, 150],
        gamma=0.1,
    )
    epoch = 0
    load_path = "./checkpoints/resnet50-pretrained/resnet50-pretrained_last.ckpt"
    if load_path:
        epoch, _ = load_checkpoint(
            load_path,
            net,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device
        )

    train(net, train_loader, val_loader, optimizer, epoch, 200, './checkpoints/resnet50-pretrained', scheduler=scheduler,
          notes='train=l3+l4+fc_SGD_lr=0.01+0.1_momentum=0.9_weight_decay=5e-4',
          device=device,
    )


if __name__ == "__main__":
    main()