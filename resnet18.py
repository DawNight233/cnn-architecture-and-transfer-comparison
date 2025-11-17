import torch
from torchvision.models import resnet18
from torch import nn

from tool import get_dataloaders, train

def main():
    net = resnet18()
    net.name = 'resnet18'
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    net.fc = nn.Linear(512, 10)

    train_loader, val_loader = get_dataloaders()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

    train(net, train_loader, val_loader, optimizer, 200, './checkpoints/resnet18', scheduler=scheduler, notes='SGD_lr=0.1_momentum=0.9_weight_decay=5e-4_MultiStepLR_100,150')

if __name__ == "__main__":
    main()