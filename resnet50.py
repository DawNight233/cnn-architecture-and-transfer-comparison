import torch
from torchvision.models import resnet50
from torch import nn

from tool import get_dataloaders, train

def main():
    net = resnet50()
    net.name = 'resnet50'
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    net.fc = nn.Linear(2048, 10)

    train_loader, val_loader = get_dataloaders()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

    train(net, train_loader, val_loader, optimizer, 0, 300, './checkpoints/resnet50', scheduler=scheduler,
          notes='SGD_lr=0.1_momentum=0.9_weight_decay=5e-4_CosineAnnealingLR_Tmax=300_LabelSmoothing=0.1')

if __name__ == "__main__":
    main()