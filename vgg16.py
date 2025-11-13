import torch
from torchvision.models import vgg16
from torch import nn

from tool import get_dataloaders, train

def main():
    net = vgg16()
    net.name = 'vgg16'
    net.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    net.classifier = nn.Sequential(
    nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
    )

    train_loader, val_loader = get_dataloaders()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)

    train(net, train_loader, val_loader, optimizer, './checkpoints/vgg16', scheduler=scheduler, notes='UseMultiStepLR')

if __name__ == "__main__":
    main()