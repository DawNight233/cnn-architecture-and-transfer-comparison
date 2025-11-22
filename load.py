import torch

def get_best_epoch_and_loss_from_checkpoint(path: str):
    ckpt = torch.load(path)
    return ckpt["epoch"], ckpt["best_loss"]

def main():
    vgg16 = "./best/vgg16_SGD_lr=0.1_momentum=0.9_weight_decay=5e-4_MultiStepLR_50,100_LabelSmoothing=0.1_2025-11-18_23-23-19best.ckpt"
    resnet18 = "./best/resnet18_SGD_lr=0.1_momentum=0.9_weight_decay=5e-4_MultiStepLR_100,150_LabelSmoothing=0.1_2025-11-18_22-44-44best.ckpt"
    resnet50 = "./best/resnet50_SGD_lr=0.1_momentum=0.9_weight_decay=5e-4_CosineAnnealingLR_Tmax=300_LabelSmoothing=0.1_2025-11-21_15-35-37best.ckpt"
    resnet50_pretrained = "./best/resnet50-pretrained_train=l3+l4+fc_SGD_lr=0.01+0.1_momentum=0.9_weight_decay=5e-4_2025-11-21_20-56-21_best.ckpt"

    vgg16_epoch, vgg16_loss = get_best_epoch_and_loss_from_checkpoint(vgg16)
    resnet18_epoch, resnet18_loss = get_best_epoch_and_loss_from_checkpoint(resnet18)
    resnet50_epoch, resnet50_loss = get_best_epoch_and_loss_from_checkpoint(resnet50)
    resnet50_pretrained_epoch, resnet50_pretrained_loss = get_best_epoch_and_loss_from_checkpoint(resnet50_pretrained)

    print(f"VGG16 Epoch: {vgg16_epoch}, Loss: {vgg16_loss}")
    print(f"ResNet18 Epoch: {resnet18_epoch}, Loss: {resnet18_loss}")
    print(f"ResNet50 Epoch: {resnet50_epoch}, Loss: {resnet50_loss}")
    print(f"ResNet50-pretrained Epoch: {resnet50_pretrained_epoch}, Loss: {resnet50_pretrained_loss}")


if __name__ == "__main__":
    main()