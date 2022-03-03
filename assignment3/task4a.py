import torch
import torchvision
from torch import nn
import utils
from dataloaders import load_cifar10, load_cifar10_aug_ResNet,load_cifar10_aug2, load_cifar10_aug3, load_cifar10_aug4, load_cifar10_aug5, load_cifar10_aug6, load_cifar10_aug7, load_cifar10_aug8
from task3 import create_plots
from trainer import ResNetTrainer

class ResNetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, 10) # No need to apply softmax,
        # as this is done in nn.CrossEntropyLoss
        for param in self.model.parameters(): # Freeze all parameters
            param.requires_grad = False
        for param in self.model.fc.parameters(): # Unfreeze the last fully-connected
            param.requires_grad = True # layer
        for param in self.model.layer4.parameters(): # Unfreeze the last 5 convolutional
            param.requires_grad = True # layers
        self.batch_size = 32

    def forward(self, x):
        x = self.model(x)
        return x


def main():
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result!

    utils.set_seed(0)
    epochs = 25
    batch_size = 32
    learning_rate = 5e-4
    #learning_rate = 3e-4
    early_stop_count = 10
    dataloaders = load_cifar10_aug_ResNet(batch_size)
    model = ResNetModel()


    trainer = ResNetTrainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders
    )

    trainer.train()

    create_plots(trainer, "task4a_sgd")

if __name__ == "__main__":
    main()
