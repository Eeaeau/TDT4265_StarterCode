import torch
import torchvision
from torch import nn
from task3 import load_cifar10
from resnet_dl import load_cifar10 
from task2 import Trainer, create_plots, print_accuracy


class ExampleModel(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = nn.Sequential(
            nn.Linear(512, 10),
        )
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True
        for param in self.model.layer4.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == "__main__":
    epochs = 10
    batch_size = 32
    learning_rate = 5e-4
    early_stop_count = 5
    dataloaders = load_cifar10(batch_size)
    model = ExampleModel(image_channels=3, num_classes=10)
    trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders
    )
    trainer.optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    trainer.train()
    create_plots(trainer, "task4a")
    print_accuracy(trainer)
