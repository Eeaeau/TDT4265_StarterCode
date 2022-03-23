import torch
import numpy as np
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler
from task2 import Trainer, print_accuracy, create_plots

mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)


def load_cifar10(batch_size, validation_fraction=0.1):
    # Note that transform train will apply the same transform for
    # validation!
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    data_train = datasets.CIFAR10('data/cifar10',
                                  train=True,
                                  download=True,
                                  transform=transform_train)

    data_test = datasets.CIFAR10('data/cifar10',
                                 train=False,
                                 download=True,
                                 transform=transform_test)

    indices = list(range(len(data_train)))
    split_idx = int(np.floor(validation_fraction * len(data_train)))

    val_indices = np.random.choice(indices, size=split_idx, replace=False)
    train_indices = list(set(indices) - set(val_indices))

    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(val_indices)

    dataloader_train = torch.utils.data.DataLoader(data_train,
                                                   sampler=train_sampler,
                                                   batch_size=batch_size,
                                                   num_workers=2)

    dataloader_val = torch.utils.data.DataLoader(data_train,
                                                 sampler=validation_sampler,
                                                 batch_size=batch_size,
                                                 num_workers=2)

    dataloader_test = torch.utils.data.DataLoader(data_test,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=2)

    return dataloader_train, dataloader_test, dataloader_val


def apply_weight_init(module):
    for m in module.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)


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

        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.LeakyReLU(.2),
            torch.nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.LeakyReLU(.2),
            torch.nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.LeakyReLU(.2),
            torch.nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.MaxPool2d(2),
            nn.LeakyReLU(.2),
            torch.nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(.2),
            torch.nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.MaxPool2d(2),
            nn.LeakyReLU(.2),
            torch.nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.LeakyReLU(.2),
            torch.nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.MaxPool2d(2),
            nn.LeakyReLU(.2),
            torch.nn.BatchNorm2d(128),
        )

        # The output of feature_extractor will be [batch_size, num_filters, 16, 16]
        self.num_output_features = 32*32*32
        self.num_output_features = 128*2*2 #sol
        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss
        self.classifier = nn.Sequential(
            nn.Linear(self.num_output_features, num_classes),
        )
        apply_weight_init(self)

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        # Run image through convolutional layers

        x = self.feature_extractor(x)
        # Reshape our input to (batch_size, num_output_features)
        x = x.view(-1, self.num_output_features)
        # Forward pass through the fully-connected layers.
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    epochs = 10
    batch_size = 64
    learning_rate = 1e-3
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
    create_plots(trainer, "task3")
    print_accuracy(trainer)
