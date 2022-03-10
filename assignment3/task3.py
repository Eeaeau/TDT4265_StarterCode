import pathlib
import matplotlib.pyplot as plt
import utils
import torch
from torch import nn
from dataloaders import load_cifar10, load_cifar10_aug1,load_cifar10_aug2, load_cifar10_aug3, load_cifar10_aug4, load_cifar10_aug5, load_cifar10_aug6, load_cifar10_aug7, load_cifar10_aug8
from trainer import Trainer
from models import modelv2, modelv3, modelv4, modelv5, modelv6



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
        # TODO: Implement this function (Task  2a)
        num_filters = 32  # Set number of filters in first conv layer
        self.num_classes = num_classes
        filter_size = 5
        pad = 2
        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=num_filters,
                kernel_size=filter_size,
                stride=1,
                padding=pad
            ),
            nn.ReLU(),
            nn.MaxPool2d([2,2], stride=2),
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=64,
                kernel_size=filter_size,
                stride=1,
                padding=pad
            ),
            nn.ReLU(),
            nn.MaxPool2d([2,2], stride=2),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=filter_size,
                stride=1,
                padding=pad
            ),
            nn.ReLU(),
            nn.MaxPool2d([2,2], stride=2),
            #nn.Flatten()


        ).to('cuda')
        # The output of feature_extractor will be [batch_size, num_filters, 16, 16]
        self.num_output_features = 128*4*4
        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss
        self.classifier = nn.Sequential(
            nn.Linear(self.num_output_features, 64),
            nn.ReLU(),
            nn.Linear(64,num_classes)
        ).to('cuda')

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        # TODO: Implement this function (Task  2a)
        batch_size = x.shape[0]
        features = self.feature_extractor(x)
        flattened = features.view(-1,self.num_output_features)
        out = self.classifier(flattened)
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out


def create_plots(trainer: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    plt.ylabel("CEL")
    plt.xlabel("Step")
    utils.plot_loss(trainer.train_history["loss"], label="Training loss", npoints_to_average=10)
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    plt.ylabel("Procentage")
    plt.xlabel("Step")
    utils.plot_loss(trainer.validation_history["accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.eps"))
    plt.show()


def main():
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result!

    utils.set_seed(0)
    epochs = 25
    batch_size = 64
    #learning_rate = 5e-2
    learning_rate = 3e-4 #adam
    early_stop_count = 10
    dataloaders = load_cifar10_aug7(batch_size)
    #model = ExampleModel(image_channels=3, num_classes=10)
    model1 = modelv6(image_channels=3, num_classes=10)
    # trainer = Trainer(
    #     batch_size,
    #     learning_rate,
    #     early_stop_count,
    #     epochs,
    #     model,
    #     dataloaders
    # )
    #trainer.train()
    #create_plots(trainer, "task3_aug7_base")
    trainer1 = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model1,
        dataloaders
    )
    trainer1.train()
    create_plots(trainer1, "task3_aug7_modv6_modv4_wdropout_adam_batnorm")

if __name__ == "__main__":
    main()
