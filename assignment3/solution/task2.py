import pathlib
import matplotlib.pyplot as plt
import utils
from torch import nn
from dataloaders import load_cifar10
from trainer import Trainer
from trainer import compute_loss_and_accuracy #sol


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
        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=num_filters,
                kernel_size=5,
                stride=1,
                padding=2
            )
        )
        self.feature_extractor = nn.Sequential( #sol
            nn.Conv2d(3, 32, 5, padding=2), #sol
            nn.MaxPool2d(2),#sol
            nn.ReLU(),#sol
            nn.Conv2d(32, 64, 5, padding=2),#sol
            nn.MaxPool2d(2),#sol
            nn.ReLU(),#sol
            nn.Conv2d(64, 128, 5, padding=2),#sol
            nn.MaxPool2d(2),#sol
            nn.ReLU()#sol
        )#sol
        # The output of feature_extractor will be [batch_size, num_filters, 16, 16]
        self.num_output_features = 32*32*32
        self.num_output_features = 128*4*4 #sol
        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss
        self.classifier = nn.Sequential(
            nn.Linear(self.num_output_features, num_classes),
        )
        self.classifier = nn.Sequential(#sol
            nn.Linear(self.num_output_features, 64),#sol
            nn.ReLU(),#sol
            nn.Linear(64, num_classes),#sol
        )#sol

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        # TODO: Implement this function (Task  2a)
        # Run image through convolutional layers#sol
        batch_size = x.shape[0]
        x = self.feature_extractor(x)#sol
        # Reshape our input to (batch_size, num_output_features)#sol
        x = x.view(-1, self.num_output_features)#sol
        # Forward pass through the fully-connected layers.#sol
        out = x
        out = self.classifier(x)#sol
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out


def create_plots(trainer: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    latex_figures = pathlib.Path("../latex/figures") #sol
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer.train_history["loss"], label="Training loss", npoints_to_average=10)
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer.validation_history["accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.savefig(latex_figures.joinpath(f"{name}_plot.png")) #sol
    plt.show()
#sol
#sol
def print_accuracy(trainer: Trainer): #sol
    datasets = { #sol
        "train": trainer.dataloader_train,#sol
        "test": trainer.dataloader_test,#sol
        "val": trainer.dataloader_val#sol
    }#sol
    trainer.load_best_model() #sol
    for dset, dl in datasets.items():#sol 
        avg_loss, accuracy = compute_loss_and_accuracy(dl, trainer.model, trainer.loss_criterion)#sol
        print(#sol
            f"Dataset: {dset}, Accuracy: {accuracy}, loss: {avg_loss}"#sol
        )#sol


def main():
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result! 
    utils.set_seed(0)
    epochs = 10
    batch_size = 64
    learning_rate = 5e-2
    early_stop_count = 4
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
    trainer.train()
    create_plots(trainer, "task2")
    print_accuracy(trainer) #sol

if __name__ == "__main__":
    main()