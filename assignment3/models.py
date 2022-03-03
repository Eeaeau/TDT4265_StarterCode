import pathlib
import matplotlib.pyplot as plt
import utils
import torch
from torch import nn
from dataloaders import load_cifar10, load_cifar10_aug1,load_cifar10_aug2, load_cifar10_aug3, load_cifar10_aug4, load_cifar10_aug5, load_cifar10_aug6, load_cifar10_aug7, load_cifar10_aug8
from trainer import Trainer

class modelv2(nn.Module):

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
                out_channels=64,
                kernel_size=filter_size,
                stride=1,
                padding=pad
            ),
            nn.ReLU(),
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

class modelv3(nn.Module):

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
                out_channels=256,
                kernel_size=filter_size,
                stride=1,
                padding=pad
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
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


class modelv4(nn.Module):

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
                out_channels=256,
                kernel_size=filter_size,
                stride=1,
                padding=pad
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
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


class modelv5(nn.Module):

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
            nn.Dropout(0.2),
            nn.MaxPool2d([2,2], stride=2),
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=64,
                kernel_size=filter_size,
                stride=1,
                padding=pad
            ),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d([2,2], stride=2),
            nn.Conv2d(
                in_channels=64,
                out_channels=256,
                kernel_size=filter_size,
                stride=1,
                padding=pad
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
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

class modelv6(nn.Module):

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
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d([2,2], stride=2),
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=64,
                kernel_size=filter_size,
                stride=1,
                padding=pad
            ),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.2),
            nn.MaxPool2d([2,2], stride=2),
            nn.Conv2d(
                in_channels=64,
                out_channels=256,
                kernel_size=filter_size,
                stride=1,
                padding=pad
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=128,
                kernel_size=filter_size,
                stride=1,
                padding=pad
            ),
            nn.BatchNorm2d(128),
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
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
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