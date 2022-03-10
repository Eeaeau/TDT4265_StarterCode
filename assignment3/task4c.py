from distutils.command.build_scripts import first_line_re
import enum
import imp
from operator import mod
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import torch
import numpy as np
import pathlib
import utils

def torch_image_to_numpy(image: torch.Tensor):
    """
    Function to transform a pytorch tensor to numpy image
    Args:
        image: shape=[3, height, width]
    Returns:
        iamge: shape=[height, width, 3] in the range [0, 1]
    """
    # Normalize to [0 - 1.0]
    image = image.detach().cpu() # Transform image to CPU memory (if on GPU VRAM)
    image = image - image.min()
    image = image / image.max()
    image = image.numpy()
    if len(image.shape) == 2: # Grayscale image, can just return
        return image
    assert image.shape[0] == 3, "Expected color channel to be on first axis. Got: {}".format(image.shape)
    image = np.moveaxis(image, 0, 2)
    return image

def layer_plotter(activations, indices, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 5))

    n_layers = len(indices)

    plt.title("Activations")

    for n, i in enumerate(indices):

        plt.subplot(2, n_layers//2, n+1)

        im =activations[i].numpy()
        plt.imshow(im)


    plt.savefig(plot_path.joinpath(f"{name}_plot.eps"))
    plt.show()


def plot_hook(model, model_in, model_out):

    indices = range(10)
    activations_extracted = model_out[0].detach().cpu()
    print("Last conv layer shape:", activations_extracted.shape)

    layer_plotter(activations_extracted, indices, "task4c_plot")

def main():
    image = Image.open("images/zebra.jpg")
    print("Image shape:", image.size)

    model = torchvision.models.resnet18(pretrained=True)
    print(model)

    # Resize, and normalize the image with the mean and standard deviation
    image_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = image_transform(image)[None]
    print("Image shape:", image.shape)

    last_conv_layer = model.layer4[1].conv2
    # execute after forwarding updates conv2 in last layer
    last_conv_layer.register_forward_hook(plot_hook)

    model(image)

if __name__ == "__main__":
    main()
