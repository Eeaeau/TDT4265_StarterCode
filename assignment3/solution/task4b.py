import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import torch
import numpy as np
image = Image.open("images/zebra.jpg")
print("Image shape:", image.size)

model = torchvision.models.resnet18(pretrained=True)
print(model)
first_conv_layer = model.conv1
print("First conv layer weight shape:", first_conv_layer.weight.shape)
print("First conv layer:", first_conv_layer)

# Resize, and normalize the image with the mean and standard deviation
image_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image = image_transform(image)[None]
print("Image shape:", image.shape)

activation = first_conv_layer(image)
print("Activation shape:", activation.shape)


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


def plot_filters(indices): #sol
    weight_data = model.conv1.weight #sol
    num_filters = len(indices)#sol
    n = 1#sol
    for i in indices:#sol
        plt.subplot(2, num_filters, n)#sol
        weight = weight_data[i].squeeze()#sol
        weight = torch_image_to_numpy(weight)#sol
        plt.imshow(weight)#sol
        activation = first_conv_layer(image)[0, i]#sol
        activation = torch_image_to_numpy(activation)#sol
        plt.subplot(2, num_filters, num_filters+n)#sol
        plt.imshow(activation, cmap="gray")#sol
        n += 1#sol
#sol
indices = [0, 1, 2, 3, 14] #sol
plt.figure(figsize=(20, 4)) #sol
plot_filters(indices)#sol
plt.savefig("../latex/figures/filters_first_layer_example.png") #sol
#plt.show()#sol
#sol
indices = [14, 26, 32, 49, 52]
plt.figure(figsize=(20, 4)) #sol
plot_filters(indices) #sol
plt.savefig("../latex/figures/filters_first_layer_solution.png") #sol
plt.show() #sol
#sol
#sol
def get_last_layer_act(model, image, indices):#sol
    x = image#sol
    for block in list(model.children())[:-2]:#sol
        print(block)#sol
        x = block(x)#sol
    act = x#sol
    img = np.zeros((act.shape[-1], act.shape[-1]* len(indices)))#sol
    n = 0#sol
    for i in indices:#sol
        a = act[0, i]#sol
        a = torch_image_to_numpy(a)#sol
        img[:, act.shape[-1]*n:act.shape[-1]*(n+1)] = a#sol
        n += 1#sol
    return img#sol
#sol
indices = list(range(10, 20)) #sol
plt.imsave("../latex/figures/last_layer_example.png", get_last_layer_act(model, image, indices), cmap="gray") #sol
#sol
indices = list(range(10))#sol
plt.imsave("../latex/figures/last_layer_solution.png", get_last_layer_act(model, image, indices), cmap="gray")#sol
