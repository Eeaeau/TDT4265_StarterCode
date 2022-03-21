# Inherit configs from the default ssd300
import torchvision
from ssd.data import TDT4265Dataset
from tops.config import LazyCall as L
from ssd.data.transforms import (
    ToTensor, Normalize, Resize,
    GroundTruthBoxesToAnchors)
from .ssd300 import train, anchors, optimizer, schedulers, backbone, model, data_train, data_val, loss_objective
from .utils import get_dataset_dir

# Keep the model, except change the backbone and number of classes
train.imshape = (128, 1024)
train.image_channels = 3
model.num_classes = 8 + 1  # Add 1 for background class


train_cpu_transform = L(torchvision.transforms.Compose)(transforms=[
    L(ToTensor)(),
    L(Resize)(imshape="${train.imshape}"),
    L(GroundTruthBoxesToAnchors)(anchors="${anchors}", iou_threshold=0.5),
])
val_cpu_transform = L(torchvision.transforms.Compose)(transforms=[
    L(ToTensor)(),
    L(Resize)(imshape="${train.imshape}"),
])
gpu_transform = L(torchvision.transforms.Compose)(transforms=[
    L(Normalize)(mean=[0.4765, 0.4774, 0.2259], std=[0.2951, 0.2864, 0.2878])
])
data_train.dataset = L(TDT4265Dataset)(
    img_folder=get_dataset_dir("tdt4265_2022"),
    transform="${train_cpu_transform}",
    annotation_file=get_dataset_dir("tdt4265_2022/train_annotations.json"))
data_val.dataset = L(TDT4265Dataset)(
    img_folder=get_dataset_dir("tdt4265_2022"),
    transform="${val_cpu_transform}",
    annotation_file=get_dataset_dir("tdt4265_2022/val_annotations.json"))
data_val.gpu_transform = gpu_transform
data_train.gpu_transform = gpu_transform

label_map = {idx: cls_name for idx, cls_name in enumerate(TDT4265Dataset.class_names)}
