# Import everything from the old dataset and only change the dataset folder.
from .tdt4265 import (
    train,
    optimizer,
    schedulers,
    loss_objective,
    model,
    backbone,
    data_train,
    data_val,
    train_cpu_transform,
    val_cpu_transform,
    gpu_transform,
    label_map,
    anchors
)
from ssd.data.transforms import (
    ToTensor, Normalize, Resize,
    GroundTruthBoxesToAnchors, RandomHorizontalFlip, RandomSampleCrop, ColorJitter, GaussianBlurr, AdjustSharpness, TrivialAugmentWide, RandomAffine, RandomAutocontrast, RandomPosterize)
from tops.config import LazyCall as L
import torchvision
from ssd.data import TDT4265Dataset
from .utils import get_dataset_dir


#adding data augmentarion

train_cpu_transform = L(torchvision.transforms.Compose)(transforms=[
    L(RandomSampleCrop)(),
    L(ToTensor)(),
    L(RandomHorizontalFlip)(),
    L(Resize)(imshape="${train.imshape}"),
    L(GroundTruthBoxesToAnchors)(anchors="${anchors}", iou_threshold=0.5),
])

gpu_transform_train = L(torchvision.transforms.Compose)(transforms=[
    # L(TrivialAugmentWide)(),
    #L(RandomAffine)(),
    # L(RandomPosterize)(),
    #L(RandomAutocontrast)(),
    #L(ColorJitter)(),
    L(Normalize)(mean=[0.4765, 0.4774, 0.2259], std=[0.2951, 0.2864, 0.2878]),
])

data_train.dataset = L(TDT4265Dataset)(
    img_folder=get_dataset_dir("tdt4265_2022"),
    transform="${train_cpu_transform}",
    annotation_file=get_dataset_dir("tdt4265_2022/train_annotations.json"))

data_train.gpu_transform = gpu_transform_train
