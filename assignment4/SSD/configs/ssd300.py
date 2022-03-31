import torch
import torchvision
from torch.optim.lr_scheduler import MultiStepLR, LinearLR
from ssd.modeling import SSD300, SSDMultiboxLoss, backbones, AnchorBoxes
from tops.config import LazyCall as L
from ssd.data.mnist import MNISTDetectionDataset
from ssd import utils
from ssd.data.transforms import  Normalize, ToTensor, GroundTruthBoxesToAnchors, ColorJitter, RandomHorizontalFlip, RandomSampleCrop
from ssd.data.mnist import MNISTDetectionDataset
from .utils import get_dataset_dir, get_output_dir


train = dict(
    batch_size=32, #32 org
    amp=True, # Automatic mixed precision
    log_interval=20,
    seed=0,
    epochs=33,
    _output_dir=get_output_dir(),
    imshape=(300, 300),
    image_channels=3
)

anchors = L(AnchorBoxes)(
    feature_sizes=[[38, 38], [19, 19], [10, 10], [5, 5], [3, 3], [1, 1]],
    # Strides is the number of pixels (in image space) between each spatial position in the feature map
    strides=[[8, 8], [16, 16], [32, 32], [64, 64], [100, 100], [300, 300]],
    min_sizes=[[15, 15], [60, 60], [111, 111], [162, 162], [213, 213], [264, 264], [315, 315]],
    # aspect ratio is defined per feature map (first index is largest feature map (38x38))
    # aspect ratio is used to define two boxes per element in the list.
    # if ratio=[2], boxes will be created with ratio 1:2 and 2:1
    # Number of boxes per location is in total 2 + 2 per aspect ratio
    aspect_ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    image_shape="${train.imshape}",
    scale_center_variance=0.1,
    scale_size_variance=0.2
)

backbone = L(backbones.ModelV3)(
    output_channels=[128, 256, 128, 128, 64, 64], #for orig
    #output_channels=[128, 128, 128, 256, 256, 256, 128, 128, 128, 64, 64, 64],
    image_channels="${train.image_channels}",
    output_feature_sizes="${anchors.feature_sizes}"
)

loss_objective = L(SSDMultiboxLoss)(anchors="${anchors}")

model = L(SSD300)(
    feature_extractor="${backbone}",
    anchors="${anchors}",
    loss_objective="${loss_objective}",
    num_classes=10 + 1  # Add 1 for background
)
optimizer = L(torch.optim.AdamW)(
    # Tip: Scale the learning rate by batch size! 2.6e-3 is set for a batch size of 32. use 2*2.6e-3 if you use 64
    lr=5e-4, weight_decay=0.0005, amsgrad = True,
)
# optimizer = L(torch.optim.SGD)(
#     # Tip: Scale the learning rate by batch size! 5e-3 is set for a batch size of 32. use 2*5e-3 if you use 64
#     lr=5e-3, momentum=0.9, weight_decay=0.0005 #org 5e-3
# )
schedulers = dict(
    linear=L(LinearLR)(start_factor=0.1, end_factor=1, total_iters=500),
    multistep=L(MultiStepLR)(milestones=[], gamma=0.1)
)


data_train = dict(
    dataset=L(MNISTDetectionDataset)(
        data_dir=get_dataset_dir("mnist_object_detection/train"),
        is_train=True,
        transform=L(torchvision.transforms.Compose)(transforms=[
            L(ToTensor)(),  # ToTensor has to be applied before conversion to anchors.
            # GroundTruthBoxesToAnchors assigns each ground truth to anchors, required to compute loss in training.
            L(GroundTruthBoxesToAnchors)(anchors="${anchors}", iou_threshold=0.5),
            #L(RandomHorizontalFlip)(),
            # L(RandomSampleCrop)(),
        ])
    ),
    dataloader=L(torch.utils.data.DataLoader)(
        dataset="${..dataset}", num_workers=0, pin_memory=True, shuffle=True, batch_size="${...train.batch_size}", collate_fn=utils.batch_collate,
        drop_last=True
    ),
    #num workers changed from 4 to 0 bc of windows?
    # GPU transforms can heavily speedup data augmentations.
    gpu_transform=L(torchvision.transforms.Compose)(transforms=[
        L(Normalize)(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #L(Normalize)(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), # Normalize has to be applied after ToTensor (GPU transform is always after CPU)
        # L(ColorJitter)(brightness=0.2, contrast=0.3, saturation=0.3, hue=0.2),
    ]),
)
data_val = dict(
    dataset=L(MNISTDetectionDataset)(
        data_dir=get_dataset_dir("mnist_object_detection/val"),
        is_train=False,
        transform=L(torchvision.transforms.Compose)(transforms=[
            L(ToTensor)()
        ])
    ),
    dataloader=L(torch.utils.data.DataLoader)(
       dataset="${..dataset}", num_workers=0, pin_memory=True, shuffle=False, batch_size="${...train.batch_size}", collate_fn=utils.batch_collate_val
    ),
    #num workers changed from 4 to 0 bc of windows?
    gpu_transform=L(torchvision.transforms.Compose)(transforms=[
        #L(Normalize)(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        L(Normalize)(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
)

label_map = {
    0: "background",
    **{i + 1: str(i) for i in range(10)}
}
