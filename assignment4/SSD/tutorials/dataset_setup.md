# Setting Up Datasets
Note that all datasets are downloaded on the TDT4265 compute server and the tulipan/cybele computers (saved under /work/datasets).

Index
- Datasets used for assignment 4:
    - [PASCAL VOC](#pascal-voc)
    - MNIST dataset will be downloaded automatic. For information, see: [hukkelas/MNIST-ObjectDetection](github.com/hukkelas/MNIST-ObjectDetection)

---


## Pascal VOC
For Pascal VOC dataset, make the folder structure like this:
```
VOCdevkit
|__ VOC2007
    |_ JPEGImages
    |_ Annotations
    |_ ImageSets
    |_ SegmentationClass
|__ VOC2012
    |_ JPEGImages
    |_ Annotations
    |_ ImageSets
    |_ SegmentationClass
|__ ...
```

You can download this if you are on the school network directly as a .zip file (Note that this dataset should be only used for educational/academic purposes).

With scp: 
```
scp [ntnu-username]@oppdal.idi.ntnu.no:/work/datasets/VOC.zip .

unzip VOC.zip
```


Or you can download it from the PASCAL VOC website:
http://host.robots.ox.ac.uk/pascal/VOC/

Note that we are using the VOC2007 TRAIN/VAL + VOC2012 TRAIN/VAL as the train set.

We use VOC2007 Test as the validation set.


## TDT4265 Dataset
To take a peek at the dataset, take a look at [visualize_dataset.ipynb](../notebooks/visualize_dataset.ipynb).

#### Getting started
- Download:
    - Everything will be done automatically with a script:  `python3 scripts/update_tdt4265_dataset.py`.
    - **REQUIRED TO READ** For the TDT4265 server, the dataset is locally available and the script will automatically setup the dataset.
    - This will take some time on computers that are not provided by us, since you need to download the dataset.
    - We recommend you to **run this file once in a while**, since labels will be automatically updated as students annotate more data.
- Change the following variables in the config file (This is already done in [train_tdt4265.yaml](../configs/train_tdt4265.yaml)):
```
DATASETS:
    TRAIN: ("tdt4265_train",)
    TEST: ("tdt4265_val", )
```
#### Dataset Information
Label format follows the standard COCO format (see [here](https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch) for more info).