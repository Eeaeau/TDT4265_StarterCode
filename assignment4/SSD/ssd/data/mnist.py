import torch
import pathlib
import numpy as np
from ssd import utils
from .mnist_object_detection.mnist_object_detection import load_dataset
from pycocotools.coco import COCO


class MNISTDetectionDataset(torch.utils.data.Dataset):

    class_names = ["__background__"] + [str(x) for x in range(10)]

    def __init__(self, data_dir: str, is_train: bool, transform=None):
        data_dir = pathlib.Path(data_dir)
        self.transform = transform
        self.images, labels, boxes_ltrb = load_dataset(data_dir, is_train)
        self.boxes_ltrb = boxes_ltrb
        self.transform = transform
        self.labels = labels

    def __getitem__(self, idx):
        image = self._read_image(idx)
        boxes, labels = self.get_annotation(idx)
        sample = dict(
            image=image, boxes=boxes, labels=labels,
            width=300, height=300, image_id=idx)
        if self.transform:
             return self.transform(sample)
        return sample

    def __len__(self):
        return len(self.images)

    def get_annotation(self, index):
        boxes = self.boxes_ltrb[index].copy().astype(np.float32)
        # SSD use label 0 as the background. Therefore +1
        labels = self.labels[index].copy().astype(np.int64) + 1
        H, W = self.images.shape[1:3]
        boxes[:, [0, 2]] /= W
        boxes[:, [1, 3]] /= H
        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64))

    def _read_image(self, image_id: int):
        return self.images[image_id][:, :, None].repeat(3, -1)
    
    def get_annotations_as_coco(self) -> COCO:
        """
            Returns bounding box annotations in COCO dataset format
        """
        coco_anns = {"annotations" : [], "images" : [], "licences" : [{"name": "", "id": 0, "url": ""}], "categories" : []}
        categories = [str(i+1) for i in range(10)]
        coco_anns["categories"] = [
            {"name": cat, "id": i+1, "supercategory": ""}
            for i, cat in enumerate(categories) 
        ]
        ann_id = 1
        for idx in range(len(self)):
            image_id = idx
            boxes_ltrb  = self.boxes_ltrb[idx]
            boxes_ltwh = utils.bbox_ltrb_to_ltwh(boxes_ltrb)
            coco_anns["images"].append({"id": image_id, "height": 300, "width": 300 })
            for box, label in zip(boxes_ltwh, self.labels[idx]):
                box = box.tolist()
                area = box[-1] * box[-2]
                coco_anns["annotations"].append({
                    "bbox": box, "area": area, "category_id": int(label+1),
                    "image_id": image_id, "id": ann_id, "iscrowd": 0, "segmentation": []}
                )
                ann_id += 1
        coco_anns["annotations"].sort(key=lambda x: x["image_id"])
        coco_anns["images"].sort(key=lambda x: x["id"])
        coco = COCO()
        coco.dataset = coco_anns
        coco.createIndex()
        return coco