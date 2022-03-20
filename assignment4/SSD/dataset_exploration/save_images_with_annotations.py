import cv2
import os
import numpy as np

from tops.config import instantiate, LazyConfig
from vizer.draw import draw_boxes
from ssd import utils
from tqdm import tqdm


def get_config(config_path):
    cfg = LazyConfig.load(config_path)
    cfg.data_train.dataloader.shuffle = False
    cfg.data_val.dataloader.shuffle = False
    cfg.train.batch_size = 1
    return cfg


def get_dataloader(cfg, dataset_to_visualize):
    if dataset_to_visualize == "train":
        # Remove GroundTruthBoxesToAnchors transform
        cfg.data_train.dataset.transform.transforms = cfg.data_train.dataset.transform.transforms[:-1]
        data_loader = instantiate(cfg.data_train.dataloader)
    else:
        cfg.data_val.dataloader.collate_fn = utils.batch_collate
        data_loader = instantiate(cfg.data_val.dataloader)

    return data_loader


def convert_boxes_coords_to_pixel_coords(boxes, width, height):
    boxes_for_first_image = boxes[0]  # This is the only image in batch
    boxes_for_first_image[:, [0, 2]] *= width
    boxes_for_first_image[:, [1, 3]] *= height
    return boxes_for_first_image.cpu().numpy()


def convert_image_to_hwc_byte(image):
    first_image_in_batch = image[0]  # This is the only image in batch
    image_pixel_values = (first_image_in_batch * 255).byte()
    image_h_w_c_format = image_pixel_values.permute(1, 2, 0)
    return image_h_w_c_format.cpu().numpy()


def visualize_boxes_on_image(batch, label_map):
    image = convert_image_to_hwc_byte(batch["image"])
    boxes = convert_boxes_coords_to_pixel_coords(batch["boxes"], batch["width"], batch["height"])
    labels = batch["labels"][0].cpu().numpy().tolist()

    image_with_boxes = draw_boxes(image, boxes, labels, class_name_map=label_map)
    return image_with_boxes


def create_viz_image(batch, label_map):
    image_without_annotations = convert_image_to_hwc_byte(batch["image"])
    image_with_annotations = visualize_boxes_on_image(batch, label_map)

    # We concatinate in the height axis, so that the images are placed on top of
    # each other
    concatinated_image = np.concatenate([
        image_without_annotations,
        image_with_annotations,
    ], axis=0)
    return concatinated_image


def create_filepath(save_folder, image_id):
    filename = "image_" + str(image_id[0].item()) + ".png"
    return os.path.join(save_folder, filename)


def save_images_with_annotations(dataloader, cfg, save_folder, num_images_to_visualize):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    print("Saving images to", save_folder)

    num_images_to_save = min(len(dataloader), num_images_to_visualize)
    dataloader = iter(dataloader)

    for i in tqdm(range(num_images_to_save)):
        batch = next(dataloader)
        viz_image = create_viz_image(batch, cfg.label_map)
        filepath = create_filepath(save_folder, batch["image_id"])
        cv2.imwrite(filepath, viz_image)


def main():
    config_path = "configs/tdt4265.py"
    cfg = get_config(config_path)
    dataset_to_visualize = "train"  # or "val"
    num_images_to_visualize = 500  # Increase this if you want to save more images

    dataloader = get_dataloader(cfg, dataset_to_visualize)
    save_folder = os.path.join("dataset_exploration", "annotation_images")
    save_images_with_annotations(dataloader, cfg, save_folder, num_images_to_visualize)


if __name__ == '__main__':
    main()
