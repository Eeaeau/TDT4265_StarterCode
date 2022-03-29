import typing
import pathlib
import cv2
import numpy as np
import tqdm
from . import mnist


def read_labels(label_path: pathlib.Path) -> typing.Tuple[np.ndarray]:
    assert label_path.is_file(), f"Did not find file: {label_path}"
    labels = []
    BBOXES_XYXY = []
    with open(label_path, "r") as fp:
        for line in list(fp.readlines())[1:]:
            label, xmin, ymin, xmax, ymax = [int(_) for _ in line.split(",")]
            labels.append(label)
            BBOXES_XYXY.append([xmin, ymin, xmax, ymax])
    boxes = np.array(BBOXES_XYXY)
    if len(boxes) == 0:
        boxes = np.zeros((0, 4))
    return np.array(labels), boxes


def calculate_iou(prediction_box, gt_box):
    x1_t, y1_t, x2_t, y2_t = gt_box
    x1_p, y1_p, x2_p, y2_p = prediction_box
    if (x2_t < x1_p or x2_p < x1_t or y2_t < y1_p or y2_p < y1_t):
        return 0.0

    # Compute intersection
    x1i = max(x1_t, x1_p)
    x2i = min(x2_t, x2_p)
    y1i = max(y1_t, y1_p)
    y2i = min(y2_t, y2_p)
    intersection = (x2i - x1i) * (y2i - y1i)

    # Compute union
    pred_area = (x2_p - x1_p) * (y2_p - y1_p)
    gt_area = (x2_t - x1_t) * (y2_t - y1_t)
    union = pred_area + gt_area - intersection
    iou = intersection / union
    assert iou >= 0 and iou <= 1
    return iou


def compute_iou_all(bbox, all_bboxes):
    ious = [0]
    for other_bbox in all_bboxes:
        ious.append(
            calculate_iou(bbox, other_bbox)
        )
    return ious


def tight_bbox(digit, orig_bbox):
    xmin, ymin, xmax, ymax = orig_bbox
    # xmin
    shift = 0
    for i in range(digit.shape[1]):
        if digit[:, i].sum() != 0:
            break
        shift += 1
    xmin += shift
    # xmax
    shift = 0
    for i in range(-1, -digit.shape[1], -1):
        if digit[:, i].sum() != 0:
            break
        shift += 1
    xmax -= shift
    ymin
    shift = 0
    for i in range(digit.shape[0]):
        if digit[i, :].sum() != 0:
            break
        shift += 1
    ymin += shift
    shift = 0
    for i in range(-1, -digit.shape[0], -1):
        if digit[i, :].sum() != 0:
            break
        shift += 1
    ymax -= shift
    return [xmin, ymin, xmax, ymax]


def dataset_exists(dirpath: pathlib.Path, num_images):
    if not dirpath.is_dir():
        return False
    error_msg = f"MNIST dataset already generated in {dirpath}, \n\tbut did not find filepath:"
    error_msg2 = f"You can delete the directory by running: rm -r {dirpath.parent}"
    assert dirpath.joinpath("images", "images.npy").is_file(),\
        f"{error_msg}, {error_msg2}"

    for image_id in range(num_images):
        label_path = dirpath.joinpath("labels", f"{image_id}.txt")
        assert label_path.is_file(),  f"{error_msg} {label_path} \n\t{error_msg2}"
    return True


def load_dataset(dirpath: pathlib.Path,
                 is_train: bool,
                 max_digit_size: int = 100,
                 min_digit_size: int = 15,
                 imsize: int =300,
                 max_digits_per_image: int = 20):
    num_images = 10000 if is_train else 1000
    X_train, Y_train, X_test, Y_test = mnist.load()
    X, Y = X_train, Y_train
    if not is_train:
        X, Y = X_test, Y_test
    generate_dataset(
        dirpath,
        num_images,
        max_digit_size,
        min_digit_size,
        imsize,
        max_digits_per_image,
        X,
        Y)
    images = []
    all_labels = []
    all_bboxes_XYXY = []
    images = np.load(dirpath.joinpath("images", "images.npy"))
    for image_id in range(len(images)):
        label_path = dirpath.joinpath("labels").joinpath(
            f"{image_id}.txt")
        labels, bboxes = read_labels(label_path)
        all_labels.append(labels)
        all_bboxes_XYXY.append(bboxes)
    return images, all_labels, all_bboxes_XYXY


def generate_dataset(dirpath: pathlib.Path,
                     num_images: int,
                     max_digit_size: int,
                     min_digit_size: int,
                     imsize: int,
                     max_digits_per_image: int,
                     mnist_images: np.ndarray,
                     mnist_labels: np.ndarray):
    if dataset_exists(dirpath, num_images):
        return
    max_image_value = 255
    assert mnist_images.dtype == np.uint8
    image_dir = dirpath.joinpath("images")
    label_dir = dirpath.joinpath("labels")
    image_dir.mkdir(exist_ok=True, parents=True)
    label_dir.mkdir(exist_ok=True, parents=True)
    all_images = np.zeros((num_images, imsize, imsize), dtype=np.uint8)
    for image_id in tqdm.trange(num_images, desc=f"Generating dataset, saving to: {dirpath}"):
        im = np.zeros((imsize, imsize), dtype=np.float32)
        labels = []
        bboxes = []
        num_images = np.random.randint(1, max_digits_per_image)
        for _ in range(num_images):
            while True:
                width = np.random.randint(min_digit_size, max_digit_size)
                x0 = np.random.randint(0, imsize-width)
                y0 = np.random.randint(0, imsize-width)
                ious = compute_iou_all([x0, y0, x0+width, y0+width], bboxes)
                if max(ious) < 0.25:
                    break
            digit_idx = np.random.randint(0, len(mnist_images))
            digit = mnist_images[digit_idx].astype(np.float32)
            digit = cv2.resize(digit, (width, width))
            label = mnist_labels[digit_idx]
            labels.append(label)
            assert im[y0:y0+width, x0:x0+width].shape == digit.shape, \
                f"imshape: {im[y0:y0+width, x0:x0+width].shape}, digit shape: {digit.shape}"
            bbox = tight_bbox(digit, [x0, y0, x0+width, y0+width])
            bboxes.append(bbox)

            im[y0:y0+width, x0:x0+width] += digit
            im[im > max_image_value] = max_image_value
        image_target_path = image_dir.joinpath(f"{image_id}.png")
        label_target_path = label_dir.joinpath(f"{image_id}.txt")
        im = im.astype(np.uint8)
        all_images[image_id] = im
        with open(label_target_path, "w") as fp:
            fp.write("label,xmin,ymin,xmax,ymax\n")
            for l, bbox in zip(labels, bboxes):
                bbox = [str(_) for _ in bbox]
                to_write = f"{l}," + ",".join(bbox) + "\n"
                fp.write(to_write)
    np.save(str(image_dir.joinpath("images.npy")), all_images)


