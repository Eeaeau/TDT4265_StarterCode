from tops.config import instantiate, LazyConfig
from ssd import utils
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import cv2
import os
from vizer.draw import draw_boxes
def get_config(config_path):
    cfg = LazyConfig.load(config_path)
    cfg.train.batch_size = 1
    return cfg

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
    filename = "image_" + str(image_id) + ".png"
    return os.path.join(save_folder, filename)



def get_dataloader(cfg, dataset_to_visualize):
    if dataset_to_visualize == "train":
        # Remove GroundTruthBoxesToAnchors transform
        cfg.data_train.dataset.transform.transforms = cfg.data_train.dataset.transform.transforms[:-1]
        data_loader = instantiate(cfg.data_train.dataloader)
    else:
        cfg.data_val.dataloader.collate_fn = utils.batch_collate
        data_loader = instantiate(cfg.data_val.dataloader)

    return data_loader

def histogramLabelDist(dataloader,labels,save_fig=False):
    #create some more in seaborn mb
    count_labels = []
    for batch in tqdm(dataloader):
        count_labels += batch['labels'].tolist()[0]
    
    x_values = np.arange(0, len(labels), 1)
    sns.histplot(data=count_labels, kde=False,bins=len(labels),discrete=True)
    plt.xticks(x_values,labels,rotation=40)
    if save_fig:
        plt.savefig('./dataset_exploration/images/data_distribution_train_updated_dataset_sns.png', bbox_inches = "tight",dpi=200)
        plt.savefig('./dataset_exploration/images/data_distribution_train_updated_dataset_sns.eps', bbox_inches = "tight",dpi=200)
    plt.show()


def histogramNumLabel(dataloader,save_fig=False):
    #create some more in seaborn mb
    count_labels = []
    for batch in tqdm(dataloader):
        count_labels.append(len(batch['labels'].tolist()[0]))
        
    sns.histplot(data=count_labels, kde=False,discrete=True)
    plt.xlabel('Number of labels')
    
    if save_fig:
        plt.savefig('./dataset_exploration/images/data_num_label_train_sns.png', bbox_inches = "tight",dpi=200)
    plt.show()

def imagesAbove25Labels(dataloader,cfg,save_folder):
    #create some more in seaborn mb
 
    i = 0
    for batch in tqdm(dataloader):
        if len(batch['labels'].tolist()[0])>25:
            viz_image = create_viz_image(batch, cfg.label_map)
            filepath = create_filepath(save_folder, i)
            cv2.imwrite(filepath, viz_image[:, :, ::-1])
            i += 1

def imagesUnderXLabels(X,dataloader,cfg,save_folder):
    #create some more in seaborn mb
 
    i = 0
    for batch in tqdm(dataloader):
        if len(batch['labels'].tolist()[0])<X:
            viz_image = create_viz_image(batch, cfg.label_map)
            filepath = create_filepath(save_folder, i)
            cv2.imwrite(filepath, viz_image[:, :, ::-1])
            i += 1

        
        
    

def analyze_something(dataloader, cfg):
    count_labels = []
    for batch in tqdm(dataloader):
        # Remove the two lines below and start analyzing :D
        #print(len(batch['labels'].tolist()[0]))
        count_labels += batch['labels'].tolist()[0]
    
        

        #print("The keys in the batch are:", batch.keys())
        #exit()

def omegaDictTolist(cfg_label_map): #mb getLabels
    labels = []
    for i in range(len(cfg_label_map)):
       labels.append(cfg_label_map[i])
    return labels

def main():
    config_path = "configs/tdt4265_updated_res34.py"
    cfg = get_config(config_path)
    dataset_to_analyze = "train"  # or "val"
    
    print("Label map is:", cfg.label_map)
    #print(cfg.label_map)
    labels = omegaDictTolist(cfg.label_map)
    dataloader = get_dataloader(cfg, dataset_to_analyze)
    #analyze_something(dataloader, labels)
    histogramLabelDist(dataloader,labels, save_fig=True)
    #histogramNumLabel(dataloader)
    #imagesAbove25Labels(dataloader,cfg, save_folder='./dataset_exploration/images/images_above25_labels_val')
    imagesUnderXLabels(3,dataloader,cfg, save_folder='./dataset_exploration/images/images_under3_labels_train')
if __name__ == '__main__':
    main()
