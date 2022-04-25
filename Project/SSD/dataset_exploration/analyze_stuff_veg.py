from audioop import mul
from operator import index
from matplotlib import collections
from tops.config import instantiate, LazyConfig
from ssd import utils
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import cv2
import os
import pandas as pd
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

def calcArea(arr):
    try:

        x = arr[2]-arr[0]
        y = arr[3]-arr[1]
        return x*y
    except Exception as err:
        print(f'Error {err}')

def calcSides(arr):
    x = arr[2]-arr[0]
    y = arr[3]-arr[1]
    return x.item(),y.item()



def numOverlapped(dataloader_train,dataloader_val):
    num_overlapped_per_frame_train = []
    num_overlapped_per_frame_val = []
    for idx_t, batch_t in enumerate(tqdm(dataloader_train)):
        overlapped = 0
        
        for i_t, box_t in enumerate(batch_t['boxes'].squeeze().numpy()):
            
            for idx, x in enumerate(batch_t['boxes'].squeeze().numpy()):
                if idx == i_t:
                    continue
                elif not (box_t[0]>=x[2]) or (box_t[2]<=x[0]) or (box_t[3]<=x[1]) or (box_t[1]>=x[3]):
                    overlapped += 1
        num_overlapped_per_frame_train.append(overlapped)
    




    for idx_v, batch_v in enumerate(tqdm(dataloader_val)):
        overlapped = 0
        
        for i_v, box_v in enumerate(batch_v['boxes'].squeeze().numpy()):
            
            for idxv, x_v in enumerate(batch_v['boxes'].squeeze().numpy()):
                if idxv == i_v:
                    continue
                elif not (box_t[0]>=x_v[2]) or (box_v[2]<=x_v[0]) or (box_v[3]<=x_v[1]) or (box_v[1]>=x_v[3]):
                    overlapped += 1
        num_overlapped_per_frame_val.append(overlapped)
    
    return num_overlapped_per_frame_train, num_overlapped_per_frame_val
            


def boxSizes(dataloader, save_fig=False):
    box_sizes = []
    for batch in tqdm(dataloader):
        # Remove the two lines below and start analyzing :D
        for x in batch['boxes'].squeeze():
            box_sizes.append(calcArea(x))
    print(max(box_sizes))
    xaxis = np.arange(0,len(box_sizes),1)
    
    sns.scatterplot(x=xaxis,y=box_sizes)
    #sns.histplot(data=box_sizes, kde=False,discrete=True)
    plt.ylabel('Size of boxes')
    plt.xlabel('Batch number')
    plt.gca().invert_yaxis()
    if save_fig:
        plt.savefig('./dataset_exploration/images/data_size_boxes_val_sns.png', bbox_inches = "tight",dpi=200)
    plt.show()
    return box_sizes


def createDataframe(dataloader_train, dataloader_val, labels):
    #df = pd.DataFrame(columns=['dataset','width', 'height', 'type', 'frame_num'])
    data_dict = {'dataset':[],'width':[], 'height':[], 'type':[], 'frame_num':[], 'box_point':[]}
    for idx_t, batch_t in enumerate(tqdm(dataloader_train)):
        for i_t, box_t in enumerate(batch_t['boxes'].squeeze()):
            w, h = calcSides(box_t)
            data_dict['width'].append(w)
            data_dict['height'].append(h)
            data_dict['type'].append(labels[batch_t['labels'][0][i_t]])
            data_dict['frame_num'].append(idx_t)
            data_dict['dataset'].append('train')
            data_dict['box_point'].append((box_t.numpy()))

    for idx_v, batch_v in enumerate(tqdm(dataloader_val)):
        for i_v, box_v in enumerate(batch_v['boxes'].squeeze()):
            w, h = calcSides(box_v)
            data_dict['width'].append(w)
            data_dict['height'].append(h)
            data_dict['type'].append(labels[batch_v['labels'][0][i_v]])
            data_dict['frame_num'].append(idx_v)
            data_dict['dataset'].append('val')
            data_dict['box_point'].append((box_v.numpy()))
    df = pd.DataFrame.from_dict(data_dict)
    
    return df
def sizeDistribution(train_width,train_height, val_width, val_height ): #sjit function
    #print(type(train_height))
    
    df = pd.DataFrame()
    df['train_width'] = pd.Series(train_width)
    df['train_height'] = pd.Series(train_height)
    df['val_width'] = pd.Series(val_width)
    df['val_height'] = pd.Series(val_height)
    
    #df = pd.DataFrame(np.array([train_width,train_height, val_width, val_height]), columns=['train_width','train_height', 'val_width', 'val_height' ])
    #df_train = pd.DataFrame(np.array([train_width,train_height]), columns=['train_width','train_height' ])
    #df_val = pd.DataFrame(np.array([val_width, val_height]), columns=[ 'val_width', 'val_height' ])
    #print(df_train.head())
    #print(df_val.head())
    #df_train['dataset'] = ['train']*len(train_height)
    #df_val['dataset'] = ['val']*len(val_height)
    #df = pd.concat([df_train,df_val], keys=['train', 'val'])
    #print(df.head())
    #sns.boxplot(df, kind='kde')
    plt.figure(1)
    data = [train_height, train_width, val_height, val_width]
    data_h = [train_height, val_height]
    data_w = [train_width, val_width]
    xlab = ['train_height', 'train_width', 'val_height', 'val_width']
    xlabW = [ 'train_width', 'val_width']
    xlabH = ['train_height', 'val_height']
    plt.subplot(211)
    plt.boxplot(data_h, labels=xlabH, showfliers=False)
    plt.subplot(212)
    plt.boxplot(data_w, labels=xlabW, showfliers=False)
    

    plt.savefig('boxplot_sides_split.png', dpi=200)
    plt.show()

def widthHight(dataloader):
    width = []
    height = []
    for batch in tqdm(dataloader):
        for x in batch['boxes'].squeeze():
            w,h = calcSides(x)
            width.append(w.item())
            height.append(h.item())
    return width, height

def classPlot(df, cls=None):
    df_nf = df.drop(['frame_num'], axis=1)
    if cls:
        sns.displot(df_nf[df_nf['type']==cls])
    else:
        sns.displot(df_nf['type'])
    plt.show()

def analyze_something(dataloader, cfg):
    count_labels = []
    for batch in tqdm(dataloader):
        # Remove the two lines below and start analyzing :D
        #print(len(batch['labels'].tolist()[0]))
        #count_labels += batch['labels'].tolist()[0]
        for i_t, box_t in enumerate(batch['boxes'].squeeze().numpy()):

            print(batch)

            print(batch)

        #print("The keys in the batch are:", batch.keys())
        break

def omegaDictTolist(cfg_label_map): #mb getLabels
    labels = []
    for i in range(len(cfg_label_map)):
       labels.append(cfg_label_map[i])
    return labels

def main():
    config_path = "configs/tdt4265_updated_res34.py"
    cfg = get_config(config_path)
    dataset_to_analyze = "val"  # or "val"
    
    print("Label map is:", cfg.label_map)
    #print(cfg.label_map)
    labels = omegaDictTolist(cfg.label_map)
    dataloader = get_dataloader(cfg, dataset_to_analyze)
    dataloader_train = get_dataloader(cfg, 'train')
    dataloader_val = get_dataloader(cfg, 'val')
    #analyze_something(dataloader, labels)
    histogramLabelDist(dataloader,labels, save_fig=True)
    #histogramNumLabel(dataloader)
    #imagesAbove25Labels(dataloader,cfg, save_folder='./dataset_exploration/images/images_above25_labels_val')
    #imagesUnderXLabels(3,dataloader,cfg, save_folder='./dataset_exploration/images/images_under3_labels_train')
    #boxSizes(dataloader,save_fig=True)
    #w_t, h_t = widthHight(dataloader_train)
    #w_v, h_v  = widthHight(dataloader_val)
    #sizeDistribution(w_t,h_t,w_v,h_v)
    t, v = numOverlapped(dataloader_train, dataloader_val)
    print(t)
    print(v)
    #df = createDataframe(dataloader_train, dataloader_val, labels)
    #print(df.head())
    #print(df.describe())
    #df = df[df['dataset']=='train']
    #df['area'] = df['width']*df['height']
    #sns.histplot(x='height', data=df, hue='type',alpha=1)
    #sns.displot(data=df, x='area', hue='type',row='dataset',kde=True)
    #sns.displot(data=df, x='height', row='type' alpha=1)
    #plt.savefig('./dataset_exploration/area_both_dist_kde.png', dpi=200)
    #plt.savefig('./dataset_exploration/area_both_dist_kde.eps', dpi=200)
    #plt.xlim(0,0.03)
    #plt.ylim(0,0.8)
    #plt.show()
    #classPlot(df, 'person')
if __name__ == '__main__':
    main()
