import torch
import numpy as np
import time
import os
import csv
import cv2
import argparse
from retinanet import model
from retinanet.dataloader import CSVDataset_event, collater, Resizer, AspectRatioBasedSampler, \
    Augmenter, \
    Normalizer
from torchvision import transforms
from torch.utils.data import DataLoader

def load_classes(csv_reader):
    result = {}

    for line, row in enumerate(csv_reader):
        line += 1

        try:
            class_name, class_id = row
        except ValueError:
            raise(ValueError('line {}: format should be \'class_name,class_id\''.format(line)))
        class_id = int(class_id)

        if class_name in result:
            raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_name] = class_id
    return result


# Draws a caption above the box in an image
def draw_caption(image, box, caption,colour):
    b = np.array(box).astype(int)
    # cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    # cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, colour, 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, colour, 1)


def Normalizer(sample):
    mean = np.array([[[0.485, 0.456, 0.406]]])
    std = np.array([[[0.229, 0.224, 0.225]]])

    return ((sample.astype(np.float32)-mean)/std)
def detect_image(image_path, model_path, class_list):
    color_map = {
        0: (255, 0, 0),    # Red for class 0
        1: (0, 255, 0),    # Green for class 1
        2: (0, 0, 255)     # Blue for class 2
        # Add more colors for more classes
    }

    with open(class_list, 'r') as f:
        classes = load_classes(csv.reader(f, delimiter=','))

    labels = {}
    for key, value in classes.items():
        labels[value] = key


    for img_name in os.listdir(image_path):

        event_file = os.path.join(image_path,img_name) 
        image = torch.from_numpy(np.load(event_file)['arr_0']) #event

        file = image_path.split('/') 

        img_file = os.path.join(parser.root_img, img_name.replace('.npz', '.png'))
        img_rgb = cv2.imread(img_file)

        height, width = image.shape[1], image.shape[2]
        images = np.full((height, width, 3), 255, dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                if image[0,y,x]>0:
                    #images[y,x]=(255,0,0)#DSEC
                    cv2.circle(images, (x, y), radius=2, color=(255, 0, 0), thickness=-5)
                elif image[0,y,x]<0:
                    #images[y,x]=(0,0,255)#DSEC
                    cv2.circle(images, (x, y), radius=2, color=(0, 0, 255), thickness=-5)
                    
        ev_img = images

        #combined = cv2.addWeighted(img_rgb[:, :, [2, 1, 0]].astype(np.float64), 0.7, (ev_img).astype(np.float64), 0.3, 0)
        # cv2.imwrite(os.path.join(save_path, os.path.splitext(img_name)[0] + '.png'), combined)
        combined = img_rgb

        img_rgb = img_rgb.astype(np.float32) / 255.0
        img_rgb = Normalizer(img_rgb)
        image = np.expand_dims(image, 0)
        img_rgb = np.expand_dims(img_rgb,0)
        img_rgb = np.transpose(img_rgb, (0, 3, 1, 2))

        with torch.no_grad():

            image = torch.from_numpy(image)
            img_rgb = torch.from_numpy(img_rgb)
            if torch.cuda.is_available():
                image = image.cuda().float()
                img_rgb = img_rgb.cuda().float()

            st = time.time()

            scores, classification, transformed_anchors = retinanet((img_rgb, image))
            idxs = np.where(scores.cpu() > 0.5)
            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]

                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                label_id = int(classification[idxs[0][j]])
                label_name = labels[label_id]
                # print(bbox, classification.shape)
                score = scores[idxs[0][j]]
                caption = '{} {:.3f}'.format(label_name, score)
                #box_color = color_map.get(label_id, (255, 255, 255))
                box_color = (0,255,0)
                draw_caption(combined, (x1, y1, x2, y2), caption,box_color)
                cv2.rectangle(combined, (x1, y1), (x2, y2), color=box_color, thickness=3)
                draw_caption(ev_img, (x1, y1, x2, y2), caption,box_color)
                cv2.rectangle(ev_img, (x1, y1), (x2, y2), color=box_color, thickness=3)

            cv2.imwrite(os.path.join(save_path,os.path.splitext(img_name)[0]+'_evt.png'), ev_img) 
            cv2.imwrite(os.path.join(save_path,os.path.splitext(img_name)[0]+'.png'), combined) 


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Simple script for visualizing result of training.')
    base_dir = '/media/data/hucao/zhenwu/hucao/PKU-DDD17_all'
    parser.add_argument('--image_dir', default= f'{base_dir}/images/test_all/dvs_events',help='Path to directory containing images')
    parser.add_argument('--class_list', default=f'{base_dir}/annotations_CSV/labels_filtered_map.csv',help='Path to CSV file listing class names (see README)')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--fusion', help='fpn_fusion, rgb, event', type=str,default='fpn_fusion')
    parser.add_argument('--root_img', default=f'{base_dir}/images/test_all/aps_images', help='dir to toot rgb images in dsec format')
    parser.add_argument('--model_path', default='/media/data/hucao/zehua/results_ddd17_day/ablation/final/csv_fpn_homographic_retinanet_retinanet101_161.pt', help='Path to model')

    parser = parser.parse_args(args=[])

    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=3, pretrained=False)
    elif parser.depth == 50:
        retinanet = model.resnet50('ddd17', num_classes=1,fusion_model=parser.fusion, pretrained=False)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    checkpoint = torch.load(parser.model_path)
    retinanet.load_state_dict(checkpoint['model_state_dict'])
    if torch.cuda.is_available():
        retinanet = retinanet.cuda()

    retinanet.training = False
    retinanet.eval()
    save_path = 'ddd17_event_img'
    os.makedirs(save_path, exist_ok=True)
    detect_image(parser.image_dir, parser.model_path, parser.class_list)
