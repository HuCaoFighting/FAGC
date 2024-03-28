import argparse
import collections

import numpy as np
import time
import math
import os

import torch
import torch.optim as optim
from torchvision import transforms
import pickle

from retinanet import model
from retinanet.dataloader import CSVDataset_event,CSVDataset_gray, collater, Resizer, AspectRatioBasedSampler, \
    Augmenter, \
    Normalizer
from torch.utils.data import DataLoader

# from retinanet import coco_eval
from retinanet import csv_eval

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--dataset_name', default='ddd17', help='dsec or ddd17')

    parser.add_argument('--dataset', default='csv', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_train', default='/media/data/hucao/zhenwu/hucao/PKU-DDD17_all/annotations_CSV/labels_filtered_train.csv',
                        help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', default='/media/data/hucao/zhenwu/hucao/PKU-DDD17_all/annotations_CSV/labels_filtered_map.csv',
                        help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')
    parser.add_argument('--root_img',default='/media/data/hucao/zhenwu/hucao/PKU-DDD17_all/images/test_all/aps_images',help='dir to root rgb images in dsec format')
    parser.add_argument('--root_event', default='/media/data/hucao/zhenwu/hucao/PKU-DDD17_all/images/test_all/dvs_events',help='dir to toot event files in dsec directory structure')
    parser.add_argument('--fusion', help='fpn_fusion, rgb, event', type=str, default='fpn_fusion')
    parser.add_argument('--checkpoint', help='location of pretrained file', default='path_to_fagc//csv_fpn_homographic_retinanet_retinanet101_121.pt')
    parser.add_argument('--csv_test', default='/media/data/hucao/zhenwu/hucao/PKU-DDD17_all/annotations_CSV/labels_filtered_test_all.csv',
                        help='Path to file containing training annotations (see readme)')
    parser.add_argument('--eval_corruption', help='evaluate on the coorupted images', type=bool, default=False)
    parser.add_argument('--corruption_group', help='corruption group number', type=int, default=0)
    parser.add_argument('--event_type', help='voxel or gray', type=str, default='voxel')


    parser = parser.parse_args(args)
    dataset_train = CSVDataset_event(dataset_name=parser.dataset_name, train_file=parser.csv_train, class_list=parser.csv_classes,root_event_dir=parser.root_event,root_img_dir=parser.root_img,
                                         transform=transforms.Compose([Normalizer(dataset_name=parser.dataset_name), Resizer(dataset_name=parser.dataset_name)]))
    dataloader_train = DataLoader(dataset_train, batch_size=8, num_workers=0, shuffle=True,collate_fn=collater)
    
    
    # Create the model
    list_models = ['fpn_fusion', 'event', 'rgb']
    if parser.fusion in  list_models:
        retinanet = model.resnet50(dataset_name=parser.dataset_name, num_classes=dataset_train.num_classes(),fusion_model=parser.fusion,pretrained=False)
    else:
        raise ValueError('Unsupported model fusion')

    use_gpu = True
    checkpoint = torch.load(parser.checkpoint)
    retinanet.load_state_dict(checkpoint['model_state_dict'])
    epoch_loss_all = checkpoint['loss']
    epoch_total = checkpoint['epoch']
    print(f'testing {parser.fusion} model')
    retinanet.eval()

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=100)

    retinanet.eval() #
    retinanet.training = False
    retinanet.module.freeze_bn()

    corruption_types = [['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur','glass_blur',
                    'motion_blur','zoom_blur', 'fog','snow','frost','brightness',
                    'contrast','elastic_transform','pixelate','jpeg_compression']] 

    corruption_list = corruption_types[parser.corruption_group] 
    print(corruption_list)
    severity_list = [1,2,3,4,5] 
    
    coco = True
    fps = 0.0
    if parser.eval_corruption: 
        for corruption in corruption_list: 
            Average_precisions = {'person':[],'large_vehicle':[],'car':[]}
            start_c = time.time()
            for severity in severity_list: 
                corruption_folder = f'/media/data/hucao/zhenwu/hucao/PKU-DDD17_all/images/test_all/aps_images_corruption/{corruption}/severity_{severity}' 
                # save_detect_folder = os.path.join(root_save_detect_folder,f'{parser.fusion}_{parser.event_type}',corruption,f'severity_{severity}')
                save_detect_folder = os.path.join('/media/data/hucao/zehua/results_ddd17_day',corruption,f'severity_{severity}')
                
                os.makedirs(save_detect_folder,exist_ok=True)  
                parser.root_img = corruption_folder

                if parser.event_type == 'voxel': #
                    dataset_val1 = CSVDataset_event(dataset_name=parser.dataset_name, train_file= parser.csv_test, class_list=parser.csv_classes,
                                            root_event_dir=parser.root_event,root_img_dir=parser.root_img, transform=transforms.Compose([Normalizer(dataset_name=parser.dataset_name), Resizer(dataset_name=parser.dataset_name)]))

                else:
                    parser.root_event = f'/mnt/8tb-disk/DATASETS/DSEC/e2vid' 
                    dataset_val1 = CSVDataset_gray(train_file= parser.csv_test, class_list=parser.csv_classes,
                                            root_event_dir=parser.root_event,root_img_dir=parser.root_img, transform=transforms.Compose([Normalizer(), Resizer()]))

                start = time.time()
                print(f'{parser.fusion}, {corruption}, severity_{severity}')
                if coco:
                    mAP = csv_eval.evaluate_coco_map(dataset_val1, retinanet,save_detection = True,save_folder = save_detect_folder,
                                load_detection = False)
                    Average_precisions['car'].append(mAP[0])
                    # Average_precisions['large_vehicle'].append(mAP[1])
                    # Average_precisions['car'].append(mAP[2])


                else:
                    mAP = csv_eval.evaluate(dataset_val1, retinanet,save_detection = False,save_folder = save_detect_folder,
                                load_detection = False)
                    Average_precisions['car'].append(mAP[0][0])
                    # Average_precisions['large_vehicle'].append(mAP[1][0])
                    # Average_precisions['car'].append(mAP[2][0])
                    # print(f'time for severity: {time_since(start)}')
                    # print('#########################################')

            print(f'{parser.fusion}, {corruption}')
            print('mAP', mAP)
            print('Average_precisions', Average_precisions)


            for label_name in ['car']:
                print('{}: {}'.format(label_name, list(np.around(np.mean(np.array(Average_precisions[label_name]),axis=1),4))))
                # print('{}: {}'.format(label_name, list(np.around(np.array(Average_precisions[label_name]),2))))
                print('mean', np.mean(list(np.around(np.mean(np.array(Average_precisions[label_name]),axis=1),4))))
                print('rpc', np.mean(list(np.around(np.mean(np.array(Average_precisions[label_name]),axis=1),4)))/0.46)
            print(f'time for corruption: {time_since(start_c)}')

            # ap_file = os.path.join(save_detect_folder,f'{corruption}_ap.txt')
            # with open(ap_file, "wb") as fp:
            #     pickle.dump(Average_precisions, fp, 0) 
            # for label_name in ['car']:
            #     print('{}: {}'.format(label_name, list(np.mean(np.around(np.array(Average_precisions[label_name]),2),axis=1))))
                # print('{}: {}'.format(label_name, list(np.around(np.mean(np.array(Average_precisions[label_name]),axis=1),3))))
                # print('{}: {}'.format(label_name, list(np.around(np.array(Average_precisions[label_name]),2))))
            print(f'time for corruption: {time_since(start_c)}')

    else: 
        Average_precisions = {'car':[]}

        if parser.event_type == 'voxel':
            dataset_val1 = CSVDataset_event(dataset_name=parser.dataset_name, train_file= parser.csv_test, class_list=parser.csv_classes,
                                            root_event_dir=parser.root_event,root_img_dir=parser.root_img, transform=transforms.Compose([Normalizer(dataset_name=parser.dataset_name), Resizer(dataset_name=parser.dataset_name)]))
        else:
            parser.root_event = f'/mnt/8tb-disk/DATASETS/DSEC/e2vid'
            dataset_val1 = CSVDataset_gray(train_file= parser.csv_test, class_list=parser.csv_classes,
                                            root_event_dir=parser.root_event,root_img_dir=parser.root_img, transform=transforms.Compose([Normalizer(), Resizer()]))

        start = time.time()
        # print(f'sensor fusion, {corruption}')
        # save_detect_folder = os.path.join(root_save_detect_folder,f'{parser.fusion}_{parser.event_type}','evaluation')
        save_detect_folder = '/media/data/hucao/zehua/results_ddd17_day/'
        os.makedirs(save_detect_folder,exist_ok=True)
        if coco:
            mAP = csv_eval.evaluate_coco_map(dataset_val1, retinanet,save_detection = False,save_folder = save_detect_folder,
                                load_detection = False)
            
            Average_precisions['car'].append(mAP[0])


        else:
            mAP = csv_eval.evaluate(dataset_val1, retinanet,save_detection = False,save_folder = save_detect_folder,
                                load_detection = False)
            
            Average_precisions['car'].append(mAP[0][0])
        
        fps = 1. / ((time.time() - start)/ 913)
        print('fps', fps)
        print('mAP', mAP)
        print('Average_precisions', Average_precisions)
                
        for label_name in ['car']:
                print('{}: {}'.format(label_name, list(np.mean(np.around(np.array(Average_precisions[label_name]),2),axis=1))))
                #print('{}: {}'.format(label_name, list(np.around(np.array(Average_precisions[label_name]),2))))
      
        # ap_file = os.path.join(save_detect_folder,f'evaluation_ap.txt')
        # with open(ap_file, "wb") as fp:
        #     pickle.dump(Average_precisions, fp, 0)
        
        print(time_since(start))

if __name__ == '__main__':
    main()
