# FRN

The official implementation for "Embracing Events and Frames with Hierarchical Feature Refinement Network for Robust Object Detection".

![output](https://github.com/HuCaoFighting/FRN/assets/66437581/63188281-6f24-4944-869f-029e4ac26bed)
![output_evt](https://github.com/HuCaoFighting/FRN/assets/66437581/f8e54dda-c623-4fda-91af-012fe24c22fe)

# Abstract
In the field of autonomous vehicles, various heterogeneous sensors, such as LiDAR, Radar, camera, etc, are combined to improve the vehicle ability of sensing accuracy and robustness. Multi-modal perception and learning has been proved to be an effective method to help vehicle understand the nature of complex environments. Event camera is a bio-inspired vision sensor that captures dynamic changes in the scene and filters out redundant information with high temporal resolution and high dynamic range. These characteristics of the event camera make it have a certain application potential in the field of autonomous vehicles. In this paper, we introduce a fully convolutional neural network with feature attention gate component (FAGC) for vehicle detection by combining frame-based and event-based vision. Both grayscale features and event features are fed into the feature attention gate component (FAGC) to generate the pixel-level attention feature coefficients to improve the feature discrimination ability of the network. Moreover, we explore the influence of different fusion strategies on the detection capability of the network. Experimental results demonstrate that our fusion method achieves the best detection accuracy and exceeds the accuracy of the method that only takes single-mode signal as input.

## Setup
- Setup python environment

This code has been tested with Python 3.8, Pytorch 1.12.1, and on Ubuntu 20.04

We recommend you to use Anaconda to create a conda environment:

```
conda create -n env python=3.8
conda activate env
```
- Install pytorch

```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 pytorch-cuda=11.3 -c pytorch -c nvidia
```

## Usage 
### Training with DSEC or PKU-DDD17 Dataset

```
python train_dsec.py
python train_ddd17.py
```
### Evaluation
You can download our pretrained weights below and modify the path of checkpoint in test_dsec.py file.
```
python test_dsec.py
```
For PKU-DDD17 Dataset
```
python test_ddd17.py
```

## Pre-trained Weights

Our pre-trained weights can be downloaded 

## Results COCO mAP@0.50.95 on DSEC 

| Method             | AP(car) | AP(person) | AP(largevehicle) | mAP@0.50.95 |
| ------------------ | ------- | ---------- | ---------------- | ----------- |
| FPN-fusion(RetinaNet) | 0.375   | 0.109      | 0.249            | 0.244       |
| DCF                | 0.348   | 0.081      | 0.284            | 0.238       |
| SAGate             | 0.315   | 0.097      | 0.166            | 0.193       |
| Self-Attention     | 0.364   | 0.104      | 0.296            | 0.255       |
| ECANet             | 0.36    | 0.097      | 0.295            | 0.251       |
| EFNet              | 0.366   | 0.104      | 0.311            | 0.260       |
| SPNet              | 0.393   | 0.124      | 0.292            | 0.276       |
| SENet              | 0.379   | 0.121      | 0.254            | 0.251       |
| CBAM               | 0.405   | 0.134      | 0.305            | 0.281       |
| FAGC               | **0.398**   | **0.144**      | **0.336**            | **0.293**       |

## Results COCO mAP@0.50 and COCO mAP@0.50.95 on DDD17
| Method                    | Test(all) mAP@0.50.95 | Test(day) mAP@0.50.95 | Test(night) mAP@0.50.95 | Test(all) mAP@0.50 | Test(day) mAP@0.50 | Test(night) mAP@0.50 |
| ------------------------- | --------------------- | --------------------- | ----------------------- | ------------------ | ------------------ | -------------------- |
| OnlyRGB                   | 0.427                 | 0.433                 | 0.406                   | 0.427              | 0.433              | 0.406                |
| OnlyEvent                 | 0.215        | 0.214     | 0.243       |0.215        | 0.214     | 0.243       |
| FPN-fusion(RetinaNet) | 0.416        | 0.432     | 0.357       |0.215        | 0.214     | 0.243       |
| DCF                   | 0.425        | 0.434     | 0.39        |0.215        | 0.214     | 0.243       |
| SAGate                | 0.434        | 0.449     | 0.38        |0.215        | 0.214     | 0.243       |
| Self-Attention        | 0.424        | 0.433     | 0.388       |0.215        | 0.214     | 0.243       |
| ECANet                | 0.408        | 0.422     | 0.361       |0.215        | 0.214     | 0.243       |
| EFNet                 | 0.416        | 0.434     | 0.351       |0.215        | 0.214     | 0.243       |
| SPNet                 | 0.433        | 0.449     | 0.371       |0.215        | 0.214     | 0.243       |
| CBAM                  | 0.428        | 0.442     | 0.38        |0.215        | 0.214     | 0.243       |
| SENet                 | 0.424        | 0.437     | 0.370       |0.215        | 0.214     | 0.243       |
| FAGC                      | **0.436**        | **0.469**     | **0.421**       |**0.436**        | **0.469**     | **0.421**       |


## Acknowledgements
The retinanet based sensor fusion model presented here builds upon this [implementation](https://github.com/abhishek1411/event-rgb-fusion/)
