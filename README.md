# FRN

The official implementation for "Embracing Events and Frames with Hierarchical Feature Refinement Network for Robust Object Detection".

![output](https://github.com/HuCaoFighting/FRN/assets/66437581/63188281-6f24-4944-869f-029e4ac26bed)
![output_evt](https://github.com/HuCaoFighting/FRN/assets/66437581/f8e54dda-c623-4fda-91af-012fe24c22fe)

# Abstract
This work addresses the major challenges in object detection for autonomous driving, particularly under demanding conditions such as motion blur, adverse weather, and image noise. Recognizing the limitations of traditional camera systems in these scenarios, this work focuses on leveraging the unique attributes of event cameras, such as their low latency and high dynamic range. These attributes offer promising solutions to complement and augment the capabilities of standard RGB cameras. To leverage these benefits, this work introduces a novel RGB-Event network architecture with a unique fusion module. This module effectively utilizes information from both RGB and event modalities, integrating attention mechanisms and AdaIN (Adaptive Instance Normalization) for enhanced performance. The effectiveness of this approach is validated using two datasets: DSEC and PKU-DDD17-Car, with additional image corruption tests to assess robustness. Results demonstrate that the proposed method significantly outperforms existing state-of-the-art RGB-Event fusion alternatives in both datasets and shows remarkable stability under various image corruption scenarios.

## Setup
- Setup python environment

This code has been tested with Python 3.8, Pytorch 2.0.1, and on Ubuntu 20.04

We recommend you to use Anaconda to create a conda environment:

```
conda create -n env python=3.8
conda activate env
```
- Install pytorch

```
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
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

Our pre-trained weights can be downloaded [here](https://drive.google.com/file/d/1g_AwWsOJHljpQYIpaeAN8YvYWh0pouaV/view?usp=sharing)

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
| Ours               | **0.499**   | **0.258**      | **0.382**            | **0.380**       |

## Results COCO mAP@0.50.95 on DDD17
| Method                    | Test(allday) | Test(day) | Test(night) |
| ------------------------- | ------------ | --------- | ----------- |
| OnlyRGB                   | 0.427        | 0.433     | 0.406       |
| OnlyEvent                 | 0.215        | 0.214     | 0.243       |
| FPN-fusion(RetinaNet) | 0.416        | 0.432     | 0.357       |
| DCF                   | 0.425        | 0.434     | 0.39        |
| SAGate                | 0.434        | 0.449     | 0.38        |
| Self-Attention        | 0.424        | 0.433     | 0.388       |
| ECANet                | 0.408        | 0.422     | 0.361       |
| EFNet                 | 0.416        | 0.434     | 0.351       |
| SPNet                 | 0.433        | 0.449     | 0.371       |
| CBAM                  | 0.428        | 0.442     | 0.38        |
| SENet                 | 0.424        | 0.437     | 0.370       |
| Ours                      | **0.460**        | **0.469**     | **0.421**       |


## Acknowledgements
The retinanet based sensor fusion model presented here builds upon this [implementation](https://github.com/abhishek1411/event-rgb-fusion/)