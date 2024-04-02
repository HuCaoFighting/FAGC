# FAGC

The official implementation for "Fusion-based Feature Attention Gate Component for Vehicle Detection based on Event Camera".

![output](/img/output.gif)
![output_evt](/img/output_evt.gif)

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

Our pre-trained weights can be downloaded [dsec](https://drive.google.com/file/d/1wdC6UrZXG4sYU0Zvbtdk3EX3C5mPWaoK/view?usp=sharing) and [ddd17](https://drive.google.com/file/d/1DjIdpDD1lMHJ99S1_JSYHmPHmd7lSS-m/view?usp=sharing)

## Results COCO mAP@0.50.95 on DSEC 

| Method             | AP(car) | AP(person) | AP(largevehicle) | mAP@0.50.95 |
| ------------------ | ------- | ---------- | ---------------- | ----------- |
| FPN-fusion(RetinaNet) | 0.375   | 0.109      | 0.249            | 0.244       |
| DCF                | 0.363   | 0.127      | 0.280            | 0.257       |
| SAGate             | 0.325   | 0.104      | 0.16            | 0.196       |
| Self-Attention     | 0.386   | 0.151      | 0.306            | 0.281       |
| ECANet             | 0.367    | 0.128      | 0.275            | 0.257       |
| EFNet              | 0.411   | 0.158      | 0.326            | 0.3       |
| SPNet              | 0.392   | 0.178      | 0.262            | 0.277       |
| SENet              | 0.384   | 0.149      | 0.26            | 0.262       |
| CBAM               | 0.377   | 0.135      | 0.270            | 0.261       |
| CMX               | 0.416   | 0.164      | 0.294            | 0.291       |
| RAM               | 0.244   | 0.108      | 0.176            | 0.176       |
| FAGC               | **0.398**   | **0.144**      | **0.336**            | **0.293**       |

## Results COCO mAP@0.50 and COCO mAP@0.50.95 on DDD17
| Method                    | Test(all) mAP@0.50.95 | Test(day) mAP@0.50.95 | Test(night) mAP@0.50.95 | Test(all) mAP@0.50 | Test(day) mAP@0.50 | Test(night) mAP@0.50 |
| ------------------------- | --------------------- | --------------------- | ----------------------- | ------------------ | ------------------ | -------------------- |
| OnlyRGB                   | 0.427                 | 0.433                 | 0.406                   | 0.827              | 0.829              | 0.825                |
| OnlyEvent                 | 0.215        | 0.214     | 0.243       |0.465        | 0.436     | 0.600       |
| FPN-fusion(RetinaNet) | 0.416        | 0.432     | 0.357       |0.819       | 0.828    | 0.789      |
| DCF                   | 0.425        | 0.434     | 0.39        |0.834        | 0.842     | 0.804       |
| SAGate                | 0.434        | 0.449     | 0.38        |0.820        | 0.825     | 0.804       |
| Self-Attention        | 0.424        | 0.433     | 0.388       |0.826        | 0.834    | 0.811       |
| ECANet                | 0.408        | 0.422     | 0.361       |0.822        | 0.831     | 0.790      |
| EFNet                 | 0.416        | 0.434     | 0.351       |0.830        | 0.844     | 0.787       |
| SPNet                 | 0.433        | 0.449     | 0.371       |0.847        | 0.861     | 0.789      |
| CBAM                  | 0.428        | 0.442     | 0.38        |0.819        | 0.823     | 0.810    |
| SENet                 | 0.424        | 0.437     | 0.370       |0.816      | 0.827     | 0.774       |
| CMX                 | 0.390        | 0.402     | 0.354       |0.804      | 0.807     | 0.796       |
| RAM                 | 0.388        | 0.392     | 0.369       |0.796      | 0.799     | 0.782       |
| FAGC                      | **0.436**        | **0.448**     | **0.395**       |**0.852**        | **0.859**     | **0.826**       |


## Acknowledgements
The retinanet based sensor fusion model presented here builds upon this [implementation](https://github.com/abhishek1411/event-rgb-fusion/)
