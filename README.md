# FAGC
The official implementation for "Fusion-based Feature Attention Gate Component for Vehicle Detection based on Event Camera".

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


## Acknowledgements
The retinanet based sensor fusion model presented here builds upon this [implementation](https://github.com/abhishek1411/event-rgb-fusion/)
