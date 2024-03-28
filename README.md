# FAGC
The official implementation for "Fusion-based Feature Attention Gate Component for Vehicle Detection based on Event Camera".

![output](https://github.com/HuCaoFighting/FRN/assets/66437581/63188281-6f24-4944-869f-029e4ac26bed)
![output_evt](https://github.com/HuCaoFighting/FRN/assets/66437581/f8e54dda-c623-4fda-91af-012fe24c22fe)


# Abstract

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
