# SSD: Single Shot MultiBox Detector

## Training on Cognata

## Setup

you will need to download the Cognata dataset. Follow instruction for the EULA at the [MLCommons Cognata webiste](https://mlcommons.org/datasets/cognata/). After signing the EULA you can download the dataset manually or using scripts on our [WIP repo](https://github.com/mlcommons/abtf-wip/tree/cognata/cognata).

> [!Note]
> We currently have the dataset on an MLCommons Google Drive. When the process for managing access for that is complete, then you won't have to download the dataset from the Cognata servers (which is slow). Until then you can use the scrips to make downloading the dataset more convenient.

Building the Docker container and running the container are the same as for the COCO dataset. Make the path to the dataset refer to the location of the Cognata dataset.

Build:

`docker build --network=host -t ssd .`

Run:

`docker run --gpus [num gpus] --rm -it -v path/to/your/cognata:/cognata -v path/to/trained_models:/trained_models --ipc=host --network=host ssd`

# Training

To train the model run and reproduce the initial test run.

`python -m torch.distributed.launch --nproc_per_node=1 train.py --model ssd --batch-size 1 --dataset Cognata --data-path /cognata --save-folder /trained_model --config test_8MP --save-path test_8mp.pth --epoch 2`

Some differences are the addition of the flags `dataset`, `config`, and `save-path` (the file name). The model parameters are trained based on a config file located in the config folder. This is being actively developed so make sure when you train the model you use the right commit.

# Config file

The config files has the image size and feature sizes for training. The image size determines the feature sizes of the model that are needed to generate the right anchor boxes. The `model_summary.py` script is used to determine the model the right feature sizes. Batch sizes and number of classes are optional for seeing the model structure with different values, but these won't affect the feature sizes.

First create a config file with the right image size and run `python model_summary.py --config [your config file]`. The the 3rd and 4th dimensions of layers starting from `ModuleList:1-13` determine the feature sizes.

An example of the output:

`python model_summary.py --config test_8MP` Feature sizes from output are (270, 480), (135, 240), (68, 120), (34, 60), (32, 58), (30, 56).

```
====================================================================================================
Layer (type:depth-idx)                             Output Shape              Param #
====================================================================================================
SSD                                                [4, 4, 788144]            --
├─ResNet: 1-1                                      [4, 1024, 270, 480]       --
│    └─Sequential: 2-1                             [4, 1024, 270, 480]       --
│    │    └─Conv2d: 3-1                            [4, 64, 1080, 1920]       9,408
│    │    └─BatchNorm2d: 3-2                       [4, 64, 1080, 1920]       128
│    │    └─ReLU: 3-3                              [4, 64, 1080, 1920]       --
│    │    └─MaxPool2d: 3-4                         [4, 64, 540, 960]         --
│    │    └─Sequential: 3-5                        [4, 256, 540, 960]        215,808
│    │    └─Sequential: 3-6                        [4, 512, 270, 480]        1,219,584
│    │    └─Sequential: 3-7                        [4, 1024, 270, 480]       7,098,368
├─ModuleList: 1-2                                  --                        --
│    └─Sequential: 2-2                             [4, 512, 135, 240]        --
│    │    └─Conv2d: 3-8                            [4, 256, 270, 480]        262,144
│    │    └─BatchNorm2d: 3-9                       [4, 256, 270, 480]        512
│    │    └─ReLU: 3-10                             [4, 256, 270, 480]        --
│    │    └─Conv2d: 3-11                           [4, 512, 135, 240]        1,179,648
│    │    └─BatchNorm2d: 3-12                      [4, 512, 135, 240]        1,024
│    │    └─ReLU: 3-13                             [4, 512, 135, 240]        --
│    └─Sequential: 2-3                             [4, 512, 68, 120]         --
│    │    └─Conv2d: 3-14                           [4, 256, 135, 240]        131,072
│    │    └─BatchNorm2d: 3-15                      [4, 256, 135, 240]        512
│    │    └─ReLU: 3-16                             [4, 256, 135, 240]        --
│    │    └─Conv2d: 3-17                           [4, 512, 68, 120]         1,179,648
│    │    └─BatchNorm2d: 3-18                      [4, 512, 68, 120]         1,024
│    │    └─ReLU: 3-19                             [4, 512, 68, 120]         --
│    └─Sequential: 2-4                             [4, 256, 34, 60]          --
│    │    └─Conv2d: 3-20                           [4, 128, 68, 120]         65,536
│    │    └─BatchNorm2d: 3-21                      [4, 128, 68, 120]         256
│    │    └─ReLU: 3-22                             [4, 128, 68, 120]         --
│    │    └─Conv2d: 3-23                           [4, 256, 34, 60]          294,912
│    │    └─BatchNorm2d: 3-24                      [4, 256, 34, 60]          512
│    │    └─ReLU: 3-25                             [4, 256, 34, 60]          --
│    └─Sequential: 2-5                             [4, 256, 32, 58]          --
│    │    └─Conv2d: 3-26                           [4, 128, 34, 60]          32,768
│    │    └─BatchNorm2d: 3-27                      [4, 128, 34, 60]          256
│    │    └─ReLU: 3-28                             [4, 128, 34, 60]          --
│    │    └─Conv2d: 3-29                           [4, 256, 32, 58]          294,912
│    │    └─BatchNorm2d: 3-30                      [4, 256, 32, 58]          512
│    │    └─ReLU: 3-31                             [4, 256, 32, 58]          --
│    └─Sequential: 2-6                             [4, 256, 30, 56]          --
│    │    └─Conv2d: 3-32                           [4, 128, 32, 58]          32,768
│    │    └─BatchNorm2d: 3-33                      [4, 128, 32, 58]          256
│    │    └─ReLU: 3-34                             [4, 128, 32, 58]          --
│    │    └─Conv2d: 3-35                           [4, 256, 30, 56]          294,912
│    │    └─BatchNorm2d: 3-36                      [4, 256, 30, 56]          512
│    │    └─ReLU: 3-37                             [4, 256, 30, 56]          --
├─ModuleList: 1-13                                 --                        (recursive)
│    └─Conv2d: 2-7                                 [4, 16, 270, 480]         147,472
├─ModuleList: 1-14                                 --                        (recursive)
│    └─Conv2d: 2-8                                 [4, 40, 270, 480]         368,680
├─ModuleList: 1-13                                 --                        (recursive)
│    └─Conv2d: 2-9                                 [4, 24, 135, 240]         110,616
├─ModuleList: 1-14                                 --                        (recursive)
│    └─Conv2d: 2-10                                [4, 60, 135, 240]         276,540
├─ModuleList: 1-13                                 --                        (recursive)
│    └─Conv2d: 2-11                                [4, 24, 68, 120]          110,616
├─ModuleList: 1-14                                 --                        (recursive)
│    └─Conv2d: 2-12                                [4, 60, 68, 120]          276,540
├─ModuleList: 1-13                                 --                        (recursive)
│    └─Conv2d: 2-13                                [4, 24, 34, 60]           55,320
├─ModuleList: 1-14                                 --                        (recursive)
│    └─Conv2d: 2-14                                [4, 60, 34, 60]           138,300
├─ModuleList: 1-13                                 --                        (recursive)
│    └─Conv2d: 2-15                                [4, 16, 32, 58]           36,880
├─ModuleList: 1-14                                 --                        (recursive)
│    └─Conv2d: 2-16                                [4, 40, 32, 58]           92,200
├─ModuleList: 1-13                                 --                        (recursive)
│    └─Conv2d: 2-17                                [4, 16, 30, 56]           36,880
├─ModuleList: 1-14                                 --                        (recursive)
│    └─Conv2d: 2-18                                [4, 40, 30, 56]           92,200
====================================================================================================
Total params: 14,059,236
Trainable params: 14,059,236
Non-trainable params: 0
Total mult-adds (T): 5.56
====================================================================================================
Input size (MB): 398.13
Forward/backward pass size (MB): 177580.85
Params size (MB): 56.24
Estimated Total Size (MB): 178035.22
```

# SSD COCO:

## Introduction

Here is my pytorch implementation of 2 models: **SSD-Resnet50** and **SSDLite-MobilenetV2**. These models are based on original model (SSD-VGG16) described in the paper [SSD: Single Shot MultiBox Detector](https://arxiv.org/pdf/1512.02325). **This implementation supports mixed precision training**.
<p align="center">
  <img src="demo/video.gif"><br/>
  <i>An example of SSD Resnet50's output.</i>
</p>

## Motivation

Why this implementation exists while there are many ssd implementations already ?

I believe that many of you when seeing this implementation have this question in your mind. Indeed there are already many implementations for SSD and its variants in Pytorch. However most of them are either: 
- over-complicated
- modularized
- many improvements added
- not evaluated/visualized

The above-mentioned points make learner hard to understand how original ssd looks like. Hence, I re-implement this well-known model, focusing on simplicity. I believe this implementation is suitable for ML/DL users from different levels, especially beginners. In compared to model described in the paper, there are some minor changes (e.g. backbone), but other parts follow paper strictly.  

## Datasets


| Dataset                | Classes |    #Train images      |    #Validation images      |
|------------------------|:---------:|:-----------------------:|:----------------------------:|
| COCO2017               |    80   |          118k         |              5k            |

  
- **COCO**:
  Download the coco images and annotations from [coco website](http://cocodataset.org/#download). Make sure to put the files as the following structure (The root folder names **coco**):
  ```
  coco
  ├── annotations
  │   ├── instances_train2017.json
  │   └── instances_val2017.json
  │── train2017
  └── val2017 
  ```
## Docker

For being convenient, I provide Dockerfile which could be used for running training as well as test phases

Assume that docker image's name is **ssd**. You already created an empty folder name **trained_models** for storing trained weights. Then you clone this repository and cd into it.

Build:

`docker build --network=host -t ssd .`

Run:

`docker run --rm -it -v path/to/your/coco:/coco -v path/to/trained_models:/trained_models --ipc=host --network=host ssd`

## How to use my code

Assume that at this step, you either already installed necessary libraries or you are inside docker container

Now, with my code, you can:

* **Train your model** by running `python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE train.py --model [ssd|ssdlite] --batch-size [int] [--amp]`. You could stop or resume your training process whenever you want. For example, if you stop your training process after 10 epochs, the next time you run the training script, your training process will continue from epoch 10. mAP evaluation, by default, will be run at the end of each epoch. **Note**: By specifying **--amp** flag, your model will be trained with mixed precision (FP32 and FP16) instead of full precision (FP32) by default. Mixed precision training reduces gpu usage and therefore allows you train your model with bigger batch size while sacrificing negligible accuracy. More infomation could be found at [apex](https://github.com/NVIDIA/apex) and [pytorch](https://pytorch.org/docs/stable/notes/amp_examples.html).
* **Test your model for COCO dataset** by running `python test_dataset.py --pretrained_model path/to/trained_model`
* **Test your model for image** by running `python test_image.py --pretrained_model path/to/trained_model --input path/to/input/file --output path/to/output/file`
* **Test your model for video** by running `python test_video.py --pretrained_model path/to/trained_model --input path/to/input/file --output path/to/output/file`

You could download my trained weight for SSD-Resnet50 at [link](https://drive.google.com/drive/folders/1_DYYDJUfwLIvGBDnM3hMFNgkVRZW6MgX?usp=sharing)
## Experiments

I trained my models by using NVIDIA RTX 2080. Below is mAP evaluation for **SSD-Resnet50** trained for 54 epochs on **COCO val2017** dataset 
<p align="center">
  <img src="demo/mAP.png"><br/>
  <i>SSD-Resnet50 evaluation.</i>
</p>
<p align="center">
  <img src="demo/tensorboard.png"><br/>
  <i>SSD-Resnet50 tensorboard for training loss curve and validation mAP curve.</i>
</p>

## Results

Some predictions are shown below:

<img src="demo/1.jpg" width="250"> <img src="demo/2.jpg" width="250"> <img src="demo/3.jpg" width="250">

<img src="demo/4.jpg" width="250"> <img src="demo/5.jpg" width="250"> <img src="demo/6.jpg" width="250">

<img src="demo/7.jpg" width="250"> <img src="demo/8.jpg" width="250"> <img src="demo/9.jpg" width="250">


## References
- Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg "SSD: Single Shot MultiBox Detector" [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325).

- My implementation is inspired by and therefore borrows many parts from [NVIDIA Deep Learning examples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD) and [ssd pytorch](https://github.com/qfgaohao/pytorch-ssd)
