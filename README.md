# SSD: Single Shot MultiBox Detector

## Training on Cognata

## Setup

You will need to download the Cognata dataset. Follow instruction for the EULA at the [MLCommons Cognata webiste](https://mlcommons.org/datasets/cognata/). After signing the EULA you can download the dataset manually or using scripts on our [WIP repo](https://github.com/mlcommons/abtf-wip/tree/cognata/cognata).

> [!Note]
> We currently have the dataset on an MLCommons Google Drive. When the process for managing access for that is complete, then you won't have to download the dataset from the Cognata servers (which is slow). Until then you can use the scrips to make downloading the dataset more convenient.

Building the Docker container and running the container are the same as for the COCO dataset. Make the path to the dataset refer to the location of the Cognata dataset.

Build:

`docker build --network=host -t ssd .`

Run:

`docker run --gpus [num gpus] --rm -it -v path/to/your/cognata:/cognata -v path/to/trained_models:/trained_models --ipc=host --network=host ssd`

# Training

To train the model run and produce an 8MP resultion test.

`torchrun --nproc_per_node= [num gpus] train.py --model ssd --batch-size 1 --dataset Cognata --data-path /cognata --save-folder /trained_models --config test_8MP --save-name test_8mp --epoch 2`

Some differences are the addition of the flags `dataset`, `config`, and `save-name` (base name of the file). The model parameters are trained based on a config file located in the config folder. This is being actively developed so make sure when you train the model you use the right commit. The checkpoints are saved along with the epoch name, e.g. test_8mpep2.pth.

If you are training from a checkpoint, add the `--pretrained-model` flag along the the path to the checkpoint.

# Evaluation
To evaluate a trained model

`torchrun --nproc_per_node=1 evaluate.py --model ssd --batch-size 1 --dataset Cognata --data-path /cognata --pretrained-model [model path] --config test_8MP`
> [!Note]
> Evaluation only works with 1 gpu.

# Test on data
To test on the dataset using the 1.0 checkpoint
`python test_dataset.py --pretrained-model [path/to/trained_model] --dataset Cognata --data-path /cognata --config baseline_8MP_ss_scales_test`

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
# Acknowledgements

This repo modifies [SSD-Pytorch](https://github.com/uvipen/SSD-pytorch).

