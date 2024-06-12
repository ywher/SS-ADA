# SS-ADA: A Semi-Supervised Active Domain Adaptation Framework for Semantic Segmentation

Official implementation of "SS-ADA: A Semi-Supervised Active Domain Adaptation Framework for Semantic Segmentation". Submitted to T-ITS on May **, 2024.

## Abstract

Semantic segmentation plays an important role in intelligent vehicles, providing pixel-level semantic information about the environment. However, the labeling budget is expensive and time-consuming when semantic segmentation model is applied to new driving scenarios. To reduce the costs, semi-supervised semantic segmentation methods have been proposed to leverage large quantities of unlabeled images. Despite this, their performance still lags behind the accuracy of supervised learning, which is desired in practical applications. A significant shortcoming is that they typically select unlabeled images for annotation randomly, neglecting the assessment of sample value for model training. In this paper, we propose a novel semi-supervised active domain adaptation (SS-ADA) framework for semantic segmentation that employs an image-level acquisition strategy. SS-ADA integrates active learning into semi-supervised semantic segmentation to achieve the accuracy of supervised learning with a limited amount of labeled data from the target domain. Additionally, we design an IoU-based class weighting strategy to alleviate the class imbalance problem using annotations from active learning. We conducted extensive experiments on synthetic-to-real and real-to-real domain adaptation settings. The results demonstrate the effectiveness of our method. SS-ADA can achieve or even surpass the accuracy of its supervised learning counterpart with only 25\% of the target labeled data when using a real-time segmentation model. The code for SS-ADA is available at \url{https://github.com/ywher/SS-ADA}.

## Environment Setup

I verified this reporitory in Ubuntu20.04 with Anaconda3, Pytorch 1.12.0, CUDA 11.3,  3090 GPU.

First, create the environment named ss-ada and activate the ss-ada environment through:

```
conda create -n ss-ada python=3.8 -y
conda activate ss-ada
```

Then install the required packages though:

```
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113

pip install -r requirements.txt 
```

Download the code from github and change the directory:

```
git clone https://github.com/ywher/SS-ADA
cd SS-ADA
```

## Dataset preparation

Here we only show how to set the GTA5-to-Cityscapes and bev2023-to-bev2024 settings. You can prepare the SYNTHIA-to-Cityscapes, Cityscapes-to-ACDC , and Cityscapes-to-FisheyeCampus datasets similarly.

[Download bev2023, bev2024](https://drive.google.com/drive/folders/1Zl7nbNrUOAjbHtXtQeLsntmvukhR6g_4?usp=sharing), GTA5, and Cityscapes datasets then organize the folder as follows:

```
|SS-ADA/data
│     ├── bev2023/
|     |   ├── image/
|     |   |   ├── train/
|     |   |   ├── val/
|     |   ├── label/
|     |   |   ├── train/
|     |   |   ├── val/
│     ├── bev2024/
|     |   ├── image/
|     |   |   ├── train/
|     |   |   ├── val/
|     |   ├── label/
|     |   |   ├── train/
|     |   |   ├── val/
│     ├── cityscapes/   
|     |   ├── leftImg8bit/
|     |   |   ├── train/
|     |   |   ├── val/
|     |   |   ├── test/
|     |   ├── gtFine/
|     |   |   ├── train/
|     |   |   ├── val/
│     ├── gtav/   
|     |   ├── images/
|     |   ├── labels/
      ...
```

## Pretrained Model

Downlaod the ImageNet pretrained ResNet18 and put it in /pretrained folder

```
https://download.pytorch.org/models/resnet18-5c106cde.pth
```

## Training and Evaluation example

We use one 3090 GPU for training and evaluation.

### Set config

Remember to change the work root from "/media/ywh/pool1/yanweihao/projects/active_learning/SS-ADA" in configs/*.yaml to your own SS-ADA root

### Train with GTA5-to-Cityscapes

TBD

### Train with bev2023-to-bev2024

#### Train with bev2023 (source only)

set the scripts/train_bisenet.sh as following:

```
dataset='bev_2023'
method='supervised_bisenetv1_tar'
exp='bisenetv1'
split='110'
config_name='parking_bev2023_bisenetv1'
```

Then run the training bash, (n_gpus=2, port=1008)

```
bash scripts/train_bisenetv1.sh 2 10008
```

#### Train with bev2024 (target only, supervised learning)

set the use source in configs/parking_bev2024_bisenetv1.yaml to False

```
source:
  use_source: False
```

set the scripts/train_bisenet.sh as following:

```
dataset='bev_2024'
method='supervised_bisenetv1_tar'
exp='bisenetv1'
split='140'
config_name='parking_bev2024_bisenetv1'
```

Then run the training bash, (n_gpus=2, port=1008)

```
bash scripts/train_bisenetv1.sh 2 10008
```

#### Train with bev2023 and bev2024 (joint training)

set the use source in configs/parking_bev2024_bisenetv1.yaml to True

```
source:
  use_source: True
  type: bev_2023
  data_root: /your_path_to_SS-ADA/SS-ADA/data/bev_2023
  data_list: /your_path_to_SS-ADA/SS-ADA/data/train_list/bev_2023_train_list.txt
```

set the scripts/train_bisenet.sh as following:

```
dataset='bev_2024'
method='supervised_bisenetv1_both'
exp='bisenetv1'
split='140'
config_name='parking_bev2024_bisenetv1'
```

Then run the training bash, (n_gpus=2, port=1008)

```
bash scripts/train_bisenetv1.sh 2 10008
```

### Train with SS-ADA

set the use source in configs/parking_bev2024_acda_bisenetv1_single,yaml to True

and set the n_sup and active.total ratio to the corresponding setting.

Here we take 50% of target labeled data,n_sup=70 and total_ratio=0.50 as example. (70, 0.5; 35, 0.25...))

```
n_sup: 70
source:
  use_source: True
active:
  init_ratio: 0.01
  total_ratio: 0.50
```

set the scripts/train_acda_bisenet_single.sh as following:

```
dataset='bev_2024'
method='ss_ada_bisenetv1_single'
exp='bisenetv1'
split='70'
config_name='parking_bev2024_acda_bisenetv1_single'
init_split=1
```

Then run the training bash, (n_gpus=1)

```
bash scripts/train_acda_bisenetv1_single
```

### Evaluation of the model

#### Eval of models on GTA5-to-Cityscapes

##### Eval of bev2023 source only

TBD

##### Eval of bev2024 supervised learning

TBD

##### Eval of bev2024 joint training

TBD

##### Eval of SS-ADA model

Our trained models can be found in the following links:

* 50% of the target labeled data: https://drive.google.com/file/d/10gki3Fk5V0ZWwiWYZ80DLY-aPBJdD9pk/view?usp=drive_link
* 25% of the target labeled data: https://drive.google.com/file/d/1ApSkCi7ELP71oEvcvqi9a0FsjGV6qKyQ/view?usp=drive_link
* 12.5% of the target labeled data: https://drive.google.com/file/d/1YThOOARCUUrAsGxyt11hv_KfGtTVKOY0/view?usp=drive_link

#### Eval of bev2023/2024

##### Eval of bev2023 source only

change the val dataset from bev2023 to bev2024 in parking_bev2023_bisenetv1.yaml

```
val:
  dataset: bev_2024
  data_root: /media/ywh/pool1/yanweihao/projects/active_learning/SS-ADA/data/bev_2024
```

set in scripts/eval_bisenetv1.sh and run "bash scripts/eval_bisenetv1.sh"

```
dataset="bev_2023"
config_path="configs/parking_bev2023_bisenetv1.yaml"
eval_mode="original"
exp_folder="supervised_bisenetv1_tar"
split=110
```

123

##### Eval of bev2024 supervised learning

set in scripts/eval_bisenetv1.sh and run "bash scripts/eval_bisenetv1.sh"

```
dataset="bev_2024"
config_path="configs/parking_bev2024_bisenetv1.yaml"
eval_mode="original"
exp_folder="supervised_bisenetv1_tar"
split=140
```

##### Eval of bev2024 joint training

set in scripts/eval_bisenetv1.sh and run "bash scripts/eval_bisenetv1.sh"

```
dataset="bev_2024"
config_path="configs/parking_bev2024_bisenetv1.yaml"
eval_mode="original"
exp_folder="supervised_bisenetv1_both"
split=140
```

Our trained models can be found in the following links:

* 50% of the target labeled data: https://drive.google.com/drive/folders/1qVgo4uKCH9ZqVFbwAdJ09yLDLQ9wnG3v?usp=sharing
* 25% of the target labeled data:https://drive.google.com/drive/folders/1NS2da6Dj3thafgNm5MqakZgJO6_FdlWr?usp=sharing

The evaluation results of our trained models on bev2024 validation set are listed in the following.

|           Setting           | best.pth (mIoU %) | latest.pth (mIoU %) |
| :-------------------------: | :---------------: | :-----------------: |
|      bev2023 sup only      |       38.09       |        38.09        |
| bev2024 supervised learning |       71.63       |        70.20        |
|   bev2024 joint training   |       74.41       |        72.79        |
|       ss-ada 50% data       |       73.09       |        72.83        |
|       ss-ada 25% data       |       71.67       |        71.06        |

# Acknowledgement

Some of the code is borrowed from [Unimatch](https://github.com/LiheYoung/UniMatch) and [BiSeNet](https://github.com/CoinCheung/BiSeNet)

Thanks a lot for their great work!

# Citation

If you use this code in your research please consider citing our work

```
TBD
```

# Contact

Weihao Yan: weihao_yan@outlook.com
