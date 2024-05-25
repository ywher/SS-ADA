# SS-ADA: A Semi-Supervised Active Domain Adaptation Framework for Semantic Segmentation

Official implementation of "SS-ADA: A Semi-Supervised Active Domain Adaptation Framework for Semantic Segmentation". Submitted to T-ITS on May **, 2024.

## Abstract

Semantic segmentation plays an important role in intelligent vehicle which provides pixel-level semantic information. However, the labeling budget for semantic segmentation is expensive and time-consuming. To reduce the cost, domain adaptation methods are proposed to transfer the knowledge from  labeled source domain to the unseen target domain. Among them, semi-supervised and active learning-based methods are proposed to improve the segmentation performance using limited labeled data from target domain. Semi-supervised-based methods suffer from the noisy pseudo-labels due to the domain shift and lack of accurate semantic guidance. Active learning-based methods requires the human annotation for better performance but neglect the potential of the unlabeled data. In this paper, we propose a novel semi-supervised active domain adaptation (SS-ADA) framework for semantic segmentation. SS-ADA combines the advantages of semi-supervised learning and active learning to achieve supervised learning accuracy using limited labeled data from the target domain. We also design the IoU-based class weighting strategy to alleviate the class imbalance problem using the annotation from active learning. We conducted extensive experiments on synthetic-to-real and real-to-real domain adaptation settings, and the results demonstrate the effectiveness of our method. SS-ADA can achieve or even surpass the accuracy of supervised learning counterpart with only 25\% of target labeled data using real-time segmentation model. The code will be released at \url{https://github.com/ywher/SS-ADA}

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
git clone https://github.com/ywher/TUFL
cd TUFL
```

## Dataset preparation

We only show how to set the Cityscapes-to-CrossCity setting. You can prepare the SYNTHIA-to-Cityscapes and GTA5-to-Cityscapes similarly.

Download Cityscapes and CrossCity dataset, then organize the folder as follows:

```
|TUFL/data
│     ├── cityscapes/   
|     |   ├── gtFine/
|     |   |   ├── train/
|     |   |   ├── val/
|     |   |   ├── test/
|     |   ├── leftImg8bit/
│     ├── NTHU/
|     |   ├── Rio/
|     |   |   ├── Images/
|     |   |   ├── Labels/
|     |   ├── Rome/
|     |   ├── Taipei/
|     |   ├── Tokyo/
      ...
```

## Training and Evaluation example

We use two 1080Ti GPUs for training and one of them for evaluation.

### Train with unsupervised focal loss in Cityscapes-to-Rio setting

First, you need to download the pretrained model of BiSeNet on Cityscapes and put it in the pretrained folder.  (You can also re-train the BiSeNet on Cityscapes.)

(Link: [https://drive.google.com/file/d/1P0G1mcNomCxUqMScKJlzSDpvB9vDf-6u/view?usp=sharing](https://drive.google.com/file/d/1P0G1mcNomCxUqMScKJlzSDpvB9vDf-6u/view?usp=sharing))

```
CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=2 ./unsup/train_paste.py --tensorboard \
--supervision_mode 'unsup' --unsup_loss 'focal' --dataset 'CityScapes' --pretrained_dataset 'CityScapes' \
--freeze_bn --max_iter 2000 --n_img_per_gpu_train 1 --n_img_per_gpu_unsup 1  --uda_confidence_thresh 0.8 --lr_start 0.5e-5 --segmentation_model 'BiSeNet' \
--warm_up_ratio 0.0 --focal_gamma 2 --target_dataset 'CrossCity' --target_city 'Rio' --n_classes 13 --loss_ohem --loss_ohem_ratio 0.5 --unsup_coeff 0.08
```

This training setting would take about 14 minutes using two 1080Ti GPUs. The results will be saved in the TUFL/outputs.

You can refer to TUFL/unsup/parse_args.py for more information about the training setting. More training instructions can be found in train_crosscity.sh

### Evaluation in Cityscapes-to-Rio setting

First, you can download our trained models from Cityscapes to Rio and put them in the **TUFL/outputs**.

(Link: [https://drive.google.com/drive/folders/1YAcjbf7Lt4iER3K-idP_olONO1ybRCP5?usp=sharing](https://drive.google.com/drive/folders/1YAcjbf7Lt4iER3K-idP_olONO1ybRCP5?usp=sharing))

You need to specify the --save_path (model path) and --city (target city name) in the evaluation. Note that the --save_path should be in the **TUFL/outputs** folder.

```
CUDA_VISIBLE_DEVICES=0 python3 ./utils/evaluate.py --dataset 'CrossCity' --dataset_mode 'test' --iou_mode 'cross' --simple_mode \
--save_pth 'CityScapes_BiSeNet_2k_Rio_unsup_focal_0.8_0.08_ohem_0.5/model_epoch_2000.pth' --city 'Rio'
```

More trained models can be found in the following links:

* Cityscapes-to-Rome: [https://drive.google.com/drive/folders/1CO_SxoiLP1lIqm4fBYk1SQnPG1wZNOBn?usp=sharing](https://drive.google.com/drive/folders/1CO_SxoiLP1lIqm4fBYk1SQnPG1wZNOBn?usp=sharing)
* Cityscapes-to-Taipei: [https://drive.google.com/drive/folders/1byndx9ykhg1hOQNlVLqn9znidnj0Rf5C?usp=sharing](https://drive.google.com/drive/folders/1byndx9ykhg1hOQNlVLqn9znidnj0Rf5C?usp=sharing)
* Cityscapes-to-Tokyo: [https://drive.google.com/drive/folders/1NXTiTlXA3pvLVXYcPUgtLZULTqiIqI1W?usp=sharing](https://drive.google.com/drive/folders/1NXTiTlXA3pvLVXYcPUgtLZULTqiIqI1W?usp=sharing)

The results of our trained models are listed in the following. The ss and ms mean single scale and multi scale testing respectively.

Note that we didn't use the Image Style Translation (IST) like CycleGAN and FDA in these experiments. Using IST, class-level threshold adjusment strategy, and Cross-domain Image Mixing (CIM, stage two) would further improve the adaptation results.

| Setting\iteration (mIoU) | 2k iteration(ss/ms) | best iteration     |
| ------------------------ | ------------------- | ------------------ |
| Cityscapes-to-Rome       | 52.27/53.35         | 53.30/53.96 (1.2k) |
| Cityscapes-to-Rio        | 56.39/58.02         | 56.49/58.53 (1.8k) |
| Cityscapes-to-Taipei     | 51.05/52.15         | 51.58/52.65 (1.4k) |
| Cityscapes-to-Tokyo      | 48.59/59.70         | 50.01/51.27 (1k)   |

# Acknowledgement

Some of the code is borrowed from [BiSeNet](https://github.com/CoinCheung/BiSeNet), [DACS](https://github.com/vikolss/DACS)

Thanks a lot for their great work!

# Citation

If you use this code in your research please consider citing our work

```
@ARTICLE{9916201,
  author={Yan, Weihao and Qian, Yeqiang and Wang, Chunxiang and Yang, Ming},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={Threshold-Adaptive Unsupervised Focal Loss for Domain Adaptation of Semantic Segmentation}, 
  year={2023},
  volume={24},
  number={1},
  pages={752-763},
  doi={10.1109/TITS.2022.3210759}}
```

# Contact

Weihao Yan: weihao_yan@outlook.com
