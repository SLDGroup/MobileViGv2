# MobileViGv2

## Scaling Graph Convolutions for Mobile Vision
[PDF](https://openaccess.thecvf.com/content/CVPR2024W/MAI/html/Avery_Scaling_Graph_Convolutions_for_Mobile_Vision_CVPRW_2024_paper.html) | [Arxiv](https://arxiv.org/abs/2406.05850)

William Avery, Mustafa Munir, and Radu Marculescu

# Overview
This repository contains the source code for Scaling Graph Convolutions for Mobile Vision


```
python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --use_env main.py --data-path ../../Datasets/ILSVRC/Data/CLS-LOC/ --model mobilevigv2 --output_dir mobilevigV2_test_results
```


# Pretrained Models

Weights trained on ImageNet-1K, COCO 2017 Object Detection and Instance Segmentation, and ADE20K Semantic Segmentation can be downloaded [here](https://huggingface.co/SLDGroup/MobileViGv2/tree/main). 

### detection
Contains all of the object detection and instance segmentation backbone code and config.

### segmentation
Contains all of the semantic segmentation backbone code and config.

### models
Contains the main MobileViGv2 model code.

### util
Contains utility scripts.

# Usage

## Installation Image Classification

```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
conda install mpi4py
pip install -r requirements.txt
```

## Image Classification

### Train image classification:
```
python -m torch.distributed.launch --nproc_per_node=num_GPUs --nnodes=num_nodes --use_env main.py --data-path /path/to/imagenet --model mobilevigv2_model --output_dir mobilevigv2_results
```
For example:
```
python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --use_env main.py --data-path ../../Datasets/ILSVRC/Data/CLS-LOC/ --model mobilevigv2_m --output_dir mobilevigv2_test_results
```
### Test image classification:
```
python -m torch.distributed.launch --nproc_per_node=num_GPUs --nnodes=num_nodes --use_env main.py --data-path /path/to/imagenet --model mobilevigv2_model --resume pretrained_model --eval
```
For example:
```
python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --use_env main.py --data-path ../../Datasets/ILSVRC/Data/CLS-LOC/ --model mobilevigv2_s --resume pretrained_model --eval
```

## Installation Object Detection and Instance Segmentation
```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install timm submitit mmcv-full mmdet==2.28
pip install -U openmim
```

## Object Detection and Instance Segmentation

Detection and instance segmentation on MS COCO 2017 is implemented based on [MMDetection](https://github.com/open-mmlab/mmdetection). We follow settings and hyper-parameters of [PVT](https://github.com/whai362/PVT/tree/v2/segmentation), [PoolFormer](https://github.com/sail-sg/poolformer), and [EfficientFormer](https://github.com/snap-research/EfficientFormer) for comparison. 

All commands for object detection and instance segmentation should be run from the /detection directory.

### Data preparation

Prepare COCO 2017 dataset according to the [instructions in MMDetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/1_exist_data_model.md#test-existing-models-on-standard-datasets).

### ImageNet Pretraining
Put ImageNet-1K pretrained weights of backbone as 
```
MobileViGv2
├── Results
│   ├── model
│   │   ├── model.pth
│   │   ├── ...
```

### Train object detection and instance segmentation:
```
python -m torch.distributed.launch --nproc_per_node num_GPUs --nnodes=num_nodes --node_rank 0 main.py configs/mask_rcnn_mobilevigv2_model --mobilevigv2_model mobilevigv2_model --work-dir Output_Directory --launcher pytorch > Output_Directory/log_file.txt 
```
For example:
```
python -m torch.distributed.launch --nproc_per_node 2 --nnodes 1 --node_rank 0 main.py configs/mask_rcnn_mobilevigv2_m_fpn_1x_coco.py --mobilevigv2_model mobilevigv2_m --work-dir detection_results/ --launcher pytorch > detection_results/mobilevigv2_m_run_test.txt 
```
### Test object detection and instance segmentation:
```
python -m torch.distributed.launch --nproc_per_node=num_GPUs --nnodes=num_nodes --node_rank 0 test.py configs/mask_rcnn_mobilevigv2_model --checkpoint Pretrained_Model --eval {bbox or segm} --work-dir Output_Directory --launcher pytorch > log_file.txt
```
For example:
```
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank 0 test.py configs/mask_rcnn_mobilevigv2_m_fpn_1x_coco.py --checkpoint Pretrained_Model.pth --eval bbox --work-dir detection_results/ --launcher pytorch > detection_results/mobilevigv2_m_run_evaluation.txt
```

## Installation Semantic Segmentation
```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
```
```
pip install -U openmim
mim install mmengine
mim install mmcv-full
```
```
mim install "mmsegmentation <=0.30.0"
```

### Train semantic segmentation:

Semantic segmentation on ADE20K is implemented based on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation). We follow settings and hyper-parameters of [PVT](https://github.com/whai362/PVT/tree/v2/segmentation), [PoolFormer](https://github.com/sail-sg/poolformer), and [EfficientFormer](https://github.com/snap-research/EfficientFormer) for comparison. 

```
python -m torch.distributed.launch --nproc_per_node 2 --nnodes 1 --node_rank 0 tools/train.py configs/sem_fpn/fpn_mobilevigv2_m_ade20k_40k.py --mobilevigv2_model mobilevigv2_m --work-dir semantic_results/ --launcher pytorch > semantic_results/mobilevigv2_m_run_semantic.txt
```

### Citation

If our code or models help your work, please cite MobileViG (CVPRW 2023), MobileViGv2 (CVPRW 2024), and GreedyViG (CVPR 2024):

```
@InProceedings{MobileViGv2_2024,
    author    = {Avery, William and Munir, Mustafa and Marculescu, Radu},
    title     = {Scaling Graph Convolutions for Mobile Vision},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2024},
    pages     = {5857-5865}
}
```

```
@InProceedings{mobilevig2023,
    author    = {Munir, Mustafa and Avery, William and Marculescu, Radu},
    title     = {MobileViG: Graph-Based Sparse Attention for Mobile Vision Applications},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2023},
    pages     = {2211-2219}
}
```

```
@InProceedings{GreedyViG_2024_CVPR,
    author    = {Munir, Mustafa and Avery, William and Rahman, Md Mostafijur and Marculescu, Radu},
    title     = {GreedyViG: Dynamic Axial Graph Construction for Efficient Vision GNNs},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {6118-6127}
}
```
