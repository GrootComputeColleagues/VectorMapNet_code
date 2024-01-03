# VectorMapNet_code
**VectorMapNet: End-to-end Vectorized HD Map Learning** ICML 2023

This is the official codebase of VectorMapNet


[Yicheng Liu](https://scholar.google.com/citations?user=vRmsgQUAAAAJ&hl=zh-CN), Yuantian Yuan, [Yue Wang](https://people.csail.mit.edu/yuewang/), [Yilun Wang](https://scholar.google.com.hk/citations?user=nUyTDosAAAAJ&hl=en/), [Hang Zhao](http://people.csail.mit.edu/hangzhao/)


**[[Paper](https://arxiv.org/pdf/2206.08920.pdf)] [[Project Page](https://tsinghua-mars-lab.github.io/vectormapnet/)]**

**Abstract:**
Autonomous driving systems require High-Definition (HD) semantic maps to navigate around urban roads. Existing solutions approach the semantic mapping problem by offline manual annotation, which suffers from serious scalability issues.  Recent learning-based methods produce dense rasterized segmentation predictions to construct maps. However, these predictions do not include instance information of individual map elements and require heuristic post-processing to obtain vectorized maps. To tackle these challenges, we introduce an end-to-end vectorized HD map learning pipeline, termed VectorMapNet. VectorMapNet takes onboard sensor observations and predicts a sparse set of polylines in the bird's-eye view. This pipeline can explicitly model the spatial relation between map elements and generate vectorized maps that are friendly to downstream autonomous driving tasks. Extensive experiments show that VectorMapNet achieve strong map learning performance on both nuScenes and Argoverse2 dataset, surpassing previous state-of-the-art methods by 14.2 mAP and 14.6mAP. Qualitatively, VectorMapNet is capable of generating comprehensive maps and capturing fine-grained details of road geometry. To the best of our knowledge, VectorMapNet is the first work designed towards end-to-end vectorized map learning from onboard observations. 

**Questions/Requests:** 
Please file an [issue](https://github.com/Tsinghua-MARS-Lab/vecmapnet/issues) or send an email to [Yicheng](moooooore66@gmail.com).


## Bibtex
If you found this paper or codebase useful, please cite our paper:
```
@inproceedings{liu2022vectormapnet,
        title={VectorMapNet: End-to-end Vectorized HD Map Learning},
        author={Liu, Yicheng and Yuantian, Yuan and Wang, Yue and Wang, Yilun and Zhao, Hang},
        booktitle={International conference on machine learning},
        year={2023},
        organization={PMLR}
    }
```


# Run VectorMapNet

## Note


## 0. Environment
```
conda create --name hdmap-opensource python==3.8
conda activate hdmap-opensource
```
0.a : Install Pytorch
```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```
0.b : Install MMCV-series aka open-mmlab
```
# Install mmcv-series
pip install mmcv-full==1.3.9 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
```
0.c  : Install mmdetection3d
0.c1 : export cuda environment in bashrc
```
export CUDA_HOME=/usr/local/cuda
```
0.c2 : Install mmdetection3d
```
cd
wget https://github.com/open-mmlab/mmdetection3d/archive/refs/tags/v0.17.3.zip
unzip v0.17.3.zip
cd  mmdetection3d-0.17.3
pip install -v -e .
```

0.d  : Install requirements.txt
cd to your VectorMapNet_code Directory
```
pip install -r requirements.txt
```

## 1. Prepare your dataset

Store your data with following structure:

```
    root of VectorMapNet
        |--datasets
            |--nuScenes
            |--Argoverse2(optional)

```

### 1.1 Generate annotation files
This just generates annotation .pkl. I gotta find out what tho

#### Preprocess nuScenes

```
python tools/data_converter/nuscenes_converter.py --data-root dataset/nuScenes/ --version v1.0-mini
```

## 2. Evaluate VectorMapNet
You can skip this step if u wish
### Download Checkpoint
| Method       | Modality    | Config | Checkpoint |
|--------------|-------------|--------|------------|
| VectorMapNet | Camera only | [config](configs/vectormapnet.py) | [model link](https://drive.google.com/file/d/1ccrlZ2HrFfpBB27kC9DkwCYWlTUpgmin/view?usp=sharing)      |


### Train VectorMapNet

In single GPU
```
python tools/train.py configs/vectormapnet.py
```

For multi GPUs
```
bash tools/dist_train.sh configs/vectormapnet.py $num_gpu
```


### Do Evaluation

In single GPU
```
python tools/test.py configs/vectormapnet.py /path/to/ckpt --eval name
```

For multi GPUs
```
bash tools/dist_test.sh configs/vectormapnet.py /path/to/ckpt $num_gpu --eval name
```


### Expected Results

| $AP_{ped}$   | $AP_{divider}$ | $AP_{boundary}$ | mAP   |
|--------------|----------------|-----------------|-------|
| 39.8 | 47.7    | 38.8          | 42.1 |


