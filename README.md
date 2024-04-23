This code is used for IJCAI-2024

Official PyTorch implementation

## 1. Train

Before training, please setup dataset directories in `dataset.py`:
```
DATA_PATH = '../data'                                   # path for cifar10, svhn
TINYIMAGENET_PATH = DATA_PATH + '/tiny-imagenet-200/'   # path for tinyimagenet 
OOD_PATH = '../data/oodds'                              # path for ood datasets
```

To train models from scratch, `sh run.sh`

## 2. Present Results

script `collect_metrics.py` helps collect and present experimental results: `python collect_metrics.py`

## 3.  Extend to Other Types of Data

Modify `class Backbone` in `csgrl.py`
