This code is used for IJCAI-2024

Official PyTorch implementation

## 1. Train

Before training, please setup dataset directories in `dataset.py`:
```
DATA_PATH = '../data'
TINYIMAGENET_PATH = DATA_PATH + '/tiny-imagenet-200/'
LARGE_OOD_PATH = '../data/largeoodds'
IMAGENET_PATH = '../data/imagenet2012'
OOD_PATH = '../data/oodds'
```

To train models from scratch, `sh run.sh`

## 2. Present Results

script `collect_metrics.py` helps collect and present experimental results: `python collect_metrics.py`

