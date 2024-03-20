This code is used for IJCAI-2024

Official PyTorch implementation

## 1. Train

Before training, please setup dataset directories in `dataset.py`:
```
DATA_PATH = ''          # path for cifar10, svhn
TINYIMAGENET_PATH = ''  # path for tinyimagenet
```

To train models from scratch, run run.sh

## 2. Evaluation

Add `--test` on training commands to restore and evaluate a pretrained model on specified data setup, e.g.,
```
python main.py --gpu 0 --ds {DATASET} --config {MODEL} --save {SAVING_NAME} --method cssr --test
```

With models trained by `sh run.sh`, script `collect_metrics.py` helps collect and present experimental results: `python collect_metrics.py`

