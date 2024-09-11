# Class-Specific Semantic Generation and Reconstruction Learning for Open Set Recognition

Official PyTorch implementation

## 1. Train

Before training, please setup dataset directories in `dataset.py` or `dataset_ood.py`:
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

# Citation
If you find this code useful for your research, please cite our paper.
```
@inproceedings{ijcai2024p226,
  title     = {Class-Specific Semantic Generation and Reconstruction Learning for Open Set Recognition},
  author    = {Liu, Haoyang and Lin, Yaojin and Li, Peipei and Hu, Jun and Hu, Xuegang},
  booktitle = {Proceedings of the Thirty-Third International Joint Conference on
               Artificial Intelligence, {IJCAI-24}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Kate Larson},
  pages     = {2045--2053},
  year      = {2024},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2024/226},
  url       = {https://doi.org/10.24963/ijcai.2024/226},
}
```
