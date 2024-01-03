import numpy as np
import argparse

import dataset_ood as dataset
import json
import metrics
import methods.csgrl

from methods import *
import os
import sys
import methods.util as util

import warnings

warnings.filterwarnings('ignore')


def save_everything(subfix="", index=None):
    # save model
    # if subfix == "":
    #     mth.save_model(saving_path + 'model' + str(index) + '.pth')
    # save training process data

    with open(saving_path_cifar100 + 'hist' + str(index) + '.json', 'w') as f:
        json.dump(history_cifar100[index], f)

    with open(saving_path_svhn + 'hist' + str(index) + '.json', 'w') as f:
        json.dump(history_svhn[index], f)



def log_history(index, epoch, data_dict1,data_dict2):

    item_cifar100 = {
        'epoch': epoch
    }
    item_svhn = {
        'epoch': epoch
    }

    item_cifar100.update(data_dict1)
    if isinstance(history_cifar100[index], list):
        history_cifar100[index].append(item_cifar100)
    print(f"Epoch:{epoch}")
    for key in data_dict1.keys():
        print("  ", key, ":", data_dict1[key])

    item_svhn.update(data_dict2)
    if isinstance(history_svhn[index], list):
        history_svhn[index].append(item_svhn)
    print(f"Epoch:{epoch}")
    for key in data_dict2.keys():
        print("  ", key, ":", data_dict2[key])

s_w_num = 21


def training_main():
    tot_epoch = config['epoch_num']
    global best_acc, best_auroc

    for epoch in range(mth.epoch, tot_epoch):
        sys.stdout.flush()
        losses = mth.train_epoch()
        acc = 0
        auroc = 0
        if epoch % 1 == 0:
            for i in range(s_w_num):
                save_everything(f'ckpt{epoch}', index=i)

        if epoch % test_interval == test_interval - 1:
            # big test with aurocs
            scores_cifar100, thresh_cifar100, pred_cifar100 = mth.knownpred_unknwonscore_test(test_loader_cifar100)
            scores_svhn, thresh_svhn, pred_svhn = mth.knownpred_unknwonscore_test(test_loader_svhn)
            acc_cifar100 = evaluation_cifar100.close_accuracy(pred_cifar100)
            acc_svhn = evaluation_svhn.close_accuracy(pred_svhn)
            for i in range(s_w_num):
                log_history(i, epoch, {
                    "loss": losses,
                    "close acc": acc_cifar100,
                    "open_detection": evaluation_cifar100.open_detection_indexes(scores_cifar100[i], thresh_cifar100),
                    "open_reco": evaluation_cifar100.open_reco_indexes(scores_cifar100[i], thresh_cifar100, pred_cifar100)
                }, {
                    "loss": losses,
                    "close acc": acc_svhn,
                    "open_detection": evaluation_svhn.open_detection_indexes(scores_svhn[i], thresh_svhn),
                    "open_reco": evaluation_svhn.open_reco_indexes(scores_svhn[i], thresh_svhn, pred_svhn)
                })
                save_everything(index=i)


def update_config_keyvalues(config, update):
    if update == "":
        return config
    spls = update.split(",")
    for spl in spls:
        key, val = spl.split(':')
        key_parts = key.split('.')
        sconfig = config
        for i in range(len(key_parts) - 1):
            sconfig = sconfig[key_parts[i]]
        org = sconfig[key_parts[-1]]
        if isinstance(org, bool):
            sconfig[key_parts[-1]] = val == 'True'
        elif isinstance(org, int):
            sconfig[key_parts[-1]] = int(val)
        elif isinstance(org, float):
            sconfig[key_parts[-1]] = float(val)
        else:
            sconfig[key_parts[-1]] = val
        print("Updating", key, "with", val, "results in", sconfig[key_parts[-1]])
    return config


def update_subconfig(cfg, u):
    for k in u.keys():
        if not k in cfg.keys() or not isinstance(cfg[k], dict):
            cfg[k] = u[k]
        else:
            update_subconfig(cfg[k], u[k])


def load_config(file):
    with open(file, "r") as f:
        config = json.load(f)
    if 'inherit' in config.keys():
        inheritfile = config['inherit']
        if inheritfile != 'None':
            parent = load_config(inheritfile)
            update_subconfig(parent, config)
            config = parent
    return config


def set_up_gpu(args):
    if args.gpu != 'cpu':
        args.gpu = ",".join([c for c in args.gpu])
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


if __name__ == "__main__":
    import torch

    util.setup_seed(0)
    # torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, required=False, default="0", help='GPU number')
    parser.add_argument('--ds', type=str, required=False, default="./exps/cifar10/ood_spl.json",
                        help='dataset setting, choose file from ./exps')
    parser.add_argument('--config', type=str, required=False, default="./configs/pcssr/cifar10.json",
                        help='model configuration, choose from ./configs')
    parser.add_argument('--save', type=str, required=False, default="_m0", help='Saving folder name')
    parser.add_argument('--method', type=str, required=False, default="cssr",
                        help='Methods : ' + ",".join(util.method_list.keys()))
    parser.add_argument('--configupdate', type=str, required=False, default="",
                        help='Update several key values in config')
    parser.add_argument('--test_interval', type=int, required=False, default=1,
                        help='The frequency of model evaluation')

    args = parser.parse_args()

    test_interval = args.test_interval
    if not args.save.endswith("/"):
        args.save += "/"

    set_up_gpu(args)

    saving_path_cifar100 = "./save/cifar10/cifar100" + args.save + "/"
    saving_path_svhn = "./save/cifar10/svhn" + args.save + "/"
    util.setup_dir(saving_path_cifar100)
    util.setup_dir(saving_path_svhn)

    if args.config != "None":
        config = load_config(args.config)
    else:
        config = {}
    config = update_config_keyvalues(config, args.configupdate)
    args.bs = config['batch_size']
    print('Config:', config)

    train_loader,test_loader_cifar100,test_loader_svhn,classnum = dataset.load_partitioned_dataset(args, args.ds)
    mth = util.method_list[args.method](config, classnum, train_loader.dataset)

    history_cifar100 = [[] for _ in range(s_w_num)]
    history_svhn = [[] for _ in range(s_w_num)]
    evaluation_cifar100 = metrics.OSREvaluation(test_loader_cifar100)
    evaluation_svhn = metrics.OSREvaluation(test_loader_svhn)

    print(f"TotalEpochs:{config['epoch_num']}")
    training_main()


