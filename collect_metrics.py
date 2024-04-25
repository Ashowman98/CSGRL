import json

import numpy as np

# margin = [5, 10, 15]
margin = [5]
datasets = ['cifar10']
splits = ['cifar100','svhn']
s_w = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
       1]
s_w = list(map(lambda x: '%.02f' % x, s_w))
s_w_num=len(s_w)

metric_name = 'open_detection.auroc'
# metric_name = 'open_reco.macro_f1'
# metric_name = 'close acc'
# metric_name = 'open_reco.oscr'
# metric_name = 'open_detection.bestdetacc'
# metric_name = 'open_detection.auprIN'
# metric_name = 'open_detection.auprOUT'

def get_metric(file,metric,is_last = True):
    with open(file,'r') as f:
        hist = json.load(f)
    # last metric
    if is_last:
        res = hist[-1]
        for m in metric.split('.'):
            if m not in res:
                res = -1
                break
            res = res[m]
    # best metric
    else:
        res = 0
        for epoch in hist:
            if epoch['epoch'] >= 285:
                break
            for m in metric.split('.'):
                if not m in epoch:
                    epoch = -1
                    break
                epoch = epoch[m]
            res = max(res,epoch)
    return res

def generate_tables(use_last = True, use_best = False):
    for ds in datasets:
        print("\nDataset",ds,"Last Epoch" if use_last else "Best Epoch")
        print('method','average',*splits,sep='\t')
        s_w_res = []
        for mth in margin:
            for i in range(s_w_num):
                if i in range(0,14) or i in range(20,21):
                # if i != 0:  # baseline+gen,without unkonwn errors
                    continue
                metrics = [get_metric(f'./save/{ds}/{s}_m'+str(mth)+'/hist'+str(i)+'.json', metric_name, use_last) for s in splits]
                # print(metrics)
                s_w_i = np.array(metrics)
                s_w_res.append(s_w_i)
                if not use_best:
                    avg = sum(metrics) / len(metrics)
                    metrics = [avg] + metrics
                    metrics = list(map(lambda x: '%.04f' % x, metrics))
                    print(str(s_w[i]), *metrics, sep='\t')
        s_w_res = np.stack(s_w_res, axis=0)
        best_res = np.max(s_w_res, axis=0)
        best_avg = sum(best_res) / len(best_res)
        best_res = list([best_avg]) + list(best_res)
        best_res = list(map(lambda x: '%.04f' % x, best_res))
        print('best', *best_res, sep='\t')




# generate_tables(True, True)
generate_tables(False, True)
