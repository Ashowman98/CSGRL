import math

import methods.wideresnet as wideresnet
from methods.augtools import HighlyCustomizableAugment, RandAugmentMC
import methods.util as util
import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
import random
from methods.util import AverageMeter
import time
from torchvision.transforms import transforms
from methods.resnet import ResNet
from torchvision import models as torchvision_models
from methods.gen_reco import *
import scipy.stats as st


class PretrainedResNet(nn.Module):

    def __init__(self, rawname, pretrain_path=None) -> None:
        super().__init__()
        if pretrain_path == 'default':
            self.model = torchvision_models.__dict__[rawname](pretrained=True)
            self.output_dim = self.model.fc.weight.shape[1]
            self.model.fc = nn.Identity()
        else:
            self.model = torchvision_models.__dict__[rawname]()
            self.output_dim = self.model.fc.weight.shape[1]
            self.model.fc = nn.Identity()
            if pretrain_path is not None:
                sd = torch.load(pretrain_path)
                self.model.load_state_dict(sd, strict=True)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        return x


class Backbone(nn.Module):

    def __init__(self, config, inchan):
        super().__init__()

        if config['backbone'] == 'wideresnet28-2':
            self.backbone = wideresnet.WideResNetBackbone(None, 28, 2, 0, config['category_model']['projection_dim'])
        elif config['backbone'] == 'wideresnet40-4':
            self.backbone = wideresnet.WideResNetBackbone(None, 40, 4, 0, config['category_model']['projection_dim'])
        elif config['backbone'] == 'wideresnet16-8':
            self.backbone = wideresnet.WideResNetBackbone(None, 16, 8, 0.4, config['category_model']['projection_dim'])
        elif config['backbone'] == 'wideresnet28-10':
            self.backbone = wideresnet.WideResNetBackbone(None, 28, 10, 0.3, config['category_model']['projection_dim'])
        elif config['backbone'] == 'resnet18':
            self.backbone = ResNet(output_dim=config['category_model']['projection_dim'], inchan=inchan)
        elif config['backbone'] == 'resnet18a':
            self.backbone = ResNet(output_dim=config['category_model']['projection_dim'], resfirststride=2,
                                   inchan=inchan)
        elif config['backbone'] == 'resnet18b':
            self.backbone = ResNet(output_dim=config['category_model']['projection_dim'], resfirststride=2,
                                   inchan=inchan)
        elif config['backbone'] == 'resnet34':
            self.backbone = ResNet(output_dim=config['category_model']['projection_dim'], num_block=[3, 4, 6, 3],
                                   inchan=inchan)
        elif config['backbone'] in ['prt_r18', 'prt_r34', 'prt_r50']:
            self.backbone = PretrainedResNet(
                {'prt_r18': 'resnet18', 'prt_r34': 'resnet34', 'prt_r50': 'resnet50'}[config['backbone']])
        elif config['backbone'] in ['prt_pytorchr18', 'prt_pytorchr34', 'prt_pytorchr50']:
            name, path = {
                'prt_pytorchr18': ('resnet18', 'default'),
                'prt_pytorchr34': ('resnet34', 'default'),
                'prt_pytorchr50': ('resnet50', 'default')
            }[config['backbone']]
            self.backbone = PretrainedResNet(name, path)
        elif config['backbone'] in ['prt_dinor18', 'prt_dinor34', 'prt_dinor50']:
            name, path = {
                'prt_dinor50': ('resnet50', './model_weights/dino_resnet50_pretrain.pth')
            }[config['backbone']]
            self.backbone = PretrainedResNet(name, path)
        else:
            bkb = config['backbone']
            raise Exception(f'Backbone \"{bkb}\" is not defined.')

        # types : ae_softmax_avg , ae_avg_softmax , avg_ae_softmax
        self.output_dim = self.backbone.output_dim
        # self.classifier = CRFClassifier(self.backbone.output_dim,numclss,config)

    def forward(self, x):
        x = self.backbone(x)
        # latent , global prob , logits
        return x


class LinearClassifier(nn.Module):

    def __init__(self, inchannels, num_class, config):
        super().__init__()
        self.gamma = config['gamma']
        self.cls = nn.Conv2d(inchannels, num_class, 1, padding=0, bias=False)

    def forward(self, x):
        x = self.cls(x)
        return x * self.gamma


def sim_conv_layer(input_channel, output_channel, kernel_size=1, padding=0, use_activation=True):
    if use_activation:
        res = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, padding=padding, bias=False),
            nn.Tanh())
    else:
        res = nn.Conv2d(input_channel, output_channel, kernel_size, padding=padding, bias=False)
    return res


class AutoEncoder(nn.Module):

    def __init__(self, inchannel, hidden_layers, latent_chan):
        super().__init__()
        layer_block = sim_conv_layer
        self.latent_size = latent_chan
        if latent_chan > 0:
            self.encode_convs = []
            self.decode_convs = []
            for i in range(len(hidden_layers)):
                h = hidden_layers[i]
                ecv = layer_block(inchannel, h, )
                dcv = layer_block(h, inchannel, use_activation=i != 0)
                inchannel = h
                self.encode_convs.append(ecv)
                self.decode_convs.append(dcv)
            self.encode_convs = nn.ModuleList(self.encode_convs)
            self.decode_convs.reverse()
            self.decode_convs = nn.ModuleList(self.decode_convs)
            self.latent_conv = layer_block(inchannel, latent_chan)
            self.latent_deconv = layer_block(latent_chan, inchannel, use_activation=(len(hidden_layers) > 0))
        else:
            self.center = nn.Parameter(torch.rand([inchannel, 1, 1]), True)

    def forward(self, x):
        if self.latent_size > 0:
            output = x
            for cv in self.encode_convs:
                output = cv(output)
            latent = self.latent_conv(output)
            output = self.latent_deconv(latent)
            for cv in self.decode_convs:
                output = cv(output)
            return output, latent
        else:
            return self.center, self.center


class CSGRLClassifier(nn.Module):

    def __init__(self, inchannels, num_class, config):
        super().__init__()
        ae_hidden = config['ae_hidden']
        ae_latent = config['ae_latent']
        self.class_aes = []
        for i in range(num_class + 1):
            ae = AutoEncoder(inchannels, ae_hidden, ae_latent)
            self.class_aes.append(ae)
        self.class_aes = nn.ModuleList(self.class_aes)
        # self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.useL1 = config['error_measure'] == 'L1'

        self.reduction = -1 if config['model'] == 'pcssr' else 1
        self.reduction *= config['gamma']

    def ae_error(self, rc, x):
        if self.useL1:
            # return torch.sum(torch.abs(rc-x) * self.reduction,dim=1,keepdim=True)
            return torch.norm(rc - x, p=1, dim=1, keepdim=True) * self.reduction
        else:
            return torch.norm(rc - x, p=2, dim=1, keepdim=True) ** 2 * self.reduction

    clip_len = 100

    def forward(self, x):
        cls_ers = []
        for i in range(len(self.class_aes)):
            rc, lt = self.class_aes[i](x)
            cls_er = self.ae_error(rc, x)
            if CSGRLClassifier.clip_len > 0:
                cls_er = torch.clamp(cls_er, -CSGRLClassifier.clip_len, CSGRLClassifier.clip_len)
            cls_ers.append(cls_er)
        logits = torch.cat(cls_ers, dim=1)
        return logits




class BackboneAndClassifier(nn.Module):

    def __init__(self, num_classes, config):
        super().__init__()
        clsblock = {'linear': LinearClassifier, 'pcssr': CSGRLClassifier, 'rcssr': CSGRLClassifier}
        self.backbone = Backbone(config, 3)
        cat_config = config['category_model']
        self.cat_cls = clsblock[cat_config['model']](self.backbone.output_dim, num_classes, cat_config)

    def forward(self, x, feature_only=False, isgen=False):
        if not isgen:
            x = self.backbone(x)
        if feature_only:
            return x
        xcls_raw = self.cat_cls(x)
        # x_com = self.reco(x)
        return x, xcls_raw


class CSGRLModel(nn.Module):

    def __init__(self, num_classes, config):
        super().__init__()

        # ------ New Arch
        self.backbone_cs = BackboneAndClassifier(num_classes, config)

        self.config = config
        self.mins = {i: [] for i in range(num_classes)}
        self.maxs = {i: [] for i in range(num_classes)}
        self.num_classes = num_classes

    def forward(self, x, reqfeature=False, isgen=False):

        # ----- New Arch
        x = self.backbone_cs(x, feature_only=reqfeature, isgen=isgen)
        if reqfeature:
            return x
        x, xcls_raw = x
        return x, xcls_raw


class CSGRLCriterion(nn.Module):

    def get_onehot_label(self, y, clsnum):
        y = torch.reshape(y, [-1, 1]).long()
        return torch.zeros(y.shape[0], clsnum).cuda().scatter_(1, y, 1)

    def __init__(self, avg_order, enable_sigma=True):
        super().__init__()
        self.avg_order = {"avg_softmax": 1, "softmax_avg": 2}[avg_order]
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.enable_sigma = enable_sigma

    def forward(self, x, y=None, prob=False, pred=False):
        if self.avg_order == 1:
            g = self.avg_pool(x).view(x.shape[0], -1)
            g = torch.softmax(g, dim=1)
        elif self.avg_order == 2:
            g = torch.softmax(x, dim=1)
            g = self.avg_pool(g).view(x.size(0), -1)
        if prob: return g
        if pred: return torch.argmax(g, dim=1)
        loss = -torch.sum(self.get_onehot_label(y, g.shape[1]) * torch.log(g), dim=1).mean()
        # if torch.isinf(loss) or torch.isnan(loss):
        #     print(1)
        return loss


def manual_contrast(x):
    s = random.uniform(0.1, 2)
    return x * s


class WrapDataset(data.Dataset):

    def __init__(self, labeled_ds, config, inchan_num=3) -> None:
        super().__init__()
        self.labeled_ds = labeled_ds

        __mean = [0.5, 0.5, 0.5][:inchan_num]
        __std = [0.25, 0.25, 0.25][:inchan_num]

        trans = [transforms.RandomHorizontalFlip()]
        if config['cust_aug_crop_withresize']:
            trans.append(transforms.RandomResizedCrop(size=util.img_size, scale=(0.25, 1)))
        elif util.img_size > 200:
            trans += [transforms.Resize(256), transforms.RandomResizedCrop(util.img_size)]
        else:
            trans.append(transforms.RandomCrop(size=util.img_size,
                                               padding=int(util.img_size * 0.125),
                                               padding_mode='reflect'))
        if config['strong_option'] == 'RA':
            trans.append(RandAugmentMC(n=2, m=10))
        elif config['strong_option'] == 'CUST':
            trans.append(HighlyCustomizableAugment(2, 10, -1, labeled_ds, config))
        elif config['strong_option'] == 'NONE':
            pass
        else:
            raise NotImplementedError()
        trans += [transforms.ToTensor(),
                  transforms.Normalize(mean=__mean, std=__std)]

        if config['manual_contrast']:
            trans.append(manual_contrast)
        strong = transforms.Compose(trans)

        if util.img_size > 200:
            self.simple = [transforms.RandomResizedCrop(util.img_size)]
        else:
            self.simple = [transforms.RandomCrop(size=util.img_size,
                                                 padding=int(util.img_size * 0.125),
                                                 padding_mode='reflect')]
        self.simple = transforms.Compose(([transforms.RandomHorizontalFlip()]) + self.simple + [
            transforms.ToTensor(),
            transforms.Normalize(mean=__mean, std=__std)] + ([manual_contrast] if config['manual_contrast'] else []))

        self.test_normalize = transforms.Compose([
            transforms.CenterCrop(util.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=__mean, std=__std)])

        td = {'strong': strong, 'simple': self.simple}
        self.aug = td[config['cat_augmentation']]
        self.test_mode = False

    def __len__(self) -> int:
        return len(self.labeled_ds)

    def __getitem__(self, index: int):
        img, lb, _ = self.labeled_ds[index]
        if self.test_mode:
            img = self.test_normalize(img)
        else:
            img = self.aug(img)
        return img, lb, index


@util.regmethod('csgrl')
class CSGRLMethod:

    def get_cfg(self, key, default):
        return self.config[key] if key in self.config else default

    def __init__(self, config, clssnum, train_set) -> None:
        self.config = config
        self.epoch = 0
        self.epoch_num = config['epoch_num']
        self.lr = config['learn_rate']
        self.lrG = config['learn_rateG']
        self.batch_size = config['batch_size']
        self.margin = config['margin']

        self.clsnum = clssnum
        self.crt = CSGRLCriterion(config['arch_type'], False)
        self.crtR = CriterionR()
        self.model = CSGRLModel(self.clsnum, config).cuda()
        self.modelopt = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=5e-4)

        self.crtG = CriterionG(clssnum).cuda()
        self.modelG = Generator_Class(self.clsnum, nc=self.model.backbone_cs.backbone.output_dim).cuda()
        self.modelG.apply(weights_init)
        self.modeloptG = torch.optim.SGD(self.modelG.parameters(), lr=self.lrG, weight_decay=5e-4)

        self.wrap_ds = WrapDataset(train_set, self.config, inchan_num=3, )
        self.wrap_loader = data.DataLoader(self.wrap_ds,
                                           batch_size=self.config['batch_size'], shuffle=True, pin_memory=True,
                                           num_workers=6)
        self.lr_schedule = util.get_scheduler(self.config, self.wrap_loader)

        self.prepared = -999

    def train_epoch(self):
        data_time = AverageMeter()
        batch_time = AverageMeter()
        train_acc = AverageMeter()

        running_loss = AverageMeter()
        running_lossG = AverageMeter()
        lossG = torch.tensor([0])

        self.model.train()
        self.modelG.train()

        endtime = time.time()
        max_dis = [0] * self.clsnum
        for i, data in enumerate(tqdm.tqdm(self.wrap_loader)):
            data_time.update(time.time() - endtime)

            self.lr = self.lr_schedule.get_lr(self.epoch, i, self.lr)
            self.lrG = self.lr_schedule.get_lr(self.epoch, i, self.lrG)
            util.set_lr([self.modelopt], self.lr)
            util.set_lr([self.modeloptG], self.lrG)

            sx, lb = data[0].cuda(), data[1].cuda()

            # lossC = self.crt(close_er, lb)
            # self.modelopt.zero_grad()
            # lossC.backward()
            # self.modelopt.step()
            if (self.epoch_num - self.epoch) <= 50:
                noise = []
                for c in range(self.clsnum):
                    noise.append(torch.randn(math.ceil(self.config['batch_size'] / self.clsnum), 100, 1, 1).cuda())
                gen_data, gen_label = self.modelG(noise)
                x, close_er = self.model(sx)
                # x_act = torch.unsqueeze(torch.abs(x).mean(dim=3).mean(dim=2).mean(dim=1), dim=1)
                max_dis = self.class_maximum_distance(
                    -close_er.reshape([close_er.shape[0], close_er.shape[1], -1]).mean(dim=2), lb, self.clsnum, max_dis)

                gen_x, gen_close_er = self.model(gen_data, isgen=True)
                # gen_x_act = torch.abs(gen_x).mean(dim=1).view(gen_x.shape[0],-1)
                lossG1 = self.crt(gen_close_er, gen_label)
                score = -torch.squeeze(gen_close_er)
                lossG2 = self.crtG(score, gen_label, max_dis, self.margin)
                self.modeloptG.zero_grad()
                lossG = lossG1 + lossG2
                # if torch.isnan(lossG)or torch.isinf(lossD):
                #     print(1)
                lossG.backward()
                self.modeloptG.step()

            if (self.epoch_num - self.epoch) > 50:
                x, close_er = self.model(sx)
                lossD = self.crt(close_er, lb)
            else:
                # x, close_er = self.model(sx)
                gen_data, gen_label = self.modelG(noise)
                gen_x, gen_close_er = self.model(gen_data.detach(), isgen=True)
                lossD1 = self.crt(close_er, lb)
                lossD2 = self.crt(gen_close_er, (torch.ones(gen_label.shape[0]) * self.clsnum).cuda())
                lossD = lossD1 + lossD2
            # if torch.isnan(lossD) or torch.isinf(lossD):
            #     print(1)

            self.modelopt.zero_grad()
            lossD.backward()
            self.modelopt.step()

            # gen_data, gen_label = self.modelG(noise)


            pred = self.crt(close_er, pred=True).cpu().numpy()
            # total_loss = loss + lossC

            nplb = data[1].numpy()
            train_acc.update((pred == nplb).sum() / pred.shape[0], pred.shape[0])
            running_loss.update(lossD.item())
            running_lossG.update(lossG.item())
            batch_time.update(time.time() - endtime)
            endtime = time.time()
        self.epoch += 1
        training_res = \
            {"LossD": running_loss.avg,
             "LossG": running_lossG.avg,
             "TrainAcc": train_acc.avg,
             "Learn Rate": self.lr,
             "DataTime": data_time.avg,
             "BatchTime": batch_time.avg}

        return training_res

    def known_prediction_test(self, test_loader):
        self.model.eval()
        self.modelG.eval()
        pred, scores, _, _ = self.scoring(test_loader)
        return pred

    def scoring(self, loader, prepare=False):
        gts = []
        # deviations = []

        know_scores = []
        unknow_scores = []
        prediction = []
        close_probs = []
        with torch.no_grad():
            for d in tqdm.tqdm(loader):
                x1 = d[0].cuda(non_blocking=True)
                gt = d[1].numpy()
                x_a, close_er = self.model(x1)
                pred = self.crt(close_er[:, 0:-1, :, :], pred=True).cpu().numpy()
                know_er = close_er[:, 0:-1]
                unknow_er = close_er[:, -1].cpu().numpy()
                know_max_er = know_er.cpu().numpy()[[i for i in range(pred.shape[0])], pred]
                # know_mean_er = know_er.mean(dim=1).cpu().numpy()
                # x_act = torch.abs(x_a).mean(dim=1).cpu().numpy()
                know_score = know_max_er
                unknow_score = unknow_er
                know_scores.append(know_score.reshape([know_score.shape[0], -1]).mean(axis=1))
                unknow_scores.append(unknow_score.reshape([unknow_score.shape[0], -1]).mean(axis=1))
                # close_prob = []

                # close_prob = 1 - prob[:, -1]
                prediction.append(pred)
                # scores.append(scr)
                gts.append(gt)
                # close_probs.append(close_prob)
        know_scores = np.concatenate(know_scores)
        unknow_scores = np.concatenate(unknow_scores)
        know_scores_stand = (know_scores - np.mean(know_scores)) / np.std(know_scores)
        unknow_scores_stand = (unknow_scores - np.mean(unknow_scores)) / np.std(unknow_scores)
        for j in range(len(CSGRLMethod.s_w)):
            close_prob_j = (1 - CSGRLMethod.s_w[j]) * know_scores_stand - (CSGRLMethod.s_w[j] * unknow_scores_stand)
            close_probs.append(close_prob_j)
        prediction = np.concatenate(prediction)
        # scores = np.concatenate(scores)
        gts = np.concatenate(gts)
        # for j in range(len(CSGRLMethod.s_w)):
        #     close_probs[j] = np.concatenate(close_probs[j])

        return prediction, gts, close_probs

    s_w = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
           1]

    def knownpred_unknwonscore_test(self, test_loader):
        self.model.eval()
        self.modelG.eval()
        pred, gts, close_probs = self.scoring(test_loader)
        return close_probs, -9999999, pred

    def save_model(self, path):
        save_dict = {
            'model': self.model.state_dict(),
            'config': self.config,
            'optimzer': self.modelopt.state_dict(),
            'epoch': self.epoch,
        }
        torch.save(save_dict, path)

    def load_model(self, path):
        save_dict = torch.load(path)
        self.model.load_state_dict(save_dict['model'])
        if 'optimzer' in save_dict and self.modelopt is not None:
            self.modelopt.load_state_dict(save_dict['optimzer'])
        self.epoch = save_dict['epoch']

    def class_maximum_distance(self, cls_er, y, clsnum, max_dis):
        for i in range(clsnum):
            index = torch.where(y == i)[0]
            if index.numel() == 0:
                continue
            temp = torch.max(cls_er[index, i]).detach()
            if max_dis[i] < temp:
                max_dis[i] = temp

        return max_dis



