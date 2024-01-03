import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator_Class(nn.Module):
    def __init__(self, num_class, nz=100, ngf=64, nc=512):
        super(Generator_Class, self).__init__()
        self.class_gen = []
        for i in range(num_class):
            gen = Generator(nz, ngf, nc).cuda()
            self.class_gen.append(gen)
        self.class_gen = nn.ModuleList(self.class_gen)

    def forward(self, x):
        cls_gens = []
        cls_genls = []
        for i in range(len(self.class_gen)):
            if len(x[i]) ==0:
                continue
            cls_gen = self.class_gen[i](x[i])
            cls_gens.append(cls_gen)
            cls_genl = (torch.ones(cls_gen.shape[0])*i).cuda()
            cls_genls.append(cls_genl)

        gendata = torch.cat(cls_gens, dim=0)
        genlabel = torch.cat(cls_genls, dim=0)
        return gendata, genlabel



class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=512):
        super(Generator, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.nc = nc

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.Conv2d(self.nz, self.ngf * 8, 1, 1, 0, bias=True),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (self.ngf*8) x 4 x 4
            nn.Conv2d(self.ngf * 8, self.ngf * 4, 1, 1, 0, bias=True),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (self.ngf*4) x 8 x 8
            nn.Conv2d(self.ngf * 4, self.ngf * 2, 1, 1, 0, bias=True),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (self.ngf*2) x 16 x 16
            nn.Conv2d(self.ngf * 2, self.ngf * 4, 1, 1, 0, bias=True),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (self.ngf) x 32 x 32
            nn.Conv2d(self.ngf * 4, self.nc, 1, 1, 0, bias=True),
            # nn.Tanh()
            # state size. (self.nc) x 64 x 64
        )

    def forward(self, input):
        if input.shape[0] == 1:
            for m in self.main.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
            output = self.main(input)
            for m in self.main.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()
            return output
        return self.main(input)


class Recognizer(nn.Module):
    def __init__(self, nc=512, ndf=64):
        super(Recognizer, self).__init__()
        self.nc = nc
        self.ndf = ndf
        self.main = nn.Sequential(
            nn.Conv2d(self.nc, self.ndf * 8, 1, 1, 0, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf * 8, self.ndf * 4, 1, 1, 0, bias=True),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf * 4, self.ndf * 2, 1, 1, 0, bias=True),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf * 2, self.ndf, 1, 1, 0, bias=True),
            nn.BatchNorm2d(self.ndf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf, 1, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

class CriterionG(nn.Module):


    def __init__(self,numclass):
        super().__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.main = nn.Sequential(nn.BatchNorm1d(numclass),
        #                         nn.Sigmoid())
        self.sig =  nn.Sigmoid()



    def forward(self, close_er, y, max_dis, margin):
        # x = self.main(-close_er)
        loss = 0
        j = 0
        for i in range(len(max_dis)):
            index = torch.where(y==i)[0]
            if len(index) == 0:
                continue
            gap = self.sig(close_er[index,i] - max_dis[i] - margin)
            gap = torch.clamp(gap,1e-7,1-1e-7)
            loss += -torch.log(gap).mean()
            j += 1
            # if torch.isinf(loss) or torch.isnan(loss):
            #     print(1)
        loss /= j
        return loss

class CriterionR(nn.Module):


    def __init__(self):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bce = nn.BCELoss()


    def forward(self,y_prob,isgen = False):
        if isgen:
            y = torch.zeros(y_prob.shape[0]).cuda()
        else:
            y = torch.ones(y_prob.shape[0]).cuda()
        g = self.avg_pool(y_prob).view(y_prob.shape[0])
        loss = self.bce(g,y)

        return loss
