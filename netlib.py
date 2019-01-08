import torch.nn as nn, torch.nn.functional as F
import numpy as np, math


###################### EASY TRIPLET NET ##########################
class TripletNet(nn.Module):
    def __init__(self, funnel_type='fully-connected'):
        super(TripletNet, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(3,32,5,1,2),nn.LeakyReLU(0.2),
                                      nn.MaxPool2d(2,2),
                                      nn.Conv2d(32,64,5,1,2),nn.LeakyReLU(0.2),
                                      nn.MaxPool2d(2,2),
                                      nn.Conv2d(64,92,5,1,2),nn.LeakyReLU(0.2),
                                      nn.MaxPool2d(2,2))

        self.funnel_type = funnel_type
        if funnel_type=='fully-connected':
            # nn.Linear(64 * 54 * 44, 256)
            self.funnel   = nn.Sequential(nn.Linear(92 * 27 * 22, 256),nn.LeakyReLU(0.2),
                                          nn.Linear(256, 256),nn.LeakyReLU(0.2),
                                          nn.Linear(256, 2))
        else:
            self.funnel   = nn.Sequential(nn.Conv2d(92,128,3,1,1),nn.LeakyReLU(0.2),
                                          nn.MaxPool2d(2,2),
                                          nn.Conv2d(128,128,3,1,1),nn.LeakyReLU(0.2),
                                          nn.Conv2d(128,64,3,1,1))

    def forward(self, anchor, pos, neg):
        anchor = self.features(anchor)
        pos    = self.features(pos)
        neg    = self.features(neg)

        if self.funnel_type=='fully-connected':
            anchor = anchor.view(anchor.size()[0], -1)
            pos    = pos.view(pos.size()[0], -1)
            neg    = neg.view(neg.size()[0], -1)

        anchor = self.funnel(anchor)
        pos    = self.funnel(pos)
        neg    = self.funnel(neg)

        if self.funnel_type=='fully-convolutional':
            anchor = F.avg_pool2d(anchor, kernel_size=anchor.size()[2:]).view(anchor.size()[0],-1)
            pos    = F.avg_pool2d(pos, kernel_size=pos.size()[2:]).view(pos.size()[0],-1)
            neg    = F.avg_pool2d(neg, kernel_size=neg.size()[2:]).view(neg.size()[0],-1)
        return anchor, pos, neg



###################### FULLY-CONVOLUTIONAL TRIPLET ALEXNET ##########################
class FC_AlexNet(nn.Module):
    def __init__(self, num_classes, in_channels=3, use_batchnorm=True, config=None):
        self.pars = {key:item for key,item in locals().items() if key not in ['self', '__class__']}

        super(FC_AlexNet, self).__init__()

        self.pars['config']   = [(96, 11, 2, 2), 'M', (256, 5, 1, 2), 'M', (384, 3, 1, 1), (384, 3, 1, 1), (256, 3, 1, 1), 'M'] if self.pars['config'] is None else self.pars['config']
        # self.pars['config']   = [(96, 11, 2, 2), 'M', (256, 5, 1, 2), 'M', (384, 3, 1, 1), (384, 3, 1, 1), (256, 3, 1, 1), 'M'] if self.pars['config'] is None else self.pars['config']
        self.output_layer     = nn.Conv2d(256,92,3,1,1)

        self.__GiveConvFeatures()
        # self.__GiveMLP()

        self.__initialize_weights()

    def __GiveConvFeatures(self):
        layers = []
        for c in self.pars['config']:
            if c == 'M':
                layers += [nn.MaxPool2d(kernel_size=3, stride=2)]
            elif c == 'ML':
                layers += [nn.MaxPool2d(kernel_size=3, stride=2), nn.LRN(local_size=5, alpha=0.0001, beta=0.75)]
            else:
                conv2d = nn.Conv2d(self.pars['in_channels'], c[0], kernel_size=c[1], stride=c[2], padding=c[3])
                if self.pars['use_batchnorm']:
                    layers += [conv2d, nn.BatchNorm2d(c[0]), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                self.pars['in_channels'] = c[0]
        self.features = nn.Sequential(*layers)



    def __GiveMLP(self):
        self.MLP = nn.Sequential(nn.Linear(256, 2048),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(0.5),
                                 nn.Linear(2048,1024),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(0.5),
                                 nn.Linear(1024, self.pars['num_classes']))


    def __initialize_weights(self):
        for idx,module in enumerate(self.modules()):
            if isinstance(module, nn.Conv2d):
                n = np.prod(module.kernel_size[:2])*module.out_channels
                for i in range(module.out_channels):
                    module.weight.data[i].normal_(0, math.sqrt(2./n))
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            elif isinstance(module, nn.Linear):
                module.weight.data.normal_(0,0.01)
                module.bias.data.zero_()



    def forward(self, anchor, pos, neg):
        anchor = self.features(anchor)
        pos    = self.features(pos)
        neg    = self.features(neg)

        anchor = self.output_layer(anchor)
        pos    = self.output_layer(pos)
        neg    = self.output_layer(neg)

        anchor = F.avg_pool2d(anchor, kernel_size=anchor.size()[2:]).view(anchor.size()[0],-1)
        pos    = F.avg_pool2d(pos, kernel_size=pos.size()[2:]).view(pos.size()[0],-1)
        neg    = F.avg_pool2d(neg, kernel_size=neg.size()[2:]).view(neg.size()[0],-1)

        return anchor, pos, neg





########################### TRIPLET LOSS ################################
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, pos, neg):
        anchor_pos_dist_L2 = (anchor-pos).pow(2).sum(1)
        anchor_neg_dist_L2 = (anchor-neg).pow(2).sum(1)
        loss = F.relu(anchor_pos_dist_L2-anchor_neg_dist_L2+self.margin).mean()
        return loss
