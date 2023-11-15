import torch
import torch.nn as nn
import torchvision.models as models
import copy

class ModelTMRCoarseModel(nn.Module):
    def __init__(self, model):
        super(ModelTMRCoarseModel,self).__init__()
        self.xfeatures = copy.deepcopy(model.features)
        self.xavgpool = copy.deepcopy(model.avgpool)
        self.xclassifier = copy.deepcopy(model.classifier)
        self.yfeatures = copy.deepcopy(model.features)
        self.yavgpool = copy.deepcopy(model.avgpool)
        self.yclassifier = copy.deepcopy(model.classifier)
        self.zfeatures = copy.deepcopy(model.features)
        self.zavgpool = copy.deepcopy(model.avgpool)
        self.zclassifier = copy.deepcopy(model.classifier)
        self.threshold = torch.tensor(1000)

    def forward(self,x):
        x, y, z = x.clone().detach(), x.clone().detach(), x.clone().detach()
        layer_num = 0
        while layer_num < len(self.xfeatures):
            x = self.xfeatures[layer_num](x)
            y = self.yfeatures[layer_num](y)
            z = self.zfeatures[layer_num](z)
            layer_num += 1
        x = self.xavgpool(x)
        x = torch.flatten(x, 1)
        y = self.yavgpool(y)
        y = torch.flatten(y, 1)
        z = self.zavgpool(z)
        z = torch.flatten(z, 1)
        layer_num = 0
        while layer_num < len(self.xclassifier):
            x = self.xclassifier[layer_num](x)
            y = self.yclassifier[layer_num](y)
            z = self.zclassifier[layer_num](z)
            layer_num += 1
        if torch.equal(x,y): #torch.max(torch.abs(torch.sub(x,y))) < self.threshold:
            out = x
        elif torch.equal(y,z): #torch.max(torch.abs(torch.sub(y,z))) < self.threshold:
            out = y
        elif torch.equal(x,z): #torch.max(torch.abs(torch.sub(x,z))) < self.threshold:
            out = z
        else:
            print("Unrecoverable Error")
            return None
        return out
