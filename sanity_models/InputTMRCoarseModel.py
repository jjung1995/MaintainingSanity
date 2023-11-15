import torch
import torch.nn as nn
import torchvision.models as models

class InputTMRCoarseModel(nn.Module):
    def __init__(self, model):
        super(InputTMRCoarseModel,self).__init__()
        self.features = model.features
        self.avgpool = model.avgpool
        self.classifier = model.classifier
        self.threshold = torch.tensor(1000)

    def forward(self,x):
        x = torch.cat([x.clone().detach(), x.clone().detach(), x.clone().detach()])
        for layer in self.features:
            x = layer(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        for layer in self.classifier:
            x = layer(x)
        if torch.equal(x[0],x[1]): #torch.max(torch.abs(torch.sub(x[0],x[1]))) < self.threshold:
            y = x[0]
        elif torch.equal(x[1],x[2]): #torch.max(torch.abs(torch.sub(x[1],x[2]))) < self.threshold:
            y = x[1]
        elif torch.equal(x[0],x[2]): #torch.max(torch.abs(torch.sub(x[0],x[2]))) < self.threshold:
            y = x[2]
        else:
            print("Unrecoverable Error")
            y = None
        return y
