import torch
import torch.nn as nn
import torchvision.models as models
import copy

class InputTMRFineModel(nn.Module):
    def __init__(self, model):
        super(InputTMRFineModel,self).__init__()
        self.features = copy.deepcopy(model.features)
        self.avgpool = copy.deepcopy(model.avgpool)
        self.classifier = copy.deepcopy(model.classifier)
        self.threshold = torch.tensor(.25)

    def forward(self,x):
        x = torch.cat([x.clone().detach(), x.clone().detach(), x.clone().detach()])
        for layer in self.features:
            x = layer(x)
            if torch.equal(x[0],x[1]): #torch.max(torch.abs(torch.sub(x[0],x[1]))) < self.threshold:
                if not torch.equal(x[0],x[2]):
                    x[2] = x[0].clone().detach()
            elif torch.equal(x[1],x[2]): #torch.max(torch.abs(torch.sub(x[1],x[2]))) < self.threshold:
                x[0] = x[1].clone().detach()
            elif torch.equal(x[0],x[2]): #torch.max(torch.abs(torch.sub(x[0],x[2]))) < self.threshold:
                x[1] = x[2].clone().detach()
            else:
                print("Unrecoverable Error")
                return None
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if torch.equal(x[0],x[1]): #torch.max(torch.abs(torch.sub(x[0],x[1]))) < self.threshold:
            if not torch.equal(x[0],x[2]):
                x[2] = x[0].clone().detach()
        elif torch.equal(x[1],x[2]): #torch.max(torch.abs(torch.sub(x[1],x[2]))) < self.threshold:
            x[0] = x[1].clone().detach()
        elif torch.equal(x[0],x[2]): #torch.max(torch.abs(torch.sub(x[0],x[2]))) < self.threshold:
            x[1] = x[2].clone().detach()
        else:
            print("Unrecoverable Error")
            return None
        for layer in self.classifier:
            x = layer(x)
            if torch.equal(x[0],x[1]): #torch.max(torch.abs(torch.sub(x[0],x[1]))) < self.threshold:
                if not torch.equal(x[0],x[2]):
                    x[2] = x[0].clone().detach()
            elif torch.equal(x[1],x[2]): #torch.max(torch.abs(torch.sub(x[1],x[2]))) < self.threshold:
                x[0] = x[1].clone().detach()
            elif torch.equal(x[0],x[2]): #torch.max(torch.abs(torch.sub(x[0],x[2]))) < self.threshold:
                x[1] = x[2].clone().detach()
            else:
                print("Unrecoverable Error")
                return None
        return x
