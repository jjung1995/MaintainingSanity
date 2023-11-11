import torch
import torch.nn as nn
import torchvision.models as models

class FineTMRModel(nn.Module):
    def __init__(self, model):
        super(FineTMRModel,self).__init__()
        self.features = model.features
        self.avgpool = model.avgpool
        self.classifier = model.classifier
        self.threshold = torch.tensor(1000)

    def forward(self,x):
        x = torch.cat([x.clone().detach(), x.clone().detach(), x.clone().detach()])
        for layer in self.features:
            x = layer(x)
            if torch.equal(x[0],x[1]): #torch.max(torch.abs(torch.sub(x[0],x[1]))) < self.threshold:
                x[2] = x[0].clone().detach()
            elif torch.equal(x[1],x[2]): #torch.max(torch.abs(torch.sub(x[1],x[2]))) < self.threshold:
                x[0] = x[1].clone().detach()
            elif torch.equal(x[0],x[2]): #torch.max(torch.abs(torch.sub(x[0],x[2]))) < self.threshold:
                x[1] = x[2].clone().detach()
            else:
                print("Unrecoverable Error")
                return x
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if torch.equal(x[0],x[1]): #torch.max(torch.abs(torch.sub(x[0],x[1]))) < self.threshold:
                x[2] = x[0].clone().detach()
        elif torch.equal(x[1],x[2]): #torch.max(torch.abs(torch.sub(x[1],x[2]))) < self.threshold:
            x[0] = x[1].clone().detach()
        elif torch.equal(x[0],x[2]): #torch.max(torch.abs(torch.sub(x[0],x[2]))) < self.threshold:
            x[1] = x[2].clone().detach()
        else:
            print("Unrecoverable Error")
            return x
        for layer in self.classifier:
            x = layer(x)
            if torch.equal(x[0],x[1]): #torch.max(torch.abs(torch.sub(x[0],x[1]))) < self.threshold:
                x[2] = x[0].clone().detach()
            elif torch.equal(x[1],x[2]): #torch.max(torch.abs(torch.sub(x[1],x[2]))) < self.threshold:
                x[0] = x[1].clone().detach()
            elif torch.equal(x[0],x[2]): #torch.max(torch.abs(torch.sub(x[0],x[2]))) < self.threshold:
                x[1] = x[2].clone().detach()
            else:
                print("Unrecoverable Error")
                return x
        return x
