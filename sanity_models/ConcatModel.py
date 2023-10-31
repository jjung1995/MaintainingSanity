import torch
import torch.nn as nn 
import torchvision.models as models

class ConcatModel(nn.Module):
    def __init__(self, model):
        super(ConcatModel,self).__init__()
        self.features = model.features
        self.avgpool = model.avgpool
        self.classifier = model.classifier
        self.threshold = torch.tensor(1000)

    def forward(self,x):
        x = torch.cat([x,x])
        layer_num = 0
        y = [x[:],x[:]]
        while layer_num < len(self.features):
            layer = self.features[layer_num]
            x = layer(x)
            y[layer_num%2] = x[:]
            s, t = x.chunk(2)
            if not torch.eq(s,t).all():
                print("Input Error Found! Layer: ", layer)
                # x = y[(layer_num+1)%2]
                # layer_num -= 1
            layer_num += 1

        while True:
            s, t = x.chunk(2)
            s = self.avgpool(s)
            t = self.avgpool(t)
            s = torch.flatten(s, 1)
            t = torch.flatten(t, 1)
            x = torch.cat([s,t])
            y[1] = x[:]
            if torch.eq(s,t).all():
                break
            break
            print("Input Error Found! Layer: ", layer)
            x = y[0]

        layer_num = 0
        while layer_num < len(self.classifier):
            layer = self.classifier[layer_num]
            x = layer(x)
            y[layer_num%2] = x[:]
            s, t = x.chunk(2)
            if not torch.eq(s,t).all():
                print("Input Error Found! Layer: ", layer)
                # x = y[(layer_num+1)%2]
                # layer_num -= 1
            layer_num += 1
        return x.chunk(2)[0]
