import torch
import torch.nn as nn 
import torchvision.models as models
import math
import bisect

# Maintaining Sanity but spatial redundancy only

def add_msanity(layer, num_san, idx_arr):
    if isinstance(layer, nn.Linear):
        ins, outs = layer.in_features, layer.out_features
        sane_layer = nn.Linear(ins,outs + num_san[outs])
        with torch.no_grad():
            sane_layer.weight[:outs] = layer.weight[:]
            sane_layer.bias[:outs] = layer.bias[:]
            for i in range(num_san[outs]-1):
                indices = idx_arr[i][:bisect.bisect_left(idx_arr[i],outs)]
                sane_layer.weight[outs+i] = -torch.sum(layer.weight[indices],0)
                sane_layer.bias[outs+i] = -torch.sum(layer.bias[indices])
            sane_layer.weight[-1] = -torch.sum(layer.weight,0)
            sane_layer.bias[-1] = -torch.sum(layer.bias)
        return sane_layer
    if isinstance(layer, nn.Conv2d):
        ins, outs = layer.in_channels, layer.out_channels
        ker, strd, pad = layer.kernel_size, layer.stride, layer.padding
        sane_layer = nn.Conv2d(ins, outs + num_san[outs], ker, strd, pad)
        with torch.no_grad():
            sane_layer.weight[:outs] = layer.weight[:]
            sane_layer.bias[:outs] = layer.bias[:]
            for i in range(num_san[outs]-1):
                indices = idx_arr[i][:bisect.bisect_left(idx_arr[i],outs)]
                sane_layer.weight[outs+i] = -torch.sum(layer.weight[indices],0)
                sane_layer.bias[outs+i] = -torch.sum(layer.bias[indices])
            sane_layer.weight[-1] = -torch.sum(layer.weight,0)
            sane_layer.bias[-1] = -torch.sum(layer.bias)
        return sane_layer
    return layer

class SimplifiedMaintainingSanityModel(nn.Module):
    def __init__(self, model):
        super(SimplifiedMaintainingSanityModel,self).__init__()
        self.threshold = 1000 
        self.num_san = {}
        self.rev_san = {}
        for layer in model.modules():
            if isinstance(layer, nn.Conv2d):
                outs = layer.out_channels
                sans = int(math.log2(outs))+2
                self.num_san[outs] = sans
                self.rev_san[outs + sans] = sans
            if isinstance(layer, nn.Linear):
                outs = layer.out_features
                sans = int(math.log2(outs))+2
                self.num_san[outs] = sans
                self.rev_san[outs + sans] = sans
        k = max(self.num_san)
        self.idx_arr = [[n-1 for n in range(1,k+1) if n&2**m == 2**m] for m in range(self.num_san[k]-1)]

        self.features = nn.Sequential(*list(add_msanity(layer, self.num_san, self.idx_arr) for layer in model.features))
        self.avgpool = model.avgpool
        self.classifier = nn.Sequential(*list(add_msanity(layer, self.num_san, self.idx_arr) for layer in model.classifier))

    def forward(self,x):
        for layer in self.features:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                # Spatial Error Detection
                sans = self.rev_san[layer.out_channels]
                if torch.max(torch.abs(torch.sum(x[:,:-sans],1)+x[:,-1])) > self.threshold:
                    print("Weight/Bias Error Found! Layer: ", layer, "Value: ", torch.max(torch.abs(torch.sum(x[:,:-sans],1)+x[:,-1])))
                    idx = -1
                    for i in range(sans-1):
                        indices = self.idx_arr[i][:bisect.bisect_left(self.idx_arr[i],layer.out_channels-sans)]
                        if torch.max(torch.abs(torch.sum(x[:,indices],1)+x[:,-sans+i])) > self.threshold:
                            idx += 2**i
                            print("Weight/Bias Error Found! Layer: ", layer, "Sanity Index: ", i, "Value: ", torch.max(torch.abs(torch.sum(x[:,indices],1)+x[:,-sans+i])))
                    print("Weight/Bias Error Found! Layer: ", layer, "Index: ", idx)
                    # Spatial Error Correction
                    with torch.no_grad():
                        layer.weight[idx] -= torch.sum(layer.weight[:-sans],0) + layer.weight[-1]
                        layer.bias[idx] -= torch.sum(layer.bias[:-sans]) + layer.bias[-1]
                    x[:,idx] -= (torch.sum(x[:,:-sans],1) + x[:,-1])
                x = x[:,:-sans]
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        for layer in self.classifier:
            x = layer(x)
            if isinstance(layer, nn.Linear):
                # Spatial Error Detection
                sans = self.rev_san[layer.out_features]
                if torch.max(torch.abs(torch.sum(x[:,:-sans],1)+x[:,-1])) > self.threshold:
                    print("Weight/Bias Error Found! Layer: ", layer, "Value: ", torch.max(torch.abs(torch.sum(x[:,:-sans],1)+x[:,-1])))
                    idx = -1
                    for i in range(sans-1):
                        indices = self.idx_arr[i][:bisect.bisect_left(self.idx_arr[i],layer.out_features-sans)]
                        if torch.max(torch.abs(torch.sum(x[:,indices],1)+x[:,-sans+i])) > self.threshold:
                            idx += 2**i
                            print("Weight/Bias Error Found! Layer: ", layer, "Sanity Index: ", i, "Value: ", torch.max(torch.abs(torch.sum(x[:,indices],1)+x[:,-sans+i])))
                    print("Weight/Bias Error Found! Layer: ", layer, "Index: ", idx)
                    # Spatial Error Correction
                    with torch.no_grad():
                        layer.weight[idx] -= torch.sum(layer.weight[:-sans],0) + layer.weight[-1]
                        layer.bias[idx] -= torch.sum(layer.bias[:-sans]) + layer.bias[-1]
                    x[:,idx] -= torch.sum(x[:,:-sans],1) + x[:,-1]
                x = x[:,:-sans]
        return x
