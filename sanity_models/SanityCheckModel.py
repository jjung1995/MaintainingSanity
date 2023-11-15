import torch
import torch.nn as nn 
import torchvision.models as models
import copy

def add_sanity_check(layer):
    # Adding spatial checksum for convolution layer
    if isinstance(layer, nn.Conv2d):
        in_dim, out_dim = layer.in_channels, layer.out_channels + 1 # One additional output channel
        ker, strd, pad = layer.kernel_size, layer.stride, layer.padding
        sanity_layer = nn.Conv2d(in_dim, out_dim, ker, strd, pad)
        with torch.no_grad():
            # weight checksum
            sanity_layer.weight[:-1] = layer.weight[:] # values other than checksum remain unchanged
            sanity_layer.weight[-1] = -torch.sum(layer.weight,0) # checksum = negative sum of weights
            # bias checksum
            sanity_layer.bias[:-1] = layer.bias[:] # values other than checksum remain unchanged
            sanity_layer.bias[-1] = -torch.sum(layer.bias) # checksum = negative sum of bias
        return sanity_layer
    
    # Adding spatial checksum for fully-connected layer
    if isinstance(layer, nn.Linear):
        in_dim, out_dim = layer.in_features, layer.out_features + 1 # One additional output feature
        sanity_layer = nn.Linear(in_dim, out_dim)
        with torch.no_grad():
            # weight checksum
            sanity_layer.weight[:-1] = layer.weight[:] # values other than checksum remain unchanged
            sanity_layer.weight[-1] = -torch.sum(layer.weight,0) # checksum = negative sum of weights
            # bias checksum
            sanity_layer.bias[:-1] = layer.bias[:] # values other than checksum remain unchanged
            sanity_layer.bias[-1] = -torch.sum(layer.bias) # checksum = negative sum of bias
        return sanity_layer
    
    # No modification if not FC or Conv2d
    return layer

class SanityCheckModel(nn.Module):
    def __init__(self, model):
        super(SanityCheckModel,self).__init__()
        # Apply Sanity-Check to each layer
        self.features = nn.Sequential(*list(add_sanity_check(layer) for layer in model.features))
        self.avgpool = copy.deepcopy(model.avgpool)
        self.classifier = nn.Sequential(*list(add_sanity_check(layer) for layer in model.classifier))
        # Threshold to determine whether mismatch is due to error or precision issue
        self.threshold = torch.tensor(1000)

    def forward(self,x):
        checksums = torch.zeros(16)
        idx = 0
        for layer in self.features:
            if isinstance(layer, nn.Conv2d):
                x = torch.cat([x,x.clone().detach() * -1])
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                b = layer.bias.data.clone() * -2
                checksums[idx] = torch.max(torch.abs(torch.sum(x,0) + b[:, None, None]))
                checksums[idx+1] = torch.max(torch.abs(torch.sum(x,1)))
                idx += 2
                x = x[:1,:-1]
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                x = torch.cat([x,x.clone().detach() * -1])
            x = layer(x)
            if isinstance(layer, nn.Linear):
                # Temporal
                b = layer.bias.data.clone() * -2
                checksums[idx] = torch.max(torch.abs(torch.sum(x,0) + b))
                checksums[idx+1] = torch.max(torch.abs(torch.sum(x,1)))
                idx += 2
                x = x[:1,:-1]
        if torch.max(checksums) > self.threshold:
            print("Error Detected")
        return x
