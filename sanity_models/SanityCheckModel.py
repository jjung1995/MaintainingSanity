import torch
import torch.nn as nn 
import torchvision.models as models

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
        self.avgpool = model.avgpool
        self.classifier = nn.Sequential(*list(add_sanity_check(layer) for layer in model.classifier))
        # Threshold to determine whether mismatch is due to error or precision issue
        self.threshold = 1000

    def forward(self,x):
        for layer in self.features:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                # Check if each checksum is 0 (or close to 0)
                print(x.size())
                if torch.max(torch.abs(torch.sum(x,1))) > self.threshold:
                    print("Error found in layer: ", layer)
                print(torch.sum(x,1))
                x = x[:,:-1] # Exclude checksum from input to next layer
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        for layer in self.classifier:
            x = layer(x)
            if isinstance(layer, nn.Linear):
                # Check if each checksum is 0 (or close to 0)
                if torch.abs(torch.sum(x,1)) > self.threshold:
                    print("Error found in layer: ", layer)
                x = x[:,:-1] # Exclude checksum from input to next layer
        return x
