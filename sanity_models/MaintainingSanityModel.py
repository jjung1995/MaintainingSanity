import torch
import torch.nn as nn 
import torchvision.models as models
import math
import bisect

def add_sanity(layer, num_san, idx_arr): 
    # num_san yields required number of sanity neurons given number of original neurons
    # idx_arr stores list of indices used to form partial checksums
    
    # Adding spatial checksum for fully-connected layer
    if isinstance(layer, nn.Linear):
        ins, outs = layer.in_features, layer.out_features
        sane_layer = nn.Linear(ins,outs + num_san[outs]) # Additional sanity features
        with torch.no_grad():
            # values other than checksums remain unchanged
            sane_layer.weight[:outs] = layer.weight[:]
            sane_layer.bias[:outs] = layer.bias[:]
            # Error location partial checksums
            for i in range(num_san[outs]-1):
                indices = idx_arr[i][:bisect.bisect_left(idx_arr[i],outs)] # get nontrivial indices for partial checksum
                sane_layer.weight[outs+i] = -torch.sum(layer.weight[indices],0)
                sane_layer.bias[outs+i] = -torch.sum(layer.bias[indices])
            # Error detection checksum
            sane_layer.weight[-1] = -torch.sum(layer.weight,0)
            sane_layer.bias[-1] = -torch.sum(layer.bias)
        return sane_layer
    
    # Adding spatial checksum for convolution layer
    if isinstance(layer, nn.Conv2d):
        ins, outs = layer.in_channels, layer.out_channels
        ker, strd, pad = layer.kernel_size, layer.stride, layer.padding
        sane_layer = nn.Conv2d(ins, outs + num_san[outs], ker, strd, pad) # Additional sanity channels
        with torch.no_grad():
            # values other than checksums remain unchanged
            sane_layer.weight[:outs] = layer.weight[:]
            sane_layer.bias[:outs] = layer.bias[:]
            # Error location partial checksums
            for i in range(num_san[outs]-1):
                indices = idx_arr[i][:bisect.bisect_left(idx_arr[i],outs)] # get nontrivial indices for partial checksum
                sane_layer.weight[outs+i] = -torch.sum(layer.weight[indices],0)
                sane_layer.bias[outs+i] = -torch.sum(layer.bias[indices])
            # Error detection checksum
            sane_layer.weight[-1] = -torch.sum(layer.weight,0)
            sane_layer.bias[-1] = -torch.sum(layer.bias)
        return sane_layer
    
    # No modification if not FC or Conv2d
    return layer

class MaintainingSanityModel(nn.Module):
    def __init__(self, model):
        super(MaintainingSanityModel,self).__init__()
        
        # num_san yields required number of sanity neurons given number of original neurons
        self.num_san = {} # {original # neurons : required # sanity neurons}
        
        # rev_san yields number of sanity neurons given total number of neurons 
        self.rev_san = {} # {# total neurons : # sanity neurons}
        
        # each layer to fill in num_san/rev_san and find maximum number of neurons in a layer 
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
        
        # idx_arr stores list of indices used to form partial checksums
        # ex) checksum_x will be the negative sum of values with indices indicated by idx_arr[x]
        self.idx_arr = [[n-1 for n in range(1,k+1) if n&2**m == 2**m] for m in range(self.num_san[k]-1)]
        
        # Apply Maintaining Sanity to each layer
        self.features = nn.Sequential(*list(add_sanity(layer, self.num_san, self.idx_arr) for layer in model.features))
        self.avgpool = model.avgpool
        self.classifier = nn.Sequential(*list(add_sanity(layer, self.num_san, self.idx_arr) for layer in model.classifier))

        # Threshold to determine whether mismatch is due to error or precision issue
        self.threshold = torch.tensor(1000)
        
    def forward(self,x):
        x = torch.cat([x,x]) # Concatenated DMR
        layer_num = 0
        y = [x[:],x[:]] # Store previous and current layer output (sliding window)
        while layer_num < len(self.features):
            layer = self.features[layer_num]
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
                """
                elif error detected in partial checksums but not total checksum: # have not rigorously checked if indexed correctly
                    for i in range(sans):
                        indices = idx_arr[i][:bisect.bisect_left(idx_arr[i],sans)] # get nontrivial indices for partial checksum
                        layer.weight[-sans+i] = -torch.sum(layer.weight[indices],0)
                        layer.bias[-sans+i] = -torch.sum(layer.bias[indices])
                """
                x = x[:,:-sans]
            y[layer_num%2] = x[:]
            s, t = x.chunk(2)
            if not torch.sum(torch.abs(torch.sub(s,t))) < self.threshold:
                print("Input Error Found! Layer: ", layer)
                x = y[(layer_num+1)%2]
                continue
            layer_num += 1

        while True:
            s, t = x.chunk(2)
            s = self.avgpool(s)
            t = self.avgpool(t)
            s = torch.flatten(s, 1)
            t = torch.flatten(t, 1)
            x = torch.cat([s,t])
            y[1] = x[:]
            if torch.sum(torch.abs(torch.sub(s,t))) < self.threshold:
                break
            print("Input Error Found! Layer: ", layer)
            x = y[0]

        layer_num = 0
        while layer_num < len(self.classifier):
            layer = self.classifier[layer_num]
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
                """
                elif error detected in partial checksums but not total checksum: # have not rigorously checked if indexed correctly
                    for i in range(sans):
                        indices = idx_arr[i][:bisect.bisect_left(idx_arr[i],sans)] # get nontrivial indices for partial checksum
                        layer.weight[-sans+i] = -torch.sum(layer.weight[indices],0)
                        layer.bias[-sans+i] = -torch.sum(layer.bias[indices])
                """
                x = x[:,:-sans]
            y[layer_num%2] = x[:]
            s, t = x.chunk(2)
            if not torch.sum(torch.abs(torch.sub(s,t))) < self.threshold:
                print("Input Error Found! Layer: ", layer)
                x = y[(layer_num+1)%2]
                layer_num -= 1
            layer_num += 1
        return x.chunk(2)[0]
