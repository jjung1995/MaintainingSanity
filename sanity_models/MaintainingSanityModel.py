import torch
import torch.nn as nn 
import torchvision.models as models
import math
import bisect
import copy

def add_msanity(layer, num_san, idx_arr):
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
        # self.threshold = torch.tensor([1e-5, 0.005, 0.1, 0.05, 0.25, 0.005, 0.001, 0.001])
        self.threshold = torch.tensor(0.25)
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
        self.avgpool = copy.deepcopy(model.avgpool)
        self.classifier = nn.Sequential(*list(add_msanity(layer, self.num_san, self.idx_arr) for layer in model.classifier))
        self.isError = False
        #self.maxValues = [0, 0, 0, 0, 0, 0, 0, 0]
        #self.passed = False
        #self.fAlarm = False

    def forward(self,x):
        # self.isError = False
        # self.passed = False
        # self.fAlarm = False
        x = torch.cat([x.clone().detach(),x.clone().detach()]).cuda()
        x0 = x.clone().detach().cuda()
        x1 = x.clone().detach().cuda()


        while True: #Until 1st conv
            x = self.features[0](x) # (0): Conv2d(3, 65, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
            sans = self.rev_san[self.features[0].out_channels]
            z = torch.sum(x[:,:-sans],1)
            # self.maxValues[0] = max(self.maxValues[0], torch.max(torch.abs(z + x[:,-1])))
            if torch.max(torch.abs(z + x[:,-1])) > self.threshold:
                self.isError = True
                layer = self.features[0]
                print("Weight/Bias Error Found! Layer: ", layer, "Value: ", torch.max(torch.abs(torch.sum(x[:,:-sans],1)+x[:,-1])))
                # self.fAlarm = True
                idx = -1
                for i in range(sans-1):
                    indices = self.idx_arr[i][:bisect.bisect_left(self.idx_arr[i],layer.out_channels-sans)]
                    if torch.max(torch.abs(torch.sum(x[:,indices],1)+x[:,-sans+i])) > self.threshold:
                        idx += 2**i
                        print("Weight/Bias Error Found! Layer: ", layer, "Sanity Index: ", i, "Value: ", torch.max(torch.abs(torch.sum(x[:,indices],1)+x[:,-sans+i])))
                print("Weight/Bias Error Found! Layer: ", layer, "Index: ", idx)
                # Spatial Error Correction
                if(idx < layer.out_channels):
                    with torch.no_grad():
                        layer.weight[idx] -= torch.sum(layer.weight[:-sans],0) + layer.weight[-1]
                        layer.bias[idx] -= torch.sum(layer.bias[:-sans]) + layer.bias[-1]
                    x[:,idx] -= (torch.sum(x[:,:-sans],1) + x[:,-1])
                # else:
                #     self.passed = True
            x = x[:,:-sans] # Exclude checksum from input to next layer
            x1 = x.clone().detach()
            # self.maxValues[0] = max(self.maxValues[0], torch.max(torch.abs(torch.sub(z[0],z[1]))))
            if torch.max(torch.abs(torch.sub(z[0],z[1]))) < self.threshold:
                break
            self.isError = True
            x = x0.clone().detach()


        while True: #Until 2nd conv
            x = self.features[1](x) # (1): ReLU(inplace=True)
            x = self.features[2](x) # (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
            x = self.features[3](x) # (3): Conv2d(64, 193, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
            sans = self.rev_san[self.features[3].out_channels]
            z = torch.sum(x[:,:-sans],1)
            # # self.maxValues[1] = max(self.maxValues[1], torch.max(torch.abs(z + x[:,-1])))
            if torch.max(torch.abs(z + x[:,-1])) > self.threshold:
                self.isError = True
                layer = self.features[3]
                print("Weight/Bias Error Found! Layer: ", layer, "Value: ", torch.max(torch.abs(torch.sum(x[:,:-sans],1)+x[:,-1])))
                # self.fAlarm = True
                idx = -1
                for i in range(sans-1):
                    indices = self.idx_arr[i][:bisect.bisect_left(self.idx_arr[i],layer.out_channels-sans)]
                    if torch.max(torch.abs(torch.sum(x[:,indices],1)+x[:,-sans+i])) > self.threshold:
                        idx += 2**i
                        print("Weight/Bias Error Found! Layer: ", layer, "Sanity Index: ", i, "Value: ", torch.max(torch.abs(torch.sum(x[:,indices],1)+x[:,-sans+i])))
                print("Weight/Bias Error Found! Layer: ", layer, "Index: ", idx)
                # Spatial Error Correction
                if(idx < layer.out_channels):
                    with torch.no_grad():
                        layer.weight[idx] -= torch.sum(layer.weight[:-sans],0) + layer.weight[-1]
                        layer.bias[idx] -= torch.sum(layer.bias[:-sans]) + layer.bias[-1]
                    x[:,idx] -= (torch.sum(x[:,:-sans],1) + x[:,-1])
                # else:
                #     self.passed = True
            x = x[:,:-sans] # Exclude checksum from input to next layer
            x0 = x.clone().detach()
            # self.maxValues[1] = max(self.maxValues[1], torch.max(torch.abs(torch.sub(z[0],z[1]))))
            if torch.max(torch.abs(torch.sub(z[0],z[1]))) < self.threshold:
                break
            self.isError = True
            x = x1.clone().detach()


        while True: #Until 3rd conv
            x = self.features[4](x) # (4): ReLU(inplace=True)
            x = self.features[5](x) # (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
            x = self.features[6](x) # (6): Conv2d(192, 385, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            sans = self.rev_san[self.features[6].out_channels]
            z = torch.sum(x[:,:-sans],1)
            # # self.maxValues[2] = max(self.maxValues[2], torch.max(torch.abs(z + x[:,-1])))
            if torch.max(torch.abs(z + x[:,-1])) > self.threshold:
                self.isError = True
                layer = self.features[6]
                print("Weight/Bias Error Found! Layer: ", layer, "Value: ", torch.max(torch.abs(torch.sum(x[:,:-sans],1)+x[:,-1])))
                # self.fAlarm = True
                idx = -1
                for i in range(sans-1):
                    indices = self.idx_arr[i][:bisect.bisect_left(self.idx_arr[i],layer.out_channels-sans)]
                    if torch.max(torch.abs(torch.sum(x[:,indices],1)+x[:,-sans+i])) > self.threshold:
                        idx += 2**i
                        print("Weight/Bias Error Found! Layer: ", layer, "Sanity Index: ", i, "Value: ", torch.max(torch.abs(torch.sum(x[:,indices],1)+x[:,-sans+i])))
                print("Weight/Bias Error Found! Layer: ", layer, "Index: ", idx)
                # Spatial Error Correction
                if(idx < layer.out_channels):
                    with torch.no_grad():
                        layer.weight[idx] -= torch.sum(layer.weight[:-sans],0) + layer.weight[-1]
                        layer.bias[idx] -= torch.sum(layer.bias[:-sans]) + layer.bias[-1]
                    x[:,idx] -= (torch.sum(x[:,:-sans],1) + x[:,-1])
                # else:
                #     self.passed = True
            x = x[:,:-sans] # Exclude checksum from input to next layer
            x1 = x.clone().detach()
            # self.maxValues[2] = max(self.maxValues[2], torch.max(torch.abs(torch.sub(z[0],z[1]))))
            if torch.max(torch.abs(torch.sub(z[0],z[1]))) < self.threshold:
                break
            self.isError = True
            x = x0.clone().detach()


        while True: #Until 4th conv
            x = self.features[7](x) # (7): ReLU(inplace=True)
            x = self.features[8](x) # (8): Conv2d(384, 257, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            sans = self.rev_san[self.features[8].out_channels]
            z = torch.sum(x[:,:-sans],1)
            # # self.maxValues[3] = max(self.maxValues[3], torch.max(torch.abs(z + x[:,-1])))
            if torch.max(torch.abs(z + x[:,-1])) > self.threshold:
                self.isError = True
                layer = self.features[8]
                print("Weight/Bias Error Found! Layer: ", layer, "Value: ", torch.max(torch.abs(torch.sum(x[:,:-sans],1)+x[:,-1])))
                # self.fAlarm = True
                idx = -1
                for i in range(sans-1):
                    indices = self.idx_arr[i][:bisect.bisect_left(self.idx_arr[i],layer.out_channels-sans)]
                    if torch.max(torch.abs(torch.sum(x[:,indices],1)+x[:,-sans+i])) > self.threshold:
                        idx += 2**i
                        print("Weight/Bias Error Found! Layer: ", layer, "Sanity Index: ", i, "Value: ", torch.max(torch.abs(torch.sum(x[:,indices],1)+x[:,-sans+i])))
                print("Weight/Bias Error Found! Layer: ", layer, "Index: ", idx)
                # Spatial Error Correction
                if(idx < layer.out_channels):
                    with torch.no_grad():
                        layer.weight[idx] -= torch.sum(layer.weight[:-sans],0) + layer.weight[-1]
                        layer.bias[idx] -= torch.sum(layer.bias[:-sans]) + layer.bias[-1]
                    x[:,idx] -= (torch.sum(x[:,:-sans],1) + x[:,-1])
                # else:
                #     self.passed = True
            x = x[:,:-sans] # Exclude checksum from input to next layer
            x0 = x.clone().detach()
            # self.maxValues[3] = max(self.maxValues[3], torch.max(torch.abs(torch.sub(z[0],z[1]))))
            if torch.max(torch.abs(torch.sub(z[0],z[1]))) < self.threshold:
                break
            self.isError = True
            x = x1.clone().detach()



        while True: #Until 5th conv
            x = self.features[9](x) # (7): ReLU(inplace=True)
            x = self.features[10](x) # (10): Conv2d(256, 257, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            sans = self.rev_san[self.features[10].out_channels]
            z = torch.sum(x[:,:-sans],1)
            # # self.maxValues[4] = max(self.maxValues[4], torch.max(torch.abs(z + x[:,-1])))
            if torch.max(torch.abs(z + x[:,-1])) > self.threshold:
                self.isError = True
                layer = self.features[10]
                print("Weight/Bias Error Found! Layer: ", layer, "Value: ", torch.max(torch.abs(torch.sum(x[:,:-sans],1)+x[:,-1])))
                # self.fAlarm = True
                idx = -1
                for i in range(sans-1):
                    indices = self.idx_arr[i][:bisect.bisect_left(self.idx_arr[i],layer.out_channels-sans)]
                    if torch.max(torch.abs(torch.sum(x[:,indices],1)+x[:,-sans+i])) > self.threshold:
                        idx += 2**i
                        print("Weight/Bias Error Found! Layer: ", layer, "Sanity Index: ", i, "Value: ", torch.max(torch.abs(torch.sum(x[:,indices],1)+x[:,-sans+i])))
                print("Weight/Bias Error Found! Layer: ", layer, "Index: ", idx)
                # Spatial Error Correction
                if(idx < layer.out_channels):
                    with torch.no_grad():
                        layer.weight[idx] -= torch.sum(layer.weight[:-sans],0) + layer.weight[-1]
                        layer.bias[idx] -= torch.sum(layer.bias[:-sans]) + layer.bias[-1]
                    x[:,idx] -= (torch.sum(x[:,:-sans],1) + x[:,-1])
                # else:
                #     self.passed = True
            x = x[:,:-sans] # Exclude checksum from input to next layer
            x1 = x.clone().detach()
            # self.maxValues[4] = max(self.maxValues[4], torch.max(torch.abs(torch.sub(z[0],z[1]))))
            if torch.max(torch.abs(torch.sub(z[0],z[1]))) < self.threshold:
                break
            self.isError = True
            x = x0.clone().detach()


        while True: #Until 1st Linear
            x = self.features[11](x) # (11): ReLU(inplace=True)
            x = self.features[12](x) # (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
            s, t = x.chunk(2)
            s = self.avgpool(s)
            t = self.avgpool(t)
            s = torch.flatten(s, 1)
            t = torch.flatten(t, 1)
            x = torch.cat([s,t])

            x = self.classifier[0](x) #(0): Dropout(p=0.5, inplace=False)
            x = self.classifier[1](x) #(1): Linear(in_features=9216, out_features=4097, bias=True)
            sans = self.rev_san[self.classifier[1].out_features]
            z = torch.sum(x[:,:-sans],1)
            # # self.maxValues[5] = max(self.maxValues[5], torch.max(torch.abs(z + x[:,-1])))
            if torch.max(torch.abs(z + x[:,-1])) > self.threshold:
                self.isError = True
                layer = self.classifier[1]
                print("Weight/Bias Error Found! Layer: ", layer, "Value: ", torch.max(torch.abs(torch.sum(x[:,:-sans],1)+x[:,-1])))
                # self.fAlarm = True
                idx = -1
                for i in range(sans-1):
                    indices = self.idx_arr[i][:bisect.bisect_left(self.idx_arr[i],layer.out_features-sans)]
                    if torch.max(torch.abs(torch.sum(x[:,indices],1)+x[:,-sans+i])) > self.threshold:
                        idx += 2**i
                        print("Weight/Bias Error Found! Layer: ", layer, "Sanity Index: ", i, "Value: ", torch.max(torch.abs(torch.sum(x[:,indices],1)+x[:,-sans+i])))
                print("Weight/Bias Error Found! Layer: ", layer, "Index: ", idx)
                # Spatial Error Correction
                if(idx < layer.out_features):
                    with torch.no_grad():
                        layer.weight[idx] -= torch.sum(layer.weight[:-sans],0) + layer.weight[-1]
                        layer.bias[idx] -= torch.sum(layer.bias[:-sans]) + layer.bias[-1]
                    x[:,idx] -= torch.sum(x[:,:-sans],1) + x[:,-1]
                # else:
                #     self.passed = True
            x = x[:,:-sans] # Exclude checksum from input to next layer
            x0 = x.clone().detach()
            # self.maxValues[5] = max(self.maxValues[5], torch.max(torch.abs(torch.sub(z[0],z[1]))))
            if torch.max(torch.abs(torch.sub(z[0],z[1]))) < self.threshold:
                break
            self.isError = True
            x = x1.clone().detach()



        while True: #Until 2nd Linear
            x = self.classifier[2](x) #(2): ReLU(inplace=True)
            x = self.classifier[3](x) #(3): Dropout(p=0.5, inplace=False)
            x = self.classifier[4](x) #(4): Linear(in_features=4096, out_features=4097, bias=True)
            sans = self.rev_san[self.classifier[4].out_features]
            z = torch.sum(x[:,:-sans],1)
            # # self.maxValues[6] = max(self.maxValues[6], torch.max(torch.abs(z + x[:,-1])))
            if torch.max(torch.abs(z + x[:,-1])) > self.threshold:
                self.isError = True
                layer = self.classifier[4]
                print("Weight/Bias Error Found! Layer: ", layer, "Value: ", torch.max(torch.abs(torch.sum(x[:,:-sans],1)+x[:,-1])))
                # self.fAlarm = True
                idx = -1
                for i in range(sans-1):
                    indices = self.idx_arr[i][:bisect.bisect_left(self.idx_arr[i],layer.out_features-sans)]
                    if torch.max(torch.abs(torch.sum(x[:,indices],1)+x[:,-sans+i])) > self.threshold:
                        idx += 2**i
                        print("Weight/Bias Error Found! Layer: ", layer, "Sanity Index: ", i, "Value: ", torch.max(torch.abs(torch.sum(x[:,indices],1)+x[:,-sans+i])))
                print("Weight/Bias Error Found! Layer: ", layer, "Index: ", idx)
                # Spatial Error Correction
                if(idx < layer.out_features):
                    with torch.no_grad():
                        layer.weight[idx] -= torch.sum(layer.weight[:-sans],0) + layer.weight[-1]
                        layer.bias[idx] -= torch.sum(layer.bias[:-sans]) + layer.bias[-1]
                    x[:,idx] -= torch.sum(x[:,:-sans],1) + x[:,-1]
                # else:
                #     self.passed = True
            x = x[:,:-sans] # Exclude checksum from input to next layer
            x1 = x.clone().detach()
            # self.maxValues[6] = max(self.maxValues[6], torch.max(torch.abs(torch.sub(z[0],z[1]))))
            if torch.max(torch.abs(torch.sub(z[0],z[1]))) < self.threshold:
                break
            self.isError = True
            x = x0.clone().detach()


        while True: #Until 3rd Linear
            x = self.classifier[5](x) #(5): ReLU(inplace=True)
            x = self.classifier[6](x) #(6): Linear(in_features=4096, out_features=1001, bias=True)
            sans = self.rev_san[self.classifier[6].out_features]
            z = torch.sum(x[:,:-sans],1)
            # # self.maxValues[7] = max(self.maxValues[7], torch.max(torch.abs(z + x[:,-1])))
            if torch.max(torch.abs(z + x[:,-1])) > self.threshold:
                self.isError = True
                layer = self.classifier[6]
                print("Weight/Bias Error Found! Layer: ", layer, "Value: ", torch.max(torch.abs(torch.sum(x[:,:-sans],1)+x[:,-1])))
                # self.fAlarm = True
                idx = -1
                for i in range(sans-1):
                    indices = self.idx_arr[i][:bisect.bisect_left(self.idx_arr[i],layer.out_features-sans)]
                    if torch.max(torch.abs(torch.sum(x[:,indices],1)+x[:,-sans+i])) > self.threshold:
                        idx += 2**i
                        print("Weight/Bias Error Found! Layer: ", layer, "Sanity Index: ", i, "Value: ", torch.max(torch.abs(torch.sum(x[:,indices],1)+x[:,-sans+i])))
                print("Weight/Bias Error Found! Layer: ", layer, "Index: ", idx)
                # Spatial Error Correction
                if(idx < layer.out_features):
                    with torch.no_grad():
                        layer.weight[idx] -= torch.sum(layer.weight[:-sans],0) + layer.weight[-1]
                        layer.bias[idx] -= torch.sum(layer.bias[:-sans]) + layer.bias[-1]
                    x[:,idx] -= torch.sum(x[:,:-sans],1) + x[:,-1]
                # else:
                #     self.passed = True
            x = x[:,:-sans] # Exclude checksum from input to next layer
            x0 = x.clone().detach()
            # self.maxValues[7] = max(self.maxValues[7], torch.max(torch.abs(torch.sub(z[0],z[1]))))
            if torch.max(torch.abs(torch.sub(z[0],z[1]))) < self.threshold:
                break
            self.isError = True
            x = x1.clone().detach()

        return x[:1], self.isError

