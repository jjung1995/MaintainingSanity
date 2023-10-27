import torch
import torch.nn as nn 
import torchvision.models as models
import time
import math
import bisect
import random

from san_models.DMRModel import DMRModel 
from san_models.DMRModel1 import DMRModel1 # simple DMR (no comparison) 
from san_models.DMRModel2 import DMRModel2 # simple DMR + torch.eq
from san_models.DMRModel3 import DMRModel3 # simple DMR + dummy if statement
from san_models.DMRModel4 import DMRModel4 # simple DMR + torch.eq + dummy if statement
from san_models.ConcatModel import ConcatModel 
from san_models.SanityCheckModel import SanityCheckModel # Sanity-Check but only spatial redundancy
from san_models.SanityCheckModel1 import SanityCheckModel1 # Sanity-Check without error checking
from san_models.SanityCheckModel2 import SanityCheckModel2 # Sanity-Check + simplified error checking
from san_models.SanityCheckModel3 import SanityCheckModel3 # Sanity-Check + checksum calculation + dummy if statement
from san_models.MaintainingSanityModel import MaintainingSanityModel 
from san_models.SimplifiedMaintainingSanityModel import SimplifiedMaintainingSanityModel # Maintaining Sanity but spatial redundancy only


alex_model = models.alexnet(pretrained=True)
alex_model.eval()

dmr_model = DMRModel(alex_model)
dmr_model.eval()

dmr_model1 = DMRModel1(alex_model)
dmr_model1.eval()

dmr_model2 = DMRModel2(alex_model)
dmr_model2.eval()

dmr_model3 = DMRModel3(alex_model)
dmr_model3.eval()

dmr_model4 = DMRModel4(alex_model)
dmr_model4.eval()

concat_model = ConcatModel(alex_model)
concat_model.eval()

sc_model = SanityCheckModel(alex_model)
sc_model.eval()

sc_model1 = SanityCheckModel1(alex_model)
sc_model1.eval()

sc_model2 = SanityCheckModel2(alex_model)
sc_model2.eval()

sc_model3 = SanityCheckModel3(alex_model)
sc_model3.eval()

ms_model = MaintainingSanityModel(alex_model)
ms_model.eval()

sms_model = SimplifiedMaintainingSanityModel(alex_model)
sms_model.eval()

batch_size = 1
h = 224
w = 224
c = 3

device = torch.device("cuda")
#test_models = [alex_model, alex_model, dmr_model, dmr_model1, dmr_model2, dmr_model3, dmr_model4]#, concat_model, sc_model, sms_model, ms_model]
#test_models = [alex_model, alex_model, sc_model, sc_model1, sc_model2, sc_model3]
test_models = [sc_model]
for m in test_models:
    m.to(device)
    runtime = 0
    runtime_cpu = 0
    for _ in range(1):
        image = torch.rand((batch_size, c, h, w)).cuda()

        start_time_cpu = time.time()
        with torch.no_grad():
            output = m(image)
        end_time_cpu = time.time()
        
        runtime_cpu += end_time_cpu - start_time_cpu
    print(m.__class__.__name__, runtime_cpu)
    
    
"""
class QuantizedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model_fp32 = model
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        
    def forward(self, x):
        x = self.quant(x)
        x = self.model_fp32(x)
        x = self.dequant(x)
        return x

q_model = QuantizedModel(dmr_model)

backend = "qnnpack"
q_model.qconfig = torch.quantization.get_default_qconfig(backend)
torch.backends.quantized.engine = backend
model_static_quantized = torch.quantization.prepare(q_model, inplace=False)
model_static_quantized = torch.quantization.convert(model_static_quantized, inplace=False)
model_static_quantized.eval()

image = torch.rand((1, 3, 224, 224))
with torch.no_grad():
    output = model_static_quantized(image)

    
class SanityCheck(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, t):
        torch.eq(x,y)
        #    print(t)
        #if not torch.sum(torch.abs(torch.sub(x,y))) < t:
        #    print("Error Found! Layer: ", layer)
        return None
"""
