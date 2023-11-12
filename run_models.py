import torch
import torch.nn as nn 
import torchvision.models as models
import time
import math
import bisect
import random

from san_models.DMRModel import DMRModel 
from san_models.ConcatModel import ConcatModel 
from san_models.SanityCheckModel import SanityCheckModel # Sanity-Check but only spatial redundancy
from san_models.MaintainingSanityModel import MaintainingSanityModel 
from san_models.SimplifiedMaintainingSanityModel import SimplifiedMaintainingSanityModel # Maintaining Sanity but spatial redundancy only


alex_model = models.alexnet(pretrained=True)
alex_model.eval()

dmr_model = DMRModel(alex_model)
dmr_model.eval()
concat_model = ConcatModel(alex_model)
concat_model.eval()
sc_model = SanityCheckModel(alex_model)
sc_model.eval()
ms_model = MaintainingSanityModel(alex_model)
ms_model.eval()
sms_model = SimplifiedMaintainingSanityModel(alex_model)
sms_model.eval()

ums_model = UnrolledMaintainingSanityModel_CheckingAfterConvFully(alex_model)
ums_model.eval()

batch_size = 1
h = 224
w = 224
c = 3

device = torch.device("cuda")
test_models = [alex_model, alex_model, dmr_model, concat_model, sc_model, sms_model, ms_model, ums_model] # must run a dummy model first

for m in test_models:
    m.to(device)
    runtime = 0
    runtime_cpu = 0
    for _ in range(100):
        image = torch.rand((batch_size, c, h, w)).cuda()

        start_time_cpu = time.time()
        with torch.no_grad():
            output = m(image)
        end_time_cpu = time.time()
        
        runtime_cpu += end_time_cpu - start_time_cpu
    print(m.__class__.__name__, runtime_cpu)
    
