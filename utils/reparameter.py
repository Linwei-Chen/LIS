import matplotlib.pyplot as plt
import torch

model = torch.load('/home/ubuntu/code/LowLight/mmdetexp/MaskRCNN600x400/r50_fpn_bs4_8cls_RAW_Dark-RAW_Normal_NoiseInvPT_LLPFPrior_8cls40.8_12E/latest.pth')
for k in model: 
    print(k)
    
for k in model['state_dict']: 
    if '.conv2.' in k:
        print(k)
        # llpf = model['state_dict']['backbone.layer2.0.llpf.weight']
        try:
            llpf_conv1 = model['state_dict'][k.replace('conv2', 'learnable_conv1')]
            llpf = model['state_dict'][k.replace('conv2', 'llpf')]
            llpf_conv1 = llpf_conv1.repeat(1, 1, 3, 3)
            llpf = llpf.reshape(-1, 1, 9).softmax(dim=-1).reshape(-1, 1, 3, 3)
            model['state_dict'][k] += llpf_conv1 * llpf
            # print(k)
        except Exception:
            pass

torch.save(model, '/home/ubuntu/code/LowLight/mmdetexp/MaskRCNN600x400/r50_fpn_bs4_8cls_RAW_Dark-RAW_Normal_NoiseInvPT_LLPFPrior_8cls40.8_12E/repara.pth')