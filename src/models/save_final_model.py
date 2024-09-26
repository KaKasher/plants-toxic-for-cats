import timm
import torch

model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=47)
model.load_state_dict(torch.load('../../models/vit_b16_224_25e_256bs_0.001lr_adamW_transforms.tar'))

torch.save(model, '../../models/plant-classifier-vitb32.pth')