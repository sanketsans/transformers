import numpy as np 
import timm 
import torch 
from ViT import VisionTransformer 

## Helpers 

def get_n_params(module):
    return sum(p.numel() for p in module.parameters() if p.required_grad) 

def assert_tensors_equal(t1, t2):
    a1, a2 = t1.detach().numpy(), t2.detach().numpy() 

    np.testing.assert_allclose(a1, a2) 

model_name = "vit_base_patch16_384"
model_official = timm.create_model(model_name, pretrained=True) 
model_official.eval() 
print(type(model_official))

custom_config = {
    "img_size" : 384, 
    "patch_size": 16,
    "in_chan" : 3,
    "dims" : 768, 
    "depth": 12, 
    "n_classes" : 1000,
    "n_heads" : 12, 
    "qkv_bias" : True, 
    "mlp_ratio" : 4, 
}

model_custom = VisionTransformer(**custom_config) 
model_custom.eval() 

for (n_o, p_o), (n_c, p_c) in zip(model_official.named_parameters(), model_custom.named_parameters()):
    assert p_o.numel() == p_c.numel() 
    print("equal") 

    p_c.data[:] = p_o.data

    assert_tensors_equal(p_c.data, p_o.data) 

inp = torch.randn(1, 3, 384, 384) 
res_c = model_custom(inp) 
res_p = model_official(inp) 

