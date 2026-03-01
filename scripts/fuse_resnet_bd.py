import torch
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import torch.nn as nn
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blockdialect_codec as bd

def fuse_conv_bn_eval(conv, bn):
    fused_conv = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=True
    )
    with torch.no_grad():
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fused_conv.weight.copy_(torch.mm(w_bn, w_conv).view(fused_conv.weight.shape))
        
        b_conv = conv.bias if conv.bias is not None else torch.zeros(conv.weight.size(0))
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fused_conv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
        
    return fused_conv

def fix_fusion(m):
    for name, child in m.named_children():
        if type(child) == nn.Sequential:
            fix_fusion(child)
        else:
            if hasattr(child, 'conv1') and hasattr(child, 'bn1') and isinstance(child.conv1, nn.Conv2d) and isinstance(child.bn1, nn.BatchNorm2d):
                setattr(child, 'conv1', fuse_conv_bn_eval(child.conv1, child.bn1))
                child.bn1 = nn.Identity()
            if hasattr(child, 'conv2') and hasattr(child, 'bn2') and isinstance(child.conv2, nn.Conv2d) and isinstance(child.bn2, nn.BatchNorm2d):
                setattr(child, 'conv2', fuse_conv_bn_eval(child.conv2, child.bn2))
                child.bn2 = nn.Identity()
            if hasattr(child, 'conv3') and hasattr(child, 'bn3') and isinstance(child.conv3, nn.Conv2d) and isinstance(child.bn3, nn.BatchNorm2d):
                setattr(child, 'conv3', fuse_conv_bn_eval(child.conv3, child.bn3))
                child.bn3 = nn.Identity()
            if hasattr(child, 'downsample') and child.downsample is not None:
                if isinstance(child.downsample[0], nn.Conv2d) and isinstance(child.downsample[1], nn.BatchNorm2d):
                    child.downsample[0] = fuse_conv_bn_eval(child.downsample[0], child.downsample[1])
                    child.downsample[1] = nn.Identity()

if __name__ == "__main__":
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model.eval()
    
    if hasattr(model, 'conv1') and hasattr(model, 'bn1') and isinstance(model.conv1, nn.Conv2d) and isinstance(model.bn1, nn.BatchNorm2d):
        setattr(model, 'conv1', fuse_conv_bn_eval(model.conv1, model.bn1))
        model.bn1 = nn.Identity()

    fix_fusion(model)

    tensors = []
    for name, param in model.named_parameters():
        if not param.dtype.is_floating_point:
            continue
        print(name, param.numel())
        arr = param.detach().cpu().numpy().astype(np.float32)
        encoded = bd.encode_tensor(arr)
        tensors.append(encoded)

    print(f"LEN: {len(tensors)}")
    bd.write_weight_blob(tensors, "resnet50_bd_fused_weights.bin")
