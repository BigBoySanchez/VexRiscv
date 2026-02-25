
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import struct
import os
import sys
import argparse
import requests
from PIL import Image
from io import BytesIO

# Add scripts/ to path for blockdialect_codec
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blockdialect_codec as bd

# Configuration
OUTPUT_BIN = "weights.bin"
OUTPUT_HEX = "weights.hex"
OUTPUT_BD_BIN = "weights_bd.bin"
OUTPUT_BD_HEX = "weights_bd.hex"
INPUT_H_A = "src/main/c/murax/hyperram_phase_a/src/input.h"
INPUT_H_B = "src/main/c/murax/hyperram_phase_b/src/input.h"
EXPECTED_H = "src/main/c/murax/hyperram_phase_a/src/expected.h"

def get_resnet110():
    """Load ResNet-110 from the cached akamaster checkpoint (1.7M params).
    The akamaster torch.hub is broken due to a missing sys.path entry
    in hubconf.py, but the repo and pretrained .th file were cached.
    We load it directly instead.
    """
    hub_dir = os.path.expanduser("~/.cache/torch/hub/akamaster_pytorch_resnet_cifar10_master")
    if hub_dir not in sys.path:
        sys.path.insert(0, hub_dir)

    from resnet import resnet110
    print("Loading ResNet-110 (akamaster, pretrained)...")
    model = resnet110()
    checkpoint_path = os.path.join(hub_dir, "pretrained_models", "resnet110-1d1ed7c2.th")
    state = torch.load(checkpoint_path, map_location="cpu")
    # akamaster checkpoints wrap state_dict under 'state_dict' key
    sd = state.get("state_dict", state)
    # Their keys are prefixed with 'module.' when trained with DataParallel
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params / 1e6:.2f}M")
    return model


def get_test_image():
    print("Downloading Test Image (CIFAR-10 bird)...")
    url = "https://github.com/YoongiKim/CIFAR-10-images/raw/master/test/bird/0000.jpg"
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

def quantize_tensor(tensor, name):
    # Simple symmetric quantization to int8
    # Int8 range: -127 to 127
    max_val = torch.max(torch.abs(tensor))
    if max_val == 0:
        scale = 1.0
    else:
        scale = max_val / 127.0
    
    q_tensor = (tensor / scale).round().clamp(-127, 127).to(torch.int8)
    return q_tensor, scale

def export_weights(model):
    """Phase A: Export int8 weights as VWB0 blob."""
    print("Exporting Phase A Weights (int8)...")
    blob = bytearray()
    
    # 1. Header (Magic, Count, etc.)
    blob += struct.pack('<I', 0x56574230) # VWB0
    blob += struct.pack('<I', 0)          # Placeholder for Total Bytes
    blob += struct.pack('<I', 0)          # Placeholder for CRC
    blob += struct.pack('<I', 0)          # Reserved
    
    weight_offset = 16
    
    # Iterate named parameters
    for name, param in model.named_parameters():
        if "weight" in name or "bias" in name:
            q_param, scale = quantize_tensor(param.data, name)
            print(f"  {name}: {tuple(param.shape)} -> range check: {q_param.min()}..{q_param.max()}")
            data_bytes = q_param.numpy().tobytes()
            blob += data_bytes
            while len(blob) % 4 != 0:
                blob += b'\x00'
                
    total_size = len(blob)
    struct.pack_into('<I', blob, 4, total_size - 16)
    
    with open(OUTPUT_BIN, "wb") as f:
        f.write(blob)

    with open(OUTPUT_HEX, "w") as f:
        for i in range(0, len(blob), 4):
            word_bytes = blob[i:i+4]
            if len(word_bytes) < 4: break
            word_val = struct.unpack('<I', word_bytes)[0]
            f.write(f"{word_val:08x}\n")
    
    print(f"Saved {OUTPUT_BIN} and {OUTPUT_HEX} ({total_size} bytes)")
    return blob

def export_blockdialect_weights(model):
    """Phase B: Export weights as BlockDialect-Lite (VWB1) blob."""
    print("\nExporting Phase B Weights (BlockDialect-Lite DialectFP4)...")
    
    encoded_tensors = []
    total_original = 0
    total_packed = 0
    
    for name, param in model.named_parameters():
        if "weight" in name or "bias" in name:
            q_param, scale = quantize_tensor(param.data, name)
            q_np = q_param.numpy()
            
            original_bytes = q_np.size
            tensor_blob = bd.encode_tensor(q_np)
            packed_bytes = len(tensor_blob) - 8  # subtract tensor header
            
            total_original += original_bytes
            total_packed += len(tensor_blob)
            
            ratio = original_bytes / packed_bytes if packed_bytes > 0 else 0
            print(f"  {name}: {tuple(param.shape)} -> {original_bytes}B → {packed_bytes}B ({ratio:.2f}x)")
            
            encoded_tensors.append(tensor_blob)
    
    blob_bytes = bd.write_weight_blob(encoded_tensors, OUTPUT_BD_BIN)
    
    # Also write raw hex for Intel HEX conversion
    with open(OUTPUT_BD_HEX, "w") as f:
        # Pad to 4-byte alignment
        padded = bytearray(blob_bytes)
        while len(padded) % 4 != 0:
            padded += b'\x00'
        for i in range(0, len(padded), 4):
            word_bytes = padded[i:i+4]
            word_val = struct.unpack('<I', word_bytes)[0]
            f.write(f"{word_val:08x}\n")
    
    overall_ratio = total_original / total_packed if total_packed > 0 else 0
    print(f"\nPhase B Summary:")
    print(f"  Original (int8): {total_original} bytes")
    print(f"  Packed (BD-Lite): {len(blob_bytes)} bytes (incl. headers)")
    print(f"  Overall compression: {overall_ratio:.2f}x")
    print(f"  Saved {OUTPUT_BD_BIN} and {OUTPUT_BD_HEX}")
    
    return blob_bytes

def export_input_header(img_tensor, header_path):
    """Export input image as C header."""
    q_img, scale = quantize_tensor(img_tensor, "input")
    q_data = q_img.numpy().flatten().tolist()
    
    os.makedirs(os.path.dirname(header_path), exist_ok=True)
    with open(header_path, "w") as f:
        f.write("#ifndef INPUT_H\n#define INPUT_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"// Scale: {scale}\n")
        f.write(f"const int8_t INPUT_DATA[{len(q_data)}] = {{\n")
        for i, val in enumerate(q_data):
            f.write(f"{val}, ")
            if (i+1) % 16 == 0: f.write("\n")
        f.write("\n};\n\n#endif\n")
    print(f"Saved {header_path}")

def export_expected_header(output_tensor):
    q_out, scale = quantize_tensor(output_tensor, "output")
    q_data = q_out.numpy().flatten().tolist()
    
    with open(EXPECTED_H, "w") as f:
        f.write("#ifndef EXPECTED_H\n#define EXPECTED_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"// Scale: {scale}\n")
        f.write(f"const int8_t EXPECTED_DATA[{len(q_data)}] = {{\n")
        for val in q_data:
            f.write(f"{val}, ")
        f.write("\n};\n\n#endif\n")
    print(f"Saved {EXPECTED_H}")

def main():
    parser = argparse.ArgumentParser(description="Generate ResNet-110 weights for VexRiscv simulation")
    parser.add_argument("--phase", choices=["a", "b", "both"], default="both",
                       help="Which phase to export: a (int8), b (BlockDialect), both (default)")
    args = parser.parse_args()
    
    model = get_resnet110()
    
    # Transform Input
    img = get_test_image()
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    input_tensor = transform(img).unsqueeze(0) # [1, 3, 32, 32]
    
    # Forward Pass (Float32 Reference)
    with torch.no_grad():
        output = model(input_tensor)
        print("Model Prediction:", torch.argmax(output, 1))

    # Export
    input_tensor_cpu = input_tensor.cpu()
    
    if args.phase in ("a", "both"):
        export_weights(model)
        export_input_header(input_tensor_cpu, INPUT_H_A)
        export_expected_header(output.cpu())
    
    if args.phase in ("b", "both"):
        export_blockdialect_weights(model)
        export_input_header(input_tensor_cpu, INPUT_H_B)

if __name__ == "__main__":
    main()
