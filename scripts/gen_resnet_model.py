
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import struct
import os
import requests
from PIL import Image
from io import BytesIO

# Configuration
OUTPUT_BIN = "weights.bin"
OUTPUT_HEX = "weights.hex"
INPUT_H = "src/main/c/murax/hyperram_phase_a/src/input.h"
EXPECTED_H = "src/main/c/murax/hyperram_phase_a/src/expected.h"

def get_resnet20():
    print("Loading ResNet-20 (cifar10)...")
    try:
        # Use torch.hub to load a pre-trained model
        # repo = 'chenyaofo/pytorch-cifar-models'
        # model = torch.hub.load(repo, 'cifar10_resnet20', pretrained=True)
        # Fallback to local definition if hub fails or just to be safe/fast
        # Actually, let's try hub first.
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
        model.eval()
        return model
    except Exception as e:
        print(f"Hub load failed: {e}. creating random model for fallback (NOT REAL WEIGHTS if this happens)")
        # This is strictly a fallback to ensure script runs, but user wants REAL weights.
        raise e

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
    print("Exporting Weights...")
    blob = bytearray()
    
    # 1. Header (Magic, Count, etc.)
    # We will fill count later
    blob += struct.pack('<I', 0x56574230) # VWB0
    blob += struct.pack('<I', 0)          # Placeholder for Total Bytes
    blob += struct.pack('<I', 0)          # Placeholder for CRC
    blob += struct.pack('<I', 0)          # Reserved
    
    weight_offset = 16
    
    # Iterate named parameters
    for name, param in model.named_parameters():
        if "weight" in name or "bias" in name:
            # Quantize
            q_param, scale = quantize_tensor(param.data, name)
            
            # Metadata: ID (Hash of name? or just index?), Type, Shape...
            # For Phase A, we just dump the raw bytes and maybe print offsets
            print(f"  {name}: {tuple(param.shape)} -> range check: {q_param.min()}..{q_param.max()}")
            
            # Append to blob
            data_bytes = q_param.numpy().tobytes()
            blob += data_bytes
            
            # Align to 4 bytes
            while len(blob) % 4 != 0:
                blob += b'\x00'
                
    # Fill Size
    total_size = len(blob)
    struct.pack_into('<I', blob, 4, total_size - 16) # Payload size
    
    # Checksum
    crc = 0
    # TODO: Calculate CRC
    
    with open(OUTPUT_BIN, "wb") as f:
        f.write(blob)

    with open(OUTPUT_HEX, "w") as f:
        for i in range(0, len(blob), 4):
            word_bytes = blob[i:i+4]
            # Unpack as little endian integer for hex string
            if len(word_bytes) < 4: break # Should be padded
            word_val = struct.unpack('<I', word_bytes)[0]
            f.write(f"{word_val:08x}\n")
    
    print(f"Saved {OUTPUT_BIN} and {OUTPUT_HEX} ({total_size} bytes)")
    return blob

def export_input_header(img_tensor):
    # img_tensor is float [1, 3, 32, 32] normalized
    # We quantize it to int8
    q_img, scale = quantize_tensor(img_tensor, "input")
    q_data = q_img.numpy().flatten().tolist()
    
    # C Header format
    with open(INPUT_H, "w") as f:
        f.write("#ifndef INPUT_H\n#define INPUT_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"// Scale: {scale}\n")
        f.write(f"const int8_t INPUT_DATA[{len(q_data)}] = {{\n")
        for i, val in enumerate(q_data):
            f.write(f"{val}, ")
            if (i+1) % 16 == 0: f.write("\n")
        f.write("\n};\n\n#endif\n")
    print(f"Saved {INPUT_H}")

def export_expected_header(output_tensor):
    # output_tensor is float [1, 10]
    # We quantize it
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
    model = get_resnet20()
    
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
    blob = export_weights(model)
    export_input_header(input_tensor_cpu)
    export_expected_header(output.cpu())

if __name__ == "__main__":
    main()
