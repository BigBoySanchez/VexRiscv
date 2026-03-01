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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blockdialect_codec as bd

# Configuration
OUTPUT_BIN = "weights.bin"
OUTPUT_HEX = "weights.hex"
OUTPUT_BD_BIN = "weights_bd.bin"
OUTPUT_BD_HEX = "weights_bd.hex"
INPUT_H_A = "src/main/c/murax/hyperram_phase_a/src/input.h"
INPUT_H_B = "src/main/c/murax/hyperram_phase_b/src/input.h"
INPUT_H_FULL = "src/main/c/murax/hyperram_phase_full/src/input.h"
EXPECTED_H_FULL = "src/main/c/murax/hyperram_phase_full/src/expected_full.h"
OUTPUT_BIN_FULL = "scripts/weights.bin"  # Python should emit FPGA test artifacts to expected location

def get_resnet(arch="resnet20", checkpoint_override=None):
    hub_dir = os.path.expanduser("~/.cache/torch/hub/akamaster_pytorch_resnet_cifar10_master")
    if hub_dir not in sys.path:
        sys.path.insert(0, hub_dir)
    
    if arch == "resnet20":
        from resnet import resnet20 as get_model
        default_ckpt = "resnet20-12fca82f.th"
    elif arch == "resnet110":
        from resnet import resnet110 as get_model
        default_ckpt = "resnet110-1d1ed7c2.th"
    else:
        raise ValueError(f"Unsupported arch: {arch}")

    print(f"Loading {arch} (akamaster, pretrained)...")
    model = get_model()
    
    checkpoint_file = checkpoint_override if checkpoint_override else default_ckpt
    checkpoint_path = os.path.join(hub_dir, "pretrained_models", checkpoint_file)
    
    state = torch.load(checkpoint_path, map_location="cpu")
    sd = state.get("state_dict", state)
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
    max_val = torch.max(torch.abs(tensor))
    if max_val == 0:
        scale = 1.0
    else:
        scale = max_val / 127.0
    q_tensor = (tensor / scale).round().clamp(-127, 127).to(torch.int8)
    return q_tensor, scale

def export_weights(model, out_bin=OUTPUT_BIN, out_hex=OUTPUT_HEX):
    print(f"Exporting Weights (int8) to {out_bin}...")
    blob = bytearray()
    
    blob += struct.pack('<I', 0x56574230) # VWB0
    blob += struct.pack('<I', 0)
    blob += struct.pack('<I', 0)
    blob += struct.pack('<I', 0)
    
    for name, param in model.named_parameters():
        if "weight" in name or "bias" in name:
            q_param, scale = quantize_tensor(param.data, name)
            data_bytes = q_param.numpy().tobytes()
            blob += data_bytes
            while len(blob) % 4 != 0:
                blob += b'\x00'
                
    total_size = len(blob)
    struct.pack_into('<I', blob, 4, total_size - 16)
    
    os.makedirs(os.path.dirname(os.path.abspath(out_bin)), exist_ok=True)
    with open(out_bin, "wb") as f:
        f.write(blob)

    if out_hex:
        with open(out_hex, "w") as f:
            for i in range(0, len(blob), 4):
                word_bytes = blob[i:i+4]
                if len(word_bytes) < 4: break
                word_val = struct.unpack('<I', word_bytes)[0]
                f.write(f"{word_val:08x}\n")
    
    print(f"Saved {out_bin} ({total_size} bytes)")
    return blob

def export_input_header(img_tensor, header_path):
    q_img, scale = quantize_tensor(img_tensor, "input")
    q_data = q_img.numpy().flatten().tolist()
    
    os.makedirs(os.path.dirname(os.path.abspath(header_path)), exist_ok=True)
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
    return q_img.numpy()

def conv2d_3x3_int8_fast(input_int8, weights_int8, out_c, stride=1, pad=1):
    in_c, h, w = input_int8.shape
    w_tensor = weights_int8.reshape((out_c, in_c, 3, 3))
    
    if pad > 0:
        inp_pad = np.pad(input_int8, ((0,0), (pad,pad), (pad,pad)), mode='constant', constant_values=0)
    else:
        inp_pad = input_int8
        
    out_h = h // stride
    out_w = w // stride
    out = np.zeros((out_c, out_h, out_w), dtype=np.int8)
    
    for oc in range(out_c):
        for y in range(out_h):
            for x in range(out_w):
                iy = y * stride
                ix = x * stride
                patch = inp_pad[:, iy:iy+3, ix:ix+3]
                s = np.sum(patch.astype(np.int32) * w_tensor[oc].astype(np.int32))
                s = s >> 7
                out[oc, y, x] = np.int8(s)
    return out

def batch_norm_int8(feature_map, bn_weight, bn_bias, has_relu=True):
    out = np.zeros_like(feature_map)
    for c in range(feature_map.shape[0]):
        w_bn = np.int32(bn_weight[c])
        b_bn = np.int32(bn_bias[c])
        val = feature_map[c].astype(np.int32)
        val = (val * w_bn) >> 6
        val = val + b_bn
        if has_relu:
            val = np.clip(val, 0, 127)
        else:
            val = np.clip(val, -128, 127)
        out[c] = val.astype(np.int8)
    return out

def option_a_downsample_int8(x, out_planes):
    c, _, _ = x.shape
    x_sub = x[:, ::2, ::2]
    pad_c = (out_planes - c) // 2
    return np.pad(x_sub, ((pad_c, pad_c), (0, 0), (0, 0)), mode='constant', constant_values=0)

def add_relu_int8(dst, src):
    res = dst.astype(np.int32) + src.astype(np.int32)
    res = np.clip(res, 0, 127)
    return res.astype(np.int8)

def avgpool8x8_to_64(input_int8):
    C = input_int8.shape[0]
    out = np.zeros(C, dtype=np.int8)
    for c in range(C):
        s = np.sum(input_int8[c].astype(np.int32))
        s = s >> 6
        out[c] = np.int8(s)
    return out

class ResNetInt8Sim:
    def __init__(self, weights_blob, n=3):
        self.blob = weights_blob
        self.offset = 0
        self.layer_hashes = []
        self.n = n

    def get_weights(self, count):
        data = self.blob[self.offset : self.offset + count]
        self.offset += count
        while self.offset % 4 != 0:
            self.offset += 1
        return np.frombuffer(data, dtype=np.int8)

    def print_hash(self, name, x):
        h = np.sum(x.astype(np.int32), dtype=np.uint32)
        print(f"Hash {name:15}: 0x{h:08x}")
        self.layer_hashes.append((name, h))

    def conv3x3(self, x, out_c, stride=1):
        in_c = x.shape[0]
        w_int8 = self.get_weights(out_c * in_c * 3 * 3)
        return conv2d_3x3_int8_fast(x, w_int8, out_c, stride=stride, pad=1)

    def bn_relu(self, x, has_relu=True):
        c = x.shape[0]
        bn_w = self.get_weights(c)
        bn_b = self.get_weights(c)
        return batch_norm_int8(x, bn_w, bn_b, has_relu=has_relu)

    def basic_block(self, block_name, x, out_c, stride=1):
        out = self.conv3x3(x, out_c, stride=stride)
        out = self.bn_relu(out, has_relu=True)
        
        out = self.conv3x3(out, out_c, stride=1)
        out = self.bn_relu(out, has_relu=False)
        
        if stride != 1 or x.shape[0] != out_c:
            shortcut = option_a_downsample_int8(x, out_c)
        else:
            shortcut = x
            
        out = add_relu_int8(out, shortcut)
        self.print_hash(block_name, out)
        return out

    def forward(self, img_int8):
        self.offset = 16 # skip VWB0 header
        
        print(f"\n--- Python Int8 Simulation (n={self.n}) ---")
        x = self.conv3x3(img_int8, 16, stride=1)
        x = self.bn_relu(x, has_relu=True)
        self.print_hash("conv1", x)
        
        for i in range(self.n):
            x = self.basic_block(f"layer1_{i}", x, 16, stride=1)
            
        x = self.basic_block("layer2_0", x, 32, stride=2)
        for i in range(1, self.n):
            x = self.basic_block(f"layer2_{i}", x, 32, stride=1)
            
        x = self.basic_block("layer3_0", x, 64, stride=2)
        for i in range(1, self.n):
            x = self.basic_block(f"layer3_{i}", x, 64, stride=1)
            
        x = avgpool8x8_to_64(x)
        self.print_hash("pool", x)
        
        fc_w = self.get_weights(10 * 64).reshape(10, 64)
        fc_b = self.get_weights(10)
        
        logits = np.zeros(10, dtype=np.int32)
        for i in range(10):
            s = np.sum(x.astype(np.int32) * fc_w[i].astype(np.int32))
            s += fc_b[i].astype(np.int32)
            logits[i] = s
            
        print("Final Logits:", logits.tolist())
        pred = np.argmax(logits)
        print("Predicted Class:", pred)
        
        self.expected_output = logits.tolist()
        return self.expected_output, int(pred), self.layer_hashes

def export_expected_full_header(logits, pred_class, hashes, header_path):
    os.makedirs(os.path.dirname(os.path.abspath(header_path)), exist_ok=True)
    with open(header_path, "w") as f:
        f.write("#ifndef EXPECTED_FULL_H\n#define EXPECTED_FULL_H\n\n")
        f.write("#include <stdint.h>\n\n")
        
        f.write("const int32_t EXPECTED_LOGITS[10] = {\n")
        f.write("    " + ", ".join(str(l) for l in logits) + "\n};\n\n")
        
        f.write(f"const int32_t EXPECTED_CLASS = {pred_class};\n\n")
        
        f.write(f"const uint32_t EXPECTED_HASHES[{len(hashes)}] = {{\n")
        for name, h in hashes:
            f.write(f"    0x{h:08x}, // {name}\n")
        f.write("};\n\n")
        
        f.write("\n#endif\n")
    print(f"Saved {header_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["a", "b", "both", "full"], default="both")
    parser.add_argument("--arch", choices=["resnet20", "resnet110"], default="resnet20")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint override")
    parser.add_argument("--outdir", default=None, help="Directory to save artifacts")
    args = parser.parse_args()
    
    model = get_resnet(args.arch, args.checkpoint)
    
    img = get_test_image()
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    input_tensor = transform(img).unsqueeze(0)
    
    n = 3 if args.arch == "resnet20" else 18
    
    if args.outdir:
        out_weights = os.path.join(args.outdir, "weights.bin")
        out_input_h = os.path.join(args.outdir, "input.h")
        out_expected_h = os.path.join(args.outdir, "expected_full.h")
    else:
        out_weights = OUTPUT_BIN_FULL if args.phase == "full" else OUTPUT_BIN
        out_input_h = INPUT_H_FULL if args.phase == "full" else INPUT_H_A
        out_expected_h = EXPECTED_H_FULL

    if args.phase in ("a", "both"):
        blob = export_weights(model, out_bin=out_weights, out_hex=OUTPUT_HEX)
        export_input_header(input_tensor[0], out_input_h)

    if args.phase == "full":
        blob = export_weights(model, out_bin=out_weights, out_hex=None)
        
        q_img_np = export_input_header(input_tensor[0], out_input_h)
        
        sim = ResNetInt8Sim(blob, n=n)
        logits, pred_class, hashes = sim.forward(q_img_np)
        
        export_expected_full_header(logits, pred_class, hashes, out_expected_h)

if __name__ == "__main__":
    main()
