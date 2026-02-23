"""
Reconstruct the CIFAR-10 input image from input.h for visual verification.
Parses the scale directly from the header comment so it works after any regen.
"""

import re
import numpy as np
from PIL import Image
import os

MEAN = [0.4914, 0.4822, 0.4465]
STD  = [0.2023, 0.1994, 0.2010]

INPUT_H = os.path.join(os.path.dirname(os.path.abspath(__file__)),
          "../src/main/c/murax/hyperram_phase_a/src/input.h")
OUT_PNG  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input_image.png")

with open(INPUT_H) as f:
    text = f.read()

# Parse scale from comment: "// Scale: 0.01912..."
scale_m = re.search(r'//\s*Scale:\s*([0-9.e+-]+)', text)
SCALE = float(scale_m.group(1)) if scale_m else 1.0
print(f"Parsed scale from header: {SCALE}")

# Parse the array body between { ... }
array_m = re.search(r'INPUT_DATA\[(\d+)\]\s*=\s*\{([^}]+)\}', text, re.DOTALL)
if not array_m:
    raise RuntimeError("Could not find INPUT_DATA array in input.h")
n_declared = int(array_m.group(1))
vals = [int(x) for x in re.findall(r'-?\d+', array_m.group(2))]
assert len(vals) == n_declared == 3072, f"Expected 3072 values, got {len(vals)}"

arr = np.array(vals, dtype=np.int8).reshape(3, 32, 32)

# Dequantize: int8 -> normalized float
flt = arr.astype(np.float32) * SCALE

# Un-normalize: reverse CIFAR-10 normalization per channel
for c in range(3):
    flt[c] = flt[c] * STD[c] + MEAN[c]

# CHW -> HWC, clamp, convert to uint8
img_hwc = np.clip(flt, 0.0, 1.0).transpose(1, 2, 0)
img_u8  = (img_hwc * 255).round().astype(np.uint8)

# Upscale 32x32 -> 256x256 (nearest-neighbour to keep pixel art look)
img_pil = Image.fromarray(img_u8, 'RGB').resize((256, 256), Image.NEAREST)
img_pil.save(OUT_PNG)
print(f"Saved: {OUT_PNG}  ({img_pil.size[0]}x{img_pil.size[1]} px)")
print(f"Array shape: {arr.shape}  int8 range: [{arr.min()}, {arr.max()}]")
