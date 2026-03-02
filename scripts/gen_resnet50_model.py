#!/usr/bin/env python3
"""
gen_resnet50_model.py — ResNet-50 FPGA export script (hard-coded to exact variant)

TARGET VARIANT (do not change without updating everywhere):
  Model   : torchvision ResNet-50, IMAGENET1K_V1 weights
  Input   : 224 × 224 × 3, RGB, float32 normalized with ImageNet mean/std
  Classes : 1000 (ImageNet-1k)
  Platform: iCEBreaker iCE40UP5K, on-board SPRAM only (no HyperRAM)
  Weights : BlockDialect VWB1 (block_size=32) stored in SPI flash at 0x2000_0000

This script produces (in --outdir, default ./scripts/resnet50_artifacts/):
  weights_bd.bin        — BlockDialect VWB1 weight blob (BN-folded, biases as float32)
  model_constants.h     — firmware header with model dimensions + blob offsets
  input.h               — deterministic 224×224 int8 test input (from funny_monkey)
  expected_fp32.h       — FP32 logits from torchvision model (golden reference)
  weight_budget.txt     — size report (params, BD blocks, flash bytes)

Weight encoding follows the paper exactly:
  - Weights : BlockDialect VWB1, block_size=32, per-block {dialect_id, shared_exp}
  - Biases  : float32 (the paper does not quantize biases; they are small vectors)
  - BN is folded into Conv before encoding (reduces runtime ops and tensor count)

Test input: scripts/funny_monkey.webp (pre-normalized tensor in funny_monkey_tensor.bin)

Reference: RESNET50_FPGA_PLAN.md Section 2 + Section 3, arXiv:2501.01144v5.
"""

from __future__ import annotations

# =============================================================================
# !! CANONICAL MODEL CONSTANTS — firmware must match these exactly !!
# =============================================================================
MODEL_VARIANT      = "torchvision_resnet50_imagenet_v1"
INPUT_H            = 224
INPUT_W            = 224
INPUT_C            = 3
N_CLASSES          = 1000
BLOCK_SIZE         = 32           # BlockDialect block size (elements per block)
FLASH_WEIGHT_BASE  = 0x20000000   # WeightStore window in SoC address map
FLASH_OFFSET_BYTES = 0x100000     # 1 MiB raw-flash offset handled by SoC flashOffset parameter
                                  # — the CPU WeightStore window (0x20000000) already points here;
                                  #   firmware must NOT add this offset again.

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# The paper never quantizes biases; they are kept in float32.
# INT8 weights are NOT exported — BlockDialect IS the weight format for this project.

# =============================================================================
import os
import sys
import struct
import hashlib
import argparse
import textwrap
from typing import List, Tuple, Dict

import numpy as np

try:
    import torch
    import torchvision.models as tvm
except ImportError as e:
    sys.exit(f"Missing dependency: {e}\n  pip install torch torchvision")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blockdialect_codec as bd


# =============================================================================
# Model loading
# =============================================================================

def load_resnet50() -> torch.nn.Module:
    """Load torchvision ResNet-50 IMAGENET1K_V1 (the canonical variant for this project)."""
    print(f"Loading {MODEL_VARIANT} ...")
    weights = tvm.ResNet50_Weights.IMAGENET1K_V1
    model = tvm.resnet50(weights=weights)
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters : {params:,}  ({params / 1e6:.2f}M)")
    print(f"  Trainable  : {trainable:,}")
    return model


# =============================================================================
# BN folding
# =============================================================================

def fold_bn(conv: torch.nn.Conv2d, bn: torch.nn.BatchNorm2d) -> Tuple[np.ndarray, np.ndarray]:
    """Fold a Conv2d + BatchNorm2d pair into a single (weight, bias) in float32.

    Returns:
        (folded_weight  [out_c, in_c, kH, kW]  float32 ndarray)
        (folded_bias    [out_c]                 float32 ndarray)

    Formula:
        w_folded = w * (gamma / sqrt(var + eps))
        b_folded = (b_conv - mean) * (gamma / sqrt(var + eps)) + beta
    """
    w = conv.weight.detach().float()
    b = conv.bias.detach().float() if conv.bias is not None else torch.zeros(w.shape[0])

    gamma  = bn.weight.detach().float()
    beta   = bn.bias.detach().float()
    mean   = bn.running_mean.detach().float()
    var    = bn.running_var.detach().float()
    eps    = bn.eps

    scale = gamma / torch.sqrt(var + eps)           # [out_c]
    w_f   = w * scale.view(-1, 1, 1, 1)
    b_f   = (b - mean) * scale + beta

    return w_f.numpy(), b_f.numpy()


def extract_folded_weights(model: torch.nn.Module) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Walk a torchvision ResNet-50 and fold all Conv+BN pairs.

    Returns an ordered dict:  layer_name -> (weight_float32, bias_float32)

    Layers that have no associated BN (the final FC) are returned as-is.
    """
    layers: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def _add(name: str, w: np.ndarray, b: np.ndarray) -> None:
        layers[name] = (w, b)
        print(f"  {name:45s}  w={w.shape}  b={b.shape}")

    print("\nFolding BN into Conv ...")

    # conv1 + bn1
    w, b = fold_bn(model.conv1, model.bn1)
    _add("conv1", w, b)

    # layers 1–4 (bottleneck blocks)
    for layer_idx, layer in enumerate([model.layer1, model.layer2, model.layer3, model.layer4], 1):
        for block_idx, block in enumerate(layer):
            pfx = f"layer{layer_idx}.{block_idx}"
            w, b = fold_bn(block.conv1, block.bn1)
            _add(f"{pfx}.conv1", w, b)
            w, b = fold_bn(block.conv2, block.bn2)
            _add(f"{pfx}.conv2", w, b)
            w, b = fold_bn(block.conv3, block.bn3)
            _add(f"{pfx}.conv3", w, b)
            if block.downsample is not None:
                w, b = fold_bn(block.downsample[0], block.downsample[1])
                _add(f"{pfx}.downsample", w, b)

    # FC (no BN)
    fc_w = model.fc.weight.detach().float().numpy()   # [1000, 2048]
    fc_b = model.fc.bias.detach().float().numpy()     # [1000]
    _add("fc", fc_w, fc_b)

    total_params = sum(w.size + b.size for w, b in layers.values())
    print(f"\n  Folded total elements: {total_params:,}  ({total_params / 1e6:.2f}M)\n")
    return layers


# =============================================================================
# Quantization helpers (input only — weights go through BlockDialect)
# =============================================================================

def quantize_per_tensor_sym(arr: np.ndarray) -> Tuple[np.ndarray, float]:
    """Symmetric per-tensor int8 quantization (used for the input tensor only).

    scale = max_abs / 127  (consistent with firmware's INT8 activation path).
    """
    max_abs = float(np.max(np.abs(arr)))
    scale = max_abs / 127.0 if max_abs != 0.0 else 1.0
    q = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
    return q, scale


# =============================================================================
# Exports: BlockDialect blob (VWB2) — indexed, random-access by tensor name
# =============================================================================

def export_weights_bd(layers: Dict[str, Tuple[np.ndarray, np.ndarray]],
                      out_path: str) -> Tuple[bytes, List]:
    """Write all BN-folded weights as a BlockDialect VWB2 indexed blob.

    VWB2 format adds a tensor index table so firmware can fetch any layer's
    weights or biases by name without sequential scanning (RESNET50_FPGA_PLAN §3.3).

    Paper fidelity notes (arXiv:2501.01144v5):
      - Weights are encoded with BlockDialect (Section 3): per-block dialect_id +
        shared_exp + 32 × 4-bit codes (sign + 3-bit index into dialect table).
      - Biases are kept in float32.  The paper never quantizes biases; they are
        low-dimensional vectors that contribute negligible storage overhead.
      - Block size = 32 (paper Section 3.1).

    Returns (blob_bytes, tensor_entries) where tensor_entries is the list of
    bd.TensorEntry objects (useful for emitting per-tensor offsets into the C header).
    """
    print(f"Exporting BlockDialect VWB2 weights → {out_path}")
    tensors_spec: List[Tuple[str, int, np.ndarray]] = []

    for name, (w, b) in layers.items():
        # weights → DTYPE_BD4 (paper Section 3)
        tensors_spec.append((f"{name}.weight", bd.DTYPE_BD4, w.astype(np.float32)))
        # biases → DTYPE_FLOAT32 (paper: biases are never quantized)
        tensors_spec.append((f"{name}.bias", bd.DTYPE_FLOAT32, b.astype(np.float32)))

    blob_bytes = bd.write_weight_blob_v2(tensors_spec, out_path)

    # Re-read entries for reporting (writer discards them after writing)
    entries, _ = bd.read_weight_blob_v2(out_path)

    n_layers = len(layers)
    print(f"  Wrote {len(blob_bytes):,} bytes  "
          f"({n_layers} layers × 2 tensors = {2*n_layers} entries, biases as float32)")
    return blob_bytes, entries


# =============================================================================
# Weight budget report
# =============================================================================

def weight_budget_report(layers: Dict[str, Tuple[np.ndarray, np.ndarray]],
                         bd_blob: bytes,
                         out_path: str) -> None:
    """Print and save a weight budget summary."""
    total_params    = sum(w.size + b.size for w, b in layers.values())
    total_w_elems   = sum(w.size for w, b in layers.values())
    total_b_elems   = sum(b.size for _, b in layers.values())
    n_bd_blocks     = sum((w.size + BLOCK_SIZE - 1) // BLOCK_SIZE for w, b in layers.values())

    BD_BLOCK_BYTES  = 18  # 2-byte meta + 16-byte packed codes (paper Section 3.1)
    bias_bytes_f32  = sum(b.size * 4 for _, b in layers.values())  # float32 biases
    bd_payload_est  = n_bd_blocks * BD_BLOCK_BYTES + 8 * len(layers) + bias_bytes_f32

    flash_available = 16 * 1024 * 1024 - FLASH_OFFSET_BYTES  # 15 MiB after firmware offset

    # Compression ratio vs naive float32
    naive_f32_bytes = total_w_elems * 4 + bias_bytes_f32
    ratio = naive_f32_bytes / max(len(bd_blob), 1)

    lines = [
        f"ResNet-50 Weight Budget Report",
        f"Model variant : {MODEL_VARIANT}",
        f"Input size    : {INPUT_C}×{INPUT_H}×{INPUT_W} = {INPUT_C*INPUT_H*INPUT_W:,} elements",
        f"Output classes: {N_CLASSES}",
        f"",
        f"Weight tensors : {len(layers)}",
        f"Total params   : {total_params:,}  ({total_params / 1e6:.2f}M)",
        f"  Weight elems : {total_w_elems:,}",
        f"  Bias elems   : {total_b_elems:,}  (stored as float32)",
        f"",
        f"BlockDialect encoding (paper Section 3, arXiv:2501.01144v5):",
        f"  Block size       : {BLOCK_SIZE} elements",
        f"  Blocks           : {n_bd_blocks:,}",
        f"  Per-block bytes  : {BD_BLOCK_BYTES}  (2-byte meta + 16-byte packed codes)",
        f"  BD payload est.  : {bd_payload_est:,} bytes  ({bd_payload_est/1024/1024:.2f} MiB)",
        f"  Actual BD blob   : {len(bd_blob):,} bytes  ({len(bd_blob)/1024/1024:.2f} MiB)",
        f"  Naive float32    : {naive_f32_bytes:,} bytes  ({naive_f32_bytes/1024/1024:.2f} MiB)",
        f"  Compression ratio: {ratio:.2f}×",
        f"",
        f"Flash available after firmware offset: {flash_available/1024/1024:.1f} MiB",
        f"  BD blob fits in flash: {'YES' if len(bd_blob) < flash_available else 'NO — TOO BIG'}",
        f"",
        f"iCEBreaker SPRAM (on-board): 4 × 32 Kib = 128 Kib = 131072 bytes",
        f"  Activation buffer note : ResNet-50 activations do NOT fit in on-chip SPRAM.",
        f"  Strategy: tile convolutions so peak activation buffer << 128 KiB.",
        f"  Maximum safe activation tile: ~64 KiB (leaving 64 KiB for firmware stack/data).",
    ]

    report = "\n".join(lines)
    print("\n" + report + "\n")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(report + "\n")
    print(f"Saved {out_path}")


# =============================================================================
# Test input — funny_monkey (local, no network)
# =============================================================================

# Path to the pre-normalized CHW float32 tensor (3×224×224, ImageNet mean/std).
# Generated once from scripts/funny_monkey.webp; never re-downloaded.
_MONKEY_TENSOR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "funny_monkey_tensor.bin")
_MONKEY_WEBP_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "funny_monkey.webp")


def get_test_image_tensor() -> torch.Tensor:
    """Return the funny_monkey input as a (1, 3, 224, 224) float32 tensor.

    Priority:
      1. Load from funny_monkey_tensor.bin (raw float32 CHW, already normalized).
      2. Fall back: render funny_monkey.webp through the standard ImageNet transform
         and save the result as funny_monkey_tensor.bin for next time.

    No network access is performed.
    """
    if os.path.exists(_MONKEY_TENSOR_PATH):
        data = np.fromfile(_MONKEY_TENSOR_PATH, dtype=np.float32)  # 150528 floats
        if data.size == INPUT_C * INPUT_H * INPUT_W:
            print(f"Loaded test input from {_MONKEY_TENSOR_PATH}")
            tensor = torch.from_numpy(data.reshape(1, INPUT_C, INPUT_H, INPUT_W))
            return tensor
        print(f"  WARNING: {_MONKEY_TENSOR_PATH} has unexpected size {data.size}, rebuilding")

    # Rebuild from .webp source
    try:
        from PIL import Image
        import torchvision.transforms as T
    except ImportError as e:
        raise RuntimeError(f"Need pillow to rebuild tensor from .webp: {e}")

    print(f"Building tensor from {_MONKEY_WEBP_PATH} ...")
    img = Image.open(_MONKEY_WEBP_PATH).convert("RGB")
    transform = T.Compose([
        T.Resize((INPUT_H, INPUT_W)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    tensor = transform(img).unsqueeze(0)   # [1, 3, 224, 224]
    arr = tensor[0].numpy().astype(np.float32)
    arr.tofile(_MONKEY_TENSOR_PATH)
    print(f"  Saved normalized tensor → {_MONKEY_TENSOR_PATH}")
    return tensor


# =============================================================================
# Export: input.h
# =============================================================================

def export_input_header(tensor_f32: torch.Tensor, out_path: str) -> np.ndarray:
    """Quantize the funny_monkey (1,3,224,224) float tensor to int8 and write as C header."""
    arr = tensor_f32[0].numpy()                   # [3, 224, 224]
    q, scale = quantize_per_tensor_sym(arr)
    flat = q.flatten().tolist()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("// AUTO-GENERATED by gen_resnet50_model.py — do not edit\n")
        f.write(f"// Model: {MODEL_VARIANT}\n")
        f.write(f"// Input: {INPUT_C}×{INPUT_H}×{INPUT_W} int8, CHW layout\n")
        f.write(f"// Quantization scale (float): {scale:.8f}\n")
        f.write(f"// (scale = max_abs / 127, symmetric per-tensor)\n\n")
        f.write("#ifndef RESNET50_INPUT_DATA_H\n#define RESNET50_INPUT_DATA_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define INPUT_H       {INPUT_H}\n")
        f.write(f"#define INPUT_W       {INPUT_W}\n")
        f.write(f"#define INPUT_C       {INPUT_C}\n")
        f.write(f"#define INPUT_NELEMS  {len(flat)}\n\n")
        f.write(f"const int8_t RESNET50_INPUT[{len(flat)}] = {{\n")
        for i, v in enumerate(flat):
            f.write(f"{v:4d},")
            if (i + 1) % 16 == 0:
                f.write("\n")
        f.write("\n};\n\n#endif /* RESNET50_INPUT_DATA_H */\n")

    print(f"Saved {out_path}  ({len(flat)} int8 values, scale={scale:.6f})")
    return q.astype(np.int8)


# =============================================================================
# Export: expected_fp32.h  (golden reference from torchvision in fp32)
# =============================================================================

def run_fp32_inference(model: torch.nn.Module,
                       tensor_f32: torch.Tensor) -> Tuple[List[float], int, str]:
    """Run FP32 inference and return (logits, top1_class, top5_str)."""
    with torch.no_grad():
        logits = model(tensor_f32)[0]   # [1000]
    top5  = torch.topk(logits, 5)
    top1_idx = int(top5.indices[0])
    top5_str = ", ".join(f"{int(i)}({float(v):.3f})" for i, v in zip(top5.indices, top5.values))
    return logits.numpy().tolist(), top1_idx, top5_str


def export_expected_fp32_header(logits: List[float], top1: int,
                                top5_str: str, out_path: str) -> None:
    """Write expected logits (fp32) and classification result as a C header."""
    # Use a compact hash so firmware can compare even without storing 1000 floats.
    logits_np = np.array(logits, dtype=np.float32)
    logits_bytes = logits_np.tobytes()
    sha = hashlib.sha256(logits_bytes).hexdigest()

    # Also compute a simple uint32 checksum (sum of bit-cast uint32 values)
    u32 = np.frombuffer(logits_bytes, dtype=np.uint32)
    checksum = int(np.sum(u32.astype(np.uint64)) & 0xFFFFFFFF)

    top5_indices = [int(i) for i in np.argsort(logits)[-5:][::-1]]
    top5_values  = [float(logits[i]) for i in top5_indices]

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("// AUTO-GENERATED by gen_resnet50_model.py — do not edit\n")
        f.write(f"// Model: {MODEL_VARIANT}\n")
        f.write(f"// FP32 golden reference (torchvision, no quantization)\n\n")
        f.write("#ifndef RESNET50_EXPECTED_FP32_H\n#define RESNET50_EXPECTED_FP32_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define EXPECTED_TOP1_CLASS   {top1}\n")
        f.write(f"// Top-5: {top5_str}\n")
        f.write(f"#define EXPECTED_LOGITS_SHA256 \"{sha}\"\n")
        f.write(f"#define EXPECTED_LOGITS_U32SUM  0x{checksum:08X}u\n\n")
        f.write(f"// Top-5 indices + raw FP32 logit values\n")
        f.write(f"const int32_t  EXPECTED_TOP5_IDX[5]   = {{ {', '.join(str(i) for i in top5_indices)} }};\n")
        f.write(f"const float    EXPECTED_TOP5_LOGIT[5] = {{ {', '.join(f'{v:.6f}f' for v in top5_values)} }};\n\n")
        f.write(f"// Full logit array (FP32, 1000 classes)\n")
        f.write(f"const float EXPECTED_LOGITS[{N_CLASSES}] = {{\n")
        for i, v in enumerate(logits):
            f.write(f"    {v:.6f}f,")
            if (i + 1) % 4 == 0:
                f.write("\n")
        f.write("\n};\n\n")
        f.write("#endif /* RESNET50_EXPECTED_FP32_H */\n")

    print(f"Saved {out_path}  (top1={top1}, sha256={sha[:16]}...)")


# =============================================================================
# Export: model_constants.h  (hard-coded constants for firmware)
# =============================================================================

def export_model_constants_header(layers: Dict[str, Tuple[np.ndarray, np.ndarray]],
                                  bd_blob: bytes,
                                  entries: List,
                                  out_path: str) -> None:
    """Write a C header with model topology + VWB2 blob layout constants for firmware.

    Includes per-tensor offsets extracted from the VWB2 tensor table so firmware
    can seek directly to any layer's weights or biases without scanning (§3.3).
    """
    # Build a name→entry lookup from the VWB2 tensor table
    entry_by_hash = {e.name_hash: e for e in entries}
    def _entry(tensor_name: str) -> object:
        h = bd.fnv1a32(tensor_name)
        e = entry_by_hash.get(h)
        if e is None:
            raise KeyError(f"Tensor {tensor_name!r} not found in VWB2 blob")
        e.name = tensor_name  # resolve name lazily
        return e

    enum_vals = list(layers.keys())

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("// AUTO-GENERATED by gen_resnet50_model.py — do not edit\n")
        f.write(f"// Model: {MODEL_VARIANT}\n\n")
        f.write("#ifndef RESNET50_MODEL_CONSTANTS_H\n#define RESNET50_MODEL_CONSTANTS_H\n\n")
        f.write("#include <stdint.h>\n\n")

        f.write("// ---- Model identity ----\n")
        f.write(f'#define RESNET50_MODEL_VARIANT  "{MODEL_VARIANT}"\n')
        f.write(f"#define RESNET50_MODEL_INPUT_H  {INPUT_H}\n")
        f.write(f"#define RESNET50_MODEL_INPUT_W  {INPUT_W}\n")
        f.write(f"#define RESNET50_MODEL_INPUT_C  {INPUT_C}\n")
        f.write(f"#define RESNET50_N_CLASSES      {N_CLASSES}\n\n")

        f.write("// ---- BlockDialect parameters ----\n")
        f.write(f"#define BD_BLOCK_SIZE           {BLOCK_SIZE}\n")
        f.write(f"#define BD_BLOCK_BYTES          18   /* 2-byte meta + 16-byte packed codes */\n")
        f.write(f"/* Biases stored as float32 in separate DTYPE_FLOAT32 tensor entries */\n\n")

        f.write("// ---- Flash / WeightStore layout ----\n")
        f.write(f"#define FLASH_WEIGHT_BASE       0x{FLASH_WEIGHT_BASE:08X}u\n")
        f.write(f"// FLASH_WEIGHT_OFFSET is the raw SPI-flash byte offset where weights are programmed.\n")
        f.write(f"// The SoC flashOffset parameter already translates CPU 0x20000000 → flash 0x{FLASH_OFFSET_BYTES:06X}.\n")
        f.write(f"// Firmware must NOT add this to FLASH_WEIGHT_BASE.\n")
        f.write(f"#define FLASH_WEIGHT_OFFSET     0x{FLASH_OFFSET_BYTES:08X}u  /* raw flash byte offset (for iceprog -o) */\n")
        f.write(f"#define WEIGHT_BLOB_ADDR        FLASH_WEIGHT_BASE  /* SoC window already starts at flash+FLASH_WEIGHT_OFFSET */\n\n")

        f.write("// ---- VWB2 blob sizes (bytes) ----\n")
        f.write(f"#define RESNET50_BD_BLOB_SIZE    {len(bd_blob)}u\n\n")

        f.write("// ---- Layer count ----\n")
        f.write(f"#define RESNET50_N_LAYERS       {len(layers)}\n\n")

        f.write("// ---- Layer enum (matches VWB2 tensor table order) ----\n")
        f.write("typedef enum {\n")
        for i, name in enumerate(enum_vals):
            cname = name.upper().replace(".", "_").replace("-", "_")
            f.write(f"    RESNET50_LAYER_{cname} = {i},\n")
        f.write(f"    RESNET50_LAYER_COUNT = {len(enum_vals)}\n")
        f.write("} Resnet50LayerId;\n\n")

        f.write("// ---- Per-layer shape info + VWB2 tensor offsets ----\n")
        f.write("// weight_offset / bias_offset: byte offsets from WEIGHT_BLOB_ADDR\n")
        f.write("//   (i.e., absolute address = WEIGHT_BLOB_ADDR + offset)\n")
        f.write("// weight_bytes / bias_bytes: byte counts for each tensor payload\n")
        f.write("// weight_blocks: number of 18-byte BD blocks in the weight tensor\n")
        f.write("typedef struct {\n")
        f.write("    const char*  name;\n")
        f.write("    uint32_t     w_elements;\n")
        f.write("    uint32_t     b_elements;\n")
        f.write("    uint32_t     w_bd_blocks;\n")
        f.write("    uint32_t     weight_offset;  /* bytes from WEIGHT_BLOB_ADDR */\n")
        f.write("    uint32_t     weight_bytes;\n")
        f.write("    uint32_t     bias_offset;    /* bytes from WEIGHT_BLOB_ADDR */\n")
        f.write("    uint32_t     bias_bytes;\n")
        f.write("} Resnet50LayerInfo;\n\n")

        # Read VWB2 header to determine data_offset (so we can compute absolute file offset)
        # The data_offset is embedded in blob bytes [20:24]
        data_offset = struct.unpack_from("<I", bd_blob, 20)[0]

        f.write(f"static const Resnet50LayerInfo RESNET50_LAYERS[{len(layers)}] = {{\n")
        for name, (w, b) in layers.items():
            n_blocks = (w.size + BLOCK_SIZE - 1) // BLOCK_SIZE
            we = _entry(f"{name}.weight")
            be = _entry(f"{name}.bias")
            # tensor_offset is relative to data_offset; absolute from blob start:
            w_abs = data_offset + we.tensor_offset
            b_abs = data_offset + be.tensor_offset
            cname = name.replace(".", "_")
            f.write(f'    /* {name} */\n')
            f.write(f'    {{"{name}", {w.size}u, {b.size}u, {n_blocks}u,\n')
            f.write(f'     0x{w_abs:08X}u, {we.tensor_bytes}u,\n')
            f.write(f'     0x{b_abs:08X}u, {be.tensor_bytes}u}},\n')
        f.write("};\n\n")

        f.write("#endif /* RESNET50_MODEL_CONSTANTS_H */\n")

    print(f"Saved {out_path}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--outdir", default="scripts/resnet50_artifacts",
        help="Directory for output artifacts (default: scripts/resnet50_artifacts/)")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print model info and weight budget estimate only, do not write files")
    args = parser.parse_args()

    od = args.outdir
    os.makedirs(od, exist_ok=True)

    # 1. Load model
    model = load_resnet50()

    # 2. Fold BN into Conv (reduces runtime ops + tensor count)
    layers = extract_folded_weights(model)

    # 3. Test image — funny_monkey, no network
    tensor_f32 = get_test_image_tensor()

    # 4. FP32 golden inference
    print("\nRunning FP32 inference (golden reference) ...")
    logits, top1, top5_str = run_fp32_inference(model, tensor_f32)
    print(f"  Top-1 class : {top1}")
    print(f"  Top-5       : {top5_str}")

    if args.dry_run:
        # Estimate BD blob size without encoding
        BD_BLOCK_BYTES = 18
        n_blocks = sum((w.size + BLOCK_SIZE - 1) // BLOCK_SIZE for w, b in layers.values())
        bias_bytes = sum(b.size * 4 for _, b in layers.values())
        est = 16 + n_blocks * BD_BLOCK_BYTES + 8 * len(layers) + bias_bytes
        print(f"\n[dry-run] Estimated BD blob: {est:,} bytes ({est/1024/1024:.2f} MiB)")
        return

    # 5. Export BlockDialect VWB2 blob (the only weight format)
    bd_blob, bd_entries = export_weights_bd(layers, os.path.join(od, "weights_bd.bin"))

    # 6. Weight budget report
    weight_budget_report(layers, bd_blob, os.path.join(od, "weight_budget.txt"))

    # 7. input.h
    export_input_header(tensor_f32, os.path.join(od, "input.h"))

    # 8. expected_fp32.h
    export_expected_fp32_header(logits, top1, top5_str,
                                os.path.join(od, "expected_fp32.h"))

    # 9. model_constants.h
    export_model_constants_header(layers, bd_blob, bd_entries, os.path.join(od, "model_constants.h"))

    print(f"\nAll artifacts written to: {od}/")
    print(textwrap.dedent(f"""\
        Next steps (RESNET50_FPGA_PLAN.md §3):
          • Review weight_budget.txt to confirm blob fits in flash.
          • Copy model_constants.h to the firmware target directory.
          • Add input.h + expected_fp32.h to the smoke-test firmware.
          • Flash weights_bd.bin at offset 0x{FLASH_OFFSET_BYTES:08x} in the SPI flash image.
    """))


if __name__ == "__main__":
    main()
