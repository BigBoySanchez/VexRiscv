#!/usr/bin/env python3
"""
quantized_reference.py — Python integer-quantized reference for ResNet-50 Milestone 1

Replicates the firmware's exact integer arithmetic:
  - BlockDialect weight decoding to half-unit int16
  - Per-block exponent shifting during accumulation
  - int8 activations, int32 accumulators
  - Bias as int32 (bias_f32 * 2^shift)
  - ReLU, clamp to [0, 127] for int8 output

IMPORTANT: This script's core BD4 math must exactly match the C firmware.
Verify with: python3 scripts/validate_math_match.py
This runs comprehensive tests comparing Python functions against C logic from bd_act.h

Usage:
  python3 scripts/quantized_reference.py [--blob PATH] [--input PATH] [--shift N]

Produces:
  - Per-stage activation hashes (sum-of-int8)
  - Final logit values (int32)
  - Top-1 class prediction
  - Output file with expected checksums for firmware comparison
"""

from __future__ import annotations

import argparse
import hashlib
import os
import struct
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import textwrap
# Insert parent directory so we can import blockdialect_codec
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import blockdialect_codec as bd

# =============================================================================
# VWB2 blob reader (mirrors weight_blob.h)
# =============================================================================

VWB2_MAGIC = 0x56574232  # 'VWB2'
VWB2_HDR_SIZE = 32
VWB2_ENTRY_SIZE = 40
DTYPE_BD4 = 0       # matches blockdialect_codec.py and VWB2_DTYPE_BD4 in weight_blob.h
DTYPE_FLOAT32 = 1   # matches blockdialect_codec.py and VWB2_DTYPE_FLOAT32


def vwb2_read_header(blob: bytes) -> dict:
    # All 8 fields are uint32 (32-byte header, all LE)
    (magic, version, tensor_count, block_size,
     table_offset, data_offset, data_bytes, reserved) = struct.unpack_from('<IIIIIIII', blob, 0)
    assert magic == VWB2_MAGIC, f"Bad magic: 0x{magic:08X}"
    return {
        'magic': magic,
        'version': version,
        'tensor_count': tensor_count,
        'block_size': block_size,
        'table_offset': table_offset,
        'data_offset': data_offset,
        'data_bytes': data_bytes,
    }


def vwb2_read_entries(blob: bytes, hdr: dict) -> List[dict]:
    entries = []
    base = hdr['table_offset']   # use header's table_offset, not hardcoded 32
    for i in range(hdr['tensor_count']):
        off = base + i * VWB2_ENTRY_SIZE
        # All fields are uint32; entry layout (40 bytes):
        #   name_hash(4), dtype(4), tensor_offset(4), tensor_bytes(4),
        #   n_elements(4), shape_ndim(4), shape[4](16)
        (name_hash, dtype, tensor_offset, tensor_bytes,
         n_elements, shape_ndim) = struct.unpack_from('<IIIIII', blob, off)
        shape4 = struct.unpack_from('<IIII', blob, off + 24)
        shape = tuple(shape4[:shape_ndim])
        entries.append({
            'name_hash':     name_hash,
            'dtype':         dtype,
            'tensor_offset': tensor_offset,
            'tensor_bytes':  tensor_bytes,
            'n_elements':    n_elements,
            'shape':         shape,
        })
    return entries


def vwb2_find(entries: List[dict], name: str) -> dict:
    h = bd.fnv1a32(name)
    for e in entries:
        if e['name_hash'] == h:
            return e
    raise KeyError(f"Tensor {name!r} (hash=0x{h:08X}) not found in blob")


def vwb2_get_bd_blocks(blob: bytes, hdr: dict, entry: dict) -> bytes:
    """Return the raw BD block bytes for a BD4 tensor entry.

    BD4 payloads start with an 8-byte sub-header (n_elements u32 | n_blocks u32)
    followed by the actual 18-byte block records.  We skip that sub-header.
    """
    payload_start = hdr['data_offset'] + entry['tensor_offset']
    payload_end   = payload_start + entry['tensor_bytes']
    BD4_SUBHDR = 8  # n_elements(4) + n_blocks(4)
    return blob[payload_start + BD4_SUBHDR : payload_end]


def vwb2_get_f32(blob: bytes, hdr: dict, entry: dict) -> np.ndarray:
    """Return float32 array for a FLOAT32 tensor entry."""
    start = hdr['data_offset'] + entry['tensor_offset']
    end   = start + entry['tensor_bytes']
    return np.frombuffer(blob[start:end], dtype=np.float32)[:entry['n_elements']].copy()


# =============================================================================
# BD decode to half-units (mirrors bd_decode_sw.h)
# =============================================================================

def bd_decode_block_hu(block_bytes: bytes) -> Tuple[np.ndarray, int, int]:
    """Decode 18-byte BD block → (int16[32] half-units, dialect_id, shared_exp).

    Returns signed half-unit values matching the C firmware exactly.
    Returns zeros if block_bytes is shorter than 18 bytes (padding region).
    """
    if len(block_bytes) < 18:
        return np.zeros(32, dtype=np.int16), 0, 0
    mhi, mlo = block_bytes[0], block_bytes[1]
    did = (mhi >> 4) & 0xF
    seb = ((mhi & 0xF) << 1) | ((mlo >> 7) & 1)

    table = bd.DIALECTS_HALF_UNITS[did]
    hu = np.zeros(32, dtype=np.int16)

    for i in range(16):
        byte = block_bytes[2 + i]
        for nibble in range(2):
            code = (byte >> (4 * (1 - nibble))) & 0xF
            sign = (code >> 3) & 1
            idx = code & 0x07
            mag = table[idx]
            hu[2 * i + nibble] = np.int16(-mag if sign else mag)

    return hu, did, seb


def bd_decode_tensor_hu(block_bytes: bytes, n_elements: int
                       ) -> List[Tuple[np.ndarray, int, int]]:
    """Decode all blocks of a tensor, returning (hu[32], did, seb) per block."""
    n_blocks = (n_elements + 31) // 32
    results = []
    for b in range(n_blocks):
        blk = block_bytes[b * 18:(b + 1) * 18]
        hu, did, seb = bd_decode_block_hu(blk)
        results.append((hu, did, seb))
    return results


def bd_decode_to_f32(block_bytes: bytes, n_elems: int) -> np.ndarray:
    """Decode BD blocks → float32 values using the exact BD formula:
       val = half_units * 0.5 * 2^(seb-16)
    This is the canonical weight dequantization path for float32 inference.
    """
    n_blocks = (n_elems + 31) // 32
    result = np.zeros(n_elems, dtype=np.float32)
    for b in range(n_blocks):
        blk = block_bytes[b * 18:(b + 1) * 18]
        hu, did, seb = bd_decode_block_hu(blk)
        # Paper formula (blockdialect_codec.py line 36):
        #   e = shared_exp_bits - 15   (FP16 exponent bias = 15)
        #   value = sign * (0.5 * half_units) * 2^e
        scale = np.float32(0.5) * np.float32(2.0 ** (int(seb) - 15))
        start, end = b * 32, min((b + 1) * 32, n_elems)
        result[start:end] = hu[:end - start].astype(np.float32) * scale
    return result


def fake_q_i8(x: np.ndarray) -> Tuple[np.ndarray, float]:
    """Symmetric per-tensor fake-quantize float32 → int8 + scale.
    Returns (quantized_int8, scale) where x ≈ int8 * scale.
    """
    m = float(np.max(np.abs(x)))
    if m == 0.0:
        return np.zeros(x.shape, dtype=np.int8), 1e-8
    scale = m / 127.0
    return np.clip(np.round(x / scale), -127, 127).astype(np.int8), scale


# =============================================================================
# Layer topology (mirrors resnet50_layers.h and gen_resnet50_model.py)
# =============================================================================

# Layer name stems for ResNet-50 (54 layers: stem + 16 bottleneck×3 + 4 downsample + FC)
STEM_LAYERS = ['conv1']

BOTTLENECK_NAMES = [
    # (stage, block_idx, [conv1, conv2, conv3], optional downsample)
    # Spatial dims for 96×96 input:
    #   After stem (7×7 s2 + maxpool s2): 24×24
    #   layer2 stride-2 first block: 24→12     layer3 stride-2: 12→6     layer4 stride-2: 6→3
    # Stage 1: 3 blocks (input 24×24, stride 1 throughout)
    ('layer1.0', 64, 64, 256, 24, 24, 1, True),
    ('layer1.1', 256, 64, 256, 24, 24, 1, False),
    ('layer1.2', 256, 64, 256, 24, 24, 1, False),
    # Stage 2: 4 blocks (first block stride 2: 24→12, rest 12×12)
    ('layer2.0', 256, 128, 512, 24, 24, 2, True),
    ('layer2.1', 512, 128, 512, 12, 12, 1, False),
    ('layer2.2', 512, 128, 512, 12, 12, 1, False),
    ('layer2.3', 512, 128, 512, 12, 12, 1, False),
    # Stage 3: 6 blocks (first block stride 2: 12→6, rest 6×6)
    ('layer3.0', 512, 256, 1024, 12, 12, 2, True),
    ('layer3.1', 1024, 256, 1024, 6, 6, 1, False),
    ('layer3.2', 1024, 256, 1024, 6, 6, 1, False),
    ('layer3.3', 1024, 256, 1024, 6, 6, 1, False),
    ('layer3.4', 1024, 256, 1024, 6, 6, 1, False),
    ('layer3.5', 1024, 256, 1024, 6, 6, 1, False),
    # Stage 4: 3 blocks (first block stride 2: 6→3, rest 3×3)
    ('layer4.0', 1024, 512, 2048, 6, 6, 2, True),
    ('layer4.1', 2048, 512, 2048, 3, 3, 1, False),
    ('layer4.2', 2048, 512, 2048, 3, 3, 1, False),
]

FC_LAYER = 'fc'
N_CLASSES = 1000

# Must match gen_resnet50_model.py
INPUT_H = 96
INPUT_W = 96
INPUT_C = 3


# =============================================================================
# Integer convolution kernels (mirrors resnet50_conv.h)
# =============================================================================

def conv_hash(act: np.ndarray) -> int:
    """Compute sum-of-int8 hash matching firmware's compute_hash()."""
    return int(np.sum(act.astype(np.int64))) & 0xFFFFFFFF


def conv_1x1(inp: np.ndarray, weight_blocks: bytes, bias_f32: np.ndarray,
             in_c: int, out_c: int, h: int, w: int,
             stride: int, shift: int) -> np.ndarray:
    """1×1 convolution with BD weights, matching firmware's conv_1x1().

    inp: int8 [in_c, h, w]
    Returns: int8 [out_c, oh, ow]
    """
    oh, ow = h // stride, w // stride
    out = np.zeros((out_c, oh, ow), dtype=np.int8)
    n_blocks_per_oc = (in_c + 31) // 32

    for oc in range(out_c):
        bias_i32 = int(bias_f32[oc] * (1 << shift))
        oc_blocks = weight_blocks[oc * n_blocks_per_oc * 18:
                                  (oc + 1) * n_blocks_per_oc * 18]
        blocks_decoded = bd_decode_tensor_hu(oc_blocks, in_c)

        for oy in range(oh):
            for ox in range(ow):
                iy, ix = oy * stride, ox * stride
                acc = np.int64(0)

                elem_done = 0
                for hu, did, seb in blocks_decoded:
                    count = min(32, in_c - elem_done)
                    block_sum = np.int64(0)
                    for i in range(count):
                        ic = elem_done + i
                        a = int(inp[ic, iy, ix])
                        block_sum += int(hu[i]) * a
                    s = seb - 16
                    if s >= 0:
                        acc += block_sum << s
                    else:
                        acc += block_sum >> (-s)
                    elem_done += count

                acc += bias_i32
                v = int(acc) >> shift
                if v < 0: v = 0
                if v > 127: v = 127
                out[oc, oy, ox] = np.int8(v)

    return out


def conv_3x3(inp: np.ndarray, weight_blocks: bytes, bias_f32: np.ndarray,
             in_c: int, out_c: int, h: int, w: int,
             stride: int, pad: int, shift: int) -> np.ndarray:
    """3×3 convolution with BD weights, matching firmware's conv_3x3().

    inp: int8 [in_c, h, w]
    Returns: int8 [out_c, oh, ow]
    """
    oh, ow = h // stride, w // stride
    out = np.zeros((out_c, oh, ow), dtype=np.int8)
    kernel_elems = in_c * 3 * 3
    n_blocks_per_oc = (kernel_elems + 31) // 32

    for oc in range(out_c):
        bias_i32 = int(bias_f32[oc] * (1 << shift))
        oc_blocks = weight_blocks[oc * n_blocks_per_oc * 18:
                                  (oc + 1) * n_blocks_per_oc * 18]
        blocks_decoded = bd_decode_tensor_hu(oc_blocks, kernel_elems)

        for oy in range(oh):
            for ox in range(ow):
                acc = np.int64(0)
                elem_done = 0

                for hu, did, seb in blocks_decoded:
                    count = min(32, kernel_elems - elem_done)
                    block_sum = np.int64(0)
                    for i in range(count):
                        flat = elem_done + i
                        ic = flat // 9
                        k_rem = flat % 9
                        ky = k_rem // 3
                        kx = k_rem % 3
                        iy = oy * stride - pad + ky
                        ix = ox * stride - pad + kx
                        a = 0
                        if 0 <= iy < h and 0 <= ix < w:
                            a = int(inp[ic, iy, ix])
                        block_sum += int(hu[i]) * a
                    s = seb - 16
                    if s >= 0:
                        acc += block_sum << s
                    else:
                        acc += block_sum >> (-s)
                    elem_done += count

                acc += bias_i32
                v = int(acc) >> shift
                if v < 0: v = 0
                if v > 127: v = 127
                out[oc, oy, ox] = np.int8(v)

    return out


def conv_7x7(inp: np.ndarray, weight_blocks: bytes, bias_f32: np.ndarray,
             in_c: int, out_c: int, h: int, w: int,
             stride: int, pad: int, shift: int) -> np.ndarray:
    """7×7 convolution matching firmware's conv_7x7 (stem)."""
    oh, ow = h // stride, w // stride
    out = np.zeros((out_c, oh, ow), dtype=np.int8)
    kernel_elems = in_c * 7 * 7
    n_blocks_per_oc = (kernel_elems + 31) // 32

    for oc in range(out_c):
        bias_i32 = int(bias_f32[oc] * (1 << shift))
        oc_blocks = weight_blocks[oc * n_blocks_per_oc * 18:
                                  (oc + 1) * n_blocks_per_oc * 18]
        blocks_decoded = bd_decode_tensor_hu(oc_blocks, kernel_elems)

        for oy in range(oh):
            for ox in range(ow):
                acc = np.int64(0)
                elem_done = 0

                for hu, did, seb in blocks_decoded:
                    count = min(32, kernel_elems - elem_done)
                    block_sum = np.int64(0)
                    for i in range(count):
                        flat = elem_done + i
                        ic = flat // 49
                        k_rem = flat % 49
                        ky = k_rem // 7
                        kx = k_rem % 7
                        iy = oy * stride - pad + ky
                        ix = ox * stride - pad + kx
                        a = 0
                        if 0 <= iy < h and 0 <= ix < w:
                            a = int(inp[ic, iy, ix])
                        block_sum += int(hu[i]) * a
                    s = seb - 16
                    if s >= 0:
                        acc += block_sum << s
                    else:
                        acc += block_sum >> (-s)
                    elem_done += count

                acc += bias_i32
                v = int(acc) >> shift
                if v < 0: v = 0
                if v > 127: v = 127
                out[oc, oy, ox] = np.int8(v)

    return out


def maxpool_3x3_s2(inp: np.ndarray) -> np.ndarray:
    """3×3 max pool stride 2 pad 1, matching firmware."""
    c, h, w = inp.shape
    oh, ow = h // 2, w // 2
    out = np.full((c, oh, ow), -128, dtype=np.int8)
    for ch in range(c):
        for oy in range(oh):
            for ox in range(ow):
                mx = -128
                for ky in range(3):
                    iy = oy * 2 - 1 + ky
                    for kx in range(3):
                        ix = ox * 2 - 1 + kx
                        if 0 <= iy < h and 0 <= ix < w:
                            v = int(inp[ch, iy, ix])
                            if v > mx:
                                mx = v
                out[ch, oy, ox] = np.int8(mx)
    return out


def avgpool_global(inp: np.ndarray) -> np.ndarray:
    """Global average pool (7×7), matching firmware."""
    c, h, w = inp.shape
    out = np.zeros(c, dtype=np.int8)
    n = h * w
    for ch in range(c):
        total = 0
        for y in range(h):
            for x in range(w):
                total += int(inp[ch, y, x])
        avg = total // n
        if avg < -128: avg = -128
        if avg > 127: avg = 127
        out[ch] = np.int8(avg)
    return out


def fc_linear(inp: np.ndarray, weight_blocks: bytes, bias_f32: np.ndarray,
              in_features: int, out_features: int) -> np.ndarray:
    """Fully connected layer, returns int64 logits."""
    logits = np.zeros(out_features, dtype=np.int64)
    n_blocks_per_oc = (in_features + 31) // 32

    for oc in range(out_features):
        oc_blocks = weight_blocks[oc * n_blocks_per_oc * 18:
                                  (oc + 1) * n_blocks_per_oc * 18]
        blocks_decoded = bd_decode_tensor_hu(oc_blocks, in_features)
        acc = np.int64(0)
        elem_done = 0

        for hu, did, seb in blocks_decoded:
            count = min(32, in_features - elem_done)
            block_sum = np.int64(0)
            for i in range(count):
                block_sum += int(hu[i]) * int(inp[elem_done + i])
            s = seb - 16
            if s >= 0:
                acc += block_sum << s
            else:
                acc += block_sum >> (-s)
            elem_done += count

        acc += _bias_scale_c(float(bias_f32[oc]), 7)
        logits[oc] = acc

    return logits


def relu_inplace(act: np.ndarray) -> np.ndarray:
    """ReLU on int8 array."""
    act[act < 0] = 0
    return act


def add_relu(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise add + ReLU, matching firmware's add_relu()."""
    out = np.clip(a.astype(np.int32) + b.astype(np.int32), 0, 127).astype(np.int8)
    return out


def add_relu_bd4(bd_a: bytes, bd_b: bytes, n_elements: int) -> bytes:
    """BD4 residual add + ReLU — paper-faithful, no int8 intermediate.

    Matches the C firmware's add_relu_bd4() exactly:
      - Unpacks each block to int8 via bd_act_unpack32 (clamped to [-128,127])
      - Adds element-wise as int32
      - Applies ReLU (clamp negatives to 0)
      - Repacks via bd_act_pack32_twostage (two-stage dialect selector)

    bd_out may alias bd_a or bd_b — safe because blocks are processed one at a time.
    """
    n_blocks = (n_elements + 31) // 32
    out = bytearray()
    for b_idx in range(n_blocks):
        off = b_idx * 18
        # Unpack to int8 — matches bd_act_unpack32() in firmware (clamps to [-128,127])
        va = bd_act_unpack_tensor(bd_a[off:off + 18], 32)
        vb = bd_act_unpack_tensor(bd_b[off:off + 18], 32)
        # Add as int32 + ReLU — matches firmware inner loop
        tmp = np.zeros(32, dtype=np.int32)
        for i in range(32):
            s = int(va[i]) + int(vb[i])
            tmp[i] = 0 if s < 0 else s
        # Repack via two-stage selector — matches bd_act_pack32_twostage()
        out += _bd4_pack_twostage(tmp)
    return bytes(out)


def add_relu_bd4_i16(bd_a: bytes, bd_b: bytes, n_elements: int) -> bytes:
    """BD4 residual add + ReLU with int16 intermediate (widened fix).

    Proposed firmware change: replace the int8 local variables in
    add_relu_bd4() with int16.  This preserves the full BD4-representable
    magnitude before the add, avoiding the precision loss that occurs when a
    convolution accumulator > 127 (or < -128) is silently clamped to int8
    during unpack, corrupting the residual sum.

    Only the unpack dtype differs from add_relu_bd4(); the add, ReLU, and
    repack steps are identical.
    """
    n_blocks = (n_elements + 31) // 32
    out = bytearray()
    for b_idx in range(n_blocks):
        off = b_idx * 18
        # Unpack to int16 — wider intermediate, no int8 clamp
        va = bd_act_unpack_tensor_i16(bd_a[off:off + 18], 32)
        vb = bd_act_unpack_tensor_i16(bd_b[off:off + 18], 32)
        # Add as int32 + ReLU — identical to add_relu_bd4()
        tmp = np.zeros(32, dtype=np.int32)
        for i in range(32):
            s = int(va[i]) + int(vb[i])
            tmp[i] = 0 if s < 0 else s
        out += _bd4_pack_twostage(tmp)
    return bytes(out)


def conv_f32(inp: np.ndarray, w: np.ndarray, b: np.ndarray,
            kH: int, kW: int, stride: int = 1, pad: int = 0) -> np.ndarray:
    """Vectorised float32 convolution via im2col + matmul.

    inp : float32 [in_c, H, W]
    w   : float32 [out_c, in_c, kH, kW]
    b   : float32 [out_c]
    Returns float32 [out_c, oH, oW]  (NO activation applied here)
    """
    from numpy.lib.stride_tricks import as_strided
    inp = np.ascontiguousarray(inp.astype(np.float32))
    if pad:
        inp = np.pad(inp, ((0, 0), (pad, pad), (pad, pad)), mode='constant')
    in_c, pH, pW = inp.shape
    oH = (pH - kH) // stride + 1
    oW = (pW - kW) // stride + 1
    # Build windows [in_c, oH, oW, kH, kW] via stride tricks
    s = inp.strides
    windows = as_strided(
        inp,
        shape=(in_c, oH, oW, kH, kW),
        strides=(s[0], s[1] * stride, s[2] * stride, s[1], s[2]),
        writeable=False,
    )
    # Transpose to (in_c, kH, kW, oH, oW) so that flattening the first 3 dims
    # matches weight layout [out_c, in_c, kH, kW] → [out_c, in_c*kH*kW]
    col = windows.transpose(0, 3, 4, 1, 2).reshape(in_c * kH * kW, oH * oW)
    w_mat = w.reshape(w.shape[0], -1).astype(np.float32)     # [out_c, in_c*kH*kW]
    out = (w_mat @ col + b[:, np.newaxis]).reshape(w.shape[0], oH, oW)
    return out


# =============================================================================
# BD activation pack/unpack (mirrors bd_act.h)
# =============================================================================

def bd_act_pack_tensor(data_i8: np.ndarray) -> bytes:
    """Pack int8 tensor to BD A4 format matching firmware."""
    flat = data_i8.flatten().astype(np.int32)
    n = len(flat)
    n_blocks = (n + 31) // 32
    result = bytearray()

    for b in range(n_blocks):
        start = b * 32
        end = min(start + 32, n)
        blk = np.zeros(32, dtype=np.int32)
        blk[:end - start] = flat[start:end]

        signs = (blk < 0).astype(np.int32)
        abs_vals = np.abs(blk)
        max_abs = int(np.max(abs_vals)) if np.any(abs_vals > 0) else 0

        # Compute shared exponent
        if max_abs == 0:
            seb = 0
        else:
            import math
            seb = int(math.floor(math.log2(max_abs))) + 12
            if seb < 0: seb = 0
            if seb > 31: seb = 31

        # Scale to half-units
        if seb >= 16:
            scaled_hu = ((abs_vals * 2) + (1 << (seb - 16))) >> (seb - 16 + 1)
        else:
            scaled_hu = abs_vals << (16 - seb)
            scaled_hu = (scaled_hu + 1) >> 1  # rounding
        scaled_hu = np.clip(scaled_hu, 0, 15)

        # Brute-force dialect selection (MSE)
        best_did = 0
        best_mse = float('inf')
        for did in range(16):
            table = np.array(bd.DIALECTS_HALF_UNITS[did], dtype=np.int32)
            # Quantize each element to nearest table entry
            expanded = scaled_hu[:, None]  # [32, 1]
            diffs = np.abs(expanded - table[None, :])  # [32, 8]
            nearest_idx = np.argmin(diffs, axis=1)  # [32]
            nearest_hu = table[nearest_idx]
            mse = float(np.sum((scaled_hu - nearest_hu) ** 2))
            if mse < best_mse:
                best_mse = mse
                best_did = did

        # Encode
        table = np.array(bd.DIALECTS_HALF_UNITS[best_did], dtype=np.int32)
        expanded = scaled_hu[:, None]
        diffs = np.abs(expanded - table[None, :])
        nearest_idx = np.argmin(diffs, axis=1)

        # Build metadata
        meta_hi = ((best_did & 0xF) << 4) | ((seb >> 1) & 0xF)
        meta_lo = ((seb & 1) << 7)
        result.append(meta_hi)
        result.append(meta_lo)

        # Pack codes
        for i in range(16):
            code_hi = (signs[2 * i] << 3) | (int(nearest_idx[2 * i]) & 0x7)
            code_lo = (signs[2 * i + 1] << 3) | (int(nearest_idx[2 * i + 1]) & 0x7)
            result.append((code_hi << 4) | code_lo)

    return bytes(result)


def bd_act_unpack_tensor(packed: bytes, n_elements: int) -> np.ndarray:
    """Unpack BD A4 back to int8, matching firmware."""
    n_blocks = (n_elements + 31) // 32
    out = np.zeros(n_elements, dtype=np.int8)

    for b in range(n_blocks):
        blk = packed[b * 18:(b + 1) * 18]
        hu, did, seb = bd_decode_block_hu(blk)

        start = b * 32
        count = min(32, n_elements - start)

        for i in range(count):
            h = int(hu[i])
            sign = 1 if h < 0 else 0
            abs_h = abs(h)
            # Reconstruct: val = sign * (0.5 * abs_h) * 2^(seb-15)
            #            = sign * abs_h * 2^(seb-16)
            s = seb - 16
            if s >= 0:
                val = abs_h << s
            else:
                val = abs_h >> (-s)
            if sign:
                val = -val
            if val < -128: val = -128
            if val > 127: val = 127
            out[start + i] = np.int8(val)

    return out


def bd_act_unpack_tensor_i16(packed: bytes, n_elements: int) -> np.ndarray:
    """Unpack BD A4 → int16 WITHOUT the int8 clamp.

    This is the 'widened add_relu' variant: the BD4-encoded value may exceed
    [-128, 127] when a convolution accumulator was packed directly to BD4
    (no int8 clamp at the layer output, as in the paper-faithful path).
    Clamping to int8 before the add throws away significant bits; widening to
    int16 preserves the full representable BD4 range (seb up to 31 →
    half-unit 15 → value up to 15 * 2^(31-16) = 491520, well within int16
    for the seb values actually emitted by _bd4_pack_twostage).
    """
    n_blocks = (n_elements + 31) // 32
    out = np.zeros(n_elements, dtype=np.int16)

    for b in range(n_blocks):
        blk = packed[b * 18:(b + 1) * 18]
        hu, did, seb = bd_decode_block_hu(blk)

        start = b * 32
        count = min(32, n_elements - start)

        for i in range(count):
            h = int(hu[i])
            sign = 1 if h < 0 else 0
            abs_h = abs(h)
            s = seb - 16
            if s >= 0:
                val = abs_h << s
            else:
                val = abs_h >> (-s)
            if sign:
                val = -val
            # Clamp to int16 range instead of int8
            if val < -32768: val = -32768
            if val > 32767: val = 32767
            out[start + i] = np.int16(val)

    return out


# =============================================================================
# Full ResNet-50 inference
# =============================================================================

def get_weight_bias(blob: bytes, hdr: dict, entries: list, layer_name: str,
                    out_c: int = 0, kernel_elems: int = 0):
    """Get (weight_bd_bytes, bias_f32) for a layer.

    If out_c and kernel_elems are supplied, pads weight bytes to the full
    per-OC size so that per-OC block addressing never reads past the end.
    This matters only for conv1 where kernel_elems=147 is not a multiple of 32.
    """
    w_entry = vwb2_find(entries, layer_name + '.weight')
    b_entry = vwb2_find(entries, layer_name + '.bias')
    w_bytes = vwb2_get_bd_blocks(blob, hdr, w_entry)
    b_f32 = vwb2_get_f32(blob, hdr, b_entry)
    if out_c > 0 and kernel_elems > 0:
        n_blocks_per_oc = (kernel_elems + 31) // 32
        required = out_c * n_blocks_per_oc * 18
        if len(w_bytes) < required:
            w_bytes = w_bytes + b'\x00' * (required - len(w_bytes))
    return w_bytes, b_f32


def run_bottleneck(inp: np.ndarray, blob: bytes, hdr: dict, entries: list,
                   block_name: str, in_c: int, mid_c: int, out_c: int,
                   h: int, w: int, stride: int, has_downsample: bool,
                   shift: int) -> np.ndarray:
    """Run one bottleneck block, returning int8 output."""
    oh, ow = h // stride, w // stride

    # conv1 (1×1)
    w1, b1 = get_weight_bias(blob, hdr, entries, f'{block_name}.conv1')
    x = conv_1x1(inp, w1, b1, in_c, mid_c, h, w, 1, shift)
    relu_inplace(x)

    # conv2 (3×3)
    w2, b2 = get_weight_bias(blob, hdr, entries, f'{block_name}.conv2')
    x = conv_3x3(x, w2, b2, mid_c, mid_c, h, w, stride, 1, shift)
    relu_inplace(x)

    # conv3 (1×1)
    w3, b3 = get_weight_bias(blob, hdr, entries, f'{block_name}.conv3')
    x = conv_1x1(x, w3, b3, mid_c, out_c, oh, ow, 1, shift)

    # Skip
    if has_downsample:
        wd, bd_ = get_weight_bias(blob, hdr, entries, f'{block_name}.downsample')
        skip = conv_1x1(inp, wd, bd_, in_c, out_c, h, w, stride, shift)
    else:
        skip = inp.copy()

    # Add + ReLU
    out = add_relu(x, skip)
    return out


def run_inference(blob: bytes, input_i8: np.ndarray,
                  shift: int = 7, verbose: bool = True,
                  stages_only: bool = False) -> dict:
    """Run full ResNet-50 inference in bit-exact integer arithmetic.

    Args:
        blob: VWB2 weight blob bytes
        input_i8: int8 [3, 96, 96] input image
        shift: right-shift for conv output quantization
        verbose: print progress
        stages_only: if True, stop after the stem (conv1+maxpool) for quick testing

    Returns:
        dict with 'logits', 'top1', 'hashes' keys
    """
    hdr = vwb2_read_header(blob)
    entries = vwb2_read_entries(blob, hdr)
    hashes = {}

    def log(msg):
        if verbose:
            print(msg)

    # ── Stem ──────────────────────────────────────────────────────────
    log("Stem: conv1 (7×7 s2) ...")
    w1, b1 = get_weight_bias(blob, hdr, entries, 'conv1', out_c=64, kernel_elems=3*7*7)
    x = conv_7x7(input_i8, w1, b1, 3, 64, 96, 96, 2, 3, shift)
    relu_inplace(x)
    log(f"  conv1 out: {x.shape}, hash=0x{conv_hash(x):08X}")
    hashes['conv1'] = conv_hash(x)

    log("Stem: maxpool 3×3 s2 ...")
    x = maxpool_3x3_s2(x)
    log(f"  pool out: {x.shape}, hash=0x{conv_hash(x):08X}")
    hashes['stem_pool'] = conv_hash(x)
    if stages_only:
        log("[--stages-only] Stopping after stem.")
        return {'logits': np.zeros(N_CLASSES, dtype=np.int32), 'top1': -1,
                'top5': [], 'hashes': hashes}
    # ── Bottleneck blocks ─────────────────────────────────────────────
    for block_info in BOTTLENECK_NAMES:
        name, in_c, mid_c, out_c, h, w, stride, has_ds = block_info
        log(f"Block {name}: {in_c}→{mid_c}→{out_c}, {h}×{w}, s={stride} ...")
        x = run_bottleneck(x, blob, hdr, entries, name,
                           in_c, mid_c, out_c, h, w, stride, has_ds, shift)
        log(f"  out: {x.shape}, hash=0x{conv_hash(x):08X}")
        hashes[name] = conv_hash(x)

    # ── Global avg pool ───────────────────────────────────────────────
    log("Global avgpool ...")
    x = avgpool_global(x)
    log(f"  avgpool out: {x.shape}, hash=0x{conv_hash(x):08X}")
    hashes['avgpool'] = conv_hash(x)

    # ── FC ────────────────────────────────────────────────────────────
    log("FC layer ...")
    wf, bf = get_weight_bias(blob, hdr, entries, 'fc')
    logits = fc_linear(x, wf, bf, 2048, N_CLASSES)
    top1 = int(np.argmax(logits.astype(np.int64)))
    log(f"  Top-1: class {top1}, logit={int(logits[top1])}")

    # Top-5
    top5_idx = np.argsort(logits.astype(np.int64))[-5:][::-1]
    log(f"  Top-5: {[(int(i), int(logits[i])) for i in top5_idx]}")

    return {
        'logits': logits,
        'top1': top1,
        'top5': top5_idx.tolist(),
        'hashes': hashes,
    }


# =============================================================================
# BD float32 inference (canonical correctness test for the BD codec)
# =============================================================================

# =============================================================================
# BD float32 inference (canonical correctness test for the BD codec)
# =============================================================================

def run_bd_inference(blob: bytes, inp_f32,
                     verbose: bool = True,
                     stages_only: bool = False) -> dict:
    """ResNet-50 forward pass: BD-decoded float32 weights, float32 activations.

    inp_f32 : float32 [3, H, W], ImageNet-normalised - the SAME tensor as used
              for the FP32 gold.  The only difference from gold is BD weight lossiness.
    """
    import numpy as _np
    from numpy.lib.stride_tricks import as_strided as _ast
    hdr = vwb2_read_header(blob)
    entries = vwb2_read_entries(blob, hdr)
    hashes = {}

    def log(msg):
        if verbose:
            print(msg)

    def load_w(name, shape):
        we = vwb2_find(entries, name + '.weight')
        be = vwb2_find(entries, name + '.bias')
        return (bd_decode_to_f32(vwb2_get_bd_blocks(blob, hdr, we), we['n_elements']).reshape(shape),
                vwb2_get_f32(blob, hdr, be))

    def mp_f32(t):
        """Float32 max-pool 3x3 stride-2 pad-1."""
        c, h, w_ = t.shape; oh, ow = h // 2, w_ // 2
        tp = _np.pad(t, ((0,0),(1,1),(1,1)), constant_values=-_np.inf)
        s = tp.strides
        wins = _ast(tp, shape=(c,oh,ow,3,3),
                    strides=(s[0],s[1]*2,s[2]*2,s[1],s[2]), writeable=False)
        return wins.reshape(c, oh, ow, 9).max(axis=-1)

    x = inp_f32.astype(_np.float32)

    # Stem
    log("Stem: conv1 (7x7 s2) ...")
    x = _np.maximum(0.0, conv_f32(x, *load_w('conv1', (64, 3, 7, 7)), 7, 7, stride=2, pad=3))
    log(f"  conv1 out: {x.shape}  range [{x.min():.3f}, {x.max():.3f}]")
    hashes['conv1'] = conv_hash(fake_q_i8(x)[0])

    log("Stem: maxpool 3x3 s2 ...")
    x = mp_f32(x)
    log(f"  pool out:  {x.shape}  range [{x.min():.3f}, {x.max():.3f}]")
    hashes['stem_pool'] = conv_hash(fake_q_i8(x)[0])

    if stages_only:
        log("[--stages-only] Stopping after stem.")
        return {'logits': _np.zeros(N_CLASSES, dtype=_np.float32),
                'top1': -1, 'top5': [], 'hashes': hashes}

    # Bottleneck blocks
    for block_info in BOTTLENECK_NAMES:
        name, in_c, mid_c, out_c, h, w_dim, stride, has_ds = block_info
        log(f"Block {name}: {in_c}->{mid_c}->{out_c}, "
            f"{x.shape[1]}x{x.shape[2]}, s={stride} ...")
        skip = x
        x = _np.maximum(0.0, conv_f32(x, *load_w(f'{name}.conv1', (mid_c, in_c, 1, 1)), 1, 1))
        x = _np.maximum(0.0, conv_f32(x, *load_w(f'{name}.conv2', (mid_c, mid_c, 3, 3)), 3, 3, stride=stride, pad=1))
        x = conv_f32(x, *load_w(f'{name}.conv3', (out_c, mid_c, 1, 1)), 1, 1)
        if has_ds:
            skip = conv_f32(skip, *load_w(f'{name}.downsample', (out_c, in_c, 1, 1)), 1, 1, stride=stride)
        x = _np.maximum(0.0, x + skip)
        log(f"  out: {x.shape}  range [{x.min():.3f}, {x.max():.3f}]")
        hashes[name] = conv_hash(fake_q_i8(x)[0])

    # Global avg pool + FC
    log("Global avgpool ...")
    x_f = x.mean(axis=(1, 2))
    log("FC layer ...")
    wfc, bfc = load_w('fc', (N_CLASSES, 2048))
    logits = wfc @ x_f + bfc
    top1 = int(_np.argmax(logits))
    top5_idx = _np.argsort(logits)[-5:][::-1]
    log(f"  Top-1 (BD): class {top1}  logit={logits[top1]:.3f}")
    log(f"  Top-5 (BD): {[(int(i), round(float(logits[i]),2)) for i in top5_idx]}")

    return {'logits': logits, 'top1': top1, 'top5': top5_idx.tolist(), 'hashes': hashes}


# =============================================================================
# Torchvision reference on dequantized int8 input (the true BD comparison gold)
# =============================================================================

def run_tv_on_f32_input(inp_f32: np.ndarray) -> dict:
    """Run torchvision ResNet-50 on a float32 input tensor.

    inp_f32 : float32 [3, H, W], ImageNet-normalised.
    Returns top-1, top-5, logits.
    """
    try:
        import torch
        import torchvision.models as M
    except ImportError:
        return None

    model = M.resnet50(weights=M.ResNet50_Weights.IMAGENET1K_V1).eval()
    inp_t = torch.from_numpy(inp_f32).unsqueeze(0)
    with torch.no_grad():
        logits = model(inp_t).squeeze().numpy()

    top5_idx = np.argsort(logits)[-5:][::-1]
    return {
        'logits': logits,
        'top1': int(top5_idx[0]),
        'top5': top5_idx.tolist(),
    }





# =============================================================================
# ResNet-1202 CIFAR-10 BD4-faithful inference (Step 7, bd-activations-fix)
#
# Dataflow mirrors resnet1202_phase3_hw_decode/main.c exactly:
#   stem  : int8 [3,32,32] → conv3×3 → BD4 [16,32,32]
#   stages: 200+200+200 BasicBlocks, all BD4 in → BD4 out
#   skip  : identity  → add_relu_bd4
#           stride=2  → zero_pad_stride2_bd4 → add_relu_bd4
#           has_proj  → conv1×1_bd4           → add_relu_bd4
#   avgpool: BD4 [64,8,8] → int8[64]
#   fc     : int8[64]     → int32 logits[10]
#
# Key: NO int8 intermediate at layer outputs.  The int32 accumulator is
# packed directly to BD4 via _bd4_pack_twostage (no int8 clamp).
# This matches the paper's §3.3 dataflow (bd-activations-fix SKILL.md).
# =============================================================================

RN1202_N_PER_STAGE  = 200
RN1202_N_CLASSES_10 = 10


def _bd4_pack_twostage(data_i32: np.ndarray) -> bytes:
    """Pack int32 array → BD4 bytes with C-exact two-stage logic.

    IMPORTANT: This must match firmware bd_act_pack32_twostage() exactly,
    including shared exponent and half-unit scaling rules from bd_act.h.
    """
    flat = np.asarray(data_i32, dtype=np.int32).ravel()
    n = len(flat)
    n_blocks = (n + 31) // 32
    result = bytearray()

    # C constants (bd_act.h)
    beneficial_lo_x2 = [20, 20, 18, 18, 16, 16, 15, 13]
    beneficial_hi_x2 = [26, 25, 23, 22, 20, 19, 17, 15]

    for b in range(n_blocks):
        blk = np.zeros(32, dtype=np.int32)
        s, e = b * 32, min((b + 1) * 32, n)
        blk[:e - s] = flat[s:e]

        signs = (blk < 0).astype(np.uint8)
        abs_vals = np.abs(blk).astype(np.int32)
        max_abs = int(np.max(abs_vals))

        # C: bd_act_compute_exp(max_abs)
        if max_abs == 0:
            seb = 0
        else:
            seb = int(np.floor(np.log2(max_abs))) + 12
            seb = max(0, min(31, seb))

        # C: bd_act_scale_hu(abs_val, seb)
        shift = 16 - seb
        if shift >= 0:
            scaled_hu = abs_vals << shift
        else:
            rsh = -shift
            scaled_hu = (abs_vals + (1 << (rsh - 1))) >> rsh
        scaled_hu = np.clip(scaled_hu, 0, 15).astype(np.uint8)

        block_maxhu = int(np.max(scaled_hu))
        if block_maxhu >= 15: pair_id = 0
        elif block_maxhu >= 14: pair_id = 1
        elif block_maxhu >= 13: pair_id = 2
        elif block_maxhu >= 12: pair_id = 3
        elif block_maxhu >= 11: pair_id = 4
        elif block_maxhu >= 10: pair_id = 5
        elif block_maxhu >= 9: pair_id = 6
        else: pair_id = 7

        lo = beneficial_lo_x2[pair_id]
        hi = beneficial_hi_x2[pair_id]
        s2 = (scaled_hu.astype(np.uint16) << 1)
        count_a = int(np.sum((s2 >= lo) & (s2 < hi)))

        best_dialect = pair_id * 2
        if count_a * 2 < 32:
            best_dialect += 1

        table = np.array(bd.DIALECTS_HALF_UNITS[best_dialect], dtype=np.int32)
        diffs = np.abs(scaled_hu.astype(np.int32)[:, None] - table[None, :])
        nearest_idx = np.argmin(diffs, axis=1).astype(np.uint8)

        codes = ((signs & 1) << 3) | (nearest_idx & 0x7)
        result += bd.pack_block(best_dialect, seb, codes)

    return bytes(result)


def _rn1202_decode_weights(w_blocks: bytes, out_c: int,
                            kernel_elems: int,
                            tap_blocked: bool = False,
                            in_c: Optional[int] = None) -> np.ndarray:
    """Decode weight BD4 blocks per-output-channel → float64 [out_c, kernel_elems].

    Decodes per-oc to handle cases where kernel_elems % 32 != 0: each output
    channel occupies ceil(kernel_elems/32) blocks with zero-padding at the end.
    Direct linear decode across oc boundaries would misread those pad bytes.
    """
    n_blocks_per_oc = (kernel_elems + 31) // 32
    w_f64 = np.zeros((out_c, kernel_elems), dtype=np.float64)
    for oc in range(out_c):
        off = oc * n_blocks_per_oc * 18
        vec = bd_decode_to_f32(
            w_blocks[off: off + n_blocks_per_oc * 18], kernel_elems
        ).astype(np.float64)

        if tap_blocked and kernel_elems % 9 == 0:
            # Blob stores conv3x3 as [OC, KY, KX, IC] when tap-blocked.
            # Convert back to math order [OC, IC, KY, KX] to match col=[IC,KY,KX].
            if in_c is None:
                in_c_local = kernel_elems // 9
            else:
                in_c_local = in_c
            vec = vec.reshape(3, 3, in_c_local).transpose(2, 0, 1).reshape(kernel_elems)

        w_f64[oc] = vec
    return w_f64


def _rn1202_conv3x3_bd4(
    bd_in: bytes, w_blocks: bytes, bias_f32: np.ndarray,
    in_c: int, out_c: int, h: int, w: int,
    stride: int, shift: int, do_relu: bool,
    tap_blocked: bool = False,
) -> bytes:
    """3×3 conv: BD4 activation input → BD4 output (paper-faithful, no int8 at output).

    Reads activations by unpacking bd_in (BD4 → int8), convolves with
    BD-decoded float64 weights, then packs the accumulator directly to BD4
    without an int8 clamp.  Matches conv3x3_bd4_hwmac() in resnet1202_conv.h.
    """
    from numpy.lib.stride_tricks import as_strided
    oh, ow_ = h // stride, w // stride
    inp = bd_act_unpack_tensor(bd_in, in_c * h * w).reshape(in_c, h, w).astype(np.float64)
    inp_p = np.pad(inp, ((0, 0), (1, 1), (1, 1)), mode='constant')
    kernel_elems = in_c * 9
    s = inp_p.strides
    windows = as_strided(
        inp_p,
        shape=(in_c, oh, ow_, 3, 3),
        strides=(s[0], s[1] * stride, s[2] * stride, s[1], s[2]),
        writeable=False,
    )
    col     = windows.transpose(0, 3, 4, 1, 2).reshape(kernel_elems, oh * ow_)
    w_f64   = _rn1202_decode_weights(
        w_blocks, out_c, kernel_elems, tap_blocked=tap_blocked, in_c=in_c
    )
    out_raw = w_f64 @ col                                               # [out_c, oh*ow]
    out_act = out_raw / float(1 << shift) + bias_f32.astype(np.float64)[:, np.newaxis]
    if do_relu:
        out_act = np.maximum(0.0, out_act)
    # Pack to BD4: round to int32, NOT clamped to int8 (paper-faithful step 7)
    return _bd4_pack_twostage(np.round(out_act).astype(np.int32).ravel())


def _rn1202_conv1x1_bd4(
    bd_in: bytes, w_blocks: bytes, bias_f32: np.ndarray,
    in_c: int, out_c: int, h: int, w: int,
    stride: int, shift: int, do_relu: bool,
) -> bytes:
    """1×1 projection conv: BD4 in → BD4 out, no int8 intermediate at output.

    Matches conv1x1_bd4_hwmac() in resnet1202_conv.h.
    Used only for Option B (has_proj) projection shortcuts.
    """
    oh, ow_ = h // stride, w // stride
    inp   = bd_act_unpack_tensor(bd_in, in_c * h * w).reshape(in_c, h, w).astype(np.float64)
    col   = inp[:, ::stride, ::stride].reshape(in_c, oh * ow_)
    w_f64 = _rn1202_decode_weights(w_blocks, out_c, in_c, tap_blocked=False)
    out_act = (w_f64 @ col) / float(1 << shift) + bias_f32.astype(np.float64)[:, np.newaxis]
    if do_relu:
        out_act = np.maximum(0.0, out_act)
    return _bd4_pack_twostage(np.round(out_act).astype(np.int32).ravel())


def _rn1202_zero_pad_stride2_bd4(bd_in: bytes, in_c: int, out_c: int,
                                  h: int, w: int) -> bytes:
    """Option A zero-pad stride-2 skip: BD4 in → BD4 out, no int8 intermediate.

    Spatially subsamples bd_in by 2, zero-pads channels [in_c, out_c), packs
    to BD4.  Matches bd4_zero_pad_stride2() in resnet1202_conv.h.
    """
    oh, ow_ = h // 2, w // 2
    inp = bd_act_unpack_tensor(bd_in, in_c * h * w).reshape(in_c, h, w).astype(np.int32)
    out = np.zeros((out_c, oh, ow_), dtype=np.int32)
    out[:in_c] = inp[:, ::2, ::2]
    return _bd4_pack_twostage(out.ravel())


def _rn1202_avgpool_bd4(bd_in: bytes, c: int, h: int, w: int) -> np.ndarray:
    """Global average pool: BD4 C×H×W → int8[C].

    Matches global_avgpool_bd4() in resnet1202_conv.h.
    """
    spatial = h * w
    inp  = bd_act_unpack_tensor(bd_in, c * h * w).reshape(c, h, w).astype(np.int32)
    avgs = inp.sum(axis=(1, 2)) // spatial
    return np.clip(avgs, -128, 127).astype(np.int8)


def _rn1202_bd4_cksum(bd_bytes: bytes) -> int:
    """Raw byte sum of BD4 buffer — matches firmware print_bd4_cksum()."""
    return int(sum(bd_bytes)) & 0xFFFFFFFF


def _rn1202_load_wb(blob: bytes, hdr: dict, entries: list, name: str):
    """Return (weight_bd_bytes, bias_f32) for a named layer."""
    we = vwb2_find(entries, name + '.weight')
    be = vwb2_find(entries, name + '.bias')
    return vwb2_get_bd_blocks(blob, hdr, we), vwb2_get_f32(blob, hdr, be)


def _rn1202_is_tap_blocked(blob: bytes, hdr: dict, entries: list) -> bool:
    """Read rn1202.layout_flags sentinel from blob (1.0=tap-blocked, 0.0=flat)."""
    try:
        e = vwb2_find(entries, 'rn1202.layout_flags')
    except KeyError:
        return False
    if e.get('dtype', DTYPE_FLOAT32) != DTYPE_FLOAT32:
        return False
    v = vwb2_get_f32(blob, hdr, e)
    return bool(v.size > 0 and float(v[0]) == 1.0)


def _hwcb_n_cb(c: int) -> int:
    return (c + 31) // 32


def _hwcb_block_off(y: int, x: int, w: int, n_cb: int, cb: int) -> int:
    return ((y * w + x) * n_cb + cb) * 18


def _bias_scale_c(f: float, shift: int) -> int:
    u = struct.unpack('<I', struct.pack('<f', np.float32(f)))[0]
    exp = ((u >> 23) & 0xFF) - 127
    if exp == -127:
        return 0
    man = (u & 0x7FFFFF) | 0x800000
    sign = -1 if (u >> 31) else 1
    total = exp - 23 + shift
    if total >= 31:
        return 0x7FFFFFFF if sign > 0 else -0x7FFFFFFF
    if total >= 0:
        mag = man << total
    elif -total < 24:
        mag = man >> (-total)
    else:
        return 0
    return mag if sign > 0 else -mag


def _predecode_w_blocks_hu(w_blocks: bytes, n_blocks: int):
    hu_list = []
    seb_list = []
    for i in range(n_blocks):
        blk = w_blocks[i * 18:(i + 1) * 18]
        hu, _, seb = bd_decode_block_hu(blk)
        hu_list.append(hu.astype(np.int32))
        seb_list.append(int(seb))
    return hu_list, seb_list


def _conv3x3_bd4_tap_stem_exact(
    input_chw: np.ndarray,
    w_blocks: bytes,
    bias_f32: np.ndarray,
    in_c: int,
    out_c: int,
    h: int,
    w: int,
    stride: int,
    out_shift: int,
    do_relu: bool,
) -> bytes:
    oh, ow = h // stride, w // stride
    n_cb_in = _hwcb_n_cb(in_c)
    n_cb_out = _hwcb_n_cb(out_c)
    n_wblocks_per_oc = 9 * n_cb_in
    w_hu, w_seb = _predecode_w_blocks_hu(w_blocks, out_c * n_wblocks_per_oc)

    bd_out = bytearray(oh * ow * n_cb_out * 18)
    bias_init = [_bias_scale_c(float(bias_f32[oc]), out_shift) for oc in range(out_c)]
    tmp = np.zeros(32, dtype=np.int32)

    for y in range(oh):
        for x in range(ow):
            accum_row = [0] * out_c
            for oc in range(out_c):
                acc = bias_init[oc]
                oc_base = oc * n_wblocks_per_oc
                for tap in range(9):
                    ky, kx = tap // 3, tap % 3
                    iy = y * stride - 1 + ky
                    ix = x * stride - 1 + kx
                    if iy < 0 or iy >= h or ix < 0 or ix >= w:
                        continue
                    tap_base = oc_base + tap * n_cb_in
                    for cb in range(n_cb_in):
                        idx = tap_base + cb
                        hu = w_hu[idx]
                        seb = w_seb[idx]
                        ic_base = cb * 32
                        count = min(32, in_c - ic_base)
                        if count <= 0:
                            continue
                        bsum = 0
                        for i in range(count):
                            bsum += int(hu[i]) * int(input_chw[ic_base + i, iy, ix])
                        sh = seb - 16
                        if sh >= 0:
                            acc += (bsum << sh)
                        else:
                            acc += (bsum >> (-sh))
                accum_row[oc] = acc

            for cb in range(n_cb_out):
                base = cb * 32
                for i in range(32):
                    oc = base + i
                    if oc < out_c:
                        v = accum_row[oc] >> out_shift
                        if do_relu and v < 0:
                            v = 0
                        tmp[i] = v
                    else:
                        tmp[i] = 0
                off = _hwcb_block_off(y, x, ow, n_cb_out, cb)
                bd_out[off:off + 18] = _bd4_pack_twostage(tmp)

    return bytes(bd_out)


def _conv3x3_bd4_tap_hwmac_exact(
    bd_in: bytes,
    w_blocks: bytes,
    bias_f32: np.ndarray,
    in_c: int,
    out_c: int,
    h: int,
    w: int,
    stride: int,
    out_shift: int,
    do_relu: bool,
) -> bytes:
    oh, ow = h // stride, w // stride
    n_cb_in = _hwcb_n_cb(in_c)
    n_cb_out = _hwcb_n_cb(out_c)
    n_wblocks_per_oc = 9 * n_cb_in
    w_hu, w_seb = _predecode_w_blocks_hu(w_blocks, out_c * n_wblocks_per_oc)

    bd_out = bytearray(oh * ow * n_cb_out * 18)
    zero_hu = np.zeros(32, dtype=np.int32)
    zero_seb = 0
    bias_init = [_bias_scale_c(float(bias_f32[oc]), out_shift) for oc in range(out_c)]
    tmp = np.zeros(32, dtype=np.int32)

    for y in range(oh):
        for x in range(ow):
            act_hu = [[None for _ in range(n_cb_in)] for _ in range(9)]
            act_seb = [[0 for _ in range(n_cb_in)] for _ in range(9)]

            for tap in range(9):
                ky, kx = tap // 3, tap % 3
                iy = y * stride - 1 + ky
                ix = x * stride - 1 + kx
                in_bounds = (0 <= iy < h and 0 <= ix < w)
                for cb in range(n_cb_in):
                    if in_bounds:
                        off = _hwcb_block_off(iy, ix, w, n_cb_in, cb)
                        hu, _, seb = bd_decode_block_hu(bd_in[off:off + 18])
                        act_hu[tap][cb] = hu.astype(np.int32)
                        act_seb[tap][cb] = int(seb)
                    else:
                        act_hu[tap][cb] = zero_hu
                        act_seb[tap][cb] = zero_seb

            accum_row = [0] * out_c
            for oc in range(out_c):
                acc = bias_init[oc]
                oc_base = oc * n_wblocks_per_oc
                for tap in range(9):
                    tap_base = oc_base + tap * n_cb_in
                    for cb in range(n_cb_in):
                        idx = tap_base + cb
                        ps = int(np.dot(w_hu[idx], act_hu[tap][cb]))
                        es = w_seb[idx] + act_seb[tap][cb]
                        sh = es - 32
                        if sh >= 0:
                            acc += (ps << sh)
                        else:
                            acc += (ps >> (-sh))
                accum_row[oc] = acc

            for cb in range(n_cb_out):
                base = cb * 32
                for i in range(32):
                    oc = base + i
                    if oc < out_c:
                        v = accum_row[oc] >> out_shift
                        if do_relu and v < 0:
                            v = 0
                        tmp[i] = v
                    else:
                        tmp[i] = 0
                off = _hwcb_block_off(y, x, ow, n_cb_out, cb)
                bd_out[off:off + 18] = _bd4_pack_twostage(tmp)

    return bytes(bd_out)


def _bd4_zero_pad_stride2_hwcb_exact(bd_in: bytes, in_c: int, out_c: int, h: int, w: int) -> bytes:
    oh, ow = h // 2, w // 2
    n_cb_in = _hwcb_n_cb(in_c)
    n_cb_out = _hwcb_n_cb(out_c)
    bd_out = bytearray(oh * ow * n_cb_out * 18)
    for oy in range(oh):
        for ox in range(ow):
            for cb in range(n_cb_out):
                doff = _hwcb_block_off(oy, ox, ow, n_cb_out, cb)
                if cb < n_cb_in:
                    soff = _hwcb_block_off(oy * 2, ox * 2, w, n_cb_in, cb)
                    bd_out[doff:doff + 18] = bd_in[soff:soff + 18]
                else:
                    bd_out[doff:doff + 18] = bytes(18)
    return bytes(bd_out)


def _add_relu_bd4_hwcb_exact(bd_a: bytes, bd_b: bytes, c: int, h: int, w: int) -> bytes:
    n_blocks = h * w * _hwcb_n_cb(c)
    return add_relu_bd4(bd_a, bd_b, n_blocks * 32)


def _avgpool_hwcb_exact(bd_in: bytes, c: int, h: int, w: int) -> np.ndarray:
    n_cb = _hwcb_n_cb(c)
    sums = np.zeros(c, dtype=np.int32)
    for y in range(h):
        for x in range(w):
            for cb in range(n_cb):
                off = _hwcb_block_off(y, x, w, n_cb, cb)
                v = bd_act_unpack_tensor(bd_in[off:off + 18], 32).astype(np.int32)
                base = cb * 32
                for i in range(32):
                    ch = base + i
                    if ch < c:
                        sums[ch] += int(v[i])
    spatial = h * w
    avgs = sums // spatial
    return np.clip(avgs, -128, 127).astype(np.int8)


def calibrate_rn1202_shifts(
    blob: bytes,
    input_i8: np.ndarray,   # int8 [3, 32, 32]
    target_bits: int = 7,
    verbose: bool = False,
) -> dict:
    """Compute per-layer optimal shift values by measuring raw accumulator ranges.

    Runs a single float64 forward pass (plain numpy, no BD4 round-trips between
    layers) through the full ResNet-1202 topology, records max(|raw_accum|) for
    every named conv layer, then derives:

        shift = max(0, floor(log2(max_abs + eps)) - (target_bits - 1))

    so that raw_accum / 2^shift targets ~[-2^(target_bits-1), 2^(target_bits-1)].
    Default target_bits=7 matches the original int8 downstream range but applies
    per-layer instead of one global value.

    Returns a dict mapping layer_name -> shift (int).
    """
    import math
    from numpy.lib.stride_tricks import as_strided

    hdr     = vwb2_read_header(blob)
    entries = vwb2_read_entries(blob, hdr)
    tap_blocked_blob = _rn1202_is_tap_blocked(blob, hdr, entries)
    result  = {}

    def load_wb(name: str):
        return _rn1202_load_wb(blob, hdr, entries, name)

    def optimal_shift(raw: np.ndarray) -> int:
        max_abs = float(np.max(np.abs(raw)))
        if max_abs < 1e-10:
            return 0
        return max(0, int(math.floor(math.log2(max_abs + 1e-10))) - (target_bits - 1))

    def conv3x3_f64(act_f64: np.ndarray, w_blocks: bytes, bias_f32: np.ndarray,
                    in_c: int, out_c: int, h: int, w: int, stride: int) -> np.ndarray:
        """Float64 3×3 conv pad=1, returns raw accumulator [out_c, oh, ow] (no shift/bias)."""
        inp_p = np.pad(act_f64, ((0, 0), (1, 1), (1, 1)), mode='constant')
        s = inp_p.strides
        oh, ow_ = h // stride, w // stride
        windows = as_strided(
            inp_p,
            shape=(in_c, oh, ow_, 3, 3),
            strides=(s[0], s[1] * stride, s[2] * stride, s[1], s[2]),
            writeable=False,
        )
        col   = windows.transpose(0, 3, 4, 1, 2).reshape(in_c * 9, oh * ow_)
        w_f64 = _rn1202_decode_weights(
            w_blocks, out_c, in_c * 9,
            tap_blocked=tap_blocked_blob,
            in_c=in_c,
        )
        raw   = (w_f64 @ col).reshape(out_c, oh, ow_)
        return raw

    # ── Stem ──────────────────────────────────────────────────────────────
    in_c_s, out_c_s, h_s, w_s = 3, 16, 32, 32
    act = input_i8.reshape(in_c_s, h_s, w_s).astype(np.float64)
    wb_s, bf_s = load_wb('conv1')
    raw_s = conv3x3_f64(act, wb_s, bf_s, in_c_s, out_c_s, h_s, w_s, stride=1)
    sh = optimal_shift(raw_s)
    result['conv1'] = sh
    act = np.maximum(0.0, raw_s / float(1 << sh) + bf_s.astype(np.float64)[:, np.newaxis, np.newaxis])
    if verbose:
        print(f"  calib conv1: max_raw={np.max(np.abs(raw_s)):.2f}  shift={sh}")

    # ── Stages ────────────────────────────────────────────────────────────
    STAGE_CFG = [
        (1, 'stage1', 16, 16, 32, 32),
        (2, 'stage2', 16, 32, 32, 32),
        (3, 'stage3', 32, 64, 16, 16),
    ]

    for stage_idx, stage_name, first_in_c, stage_out_c, first_h, first_w in STAGE_CFG:
        cur_in_c     = first_in_c
        cur_h, cur_w = first_h, first_w

        for blk_idx in range(RN1202_N_PER_STAGE):
            pfx        = f"{stage_name}.{blk_idx}"
            is_first   = (blk_idx == 0)
            blk_stride = 2 if (is_first and stage_idx > 1) else 1
            in_c       = cur_in_c
            out_c      = stage_out_c
            h, w_      = cur_h, cur_w
            oh, ow_    = h // blk_stride, w_ // blk_stride

            # conv1
            wa, ba = load_wb(f'{pfx}.conv1')
            raw_a  = conv3x3_f64(act, wa, ba, in_c, out_c, h, w_, blk_stride)
            sh_a   = optimal_shift(raw_a)
            result[f'{pfx}.conv1'] = sh_a
            mid = np.maximum(0.0, raw_a / float(1 << sh_a) + ba.astype(np.float64)[:, np.newaxis, np.newaxis])

            # conv2
            wb_l, bb = load_wb(f'{pfx}.conv2')
            raw_b    = conv3x3_f64(mid, wb_l, bb, out_c, out_c, oh, ow_, 1)
            sh_b     = optimal_shift(raw_b)
            result[f'{pfx}.conv2'] = sh_b
            out_act  = raw_b / float(1 << sh_b) + bb.astype(np.float64)[:, np.newaxis, np.newaxis]

            # Skip + ReLU (float64, no BD4)
            if is_first and stage_idx > 1:
                skip_pad = np.zeros((out_c, oh, ow_), dtype=np.float64)
                skip_pad[:in_c] = act[:, ::2, ::2]
                act = np.maximum(0.0, out_act + skip_pad)
            else:
                act = np.maximum(0.0, out_act + act)

            cur_in_c     = out_c
            cur_h, cur_w = oh, ow_

    if verbose:
        vals = list(result.values())
        print(f"  calib done: {len(vals)} layers  "
              f"shift range [{min(vals)}, {max(vals)}]  "
              f"mean={sum(vals)/len(vals):.1f}")

    return result


def run_rn1202_bd4_inference(
    blob: bytes,
    input_i8: np.ndarray,   # int8 [3, 32, 32] CIFAR-10 image (CHW)
    shift: int = 7,
    verbose: bool = True,
    has_proj: bool = False,  # True only for Option B (trained-from-scratch) checkpoints
    wide_add_relu: bool = False,  # True → use add_relu_bd4_i16 (int16 intermediate)
    shifts: Optional[dict] = None,  # Per-layer overrides; falls back to global `shift`
) -> dict:
    """ResNet-1202 CIFAR-10 inference: paper-faithful BD4 activation path.

    Matches resnet1202_phase3_hw_decode/main.c:
      - Activations are stored as BD4 between every pair of layers
      - Layer output = int32 accumulator → BD4-pack (no int8 clamp)
      - Residual add via add_relu_bd4 (unpack both, add, ReLU, repack to BD4)
      - No int8 path anywhere except the 64-element avgpool output feeding FC

    wide_add_relu=True selects add_relu_bd4_i16, which widens the unpack
    intermediate from int8 to int16, avoiding precision loss in the residual
    add for activations whose BD4-encoded magnitude exceeds 127.

    Returns dict: {logits, top1, hashes (BD4 raw-byte checksums per stage)}.
    """
    from numpy.lib.stride_tricks import as_strided
    hdr     = vwb2_read_header(blob)
    entries = vwb2_read_entries(blob, hdr)
    tap_blocked_blob = _rn1202_is_tap_blocked(blob, hdr, entries)

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    if verbose:
        print(f"rn1202 blob layout: {'tap-blocked' if tap_blocked_blob else 'flat'}")

    def load_wb(name: str):
        return _rn1202_load_wb(blob, hdr, entries, name)

    # Select add+ReLU function based on wide_add_relu flag
    _add_relu_fn = add_relu_bd4_i16 if wide_add_relu else add_relu_bd4

    def layer_shift(name: str) -> int:
        """Return per-layer shift from calibration table, or global `shift` fallback."""
        if shifts is not None and name in shifts:
            return int(shifts[name])
        return shift

    hashes: dict = {}

    # ── Stem: int8 [3,32,32] → conv3×3 → BD4 [16,32,32] ──────────────────
    log("rn1202 stem: conv3x3 int8→BD4 ...")
    in_c_s, out_c_s, h_s, w_s = 3, 16, 32, 32
    wb_s, bf_s = load_wb('conv1')
    if tap_blocked_blob:
        bd_cur = _conv3x3_bd4_tap_stem_exact(
            input_i8.reshape(in_c_s, h_s, w_s), wb_s, bf_s,
            in_c_s, out_c_s, h_s, w_s,
            stride=1,
            out_shift=layer_shift('conv1'),
            do_relu=True,
        )
    else:
        kernel_elems_s = in_c_s * 9
        inp_s = input_i8.reshape(in_c_s, h_s, w_s).astype(np.float64)
        inp_p = np.pad(inp_s, ((0, 0), (1, 1), (1, 1)), mode='constant')
        sst   = inp_p.strides
        windows_s = as_strided(
            inp_p,
            shape=(in_c_s, h_s, w_s, 3, 3),
            strides=(sst[0], sst[1], sst[2], sst[1], sst[2]),
            writeable=False,
        )
        col_s = windows_s.transpose(0, 3, 4, 1, 2).reshape(kernel_elems_s, h_s * w_s)
        w_f64_s = _rn1202_decode_weights(
            wb_s, out_c_s, kernel_elems_s,
            tap_blocked=tap_blocked_blob,
            in_c=in_c_s,
        )
        out_s = w_f64_s @ col_s
        out_s = np.maximum(0.0, out_s / float(1 << layer_shift('conv1')) + bf_s.astype(np.float64)[:, np.newaxis])
        bd_cur = _bd4_pack_twostage(np.round(out_s).astype(np.int32).ravel())
    hashes['conv1'] = _rn1202_bd4_cksum(bd_cur)
    log(f"  conv1 BD4 cksum=0x{hashes['conv1']:08X}")

    # ── Stages 1-3: BasicBlocks, all BD4 ──────────────────────────────────
    # Each tuple: (stage_idx, name, first_in_c, out_c, first_h, first_w)
    # first_h/w = spatial dims ENTERING block 0 of that stage.
    STAGE_CFG = [
        (1, 'stage1', 16, 16, 32, 32),
        (2, 'stage2', 16, 32, 32, 32),
        (3, 'stage3', 32, 64, 16, 16),
    ]

    for stage_idx, stage_name, first_in_c, stage_out_c, first_h, first_w in STAGE_CFG:
        log(f"rn1202 {stage_name}: {RN1202_N_PER_STAGE} blocks ...")
        cur_in_c     = first_in_c
        cur_h, cur_w = first_h, first_w

        for blk_idx in range(RN1202_N_PER_STAGE):
            pfx        = f"{stage_name}.{blk_idx}"
            is_first   = (blk_idx == 0)
            blk_stride = 2 if (is_first and stage_idx > 1) else 1
            in_c       = cur_in_c
            out_c      = stage_out_c
            h, w_      = cur_h, cur_w
            oh, ow_    = h // blk_stride, w_ // blk_stride
            out_elems  = out_c * oh * ow_

            if tap_blocked_blob:
                wa, ba = load_wb(f'{pfx}.conv1')
                bd_mid = _conv3x3_bd4_tap_hwmac_exact(
                    bd_cur, wa, ba, in_c, out_c, h, w_, blk_stride,
                    layer_shift(f'{pfx}.conv1'), do_relu=True)

                wb, bb = load_wb(f'{pfx}.conv2')
                bd_out = _conv3x3_bd4_tap_hwmac_exact(
                    bd_mid, wb, bb, out_c, out_c, oh, ow_, 1,
                    layer_shift(f'{pfx}.conv2'), do_relu=False)

                if is_first and has_proj and stage_idx > 1:
                    wp, bp = load_wb(f'{stage_name}.0.proj')
                    bd_skip = _rn1202_conv1x1_bd4(
                        bd_cur, wp, bp, in_c, out_c, h, w_, blk_stride, shift, do_relu=False)
                    bd_out = _add_relu_fn(bd_out, bd_skip, out_elems)
                elif is_first and stage_idx > 1:
                    bd_skip = _bd4_zero_pad_stride2_hwcb_exact(bd_cur, in_c, out_c, h, w_)
                    bd_out = _add_relu_bd4_hwcb_exact(bd_out, bd_skip, out_c, oh, ow_)
                else:
                    bd_out = _add_relu_bd4_hwcb_exact(bd_out, bd_cur, out_c, h, w_)
            else:
                # conv_a: BD4 in → BD4 mid (with ReLU)
                wa, ba = load_wb(f'{pfx}.conv1')
                bd_mid = _rn1202_conv3x3_bd4(
                    bd_cur, wa, ba, in_c, out_c, h, w_, blk_stride,
                    layer_shift(f'{pfx}.conv1'), do_relu=True,
                    tap_blocked=tap_blocked_blob)

                # conv_b: BD4 mid → BD4 out (no ReLU)
                wb, bb = load_wb(f'{pfx}.conv2')
                bd_out = _rn1202_conv3x3_bd4(
                    bd_mid, wb, bb, out_c, out_c, oh, ow_, 1,
                    layer_shift(f'{pfx}.conv2'), do_relu=False,
                    tap_blocked=tap_blocked_blob)

                # Skip connection
                if is_first and has_proj and stage_idx > 1:
                    # Option B: learnable conv1×1 projection
                    wp, bp = load_wb(f'{stage_name}.0.proj')
                    bd_skip = _rn1202_conv1x1_bd4(
                        bd_cur, wp, bp, in_c, out_c, h, w_, blk_stride, shift, do_relu=False)
                    bd_out = _add_relu_fn(bd_out, bd_skip, out_elems)
                elif is_first and stage_idx > 1:
                    # Option A: zero-pad + stride-2 (default akamaster weights)
                    bd_skip = _rn1202_zero_pad_stride2_bd4(bd_cur, in_c, out_c, h, w_)
                    bd_out  = _add_relu_fn(bd_out, bd_skip, out_elems)
                else:
                    # Identity skip: bd_cur has the same shape as bd_out
                    bd_out = _add_relu_fn(bd_out, bd_cur, out_elems)

            bd_cur   = bd_out
            cur_in_c = out_c
            cur_h, cur_w = oh, ow_

        cksum = _rn1202_bd4_cksum(bd_cur)
        hashes[stage_name] = cksum
        log(f"  {stage_name} done  BD4 cksum=0x{cksum:08X}")

    # ── Global avgpool: BD4 64×8×8 → int8[64] ─────────────────────────────
    log("rn1202 avgpool: BD4 64x8x8 -> int8[64] ...")
    if tap_blocked_blob:
        avgpool_out = _avgpool_hwcb_exact(bd_cur, 64, 8, 8)
    else:
        avgpool_out = _rn1202_avgpool_bd4(bd_cur, 64, 8, 8)

    # ── FC: int8[64] → int32 logits[10] ───────────────────────────────────
    log("rn1202 fc: int8[64] -> logits[10] ...")
    wfc, bfc = load_wb('fc')
    logits = fc_linear(avgpool_out, wfc, bfc, 64, RN1202_N_CLASSES_10)
    top1   = int(np.argmax(logits))
    CIFAR10_NAMES = ["airplane","automobile","bird","cat","deer",
                     "dog","frog","horse","ship","truck"]
    log(f"  Top-1: class {top1} ({CIFAR10_NAMES[top1]})  logit={int(logits[top1])}")

    return {'logits': logits, 'top1': top1, 'hashes': hashes}


def export_rn1202_checksums(result: dict, path: str,
                            cal_shifts: Optional[dict] = None) -> None:
    """Write C header with BD4 raw-byte checksums for ResNet-1202 firmware comparison.

    Produces #defines matching the names printed by firmware's print_bd4_cksum(),
    replacing (or extending alongside) the old int8 u32sum hashes from
    gen_resnet1202_model.py's export_quantized_ref_header().

    If cal_shifts is provided, also emits per-stage CONV_SHIFT defines for the
    firmware (calibrated by calibrate_rn1202_shifts).
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        f.write("// AUTO-GENERATED by quantized_reference.py --model rn1202 — do not edit\n")
        f.write("// ResNet-1202 CIFAR-10: paper-faithful BD4 activation path\n")
        f.write("// Checksums are raw uint8 byte-sum of BD4 buffers,\n")
        f.write("// matching firmware print_bd4_cksum() in resnet1202_phase3_hw_decode.\n\n")
        f.write("#ifndef RN1202_QUANTIZED_REF_H\n")
        f.write("#define RN1202_QUANTIZED_REF_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define RN1202_BD4_TOP1    {result['top1']}\n\n")

        # Per-stage calibrated shifts
        if cal_shifts is not None:
            f.write("// Per-stage calibrated output shifts (from calibrate_rn1202_shifts)\n")
            f.write("// shift = max(0, floor(log2(max_accum)) - 6) per layer,\n")
            f.write("// then collapsed to per-stage max.\n")
            import collections
            stage_max = {'conv1': 0, 'stage1': 0, 'stage2': 0, 'stage3': 0}
            for name, s in cal_shifts.items():
                if name == 'conv1':
                    stage_max['conv1'] = max(stage_max['conv1'], s)
                elif name.startswith('stage1'):
                    stage_max['stage1'] = max(stage_max['stage1'], s)
                elif name.startswith('stage2'):
                    stage_max['stage2'] = max(stage_max['stage2'], s)
                elif name.startswith('stage3'):
                    stage_max['stage3'] = max(stage_max['stage3'], s)
            f.write(f"#define RN1202_SHIFT_STEM    {stage_max['conv1']}\n")
            f.write(f"#define RN1202_SHIFT_STAGE1  {stage_max['stage1']}\n")
            f.write(f"#define RN1202_SHIFT_STAGE2  {stage_max['stage2']}\n")
            f.write(f"#define RN1202_SHIFT_STAGE3  {stage_max['stage3']}\n\n")

        f.write("// Raw BD4 buffer byte-sums at stage boundaries\n")
        for name, h in result['hashes'].items():
            cname = name.replace('.', '_').upper()
            f.write(f"#define RN1202_BD4_CKSUM_{cname}  0x{h:08X}u\n")
        f.write("\n#endif /* RN1202_QUANTIZED_REF_H */\n")
    print(f"Saved {path}")


def export_checksums(result: dict, path: str, label: str = 'BD float32') -> None:
    """Write a C header with expected int8 hashes for firmware comparison."""
    with open(path, 'w') as f:
        f.write("// AUTO-GENERATED by quantized_reference.py — do not edit\n")
        f.write(f"// BD float32 inference reference ({label})\n\n")
        f.write("#ifndef RESNET50_QUANTIZED_REF_H\n")
        f.write("#define RESNET50_QUANTIZED_REF_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define BD_REF_TOP1   {result['top1']}\n\n")
        f.write("// Per-stage activation hashes (sum-of-fake-quantised-int8, uint32)\n")
        f.write("// Generated by: BD weights decoded to float32, per-tensor sym fake-quant at each layer\n")
        for name, h in result['hashes'].items():
            cname = name.replace('.', '_').upper()
            f.write(f"#define BD_REF_HASH_{cname}  0x{h:08X}u\n")
        f.write(f"\n// Top-5 predictions\n")
        f.write(f"static const int BD_REF_TOP5[5] = {{ {', '.join(str(i) for i in result['top5'])} }};\n")
        f.write("\n#endif /* RESNET50_QUANTIZED_REF_H */\n")
    print(f"Saved {path}")


# =============================================================================
# Main
# =============================================================================

def _read_gold_top1(gold_path: str) -> Optional[int]:
    """Parse top-1 class from a generated C header."""
    import re
    try:
        with open(gold_path) as f:
            for line in f:
                m = re.search(r'#define\s+\S*TOP1\S*\s+(\d+)', line)
                if m:
                    return int(m.group(1))
    except FileNotFoundError:
        pass
    return None


def main():
    parser = argparse.ArgumentParser(
        description='BD inference reference: ResNet-50 (float32) or ResNet-1202 (BD4-faithful)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            --model resnet50 (default):
              Runs two passes and compares them:
                1) FP32 gold  — top-1 read from expected_fp32.h
                2) BD test    — BD weights decoded to float32, fake-quant activations

            --model rn1202:
              Paper-faithful BD4 activation path (bd-activations-fix Step 7):
                - Activations stored as BD4 (not int8) between every layer
                - Layer output: int32 accumulator → BD4-pack (no int8 clamp)
                - Residual add via add_relu_bd4 (unpack, add, ReLU, repack)
              Writes BD4 raw-byte checksums to resnet1202_artifacts/quantized_ref.h.
        """))
    parser.add_argument('--model', default='resnet50', choices=['resnet50', 'rn1202'],
                        help='Which model to run (default: resnet50)')
    parser.add_argument('--blob', default=None,
                        help='Path to VWB2 weight blob (auto-selected per model if omitted)')
    parser.add_argument('--input', default=None,
                        help='Input tensor file (auto-selected per model if omitted)')
    parser.add_argument('--output', default=None,
                        help='Output C header (auto-selected per model if omitted)')
    parser.add_argument('--shift', type=int, default=7,
                        help='Accumulator right-shift for rn1202 (default: 7)')
    parser.add_argument('--has-proj', action='store_true',
                        help='rn1202: use Option B (conv1×1 projection) shortcuts')
    parser.add_argument('--gold', default='scripts/resnet50_artifacts/expected_fp32.h',
                        help='FP32 gold header (resnet50 only)')
    parser.add_argument('--stages-only', action='store_true',
                        help='resnet50 only: stop after stem')
    args = parser.parse_args()

    # ── ResNet-1202 BD4-faithful path ─────────────────────────────────────────
    if args.model == 'rn1202':
        def rs(p):
            return p if os.path.isabs(p) else str(SCRIPT_DIR.parent / p)

        blob_path = rs(args.blob or 'scripts/resnet1202_artifacts/weights_bd.bin')
        inp_path  = rs(args.input or 'scripts/resnet1202_artifacts/input_32x32.raw')
        out_path  = rs(args.output or 'scripts/resnet1202_artifacts/quantized_ref.h')

        print(f"Loading blob: {blob_path}")
        with open(blob_path, 'rb') as f:
            blob = f.read()
        hdr = vwb2_read_header(blob)
        print(f"VWB2: {hdr['tensor_count']} tensors, {hdr['data_bytes']} data bytes")

        if not os.path.exists(inp_path):
            sys.exit(f"Input not found: {inp_path}  - run gen_resnet1202_model.py first")
        input_i8 = np.fromfile(inp_path, dtype=np.int8).reshape(3, 32, 32)
        print(f"Input : {inp_path}  (3×32×32 int8)")

        CIFAR10 = ["airplane","automobile","bird","cat","deer",
                   "dog","frog","horse","ship","truck"]
        SEP = '=' * 62

        # ── Single inference run (matches C firmware exactly) ─────────────
        print(f"\n{SEP}")
        print(" ResNet-1202  |  BD4-faithful inference (global shift, int8 add_relu)")
        print(SEP)
        result = run_rn1202_bd4_inference(
            blob, input_i8,
            shift=args.shift,
            verbose=True,
            has_proj=args.has_proj,
            wide_add_relu=False,
        )
        top1 = result['top1']
        print(f"\n  Top-1: class {top1} ({CIFAR10[top1]})  logit={int(result['logits'][top1])}")

        export_rn1202_checksums(result, out_path)
        print("\nDone.")
        return

    # ── ResNet-50 path ────────────────────────────────────────────────────────
    def resolve(p):
        return p if os.path.isabs(p) else str(SCRIPT_DIR.parent / p)

    # ── Load blob ─────────────────────────────────────────────────────────────
    blob_path = resolve(args.blob or 'scripts/resnet50_artifacts/weights_bd.bin')
    print(f"Loading blob: {blob_path}")
    with open(blob_path, 'rb') as f:
        blob = f.read()
    hdr = vwb2_read_header(blob)
    print(f"VWB2: {hdr['tensor_count']} tensors, {hdr['data_bytes']} data bytes")

    # Load float32 input tensor
    inp_path = resolve(args.input or 'scripts/funny_monkey_tensor.bin')
    if not os.path.exists(inp_path):
        sys.exit(f"Input not found: {inp_path}  - run gen_resnet50_model.py first")
    inp_f32 = np.fromfile(inp_path, dtype=np.float32).reshape(INPUT_C, INPUT_H, INPUT_W)
    print(f"Input : {inp_path}  ({INPUT_C}x{INPUT_H}x{INPUT_W} float32)")

    SEP = '=' * 62

    # FP32 gold (torchvision, same float32 input)
    print(f"\n{SEP}")
    print("FP32 GOLD  (torchvision ResNet-50, same float32 input)")
    print(SEP)
    print("  Running torchvision forward pass ...")
    tv_result = run_tv_on_f32_input(inp_f32)
    if tv_result is not None:
        tv_top1 = tv_result['top1']
        tv_top5 = tv_result['top5']
        print(f"  Top-1 : class {tv_top1}  logit={float(tv_result['logits'][tv_top1]):.3f}")
        print(f"  Top-5 : {[(int(i), round(float(tv_result['logits'][i]),2)) for i in tv_top5]}")
    else:
        tv_top1 = None
        print("  (torchvision not available)")

    # ── BD float32 inference ──────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("BD TEST  (BD-decoded float32 weights, same float32 input)")
    print(SEP)
    result = run_bd_inference(blob, inp_f32, verbose=True, stages_only=args.stages_only)

    # ── Comparison ────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("COMPARISON SUMMARY")
    print(SEP)
    bd_top1 = result['top1']
    bd_top5 = result['top5']
    bd_logit = float(result['logits'][bd_top1])
    match_tv = (tv_top1 is not None and bd_top1 == tv_top1)
    print(f"  FP32 gold top-1  : class {tv_top1}")
    print(f"  BD test   top-1  : class {bd_top1}  logit={bd_logit:.3f}")
    print(f"  Match            : {'YES ✓' if match_tv else 'NO ✗'}")
    if not match_tv and tv_top1 is not None:
        if tv_top1 in bd_top5:
            print(f"  Note: gold ({tv_top1}) in BD top-5 — minor BD quantisation drift, acceptable")
        elif bd_top1 in (tv_result['top5'] if tv_result else []):
            print(f"  Note: BD top-1 ({bd_top1}) in gold top-5 — minor BD quantisation drift, acceptable")
        else:
            print(f"  WARNING: top-1 mismatch with no top-5 overlap — check BD codec")

    # ── Export ────────────────────────────────────────────────────────────────
    if not args.stages_only:
        out_path = resolve(args.output or 'scripts/resnet50_artifacts/quantized_ref.h')
        export_checksums(result, out_path, label=f'BD float32, {INPUT_H}×{INPUT_W}')

    print("\nDone.")


if __name__ == '__main__':
    main()
