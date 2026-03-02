#!/usr/bin/env python3
"""
quantized_reference.py — Python integer-quantized reference for ResNet-50 Milestone 1

Replicates the firmware's exact integer arithmetic:
  - BlockDialect weight decoding to half-unit int16
  - Per-block exponent shifting during accumulation
  - int8 activations, int32 accumulators
  - Bias as int32 (bias_f32 * 2^shift)
  - ReLU, clamp to [0, 127] for int8 output

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


# =============================================================================
# Layer topology (mirrors resnet50_layers.h and gen_resnet50_model.py)
# =============================================================================

# Layer name stems for ResNet-50 (54 layers: stem + 16 bottleneck×3 + 4 downsample + FC)
STEM_LAYERS = ['conv1']

BOTTLENECK_NAMES = [
    # (stage, block_idx, [conv1, conv2, conv3], optional downsample)
    # Stage 1: 3 blocks
    ('layer1.0', 64, 64, 256, 56, 56, 1, True),
    ('layer1.1', 256, 64, 256, 56, 56, 1, False),
    ('layer1.2', 256, 64, 256, 56, 56, 1, False),
    # Stage 2: 4 blocks
    ('layer2.0', 256, 128, 512, 56, 56, 2, True),
    ('layer2.1', 512, 128, 512, 28, 28, 1, False),
    ('layer2.2', 512, 128, 512, 28, 28, 1, False),
    ('layer2.3', 512, 128, 512, 28, 28, 1, False),
    # Stage 3: 6 blocks
    ('layer3.0', 512, 256, 1024, 28, 28, 2, True),
    ('layer3.1', 1024, 256, 1024, 14, 14, 1, False),
    ('layer3.2', 1024, 256, 1024, 14, 14, 1, False),
    ('layer3.3', 1024, 256, 1024, 14, 14, 1, False),
    ('layer3.4', 1024, 256, 1024, 14, 14, 1, False),
    ('layer3.5', 1024, 256, 1024, 14, 14, 1, False),
    # Stage 4: 3 blocks
    ('layer4.0', 1024, 512, 2048, 14, 14, 2, True),
    ('layer4.1', 2048, 512, 2048, 7, 7, 1, False),
    ('layer4.2', 2048, 512, 2048, 7, 7, 1, False),
]

FC_LAYER = 'fc'
N_CLASSES = 1000


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
    """Fully connected layer, returns int32 logits."""
    logits = np.zeros(out_features, dtype=np.int32)
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

        acc += int(bias_f32[oc] * 128.0)
        logits[oc] = np.int32(int(acc) & 0xFFFFFFFF)

    return logits


def relu_inplace(act: np.ndarray) -> np.ndarray:
    """ReLU on int8 array."""
    act[act < 0] = 0
    return act


def add_relu(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise add + ReLU, matching firmware's add_relu()."""
    out = np.clip(a.astype(np.int32) + b.astype(np.int32), 0, 127).astype(np.int8)
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
        wd, bd_ = get_weight_bias(blob, hdr, entries, f'{block_name}.downsample.0')
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
        input_i8: int8 [3, 224, 224] input image
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
    x = conv_7x7(input_i8, w1, b1, 3, 64, 224, 224, 2, 3, shift)
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
# Export checksums header
# =============================================================================

def export_checksums(result: dict, path: str, shift: int) -> None:
    """Write a C header with expected int8 hashes for firmware comparison."""
    with open(path, 'w') as f:
        f.write("// AUTO-GENERATED by quantized_reference.py — do not edit\n")
        f.write(f"// Integer quantized reference (shift={shift})\n\n")
        f.write("#ifndef RESNET50_QUANTIZED_REF_H\n")
        f.write("#define RESNET50_QUANTIZED_REF_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define QUANT_REF_SHIFT  {shift}\n")
        f.write(f"#define QUANT_REF_TOP1   {result['top1']}\n\n")
        f.write("// Per-stage activation hashes (sum-of-int8, uint32)\n")
        for name, h in result['hashes'].items():
            cname = name.replace('.', '_').upper()
            f.write(f"#define QUANT_REF_HASH_{cname}  0x{h:08X}u\n")
        f.write(f"\n// Top-5 predictions\n")
        f.write(f"static const int QUANT_REF_TOP5[5] = {{ {', '.join(str(i) for i in result['top5'])} }};\n")
        f.write("\n#endif /* RESNET50_QUANTIZED_REF_H */\n")
    print(f"Saved {path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='ResNet-50 integer quantized reference')
    parser.add_argument('--blob', default='scripts/resnet50_artifacts/weights_bd.bin',
                        help='Path to VWB2 weight blob')
    parser.add_argument('--input', default=None,
                        help='Path to raw int8 input (3×224×224, CHW). Default: zeros')
    parser.add_argument('--shift', type=int, default=7,
                        help='Conv output right-shift for int8 quantization')
    parser.add_argument('--output', default='scripts/resnet50_artifacts/quantized_ref.h',
                        help='Output C header with expected checksums')
    parser.add_argument('--stages-only', action='store_true',
                        help='Only run stem + one block (for quick testing)')
    args = parser.parse_args()

    # Load blob
    blob_path = args.blob
    if not os.path.isabs(blob_path):
        blob_path = os.path.join(SCRIPT_DIR.parent, blob_path)
    print(f"Loading blob: {blob_path}")
    with open(blob_path, 'rb') as f:
        blob = f.read()

    hdr = vwb2_read_header(blob)
    print(f"VWB2: {hdr['tensor_count']} tensors, {hdr['data_bytes']} data bytes")

    # Load or create input
    if args.input:
        inp = np.fromfile(args.input, dtype=np.int8).reshape(3, 224, 224)
        print(f"Input from {args.input}")
    else:
        inp = np.zeros((3, 224, 224), dtype=np.int8)
        print("Using zero input (bias-only validation)")

    # Run
    result = run_inference(blob, inp, shift=args.shift, stages_only=args.stages_only)

    # Export
    out_path = args.output
    if not os.path.isabs(out_path):
        out_path = os.path.join(SCRIPT_DIR.parent, out_path)
    export_checksums(result, out_path, args.shift)

    print("\nDone.")


if __name__ == '__main__':
    main()
