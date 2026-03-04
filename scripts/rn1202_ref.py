#!/usr/bin/env python3
"""
rn1202_ref.py — ResNet-1202 CIFAR-10 integer-quantised reference (BD4-faithful)

Mirrors the C firmware function-by-function.  Every function has a comment
naming the C source it mirrors.  No ResNet-50 path, no backwards-compat shims.

C sources mirrored:
  weight_blob.h          → VWB2 blob parsing
  bd_decode_sw.h         → BD_DIALECT_TABLE, bd_decode_block_hu
  bd_act.h               → bd_act_unpack32/tensor, bd_act_compute_exp,
                            bd_act_scale_hu, bd_act_nearest_idx,
                            bd_act_pack32_twostage, quantize_output_bd4
  resnet1202_conv.h      → bias_scale, conv3x3_bd4, conv1x1_bd4,
                            add_relu_bd4, global_avgpool_bd4, fc_linear,
                            run_basic_block_bd4
  resnet1202_layers.h    → rn1202_block_conf (tensor IDs + BasicBlockConf)

Inference path: run_basic_block_bd4 (bulk-unpack optimisation):
  bd_in (BD4) → unpack → int8 → conv3x3_bd4 → bd_mid (BD4)
  bd_mid      → unpack → int8 → conv3x3_bd4 → bd_out (BD4)
  skip: identity → add_relu_bd4(bd_out, bd_in)
        stride=2 → zero-pad + pack → add_relu_bd4
        has_proj → conv1x1_bd4    → add_relu_bd4

Usage:
  python3 scripts/rn1202_ref.py --blob <path/to/weights.bin> \\
                                 --input <path/to/input_32x32.raw> \\
                                 [--output ref.h]

Weight layout (tap-blocked vs flat) is auto-detected from the blob sentinel.
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# BD_DIALECT_TABLE  (mirrors BD_DIALECT_TABLE in bd_decode_sw.h / bd_decode_sw.c)
# 16 dialects × 8 half-unit magnitudes.  Identical to blockdialect_codec.py
# DIALECTS_HALF_UNITS and to the hardware decoder in BlockDialectDecoder.scala.
# ---------------------------------------------------------------------------
BD_DIALECT_TABLE: list[list[int]] = [
    [0, 1, 2, 3, 4,  6, 11, 15],  # dialect  0
    [0, 1, 2, 3, 4,  6,  9, 15],  # dialect  1
    [0, 1, 2, 3, 4,  6, 11, 14],  # dialect  2
    [0, 1, 2, 3, 4,  6,  9, 14],  # dialect  3
    [0, 1, 2, 3, 4,  6, 10, 13],  # dialect  4
    [0, 1, 2, 3, 4,  6,  8, 13],  # dialect  5
    [0, 1, 2, 3, 4,  6, 10, 12],  # dialect  6
    [0, 1, 2, 3, 4,  6,  8, 12],  # dialect  7
    [0, 1, 2, 3, 4,  6,  9, 11],  # dialect  8
    [0, 1, 2, 3, 4,  6,  7, 11],  # dialect  9
    [0, 1, 2, 3, 4,  6,  9, 10],  # dialect 10
    [0, 1, 2, 3, 4,  6,  7, 10],  # dialect 11
    [0, 1, 2, 3, 4,  6,  8,  9],  # dialect 12
    [0, 1, 2, 3, 4,  6,  7,  9],  # dialect 13
    [0, 1, 2, 3, 4,  6,  7,  8],  # dialect 14
    [0, 1, 2, 3, 4,  5,  6,  8],  # dialect 15
]

# Pre-built numpy arrays for fast lookup
_DIALECT_NP = [np.asarray(d, dtype=np.int32) for d in BD_DIALECT_TABLE]

# BD4 block constants (mirrors BD_BLOCK_BYTES / BD_BLOCK_ELEMS in bd_decode_sw.h)
BD_BLOCK_BYTES  = 18
BD_BLOCK_ELEMS  = 32

# ---------------------------------------------------------------------------
# Calibrated per-layer output shifts  (mirrors main.c in resnet1202_phase3_hw_decode)
# shift = max(0, floor(log2(max_abs_accumulator)) - 6)
# The block convolutions accumulate < 64 so shift=0.
# Only the stem (3→16, 3×3, raw int8 input 0–127) reaches ~277, needing shift=2.
# shift=7 was the old bring-up default and zeroed every layer; never use it.
# ---------------------------------------------------------------------------
CONV_SHIFT_STEM   = 2
CONV_SHIFT_STAGE1 = 0
CONV_SHIFT_STAGE2 = 0
CONV_SHIFT_STAGE3 = 0

# =============================================================================
# VWB2 blob parser  (mirrors weight_blob.h)
# =============================================================================

VWB2_MAGIC      = 0x56574232   # 'VWB2'
VWB2_VERSION    = 1
VWB2_BLOCK_SIZE = 32
VWB2_HDR_BYTES  = 32           # sizeof(vwb2_header_t)
VWB2_ENTRY_BYTES = 40          # sizeof(vwb2_entry_t)
VWB2_BD4_SUBHDR  = 8           # sizeof(vwb2_bd4_hdr_t): n_elements + n_blocks

DTYPE_BD4     = 0
DTYPE_FLOAT32 = 1


def _vwb2_parse(blob: bytes) -> tuple[dict, list[dict]]:
    """Parse VWB2 header and tensor table.

    Mirrors vwb2_verify_header + vwb2_table iteration from weight_blob.h.
    Returns (hdr, entries) where entries[i] corresponds to VWB2 tensor slot i.
    """
    magic, version, tensor_count, block_size, \
        table_offset, data_offset, data_bytes, _reserved = \
        struct.unpack_from('<IIIIIIII', blob, 0)
    assert magic      == VWB2_MAGIC,      f"Bad VWB2 magic: 0x{magic:08X}"
    assert version    == VWB2_VERSION,    f"Bad VWB2 version: {version}"
    assert block_size == VWB2_BLOCK_SIZE, f"Bad block_size: {block_size}"

    hdr = {
        'tensor_count': tensor_count,
        'table_offset': table_offset,
        'data_offset':  data_offset,
    }
    entries: list[dict] = []
    for i in range(tensor_count):
        off = table_offset + i * VWB2_ENTRY_BYTES
        name_hash, dtype, tensor_offset, tensor_bytes, n_elements, shape_ndim = \
            struct.unpack_from('<IIIIII', blob, off)
        entries.append({
            'dtype':          dtype,
            'tensor_offset':  tensor_offset,
            'tensor_bytes':   tensor_bytes,
            'n_elements':     n_elements,
        })
    return hdr, entries


def _weight_blocks(blob: bytes, hdr: dict, entries: list[dict], tid: int) -> bytes:
    """Return raw BD4 block bytes for weight tensor at tensor_id.

    Mirrors rn1202_weight_blocks() in resnet1202_conv.h:
      weight entry = table[tensor_id * 2]
      data pointer = vwb2_bd4_blocks(hdr, e)  (skips 8-byte vwb2_bd4_hdr_t)
    """
    e = entries[tid * 2]
    start = hdr['data_offset'] + e['tensor_offset'] + VWB2_BD4_SUBHDR
    end   = hdr['data_offset'] + e['tensor_offset'] + e['tensor_bytes']
    return blob[start:end]


def _weight_block_raw_base(blob: bytes, hdr: dict, entries: list[dict], tid: int) -> int:
    """Return byte offset of the first weight block within 'blob'.

    The firmware's rn1202_weight_blocks() returns vwb2_bd4_blocks(hdr, e),
    a raw pointer that sits 8 bytes past the tensor's data start.  The
    firmware reads at  ptr + block_idx * BD_BLOCK_BYTES  with no bounds
    check on n_blocks, so it reads into adjacent tensors when the requested
    block_idx exceeds the declared n_blocks.  This function returns the
    matching byte offset so Python can replicate that exact behaviour:
        blob[base + idx * BD_BLOCK_BYTES : base + (idx+1) * BD_BLOCK_BYTES]
    """
    e = entries[tid * 2]
    return hdr['data_offset'] + e['tensor_offset'] + VWB2_BD4_SUBHDR


def _bias_f32(blob: bytes, hdr: dict, entries: list[dict], tid: int) -> np.ndarray:
    """Return float32 bias array for tensor at tensor_id.

    Mirrors rn1202_bias_f32() in resnet1202_conv.h:
      bias entry = table[tensor_id * 2 + 1]
      data pointer = vwb2_float32_data(hdr, e)
    """
    e = entries[tid * 2 + 1]
    start = hdr['data_offset'] + e['tensor_offset']
    end   = start + e['tensor_bytes']
    return np.frombuffer(blob[start:end], dtype=np.float32)[:e['n_elements']].copy()


# =============================================================================
# BD4 decode  (mirrors bd_decode_sw.h + bd_act.h)
# =============================================================================

def bd_decode_block_hu(block: bytes) -> tuple[np.ndarray, int, int]:
    """Decode one 18-byte BD4 block → (int16[32] half-units, dialect_id, shared_exp).

    Mirrors bd_decode_block_hu() in bd_decode_sw.h.
    Sign convention: negative when sign bit of code is 1.
    """
    mhi, mlo = block[0], block[1]
    did = (mhi >> 4) & 0xF
    seb = ((mhi & 0xF) << 1) | ((mlo >> 7) & 1)
    table = BD_DIALECT_TABLE[did]
    hu = np.empty(32, dtype=np.int16)
    for i in range(16):
        byte = block[2 + i]
        for nibble in range(2):
            code = (byte >> (4 * (1 - nibble))) & 0xF
            sign = (code >> 3) & 1
            mag  = table[code & 0x07]
            hu[2 * i + nibble] = np.int16(-mag if sign else mag)
    return hu, did, seb


def bd_act_unpack32(block: bytes, out: np.ndarray) -> None:
    """Unpack 18-byte BD4 activation block → 32 int8 values, in-place.

    Mirrors bd_act_unpack32() in bd_act.h:
      value[i] = hu[i] * 2^(seb - 16),  saturated to [-128, +127]
    Note: right shift is arithmetic truncation (matching C >> on int32).
    """
    hu, _did, seb = bd_decode_block_hu(block)
    shift = seb - 16
    for i in range(32):
        v = int(hu[i])
        if shift >= 0:
            v = v << shift
        else:
            v = v >> (-shift)   # arithmetic, rounds toward -inf (C behaviour)
        if v >  127: v =  127
        if v < -128: v = -128
        out[i] = v


def bd_act_unpack_tensor(
    bd_blocks: bytes, n_blocks: int, output: np.ndarray, n_elements: int
) -> None:
    """Unpack n_blocks BD4 blocks into output[n_elements] int8 array.

    Mirrors bd_act_unpack_tensor() in bd_act.h.
    Zero-pads the last partial block automatically.
    """
    tmp = np.empty(32, dtype=np.int8)
    pos = 0
    for b in range(n_blocks):
        if pos >= n_elements:
            break
        bd_act_unpack32(bd_blocks[b * BD_BLOCK_BYTES:(b + 1) * BD_BLOCK_BYTES], tmp)
        count = min(32, n_elements - pos)
        output[pos:pos + count] = tmp[:count]
        pos += count


def bd_act_unpack_tensor_hwcb(
    bd_in: bytes,
    out_c: int, h: int, w: int,
) -> np.ndarray:
    """Unpack HWCB BD4 tensor → int8 CHW ndarray.

    Mirrors HWCB block traversal: for each (y, x, cb) block, unpack the
    32 slots and distribute channel values [oc_lo..oc_hi) into CHW output.
    Returns flat int8 array of length out_c * h * w in CHW order.
    """
    n_cb   = (out_c + 31) // 32
    result = np.zeros(out_c * h * w, dtype=np.int8)
    tmp    = np.empty(32, dtype=np.int8)
    for y in range(h):
        for x in range(w):
            for cb in range(n_cb):
                oc_lo = cb * 32
                oc_hi = min(oc_lo + 32, out_c)
                idx   = y * w * n_cb + x * n_cb + cb
                bd_act_unpack32(bd_in[idx * BD_BLOCK_BYTES:(idx + 1) * BD_BLOCK_BYTES], tmp)
                for i in range(oc_hi - oc_lo):
                    result[(oc_lo + i) * h * w + y * w + x] = tmp[i]
    return result


# ---------------------------------------------------------------------------
# HWCB block-offset helper  (mirrors hwcb_block_ptr in bd_act.h)
# ---------------------------------------------------------------------------

def _hwcb_block_off(y: int, x: int, w: int, n_cb: int, cb: int) -> int:
    """Byte offset of HWCB block (y, x, cb) in a bd_in/bd_out buffer.

    HWCB layout: blocks stored row-major [y, x, cb], each BD_BLOCK_BYTES wide.
    Mirrors hwcb_block_ptr / hwcb_block_ptr_r in bd_act.h.
    """
    return (y * w * n_cb + x * n_cb + cb) * BD_BLOCK_BYTES


# ---------------------------------------------------------------------------
# conv3x3_bd4_tap_hwmac: HWCB BD4 → HWCB BD4 (no int8 intermediate)
# Mirrors conv3x3_bd4_tap_hwmac() in resnet1202_conv.h.
# ---------------------------------------------------------------------------

def conv3x3_bd4_tap_hwmac(
    bd_in:     bytes,        # HWCB BD4 input, in_c × h × w
    blob:      bytes,        # full weight blob
    w_base:    int,          # byte offset of first weight block in blob
    bias:      np.ndarray,   # float32[out_c]
    in_c: int, out_c: int,
    h: int, w: int,
    stride: int,
    out_shift: int,
    do_relu: bool,
) -> bytes:
    """3×3 tap-blocked conv: HWCB BD4 × BD4 weights → HWCB BD4 output.

    Exact mirror of conv3x3_bd4_tap_hwmac() in resnet1202_conv.h.
    Activation blocks are decoded as half-units (no int8 intermediate),
    accumulation formula: acc += (w_hu · act_hu) * 2^(w_seb + act_seb - 32).
    """
    oh, ow       = h // stride, w // stride
    n_cb_in      = (in_c  + 31) // 32
    n_cb_out     = (out_c + 31) // 32
    n_wblocks_per_oc = 9 * n_cb_in

    # Pre-decode all weight blocks once (out_c * n_wblocks_per_oc total)
    total_w = out_c * n_wblocks_per_oc
    w_hu  = np.empty((total_w, 32), dtype=np.int32)
    w_seb = np.empty(total_w,       dtype=np.int32)
    for i in range(total_w):
        off = w_base + i * BD_BLOCK_BYTES
        hu, _did, seb = bd_decode_block_hu(blob[off:off + BD_BLOCK_BYTES])
        w_hu[i]  = hu.astype(np.int32)
        w_seb[i] = seb

    # Pre-compute bias
    bias_init = [_i32(bias_scale(float(bias[oc]), out_shift)) for oc in range(out_c)]

    # Zero block for out-of-bounds (padding) taps
    zero_hu  = np.zeros(32, dtype=np.int32)
    zero_seb = 0

    bd_out = bytearray(oh * ow * n_cb_out * BD_BLOCK_BYTES)
    tmp    = np.zeros(32, dtype=np.int32)

    for y in range(oh):
        for x in range(ow):
            # Step 1: prefetch activation half-units for all 9 taps × n_cb_in
            act_hu  = [[None]*n_cb_in for _ in range(9)]
            act_seb = [[0]   *n_cb_in for _ in range(9)]
            for tap in range(9):
                ky, kx = tap // 3, tap % 3
                iy = y * stride - 1 + ky
                ix = x * stride - 1 + kx
                in_bounds = (0 <= iy < h and 0 <= ix < w)
                for cb in range(n_cb_in):
                    if in_bounds:
                        off = _hwcb_block_off(iy, ix, w, n_cb_in, cb)
                        hu, _did, seb = bd_decode_block_hu(
                            bd_in[off:off + BD_BLOCK_BYTES])
                        act_hu[tap][cb]  = hu.astype(np.int32)
                        act_seb[tap][cb] = int(seb)
                    else:
                        act_hu[tap][cb]  = zero_hu
                        act_seb[tap][cb] = zero_seb

            # Step 2: per-output-channel dot products
            accum_row = [0] * out_c
            for oc in range(out_c):
                acc = bias_init[oc]
                oc_base = oc * n_wblocks_per_oc
                for tap in range(9):
                    tap_base = oc_base + tap * n_cb_in
                    for cb in range(n_cb_in):
                        idx = tap_base + cb
                        # Use plain Python ints throughout — avoids numpy int32 overflow
                        ps  = int(np.dot(w_hu[idx], act_hu[tap][cb]))
                        es  = int(w_seb[idx]) + int(act_seb[tap][cb])
                        sh  = es - 32
                        if sh >= 0:
                            acc = _i32(acc + _i32(ps << sh))
                        else:
                            acc = _i32(acc + (ps >> (-sh)))
                accum_row[oc] = acc

            # Step 3: pack HWCB output for (y, x)
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
                bd_act_pack32_twostage(tmp, bd_out, off)

    return bytes(bd_out)


# ---------------------------------------------------------------------------
# int32 wrap helper  (matches C int32_t wrapping arithmetic)
# ---------------------------------------------------------------------------

def _i32(x: int) -> int:
    """Truncate Python int to signed 32-bit range, matching C int32_t overflow."""
    x = int(x) & 0xFFFFFFFF
    return x if x < 0x80000000 else x - 0x100000000

# ---------------------------------------------------------------------------
# Activation packing helpers  (mirrors bd_act.h)
# ---------------------------------------------------------------------------

def _bd_act_compute_exp(max_abs: int) -> int:
    """Compute shared exponent bits for a block.

    Mirrors bd_act_compute_exp() in bd_act.h:
      Find floor(log2(max_abs)) via repeated right-shift (CLZ-equivalent).
      seb = log2_val + 12, clamped to [0, 31].
    """
    if max_abs == 0:
        return 0
    log2_val = max_abs.bit_length() - 1   # floor(log2(max_abs))
    seb = log2_val + 12
    if seb < 0:  seb = 0
    if seb > 31: seb = 31
    return seb


def _bd_act_scale_hu(abs_val: int, seb: int) -> int:
    """Scale |value| to half-unit grid given shared_exp_bits.

    Mirrors bd_act_scale_hu() in bd_act.h:
      shift = 16 - seb
      if shift >= 0: scaled = abs_val << shift          (exact)
      else:          scaled = round(abs_val >> -shift)  (round-to-nearest)
      clamped to [0, 15]
    """
    shift = 16 - seb
    if shift >= 0:
        scaled = abs_val << shift
    else:
        rsh = -shift
        scaled = (abs_val + (1 << (rsh - 1))) >> rsh
    if scaled > 15: scaled = 15
    if scaled < 0:  scaled = 0
    return scaled


def _bd_act_nearest_idx(target_hu: int, dialect_id: int) -> int:
    """Find nearest index in a dialect for a half-unit target value.

    Mirrors bd_act_nearest_idx() in bd_act.h (linear scan, first-wins on tie).
    """
    d = BD_DIALECT_TABLE[dialect_id]
    best_i    = 0
    best_dist = abs(target_hu - d[0])
    for i in range(1, 8):
        dist = abs(target_hu - d[i])
        if dist < best_dist:
            best_dist = dist
            best_i    = i
    return best_i


# C constants for two-stage dialect selection  (mirrors bd_act.h)
_BD_BENEFICIAL_LO_X2 = [20, 20, 18, 18, 16, 16, 15, 13]
_BD_BENEFICIAL_HI_X2 = [26, 25, 23, 22, 20, 19, 17, 15]


def bd_act_pack32_twostage(vals: np.ndarray, block_out: bytearray, offset: int) -> None:
    """Pack 32 signed int32 values into 18-byte BD4 block.

    Mirrors bd_act_pack32_twostage() in bd_act.h exactly, including:
      - Step 1: signs + max_abs
      - Step 2: shared exponent via _bd_act_compute_exp
      - Step 3: scale to half-units via _bd_act_scale_hu; track block_maxhu
      - Stage 1: map block_maxhu → pair_id (bd_act_pair_from_maxhu)
      - Stage 2: count elements in dialect-A's beneficial range (doubled thresholds)
      - Step 5: quantize to nearest in chosen dialect (_bd_act_nearest_idx)
      - Step 6: pack meta + codes into 18 bytes
    """
    # Step 1: signs and max_abs
    signs    = np.empty(32, dtype=np.uint8)
    abs_vals = np.empty(32, dtype=np.int32)
    max_abs  = 0
    for i in range(32):
        v = int(vals[i])
        if v < 0:
            signs[i]  = 1
            v = -v
        else:
            signs[i]  = 0
        abs_vals[i] = v
        if v > max_abs:
            max_abs = v

    # Step 2: shared exponent
    seb = _bd_act_compute_exp(max_abs)

    # Step 3: scale to half-units; track block_maxhu
    scaled_hu  = np.empty(32, dtype=np.uint8)
    block_maxhu = 0
    for i in range(32):
        h = _bd_act_scale_hu(int(abs_vals[i]), seb)
        scaled_hu[i] = h
        if h > block_maxhu:
            block_maxhu = h

    # Stage 1: map block_maxhu → pair_id  (mirrors bd_act_pair_from_maxhu)
    if   block_maxhu >= 15: pair_id = 0
    elif block_maxhu >= 14: pair_id = 1
    elif block_maxhu >= 13: pair_id = 2
    elif block_maxhu >= 12: pair_id = 3
    elif block_maxhu >= 11: pair_id = 4
    elif block_maxhu >= 10: pair_id = 5
    elif block_maxhu >=  9: pair_id = 6
    else:                   pair_id = 7

    # Stage 2: count elements in dialect A's beneficial range
    # Condition (doubled to avoid fractions): lo_x2 <= 2*scaled_hu < hi_x2
    lo = _BD_BENEFICIAL_LO_X2[pair_id]
    hi = _BD_BENEFICIAL_HI_X2[pair_id]
    count_a = 0
    for i in range(32):
        s2 = int(scaled_hu[i]) << 1   # 2 * scaled_hu[i], same as C uint8 << 1
        if s2 >= lo and s2 < hi:
            count_a += 1

    # Choose dialect: A (even) if majority in beneficial range, else B (odd)
    best_dialect = pair_id * 2
    if count_a * 2 < 32:
        best_dialect += 1

    # Step 5: quantize to nearest in chosen dialect
    codes = np.empty(32, dtype=np.uint8)
    for i in range(32):
        idx      = _bd_act_nearest_idx(int(scaled_hu[i]), best_dialect)
        codes[i] = (int(signs[i]) << 3) | (idx & 0x07)

    # Step 6: pack into 18-byte block
    # Meta word (big-endian u16): dialect_id[15:12] | shared_exp[11:7] | zeros[6:0]
    meta = ((best_dialect & 0xF) << 12) | ((seb & 0x1F) << 7)
    block_out[offset + 0] = (meta >> 8) & 0xFF
    block_out[offset + 1] =  meta       & 0xFF
    for i in range(16):
        block_out[offset + 2 + i] = (int(codes[2 * i]) << 4) | (int(codes[2 * i + 1]) & 0x0F)


def quantize_output_bd4(
    accum: np.ndarray, n_elements: int, out_shift: int, do_relu: bool
) -> bytes:
    """Quantize int32/int64 accumulator → BD4, 32 elements per block.

    Mirrors quantize_output_bd4() in bd_act.h:
      v = accum[i] >> out_shift
      if do_relu and v < 0: v = 0
      pack 32 at a time via bd_act_pack32_twostage (no int8 clamp)
    """
    n_blocks  = (n_elements + 31) // 32
    result    = bytearray(n_blocks * BD_BLOCK_BYTES)
    tmp       = np.zeros(32, dtype=np.int32)
    for b in range(n_blocks):
        base  = b * 32
        count = min(32, n_elements - base)
        for i in range(32):
            if i < count:
                v = int(accum[base + i]) >> out_shift
                if do_relu and v < 0:
                    v = 0
                tmp[i] = v
            else:
                tmp[i] = 0
        bd_act_pack32_twostage(tmp, result, b * BD_BLOCK_BYTES)
    return bytes(result)


def quantize_output_bd4_hwcb(
    accum: np.ndarray,   # int32 flat [out_c * oh * ow] or shaped [out_c, oh, ow]
    out_c: int, oh: int, ow: int,
    out_shift: int, do_relu: bool,
) -> bytes:
    """Quantize int32 accumulator → HWCB BD4 (H×W×n_cb blocks).

    Mirrors the HWCB output packing in conv3x3_bd4_tap() in resnet1202_conv.h:
      For each (y, x) pixel and channel-block cb:
        slots [0..count-1] = out_c channel values at that pixel (shifted+relu)
        slots [count..31]  = 0 (zero-pad)
      block(y, x, cb) at byte offset (y*ow*n_cb + x*n_cb + cb) * BD_BLOCK_BYTES.
    Total bytes = oh * ow * n_cb * BD_BLOCK_BYTES.
    """
    n_cb    = (out_c + 31) // 32
    n_total = oh * ow * n_cb
    result  = bytearray(n_total * BD_BLOCK_BYTES)
    tmp     = np.zeros(32, dtype=np.int32)
    acc_3d  = accum.reshape(out_c, oh, ow)
    for y in range(oh):
        for x in range(ow):
            for cb in range(n_cb):
                oc_lo = cb * 32
                oc_hi = min(oc_lo + 32, out_c)
                count = oc_hi - oc_lo
                for i in range(32):
                    if i < count:
                        v = int(acc_3d[oc_lo + i, y, x]) >> out_shift
                        if do_relu and v < 0:
                            v = 0
                        tmp[i] = v
                    else:
                        tmp[i] = 0
                idx = y * ow * n_cb + x * n_cb + cb
                bd_act_pack32_twostage(tmp, result, idx * BD_BLOCK_BYTES)
    return bytes(result)


# =============================================================================
# conv3x3_bd4_tap: mirrors firmware conv3x3_bd4_tap() in resnet1202_conv.h
# =============================================================================

def conv3x3_bd4_tap(
    inp: np.ndarray,  # int8 CHW flat, length = in_c * h * w
    blob: bytes,      # full weight blob (firmware reads past tensor boundary)
    w_base: int,      # byte offset of first weight block in blob
    bias: np.ndarray, # float32[out_c]
    in_c: int, out_c: int,
    h: int, w: int,
    stride: int,
    out_shift: int,
    do_relu: bool,
) -> bytes:
    """3x3 convolution with tap-blocked weights -> HWCB BD4 output.

    Exact mirror of conv3x3_bd4_tap() in resnet1202_conv.h.
    Weight block index for (oc, tap, cb):
        oc * n_wblocks_per_oc + tap * n_cb_in + cb
    where n_cb_in = ceil(in_c/32), n_wblocks_per_oc = 9 * n_cb_in.
    Reads blob[w_base + idx*BD_BLOCK_BYTES] directly, mirroring the firmware
    pointer arithmetic (which reads past the tensor's declared n_blocks).
    """
    oh, ow           = h // stride, w // stride
    n_cb_in          = (in_c  + 31) // 32
    n_cb_out         = (out_c + 31) // 32
    n_wblocks_per_oc = 9 * n_cb_in

    # Pre-decode all weight blocks (out_c * n_wblocks_per_oc total).
    total_w_blocks = out_c * n_wblocks_per_oc
    w_hu  = np.empty((total_w_blocks, 32), dtype=np.int16)
    w_seb = np.empty(total_w_blocks,       dtype=np.int32)
    for i in range(total_w_blocks):
        off = w_base + i * BD_BLOCK_BYTES
        w_hu[i], _, w_seb[i] = bd_decode_block_hu(blob[off : off + BD_BLOCK_BYTES])

    # Pre-compute bias (already in output-shift units, like firmware bias_init[])
    bias_init = np.array([_i32(bias_scale(float(bias[oc]), out_shift))
                          for oc in range(out_c)], dtype=np.int32)

    inp_3d     = inp.reshape(in_c, h, w).astype(np.int32)

    n_total = oh * ow * n_cb_out
    result  = bytearray(n_total * BD_BLOCK_BYTES)
    tmp     = np.zeros(32, dtype=np.int32)

    for y in range(oh):
        for x in range(ow):
            accum_row = bias_init.copy()

            for tap in range(9):
                ky = tap // 3
                kx = tap  % 3
                iy = y * stride - 1 + ky
                ix = x * stride - 1 + kx
                if iy < 0 or iy >= h or ix < 0 or ix >= w:
                    continue

                for cb in range(n_cb_in):
                    ic_base = cb * 32
                    count   = min(in_c - ic_base, 32)
                    act_col = inp_3d[ic_base:ic_base + count, iy, ix].astype(np.int64)

                    for oc in range(out_c):
                        blk   = oc * n_wblocks_per_oc + tap * n_cb_in + cb
                        seb   = int(w_seb[blk])
                        hu    = w_hu[blk, :count].astype(np.int64)
                        bsum  = _i32(int(np.dot(hu, act_col)))
                        shift = seb - 16
                        if shift >= 0:
                            accum_row[oc] = _i32(int(accum_row[oc]) + (bsum << shift))
                        else:
                            accum_row[oc] = _i32(int(accum_row[oc]) + (bsum >> (-shift)))

            # Pack HWCB output for (y, x)
            for cb in range(n_cb_out):
                oc_lo = cb * 32
                for i in range(32):
                    oc = oc_lo + i
                    if oc < out_c:
                        v = _i32(int(accum_row[oc])) >> out_shift
                        if do_relu and v < 0:
                            v = 0
                        tmp[i] = v
                    else:
                        tmp[i] = 0
                idx = y * ow * n_cb_out + x * n_cb_out + cb
                bd_act_pack32_twostage(tmp, result, idx * BD_BLOCK_BYTES)

    return bytes(result)


# =============================================================================
# bias_scale  (mirrors bias_scale() in resnet1202_conv.h)
# =============================================================================

def bias_scale(f: float, shift: int) -> int:
    """IEEE 754 bias scaling: integer equivalent of (int32_t)(f * 2^shift).

    Mirrors bias_scale() in resnet1202_conv.h.  Avoids soft-float multiply
    by manipulating the IEEE 754 exponent directly:
      exp   = biased_exponent - 127
      man   = mantissa | 0x800000  (hidden bit)
      total = exp - 23 + shift     (exponent of result integer)
      mag   = man << total  or  man >> -total
    Returns a Python int (full precision; caller truncates if needed).
    """
    u    = struct.unpack('<I', struct.pack('<f', float(np.float32(f))))[0]
    exp  = int((u >> 23) & 0xFF) - 127
    if exp == -127:          # zero or denormal
        return 0
    man  = (u & 0x7FFFFF) | 0x800000
    sign = -1 if (u >> 31) else 1
    total = exp - 23 + shift
    if total >= 31:
        # Overflow: mirrors C's saturating behaviour
        return 0x7FFFFFFF if sign > 0 else -(0x7FFFFFFF)
    if total >= 0:
        mag = man << total
    elif -total < 24:
        mag = man >> (-total)   # truncates toward zero (C unsigned >> behaviour)
    else:
        return 0
    return mag if sign > 0 else -mag


# =============================================================================
# Convolution kernels  (mirrors resnet1202_conv.h)
# =============================================================================

def conv3x3_bd4(
    inp: np.ndarray,          # int8 flat array, length = in_c * h * w  (CHW order)
    w_blocks: bytes,
    bias: np.ndarray,         # float32[out_c]
    in_c: int, out_c: int,
    h: int, w: int,
    stride: int,
    out_shift: int,
    do_relu: bool,
    tap_blocked: bool = False,
) -> bytes:
    """3×3 convolution with BD4 weights → BD4 output (no int8 at output).

    Mirrors conv3x3_bd4() in resnet1202_conv.h.

    Weights are stored in the blob as a flat BD4 stream (encode_tensor encodes
    the full [OC,*kernel] tensor without per-OC block padding).  Elements for
    OC `oc` and kernel position `k` live at global flat index oc*kernel_elems+k,
    which may span block boundaries.  We pre-decode all blocks then slice.

    tap_blocked=True: weight tensor was permuted [OC,IC,KY,KX]→[OC,KY,KX,IC]
    before encoding, so kernel_elems are ordered [KY,KX,IC] within each OC.
    act_col is built with the matching [KY,KX,IC] column ordering.
    """
    oh, ow       = h // stride, w // stride
    out_elems    = out_c * oh * ow
    kernel_elems = in_c * 9
    n_total_elems  = kernel_elems * out_c
    n_total_blocks = (n_total_elems + 31) // 32

    # -- Build activation column matrix (im2col) ----------------------------
    # inp_padded: [in_c, h+2, w+2] with zero boundary padding
    inp_3d     = inp.reshape(in_c, h, w).astype(np.int32)
    inp_padded = np.pad(inp_3d, ((0, 0), (1, 1), (1, 1)), mode='constant')
    from numpy.lib.stride_tricks import as_strided as _ast
    s = inp_padded.strides
    # windows: [in_c, oh, ow, 3, 3]
    windows = _ast(
        inp_padded,
        shape=(in_c, oh, ow, 3, 3),
        strides=(s[0], s[1] * stride, s[2] * stride, s[1], s[2]),
        writeable=False,
    )
    if tap_blocked:
        # Weight flat order within OC: [KY, KX, IC] → flat = ky*3*in_c + kx*in_c + ic
        # Transpose windows to [oh, ow, 3, 3, in_c] → reshape to [oh*ow, 9*in_c]
        act_col = windows.transpose(1, 2, 3, 4, 0).reshape(oh * ow, kernel_elems)
    else:
        # Weight flat order within OC: [IC, KY, KX] → flat = ic*9 + ky*3 + kx
        # Transpose windows to [oh, ow, in_c, 3, 3] → reshape to [oh*ow, in_c*9]
        act_col = windows.transpose(1, 2, 0, 3, 4).reshape(oh * ow, kernel_elems)

    # -- Pre-decode all weight blocks (flat stream) -------------------------
    # The blob stores the full [OC, kernel_elems] weight tensor as one
    # contiguous flat BD4 stream with no per-OC block padding.
    all_hu  = np.empty((n_total_blocks, 32), dtype=np.int16)
    all_seb = np.empty(n_total_blocks,       dtype=np.int32)
    for blk in range(n_total_blocks):
        all_hu[blk], _did, all_seb[blk] = bd_decode_block_hu(
            w_blocks[blk * BD_BLOCK_BYTES:(blk + 1) * BD_BLOCK_BYTES])

    # -- Accumulate ----------------------------------------------------------
    accum = np.zeros(out_elems, dtype=np.int32)

    for oc in range(out_c):
        b_i32   = _i32(bias_scale(float(bias[oc]), out_shift))
        oc_acc  = np.zeros(oh * ow, dtype=np.int32)
        g_start = oc * kernel_elems          # first global flat elem of this OC
        g_end   = g_start + kernel_elems     # one past last
        blk_lo  = g_start  // 32
        blk_hi  = (g_end - 1) // 32

        for blk in range(blk_lo, blk_hi + 1):
            # Global flat index range inside this block that belongs to this OC
            lo_g   = max(blk * 32, g_start)
            hi_g   = min(blk * 32 + 32, g_end)
            pos_lo = lo_g - blk * 32    # position within the 32-element block
            pos_hi = hi_g - blk * 32
            k_lo   = lo_g - g_start     # kernel-element index within OC
            k_hi   = hi_g - g_start

            hu_slice  = all_hu[blk, pos_lo:pos_hi].astype(np.int64)
            act_slice = act_col[:, k_lo:k_hi].astype(np.int64)
            bsums_i64 = act_slice @ hu_slice          # [oh*ow]
            bsums     = bsums_i64.astype(np.int32)    # wrap to int32 like C

            shift = int(all_seb[blk]) - 16
            if shift >= 0:
                oc_acc = (oc_acc.astype(np.int64) + (bsums.astype(np.int64) << shift)).astype(np.int32)
            else:
                shifted = bsums.astype(np.int32) >> (-shift)
                oc_acc  = (oc_acc.astype(np.int64) + shifted.astype(np.int64)).astype(np.int32)

        oc_acc = (oc_acc.astype(np.int64) + b_i32).astype(np.int32)
        accum[oc * oh * ow:(oc + 1) * oh * ow] = oc_acc

    if tap_blocked:
        return quantize_output_bd4_hwcb(accum, out_c, oh, ow, out_shift, do_relu)
    return quantize_output_bd4(accum, out_elems, out_shift, do_relu)


def conv1x1_bd4(
    inp: np.ndarray,          # int8 flat array, length = in_c * h * w  (CHW)
    w_blocks: bytes,
    bias: np.ndarray,         # float32[out_c]
    in_c: int, out_c: int,
    h: int, w: int,
    stride: int,
    out_shift: int,
    do_relu: bool,
) -> bytes:
    """1×1 projection convolution with BD4 weights → BD4 output.

    Mirrors conv1x1_bd4() in resnet1202_conv.h.  Kernel is 1×1 so no
    spatial gather — just subsampled input channels.

    Weights are stored as a flat BD4 stream (no per-OC block padding).
    Elements for OC `oc` and channel `ic` live at global flat index
    oc*in_c + ic, which may span block boundaries for small in_c.
    """
    oh, ow         = h // stride, w // stride
    out_elems      = out_c * oh * ow
    kernel_elems   = in_c
    n_total_elems  = kernel_elems * out_c
    n_total_blocks = (n_total_elems + 31) // 32

    # Column matrix: inp[:, ::stride, ::stride] reshaped to [oh*ow, in_c]
    inp_3d  = inp.reshape(in_c, h, w).astype(np.int32)
    act_col = inp_3d[:, ::stride, ::stride].reshape(in_c, oh * ow).T  # [oh*ow, in_c]

    # Pre-decode all weight blocks (flat stream)
    all_hu  = np.empty((n_total_blocks, 32), dtype=np.int16)
    all_seb = np.empty(n_total_blocks,       dtype=np.int32)
    for blk in range(n_total_blocks):
        all_hu[blk], _did, all_seb[blk] = bd_decode_block_hu(
            w_blocks[blk * BD_BLOCK_BYTES:(blk + 1) * BD_BLOCK_BYTES])

    accum = np.zeros(out_elems, dtype=np.int32)

    for oc in range(out_c):
        b_i32   = _i32(bias_scale(float(bias[oc]), out_shift))
        oc_acc  = np.zeros(oh * ow, dtype=np.int32)
        g_start = oc * kernel_elems
        g_end   = g_start + kernel_elems
        blk_lo  = g_start  // 32
        blk_hi  = (g_end - 1) // 32

        for blk in range(blk_lo, blk_hi + 1):
            lo_g   = max(blk * 32, g_start)
            hi_g   = min(blk * 32 + 32, g_end)
            pos_lo = lo_g - blk * 32
            pos_hi = hi_g - blk * 32
            k_lo   = lo_g - g_start
            k_hi   = hi_g - g_start

            hu_slice  = all_hu[blk, pos_lo:pos_hi].astype(np.int64)
            act_slice = act_col[:, k_lo:k_hi].astype(np.int64)
            bsums_i64 = act_slice @ hu_slice
            bsums     = bsums_i64.astype(np.int32)

            shift = int(all_seb[blk]) - 16
            if shift >= 0:
                oc_acc = (oc_acc.astype(np.int64) + (bsums.astype(np.int64) << shift)).astype(np.int32)
            else:
                shifted = bsums.astype(np.int32) >> (-shift)
                oc_acc  = (oc_acc.astype(np.int64) + shifted.astype(np.int64)).astype(np.int32)

        oc_acc = (oc_acc.astype(np.int64) + b_i32).astype(np.int32)
        accum[oc * oh * ow:(oc + 1) * oh * ow] = oc_acc

    return quantize_output_bd4(accum, out_elems, out_shift, do_relu)


def add_relu_bd4(bd_a: bytes, bd_b: bytes, n_elements: int) -> bytes:
    """BD4 residual add + ReLU.

    Mirrors add_relu_bd4() in resnet1202_conv.h:
      For each block:
        bd_act_unpack32(bd_a) → va[32] int8
        bd_act_unpack32(bd_b) → vb[32] int8
        tmp[i] = max(0, (int32)va[i] + (int32)vb[i])   ← ReLU
        bd_act_pack32_twostage(tmp) → output block
    bd_out may alias bd_a or bd_b (safe: one block at a time).
    """
    n_blocks = (n_elements + 31) // 32
    result   = bytearray(n_blocks * BD_BLOCK_BYTES)
    va  = np.empty(32, dtype=np.int8)
    vb  = np.empty(32, dtype=np.int8)
    tmp = np.empty(32, dtype=np.int32)
    for b in range(n_blocks):
        off = b * BD_BLOCK_BYTES
        bd_act_unpack32(bd_a[off:off + BD_BLOCK_BYTES], va)
        bd_act_unpack32(bd_b[off:off + BD_BLOCK_BYTES], vb)
        for i in range(32):
            s = int(va[i]) + int(vb[i])
            tmp[i] = 0 if s < 0 else s
        bd_act_pack32_twostage(tmp, result, off)
    return bytes(result)


def add_relu_bd4_hwcb(
    bd_a: bytes, bd_b: bytes,
    out_c: int, h: int, w: int,
) -> bytes:
    """BD4 residual add + ReLU in HWCB layout.

    Mirrors add_relu_bd4_hwcb() in resnet1202_conv.h.
    Both bd_a and bd_b are HWCB-packed with (out_c, h, w).
    Iterates over all h*w*n_cb blocks identically to add_relu_bd4_hwcb().
    """
    n_cb    = (out_c + 31) // 32
    n_total = h * w * n_cb
    result  = bytearray(n_total * BD_BLOCK_BYTES)
    va  = np.empty(32, dtype=np.int8)
    vb  = np.empty(32, dtype=np.int8)
    tmp = np.empty(32, dtype=np.int32)
    for i in range(n_total):
        off = i * BD_BLOCK_BYTES
        bd_act_unpack32(bd_a[off:off + BD_BLOCK_BYTES], va)
        bd_act_unpack32(bd_b[off:off + BD_BLOCK_BYTES], vb)
        for j in range(32):
            s = int(va[j]) + int(vb[j])
            tmp[j] = 0 if s < 0 else s
        bd_act_pack32_twostage(tmp, result, off)
    return bytes(result)


def global_avgpool_bd4(bd_in: bytes, c: int, h: int, w: int) -> np.ndarray:
    """Global average pool: BD4 C×H×W → int8[C].

    Mirrors global_avgpool_bd4() in resnet1202_conv.h (bulk-unpack path):
      Unpack entire BD4 tensor to int8, sum over H×W per channel,
      divide by spatial = H*W (integer truncation), saturate to int8.
    """
    n_elements = c * h * w
    n_blocks   = (n_elements + 31) // 32
    buf        = np.zeros(n_elements, dtype=np.int8)
    bd_act_unpack_tensor(bd_in, n_blocks, buf, n_elements)
    # sum over H*W per channel: buf is CHW
    buf_i32 = buf.reshape(c, h * w).astype(np.int32)
    sums    = buf_i32.sum(axis=1)              # [c]
    avgs    = sums // (h * w)                  # integer truncation toward 0
    avgs    = np.clip(avgs, -128, 127)
    return avgs.astype(np.int8)


def global_avgpool_hwcb(bd_in: bytes, c: int, h: int, w: int) -> np.ndarray:
    """Global average pool from HWCB BD4: C×H×W → int8[C].

    Mirrors global_avgpool_hwcb() in resnet1202_conv.h.
    Unpacks via bd_act_unpack_tensor_hwcb then reduces over H*W.
    """
    buf   = bd_act_unpack_tensor_hwcb(bd_in, c, h, w)
    buf3d = buf.reshape(c, h * w).astype(np.int32)
    sums  = buf3d.sum(axis=1)
    avgs  = sums // (h * w)
    avgs  = np.clip(avgs, -128, 127)
    return avgs.astype(np.int8)


def fc_linear(
    inp: np.ndarray,      # int8[in_c]
    w_blocks: bytes,
    bias: Optional[np.ndarray],   # float32[out_c] or None
    in_c: int, out_c: int,
) -> np.ndarray:
    """Fully connected layer: int8 input → int32 logits.

    Mirrors fc_linear() in resnet1202_conv.h.
    Note: bias uses hardcoded shift=7 (matching the C source).

    For FC, in_c=64 and out_c=10 → kernel_elems=64=2×32, so OC blocks align
    with block boundaries (64%32==0) and flat == OC-padded.  We use the same
    flat-stream logic as the other conv functions for consistency.
    """
    kernel_elems   = in_c
    n_total_elems  = kernel_elems * out_c
    n_total_blocks = (n_total_elems + 31) // 32
    logits  = np.zeros(out_c, dtype=np.int32)
    inp_i32 = inp.astype(np.int32)

    # Pre-decode all weight blocks (flat stream)
    all_hu  = np.empty((n_total_blocks, 32), dtype=np.int16)
    all_seb = np.empty(n_total_blocks,       dtype=np.int32)
    for blk in range(n_total_blocks):
        all_hu[blk], _did, all_seb[blk] = bd_decode_block_hu(
            w_blocks[blk * BD_BLOCK_BYTES:(blk + 1) * BD_BLOCK_BYTES])

    for oc in range(out_c):
        acc     = 0   # Python int; i32-wrap at each step
        g_start = oc * kernel_elems
        g_end   = g_start + kernel_elems
        blk_lo  = g_start  // 32
        blk_hi  = (g_end - 1) // 32

        for blk in range(blk_lo, blk_hi + 1):
            lo_g   = max(blk * 32, g_start)
            hi_g   = min(blk * 32 + 32, g_end)
            pos_lo = lo_g - blk * 32
            pos_hi = hi_g - blk * 32
            k_lo   = lo_g - g_start
            k_hi   = hi_g - g_start
            count  = pos_hi - pos_lo

            hu_slice = all_hu[blk, pos_lo:pos_hi].astype(np.int64)
            bsum = _i32(int(np.dot(hu_slice, inp_i32[k_lo:k_hi].astype(np.int64))))
            shift = int(all_seb[blk]) - 16
            if shift >= 0:
                acc = _i32(acc + (bsum << shift))
            else:
                acc = _i32(acc + (bsum >> (-shift)))

        if bias is not None:
            acc = _i32(acc + bias_scale(float(bias[oc]), 7))   # hardcoded shift=7, mirrors C
        logits[oc] = acc

    return logits


# =============================================================================
# BasicBlock  (mirrors run_basic_block_bd4 in resnet1202_conv.h)
# =============================================================================

def _zero_pad_stride2_bd4(
    bd_in: bytes, in_c: int, out_c: int, ih: int, iw: int
) -> bytes:
    """Option A stride-2 zero-pad skip for downsampling block (BD4 output).

    Mirrors the stride==2, !has_proj path in run_basic_block_bd4():
      1. Unpack bd_in → int8 buffer
      2. Zero-fill int32 scratch of size out_c * oh * ow
      3. Copy stride-2 sampled in_c channels into scratch (as int32)
      4. Pack to BD4 via bd_act_pack32_twostage (32 elements per block)
    Channels [in_c, out_c) are zero-filled (Option A zero-padding).
    """
    oh, ow     = ih // 2, iw // 2
    in_elems   = in_c * ih * iw
    out_elems  = out_c * oh * ow
    n_in_blks  = (in_elems  + 31) // 32

    unpack_buf = np.zeros(in_elems, dtype=np.int8)
    bd_act_unpack_tensor(bd_in, n_in_blks, unpack_buf, in_elems)

    # Zero-fill output scratch; copy stride-2 sampled input channels
    accum_tmp  = np.zeros(out_elems, dtype=np.int32)
    out_spatial = oh * ow
    for ic in range(in_c):
        src_plane = unpack_buf[ic * ih * iw:(ic + 1) * ih * iw].reshape(ih, iw)
        dst_plane = src_plane[::2, ::2].astype(np.int32)   # stride-2 subsample
        accum_tmp[ic * out_spatial:(ic + 1) * out_spatial] = dst_plane.ravel()
    # channels [in_c, out_c) remain 0 — Option A zero-padding

    # Pack to BD4 with bd_act_pack32_twostage (no out_shift, no relu)
    n_blocks = (out_elems + 31) // 32
    result   = bytearray(n_blocks * BD_BLOCK_BYTES)
    tmp      = np.zeros(32, dtype=np.int32)
    for b in range(n_blocks):
        base  = b * 32
        count = min(32, out_elems - base)
        for i in range(32):
            tmp[i] = int(accum_tmp[base + i]) if i < count else 0
        bd_act_pack32_twostage(tmp, result, b * BD_BLOCK_BYTES)
    return bytes(result)


def _zero_pad_stride2_hwcb(
    bd_in: bytes, in_c: int, out_c: int, ih: int, iw: int
) -> bytes:
    """Option A stride-2 zero-pad skip in HWCB BD4 format.

    Mirrors bd4_zero_pad_stride2_hwcb() in resnet1202_conv.h:
      cb < n_cb_in  → copy HWCB block from stride-2 source position verbatim
      cb >= n_cb_in → zero block (channels [in_c, out_c) are zero-padded)
    Returns HWCB BD4 of (out_c, oh, ow).
    """
    oh, ow    = ih // 2, iw // 2
    n_cb_in   = (in_c  + 31) // 32
    n_cb_out  = (out_c + 31) // 32
    n_total   = oh * ow * n_cb_out
    result    = bytearray(n_total * BD_BLOCK_BYTES)
    tmp       = np.zeros(32, dtype=np.int32)
    for oy in range(oh):
        for ox in range(ow):
            for cb in range(n_cb_out):
                dst_idx = oy * ow * n_cb_out + ox * n_cb_out + cb
                dst_off = dst_idx * BD_BLOCK_BYTES
                if cb < n_cb_in:
                    # copy source HWCB block at (2*oy, 2*ox, cb) verbatim
                    src_idx = (oy * 2) * iw * n_cb_in + (ox * 2) * n_cb_in + cb
                    src_off = src_idx * BD_BLOCK_BYTES
                    result[dst_off:dst_off + BD_BLOCK_BYTES] = bd_in[src_off:src_off + BD_BLOCK_BYTES]
                else:
                    # zero-pad channel block (write a dialect-0 / exp-0 / all-zero block)
                    for i in range(32):
                        tmp[i] = 0
                    bd_act_pack32_twostage(tmp, result, dst_off)
    return bytes(result)


def run_basic_block_bd4(
    in_c: int, out_c: int,
    ih: int, iw: int,
    stride: int,
    has_proj: bool,
    tid_conv_a: int, tid_conv_b: int, tid_proj: int,
    shift_a: int, shift_b: int, shift_proj: int,
    blob: bytes, hdr: dict, entries: list[dict],
    bd_in: bytes,
    tap_blocked: bool = False,
) -> bytes:
    """One BasicBlock: BD4 in → BD4 out (bulk-unpack optimisation).

    Mirrors run_basic_block_bd4() in resnet1202_conv.h.

    Dataflow:
      1. bd_in (BD4) → bulk-unpack → int8
      2. conv3x3_bd4(int8 → bd_mid, conv_a weights, relu=1)
      3. bd_mid       → bulk-unpack → int8
      4. conv3x3_bd4(int8 → bd_out, conv_b weights, relu=0)
      5. Skip connection:
           has_proj → unpack bd_in again → conv1x1_bd4 → bd_skip
                      add_relu_bd4(bd_out, bd_skip)
           stride=2 → _zero_pad_stride2_bd4(bd_in) → bd_skip
                      add_relu_bd4(bd_out, bd_skip)
           identity → add_relu_bd4(bd_out, bd_in)
    """
    oh, ow    = ih // stride, iw // stride
    in_elems  = in_c * ih * iw
    out_elems = out_c * oh * ow
    n_in_blks = (in_elems  + 31) // 32

    ba = _bias_f32(blob, hdr, entries, tid_conv_a)
    bb = _bias_f32(blob, hdr, entries, tid_conv_b)

    if tap_blocked:
        # ── tap-blocked path: HWCB BD4 → conv3x3_bd4_tap_hwmac → HWCB BD4 ──
        # Mirrors run_basic_block_bd4_tap() in resnet1202_conv.h.
        # No int8 intermediate: activation half-units are used directly.
        wa_base = _weight_block_raw_base(blob, hdr, entries, tid_conv_a)
        bd_mid  = conv3x3_bd4_tap_hwmac(bd_in,  blob, wa_base, ba,
                                         in_c, out_c, ih, iw, stride, shift_a, do_relu=True)
        wb_base = _weight_block_raw_base(blob, hdr, entries, tid_conv_b)
        bd_out  = conv3x3_bd4_tap_hwmac(bd_mid, blob, wb_base, bb,
                                         out_c, out_c, oh, ow, 1,      shift_b, do_relu=False)
    else:
        # ── flat BD4 bulk-unpack path ─────────────────────────────────────
        wa = _weight_blocks(blob, hdr, entries, tid_conv_a)
        unpack_in = np.zeros(in_elems, dtype=np.int8)
        bd_act_unpack_tensor(bd_in, n_in_blks, unpack_in, in_elems)
        bd_mid = conv3x3_bd4(unpack_in, wa, ba, in_c, out_c, ih, iw, stride, shift_a, do_relu=True,
                             tap_blocked=False)

        wb = _weight_blocks(blob, hdr, entries, tid_conv_b)
        n_mid_blks = (out_elems + 31) // 32
        unpack_mid = np.zeros(out_elems, dtype=np.int8)
        bd_act_unpack_tensor(bd_mid, n_mid_blks, unpack_mid, out_elems)
        bd_out = conv3x3_bd4(unpack_mid, wb, bb, out_c, out_c, oh, ow, 1, shift_b, do_relu=False,
                             tap_blocked=False)

    # ── Step 5: skip connection + ReLU ───────────────────────────────────
    if has_proj:
        # Option B: learnable conv1×1 projection (has_proj=False in current use)
        if tap_blocked:
            unpack_in2 = bd_act_unpack_tensor_hwcb(bd_in, in_c, ih, iw)
        else:
            unpack_in2 = np.zeros(in_elems, dtype=np.int8)
            bd_act_unpack_tensor(bd_in, n_in_blks, unpack_in2, in_elems)
        wp    = _weight_blocks(blob, hdr, entries, tid_proj)
        bp    = _bias_f32(blob, hdr, entries, tid_proj)
        bd_skip = conv1x1_bd4(unpack_in2, wp, bp, in_c, out_c, ih, iw, stride, shift_proj, do_relu=False)
        bd_out  = add_relu_bd4(bd_out, bd_skip, out_elems)
    elif stride == 2:
        # Option A: zero-pad + stride-2 skip (no learnable weights)
        if tap_blocked:
            bd_skip = _zero_pad_stride2_hwcb(bd_in, in_c, out_c, ih, iw)
            bd_out  = add_relu_bd4_hwcb(bd_out, bd_skip, out_c, oh, ow)
        else:
            bd_skip = _zero_pad_stride2_bd4(bd_in, in_c, out_c, ih, iw)
            bd_out  = add_relu_bd4(bd_out, bd_skip, out_elems)
    else:
        # Identity skip: same spatial + channel dims
        if tap_blocked:
            bd_out = add_relu_bd4_hwcb(bd_out, bd_in, out_c, oh, ow)
        else:
            bd_out = add_relu_bd4(bd_out, bd_in, out_elems)

    return bd_out


# =============================================================================
# Layer topology  (mirrors resnet1202_layers.h)
# =============================================================================

RN1202_N_PER_STAGE = 200
RN1202_N_CLASSES   = 10

# Channel counts per stage (index 0 unused; index 1/2/3 = stage 1/2/3)
_CHANS = [0, 16, 32, 64]

# Input spatial dims at the *entry to block 0* of each stage.
# Stage 1: 32×32 (from conv1 stem, no downsampling)
# Stage 2: 32×32 (from stage1 output)
# Stage 3: 16×16 (from stage2 output, which halved due to block0 stride=2)
_STAGE_ENTRY_H = [0, 32, 32, 16]
_STAGE_ENTRY_W = [0, 32, 32, 16]

# TID_STAGE2_FIRST depends on whether stage2 block0 has a projection tensor.
# Both Option A and Option B use TID_STAGE2_FIRST = 401 (fixed by layout).
_TID_CONV1        = 0
_TID_STAGE1_FIRST = 1
_TID_STAGE2_FIRST = 401   # = _TID_STAGE1_FIRST + 200*2 = 401


def _tid_stage3_first(has_proj: bool) -> int:
    """First tensor ID for stage 3.

    Mirrors RN1202_TID_STAGE3_FIRST in resnet1202_layers.h:
      = TID_STAGE2_FIRST + B0_TENSORS + (200-1)*2
      where B0_TENSORS = 2 (Option A) or 3 (Option B)
    """
    b0 = 3 if has_proj else 2
    return _TID_STAGE2_FIRST + b0 + 398


def _tid_fc(has_proj: bool) -> int:
    """Tensor ID for the FC classifier.

    Mirrors RN1202_TID_FC in resnet1202_layers.h.
    """
    b0   = 3 if has_proj else 2
    s3f  = _tid_stage3_first(has_proj)
    return s3f + b0 + 398


def _block_tids(stage: int, block_idx: int, has_proj: bool) -> tuple[int, int, int]:
    """Return (tid_conv_a, tid_conv_b, tid_proj) for a given block.

    Mirrors the switch(stage) inside rn1202_block_conf() in resnet1202_layers.h.
    tid_proj = 0xFFFF when has_proj is False or block is not a projection block.
    """
    b0 = 3 if has_proj else 2   # tensors used by block 0 of stage 2 and 3

    if stage == 1:
        base     = _TID_STAGE1_FIRST + block_idx * 2
        return base, base + 1, 0xFFFF

    elif stage == 2:
        if block_idx == 0:
            base    = _TID_STAGE2_FIRST
            tid_p   = base + 2 if has_proj else 0xFFFF
            return base, base + 1, tid_p
        else:
            base = _TID_STAGE2_FIRST + b0 + (block_idx - 1) * 2
            return base, base + 1, 0xFFFF

    else:  # stage == 3
        s3f = _tid_stage3_first(has_proj)
        if block_idx == 0:
            tid_p = s3f + 2 if has_proj else 0xFFFF
            return s3f, s3f + 1, tid_p
        else:
            base = s3f + b0 + (block_idx - 1) * 2
            return base, base + 1, 0xFFFF


# =============================================================================
# Full ResNet-1202 inference
# =============================================================================

def _bd4_cksum(bd_bytes: bytes) -> int:
    """Raw uint32 byte-sum of a BD4 buffer.  Matches firmware print_bd4_cksum()."""
    return int(sum(bd_bytes)) & 0xFFFFFFFF


def _is_tap_blocked(blob: bytes, hdr: dict, entries: list[dict]) -> bool:
    """Read the rn1202.layout_flags sentinel from the blob.

    Mirrors gen_resnet1202_model.py: the last entry is a float32 sentinel
    set to 1.0 for tap-blocked [OC,KY,KX,IC] layout, 0.0 for flat
    [OC,IC,KY,KX] layout.  Returns False if the sentinel is absent.
    """
    # Sentinel is always the last tensor entry (odd index when biases follow weights)
    last_e = entries[-1]
    if last_e.get('dtype', DTYPE_FLOAT32) != DTYPE_FLOAT32:
        return False
    start = hdr['data_offset'] + last_e['tensor_offset']
    end   = start + last_e['tensor_bytes']
    val   = np.frombuffer(blob[start:end], dtype=np.float32)
    return bool(val.size > 0 and float(val[0]) == 1.0)


def run_rn1202(
    blob: bytes,
    input_i8: np.ndarray,   # int8 [3, 32, 32] CIFAR-10 image (CHW)
    verbose: bool  = True,
) -> dict:
    """ResNet-1202 CIFAR-10 BD4-faithful integer inference.

    Mirrors the main inference loop in resnet1202_phase3_hw_decode/main.c.
    Uses calibrated per-layer shifts (CONV_SHIFT_STEM / CONV_SHIFT_STAGE{1,2,3}).
    Weight layout (tap-blocked vs flat [OC,IC,KY,KX]) is auto-detected from
    the blob's rn1202.layout_flags sentinel tensor.
    Option A (zero-pad stride-2 skip) is always used; Option B is not supported.

    Dataflow:
      Stem    : int8 [3,32,32]  → conv3x3_bd4(shift=CONV_SHIFT_STEM) → BD4 [16,32,32]
      Stage 1 : 200 BasicBlocks (16ch, 32×32, stride=1, identity skip, shift=CONV_SHIFT_STAGE1)
      Stage 2 : 200 BasicBlocks (16→32ch first; 32ch×16×16 rest; Option A skip for block0, shift=CONV_SHIFT_STAGE2)
      Stage 3 : 200 BasicBlocks (32→64ch first; 64ch×8×8 rest; Option A skip for block0, shift=CONV_SHIFT_STAGE3)
      Avgpool : BD4 [64,8,8] → int8[64]
      FC      : int8[64] → int32 logits[10]

    Returns dict with keys: 'logits' (int64[10]), 'top1' (int), 'hashes' (dict).
    """
    hdr, entries = _vwb2_parse(blob)

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    tap_blocked = _is_tap_blocked(blob, hdr, entries)
    log(f"Weight layout: {'tap-blocked [OC,KY,KX,IC]' if tap_blocked else 'flat [OC,IC,KY,KX]'}")

    # Option A (zero-pad skip) is the only supported skip; has_proj always False
    has_proj = False

    hashes: dict = {}
    tid_fc = _tid_fc(has_proj)

    # ── Stem: int8 [3,32,32] → conv3×3 (pad=1, stride=1) → BD4 [16,32,32] ──
    log("stem  conv3x3 int8→BD4 ...")
    wa  = _weight_blocks(blob, hdr, entries, _TID_CONV1)
    ba  = _bias_f32(blob, hdr, entries, _TID_CONV1)
    if tap_blocked:
        wa_base = _weight_block_raw_base(blob, hdr, entries, _TID_CONV1)
        bd_cur  = conv3x3_bd4_tap(
            input_i8.flatten().astype(np.int8), blob, wa_base, ba,
            in_c=3, out_c=16, h=32, w=32,
            stride=1, out_shift=CONV_SHIFT_STEM, do_relu=True,
        )
    else:
        bd_cur = conv3x3_bd4(
            input_i8.flatten().astype(np.int8), wa, ba,
            in_c=3, out_c=16, h=32, w=32,
            stride=1, out_shift=CONV_SHIFT_STEM, do_relu=True,
            tap_blocked=False,
        )
    hashes['conv1'] = _bd4_cksum(bd_cur)
    log(f"  conv1 BD4 cksum=0x{hashes['conv1']:08X}")

    # ── Stages 1–3 ──────────────────────────────────────────────────────────
    # Each stage has 200 BasicBlocks.
    # Spatial dims are tracked here because rn1202_block_conf's IN_H/IN_W only
    # covers block 0's input; subsequent blocks inherit the post-stride dims.
    # Per-stage calibrated shifts mirror CONV_SHIFT_STAGE{1,2,3} in main.c
    _STAGE_SHIFTS = {1: CONV_SHIFT_STAGE1, 2: CONV_SHIFT_STAGE2, 3: CONV_SHIFT_STAGE3}

    STAGE_INFO = [
        (1, 16, 16, 32, 32),   # (stage, first_in_c, out_c, entry_h, entry_w)
        (2, 16, 32, 32, 32),
        (3, 32, 64, 16, 16),
    ]

    for stage_idx, first_in_c, stage_out_c, entry_h, entry_w in STAGE_INFO:
        stage_shift = _STAGE_SHIFTS[stage_idx]
        log(f"stage {stage_idx}: {RN1202_N_PER_STAGE} blocks  "
            f"(in_c={first_in_c}→{stage_out_c}, entry_spatial={entry_h}×{entry_w}, "
            f"shift={stage_shift}) ...")
        cur_in_c     = first_in_c
        cur_h, cur_w = entry_h, entry_w

        for blk_idx in range(RN1202_N_PER_STAGE):
            is_ds   = (stage_idx >= 2 and blk_idx == 0)
            stride  = 2 if is_ds else 1
            in_c    = cur_in_c

            tid_a, tid_b, tid_p = _block_tids(stage_idx, blk_idx, has_proj)
            block_proj = has_proj and is_ds

            bd_cur = run_basic_block_bd4(
                in_c        = in_c,
                out_c       = stage_out_c,
                ih          = cur_h,
                iw          = cur_w,
                stride      = stride,
                has_proj    = block_proj,
                tid_conv_a  = tid_a,
                tid_conv_b  = tid_b,
                tid_proj    = tid_p,
                shift_a     = stage_shift,
                shift_b     = stage_shift,
                shift_proj  = stage_shift,
                blob        = blob,
                hdr         = hdr,
                entries     = entries,
                bd_in       = bd_cur,
                tap_blocked = tap_blocked,
            )
            cur_in_c     = stage_out_c
            cur_h       //= stride
            cur_w       //= stride

        cksum = _bd4_cksum(bd_cur)
        hashes[f'stage{stage_idx}'] = cksum
        log(f"  stage{stage_idx} done  BD4 cksum=0x{cksum:08X}")

    # ── Global avgpool: BD4 [64,8,8] → int8[64] ─────────────────────────
    log("avgpool BD4[64,8,8] → int8[64] ...")
    if tap_blocked:
        avgpool_out = global_avgpool_hwcb(bd_cur, c=64, h=8, w=8)
    else:
        avgpool_out = global_avgpool_bd4(bd_cur, c=64, h=8, w=8)

    # ── FC: int8[64] → int32 logits[10] ──────────────────────────────────
    log("fc int8[64] → logits[10] ...")
    wfc = _weight_blocks(blob, hdr, entries, tid_fc)
    bfc = _bias_f32(blob, hdr, entries, tid_fc)
    logits = fc_linear(avgpool_out, wfc, bfc, in_c=64, out_c=RN1202_N_CLASSES)

    top1 = int(np.argmax(logits))
    CIFAR10 = ["airplane", "automobile", "bird", "cat", "deer",
               "dog",      "frog",       "horse","ship","truck"]
    log(f"  Top-1: class {top1} ({CIFAR10[top1]})  logit={int(logits[top1])}")

    return {'logits': logits, 'top1': top1, 'hashes': hashes}


# =============================================================================
# Output: C header with BD4 checksums
# =============================================================================

def export_ref_header(result: dict, path: str) -> None:
    """Write C header with per-stage BD4 byte-sum checksums.

    Produces #defines that match the firmware's print_bd4_cksum() output,
    directly usable in resnet1202.mk / regression checks.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    with open(path, 'w') as f:
        f.write("// AUTO-GENERATED by scripts/rn1202_ref.py — do not edit\n")
        f.write("// ResNet-1202 CIFAR-10 BD4-faithful reference\n")
        f.write("// Checksums = raw uint8 byte-sum of BD4 buffers,\n")
        f.write("// matching print_bd4_cksum() in resnet1202_conv.h.\n\n")
        f.write("#ifndef RN1202_QUANTIZED_REF_H\n")
        f.write("#define RN1202_QUANTIZED_REF_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define RN1202_REF_TOP1  {result['top1']}\n\n")
        f.write("// Per-stage BD4 buffer byte-sums\n")
        for name, h in result['hashes'].items():
            cname = name.upper()
            f.write(f"#define RN1202_BD4_CKSUM_{cname}  0x{h:08X}u\n")
        f.write("\n#endif /* RN1202_QUANTIZED_REF_H */\n")
    print(f"Saved {path}")


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    SCRIPT_DIR = Path(__file__).resolve().parent
    ROOT       = SCRIPT_DIR.parent

    def rel(p: str) -> str:
        return p if os.path.isabs(p) else str(ROOT / p)

    parser = argparse.ArgumentParser(
        description='ResNet-1202 CIFAR-10 BD4-faithful integer inference reference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Mirrors resnet1202_phase3_hw_decode/main.c exactly.
All activations stored as BD4; no int8 clamp at layer outputs.
Residual add via add_relu_bd4 (unpack, add as int32, ReLU, repack).
Calibrated shifts: stem={CONV_SHIFT_STEM}, stage1={CONV_SHIFT_STAGE1},
  stage2={CONV_SHIFT_STAGE2}, stage3={CONV_SHIFT_STAGE3}.
Weight layout (tap-blocked vs flat) is auto-detected from the blob sentinel.
        """.format(
            CONV_SHIFT_STEM=CONV_SHIFT_STEM,
            CONV_SHIFT_STAGE1=CONV_SHIFT_STAGE1,
            CONV_SHIFT_STAGE2=CONV_SHIFT_STAGE2,
            CONV_SHIFT_STAGE3=CONV_SHIFT_STAGE3,
        ))
    parser.add_argument('--blob',
        default=None,
        help='VWB2 weight blob (default: scripts/resnet1202_artifacts/weights_bd.bin)')
    parser.add_argument('--input',
        default=None,
        help='int8 CHW 3×32×32 input (default: scripts/resnet1202_artifacts/input_32x32.raw)')
    parser.add_argument('--output',
        default=None,
        help='Output C header (default: scripts/resnet1202_artifacts/quantized_ref.h)')
    parser.add_argument('--has-proj', action='store_true',
        help='Use Option B learnable 1×1 projection shortcuts (default: Option A zero-pad) <- WE USE THIS IN C CODE')
    args = parser.parse_args()

    blob_path = rel(args.blob   or 'scripts/resnet1202_artifacts/weights_bd.bin')
    inp_path  = rel(args.input  or 'scripts/resnet1202_artifacts/input_32x32.raw')
    out_path  = rel(args.output or 'scripts/resnet1202_artifacts/quantized_ref.h')

    # Load blob
    if not os.path.exists(blob_path):
        sys.exit(f"Blob not found: {blob_path}")
    with open(blob_path, 'rb') as fh:
        blob = fh.read()
    hdr, entries = _vwb2_parse(blob)
    print(f"VWB2 blob: {hdr['tensor_count']} tensors  ({len(blob)} bytes)")

    # Load input
    if not os.path.exists(inp_path):
        sys.exit(f"Input not found: {inp_path}  — run gen_resnet1202_model.py first")
    input_i8 = np.fromfile(inp_path, dtype=np.int8).reshape(3, 32, 32)
    print(f"Input: {inp_path}  (3×32×32 int8)")

    sep = '=' * 64
    print(f"\n{sep}")
    print(f"  ResNet-1202  |  BD4-faithful  |  "
          f"stem_shift={CONV_SHIFT_STEM}  stage_shifts={CONV_SHIFT_STAGE1}/{CONV_SHIFT_STAGE2}/{CONV_SHIFT_STAGE3}  |  "
          f"Option A (zero-pad)")
    print(sep)

    result = run_rn1202(
        blob, input_i8,
        verbose = True,
    )

    CIFAR10 = ["airplane", "automobile", "bird", "cat", "deer",
               "dog",      "frog",       "horse","ship","truck"]
    top1 = result['top1']
    print(f"\nTop-1: class {top1} ({CIFAR10[top1]})  logit={int(result['logits'][top1])}")

    export_ref_header(result, out_path)
    print("Done.")


if __name__ == '__main__':
    main()
