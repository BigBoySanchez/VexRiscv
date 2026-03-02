#!/usr/bin/env python3
"""
BlockDialect Codec (DialectFP4, weights-focused)

This module implements an *offline* (weights) BlockDialect-style encoder/decoder:

- 16-dialect DialectFP4 formatbook (Figure 4 in arXiv:2501.01144v5)
- Block size: 32 elements (1D blocks)
- Per-block metadata:
    - dialect_id: 4 bits
    - shared_exp:  5 bits  (FP16 exponent bits, stored as 0..31)
- Per-element data:
    - 4-bit code = 1-bit sign + 3-bit index into the selected dialect's 8 magnitudes

Important correction vs the previous version:
    BlockDialect is NOT “INT8 quantization + a novel encoding”.
    It is a *block-scaled FP4-like* representation (4-bit indices + per-block exponent + per-block dialect).

This codec targets the paper’s hardware-friendly representation:
    DialectFP4 magnitudes are multiples of 0.5 in [0, 7.5].
    We store magnitudes in “half-units” as integers 0..15 such that:
        real_magnitude = 0.5 * half_units

For weights (offline), we select the per-block dialect using MSE across all 16 dialects.

Binary format per block (18 bytes):
  [0:2]   metadata (big-endian uint16):
            bits 15..12 : dialect_id (0..15)
            bits 11..7  : shared_exp (0..31)  (FP16 exponent bits)
            bits  6..0  : 0 (padding)
  [2:18]  packed_codes: 32 × 4 bits = 16 bytes
            byte i = (code[2i] << 4) | code[2i+1]

Decoding formula:
  Let shared_exp_bits be the stored 5-bit value (0..31).
  Let e = shared_exp_bits - 15  (FP16 exponent bias is 15).
  Let half_units = dialect_table[idx] (0..15).
  Then:
      value = sign * (0.5 * half_units) * 2^e

Notes:
- The paper’s activation quantization uses a 2-stage dialect selector and a 5-bit intermediate
  for efficient on-the-fly selection/rounding. This module implements the simpler offline
  weight path (exact MSE over dialects), which the paper explicitly allows for weights.

Reference: 2501.01144v5 (BlockDialect), Figure 4 and Section 3.2/3.3.
"""

from __future__ import annotations

import math
import struct
from typing import List, Tuple

import numpy as np

# =============================================================================
# DialectFP4 formatbook (Figure 4) — stored in 0.5 “half-units”
# =============================================================================

BLOCK_SIZE = 32
FP16_EXP_BIAS = 15

# Each dialect has 8 unsigned magnitudes, in units of 0.5.
# So real magnitude = 0.5 * HALF_UNITS[d][idx]
#
# Figure 4 lists values descending; here we store ascending half-units.
# Example: Dialect 0 magnitudes are [0, 0.5, 1, 1.5, 2, 3, 5.5, 7.5]
#          => half-units [0, 1, 2, 3, 4, 6, 11, 15]
DIALECTS_HALF_UNITS: List[List[int]] = [
    [0, 1, 2, 3, 4, 6, 11, 15],  # 0:  7.5, 5.5, 3, 2, 1.5, 1, 0.5, 0
    [0, 1, 2, 3, 4, 6,  9, 15],  # 1:  7.5, 4.5, 3, 2, 1.5, 1, 0.5, 0
    [0, 1, 2, 3, 4, 6, 11, 14],  # 2:  7.0, 5.5, ...
    [0, 1, 2, 3, 4, 6,  9, 14],  # 3:  7.0, 4.5, ...
    [0, 1, 2, 3, 4, 6, 10, 13],  # 4:  6.5, 5.0, ...
    [0, 1, 2, 3, 4, 6,  8, 13],  # 5:  6.5, 4.0, ...
    [0, 1, 2, 3, 4, 6, 10, 12],  # 6:  6.0, 5.0, ...
    [0, 1, 2, 3, 4, 6,  8, 12],  # 7:  6.0, 4.0, ...
    [0, 1, 2, 3, 4, 6,  9, 11],  # 8:  5.5, 4.5, ...
    [0, 1, 2, 3, 4, 6,  7, 11],  # 9:  5.5, 3.5, ...
    [0, 1, 2, 3, 4, 6,  9, 10],  # 10: 5.0, 4.5, ...
    [0, 1, 2, 3, 4, 6,  7, 10],  # 11: 5.0, 3.5, ...
    [0, 1, 2, 3, 4, 6,  8,  9],  # 12: 4.5, 4.0, ...
    [0, 1, 2, 3, 4, 6,  7,  9],  # 13: 4.5, 3.5, ...
    [0, 1, 2, 3, 4, 6,  7,  8],  # 14: 4.0, 3.5, ...
    [0, 1, 2, 3, 4, 5,  6,  8],  # 15: 4.0, 3.0, 2.5, 2.0, ...
]

_DIALECT_ARRAYS = [np.asarray(d, dtype=np.int32) for d in DIALECTS_HALF_UNITS]


# =============================================================================
# Helpers: FP16 exponent bits and block preprocessing
# =============================================================================

def _fp16_exponent_bits(x: float) -> int:
    """Return the FP16 exponent bits (0..31) of |x|.

    For x == 0, returns 0.
    For subnormals, exponent bits are 0 (as in IEEE-754).
    """
    ax = float(abs(x))
    if ax == 0.0:
        return 0
    h = np.float16(ax)
    bits = np.frombuffer(h.tobytes(), dtype=np.uint16)[0]
    return int((bits >> 10) & 0x1F)


def _compute_shared_exponent_bits(block: np.ndarray) -> int:
    """Compute 5-bit shared exponent (FP16 exponent bits) for a float block.

    Paper guidance (Section 3.2): choose a shared exponent based on the block max,
    adjusted so the expression range [0, 8) comfortably covers FP4’s range [0, 6].

    A practical way to approximate this for weights:
      shared_exp_bits = exp_bits(max_abs) - 2  (clamped to [0, 31])

    This maps the block max into roughly [4, 8) after scaling by 2^(-e),
    leaving headroom for rounding/clipping while preserving precision.
    """
    block = np.asarray(block, dtype=np.float32)
    max_abs = float(np.max(np.abs(block)))
    if max_abs == 0.0:
        return 0
    emax_bits = _fp16_exponent_bits(max_abs)
    shared = emax_bits - 2
    return int(min(31, max(0, shared)))


def _scaled_half_units(block_abs: np.ndarray, shared_exp_bits: int) -> np.ndarray:
    """Convert |x| to scaled magnitudes in half-units (0..15) for quantization.

    We target magnitudes in [0, 7.5] with 0.5 granularity:
        scaled_half = round( |x| / (0.5 * 2^e) )
    where e = shared_exp_bits - FP16_EXP_BIAS.

    Returns int32 array, same shape as block_abs, clamped to [0, 15].
    """
    e = int(shared_exp_bits) - FP16_EXP_BIAS  # unbiased exponent (can be negative)
    # Compute |x| / (0.5 * 2^e) = |x| * 2^(1 - e)
    # Use ldexp for stability: ldexp(a, k) = a * 2^k
    scaled = np.rint(np.ldexp(block_abs.astype(np.float32), 1 - e)).astype(np.int32)
    return np.clip(scaled, 0, 15)


# =============================================================================
# Dialect selection and quantization
# =============================================================================

def _select_dialect_mse(scaled_half: np.ndarray) -> int:
    """Offline (weights) dialect selection via exact MSE over all dialects."""
    best_dialect = 0
    best_mse = float("inf")

    scaled_half = scaled_half.astype(np.int32)
    for d_idx, d_arr in enumerate(_DIALECT_ARRAYS):
        diffs = np.abs(scaled_half[:, None] - d_arr[None, :])  # (N, 8)
        nearest_idx = np.argmin(diffs, axis=1)
        quantized = d_arr[nearest_idx]
        mse = float(np.mean((scaled_half - quantized) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_dialect = d_idx

    return best_dialect


def _nearest_index_in_dialect(half_units: int, dialect_idx: int) -> int:
    """Return 3-bit index (0..7) of the nearest representable magnitude."""
    d = DIALECTS_HALF_UNITS[dialect_idx]
    best_i = 0
    best_dist = abs(half_units - d[0])
    for i in range(1, 8):
        dist = abs(half_units - d[i])
        if dist < best_dist:
            best_dist = dist
            best_i = i
    return best_i


def encode_block(block: np.ndarray) -> Tuple[int, int, np.ndarray]:
    """Encode one float block (length BLOCK_SIZE) into (dialect_id, shared_exp_bits, codes[32])."""
    block = np.asarray(block, dtype=np.float32)
    assert block.size == BLOCK_SIZE

    signs = (block < 0).astype(np.uint8)
    mags = np.abs(block)

    shared_exp_bits = _compute_shared_exponent_bits(block)
    scaled_half = _scaled_half_units(mags, shared_exp_bits)

    dialect_id = _select_dialect_mse(scaled_half)

    codes = np.zeros(BLOCK_SIZE, dtype=np.uint8)
    for i in range(BLOCK_SIZE):
        idx = _nearest_index_in_dialect(int(scaled_half[i]), dialect_id)
        codes[i] = ((signs[i] & 1) << 3) | (idx & 0x7)

    return dialect_id, shared_exp_bits, codes


def decode_block(dialect_id: int, shared_exp_bits: int, codes: np.ndarray) -> np.ndarray:
    """Decode one block to float32 values (length BLOCK_SIZE)."""
    dialect = _DIALECT_ARRAYS[int(dialect_id)]
    e = int(shared_exp_bits) - FP16_EXP_BIAS

    codes = np.asarray(codes, dtype=np.uint8).reshape(-1)
    assert codes.size == BLOCK_SIZE

    out = np.zeros(BLOCK_SIZE, dtype=np.float32)
    for i in range(BLOCK_SIZE):
        code = int(codes[i]) & 0x0F
        sign = (code >> 3) & 1
        idx = code & 0x07
        half_units = float(dialect[idx])  # 0..15
        mag = np.ldexp(0.5 * half_units, e)  # (0.5*half_units) * 2^e
        out[i] = -mag if sign else mag
    return out


# =============================================================================
# Packing / unpacking blocks
# =============================================================================

def pack_block(dialect_id: int, shared_exp_bits: int, codes: np.ndarray) -> bytes:
    """Pack a single block into its binary representation (18 bytes)."""
    meta = ((int(dialect_id) & 0xF) << 12) | ((int(shared_exp_bits) & 0x1F) << 7)
    data = struct.pack(">H", meta)

    codes = np.asarray(codes, dtype=np.uint8).reshape(-1)
    assert codes.size == BLOCK_SIZE

    packed = bytearray(16)
    for i in range(16):
        hi = int(codes[2 * i]) & 0x0F
        lo = int(codes[2 * i + 1]) & 0x0F
        packed[i] = (hi << 4) | lo

    return data + bytes(packed)


def unpack_block(data: bytes) -> Tuple[int, int, np.ndarray]:
    """Unpack 18 bytes into (dialect_id, shared_exp_bits, codes[32])."""
    if len(data) < 18:
        raise ValueError("Need at least 18 bytes to unpack a block")
    meta = struct.unpack(">H", data[0:2])[0]
    dialect_id = (meta >> 12) & 0xF
    shared_exp_bits = (meta >> 7) & 0x1F

    codes = np.zeros(BLOCK_SIZE, dtype=np.uint8)
    for i in range(16):
        b = data[2 + i]
        codes[2 * i] = (b >> 4) & 0x0F
        codes[2 * i + 1] = b & 0x0F

    return int(dialect_id), int(shared_exp_bits), codes


# =============================================================================
# Tensor-level encode/decode (float32)
# =============================================================================

def encode_tensor(tensor: np.ndarray) -> bytes:
    """Encode a float tensor into BlockDialect binary format.

    The tensor is flattened and split into BLOCK_SIZE blocks.
    Last block is zero-padded if needed.

    Returns bytes:
        n_elements(u32 LE) | n_blocks(u32 LE) | block_data...
    """
    flat = np.asarray(tensor, dtype=np.float32).reshape(-1)
    n_elements = int(flat.size)
    n_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    padded_len = n_blocks * BLOCK_SIZE
    if padded_len > n_elements:
        flat = np.concatenate([flat, np.zeros(padded_len - n_elements, dtype=np.float32)])

    blob = bytearray(struct.pack("<II", n_elements, n_blocks))
    for b in range(n_blocks):
        block = flat[b * BLOCK_SIZE:(b + 1) * BLOCK_SIZE]
        d_id, s_exp, codes = encode_block(block)
        blob += pack_block(d_id, s_exp, codes)
    return bytes(blob)


def decode_tensor(data: bytes, offset: int = 0) -> Tuple[np.ndarray, int]:
    """Decode one encoded tensor from data[offset:].

    Returns:
        (tensor_float32_flat, bytes_consumed)
    """
    n_elements, n_blocks = struct.unpack_from("<II", data, offset)
    pos = offset + 8

    out = np.zeros(int(n_blocks) * BLOCK_SIZE, dtype=np.float32)
    for b in range(int(n_blocks)):
        d_id, s_exp, codes = unpack_block(data[pos:pos + 18])
        out[b * BLOCK_SIZE:(b + 1) * BLOCK_SIZE] = decode_block(d_id, s_exp, codes)
        pos += 18

    return out[:int(n_elements)], pos - offset


# =============================================================================
# Weight-blob helpers (simple container; shapes/names are NOT stored)
# =============================================================================

MAGIC_BD = 0x56574231  # 'VWB1'

def write_weight_blob(tensors: List[bytes], output_path: str) -> bytes:
    """Write a list of encoded tensor blobs into a single weight file.

    File format:
        magic(u32 LE) | payload_size(u32 LE) | block_size(u32 LE) | reserved(u32 LE) | payload...

    payload is a concatenation of tensor blobs (from encode_tensor),
    each padded to 4-byte alignment.

    NOTE: tensor names and shapes are not stored. You must know them externally.
    """
    blob = bytearray()
    blob += struct.pack("<I", MAGIC_BD)
    blob += struct.pack("<I", 0)           # payload size placeholder
    blob += struct.pack("<I", BLOCK_SIZE)
    blob += struct.pack("<I", 0)           # reserved

    for t in tensors:
        blob += t
        while len(blob) % 4 != 0:
            blob += b"\x00"

    struct.pack_into("<I", blob, 4, len(blob) - 16)

    with open(output_path, "wb") as f:
        f.write(blob)
    return bytes(blob)


def read_weight_blob(input_path: str) -> List[np.ndarray]:
    """Read a weight blob and decode all tensors (flattened float32 arrays)."""
    with open(input_path, "rb") as f:
        data = f.read()

    magic = struct.unpack_from("<I", data, 0)[0]
    if magic != MAGIC_BD:
        raise ValueError(f"Bad magic: 0x{magic:08X} (expected 0x{MAGIC_BD:08X})")
    payload_size = struct.unpack_from("<I", data, 4)[0]
    block_size = struct.unpack_from("<I", data, 8)[0]
    if block_size != BLOCK_SIZE:
        raise ValueError(f"Unsupported BLOCK_SIZE {block_size}, expected {BLOCK_SIZE}")

    tensors: List[np.ndarray] = []
    pos = 16
    end = 16 + int(payload_size)

    while pos < end:
        tensor, consumed = decode_tensor(data, pos)
        tensors.append(tensor)
        pos += consumed
        while pos < end and (pos % 4) != 0:
            pos += 1

    return tensors


# =============================================================================
# VWB2: indexed weight blob with tensor table (supports random access by name)
# =============================================================================
#
# Binary layout (all integers little-endian):
#
# File header — 32 bytes:
#   [0:4]   magic        u32 = 0x56574232  ('VWB2')
#   [4:8]   version      u32 = 1
#   [8:12]  tensor_count u32
#   [12:16] block_size   u32 = 32
#   [16:20] table_offset u32 = 32   (byte offset of tensor table from file start)
#   [20:24] data_offset  u32        (byte offset of payload from file start)
#   [24:28] data_bytes   u32        (total payload bytes)
#   [28:32] reserved     u32 = 0
#
# Tensor table — tensor_count × 40 bytes each:
#   [0:4]   name_hash    u32 (FNV-1a 32-bit of tensor name UTF-8 string)
#   [4:8]   dtype        u32 (DTYPE_BD4=0 or DTYPE_FLOAT32=1)
#   [8:12]  tensor_offset u32 (byte offset of tensor data from data_offset)
#   [12:16] tensor_bytes  u32 (byte count of this tensor's data)
#   [16:20] n_elements    u32
#   [20:24] shape_ndim    u32 (0..4)
#   [24:40] shape[0..3]   4 × u32 (unused dims are 0)
#
# Payload — tensor data packed back-to-back, 4-byte aligned:
#   DTYPE_BD4    tensors: encode_tensor() output (8-byte hdr + N×18 block bytes)
#   DTYPE_FLOAT32 tensors: raw float32 array (n_elements × 4 bytes)
#
# See VWB2_SPEC.md for the full specification and firmware integration guide.

MAGIC_BD_V2      = 0x56574232  # 'VWB2'
VWB2_VERSION     = 1
VWB2_HDR_SIZE    = 32          # bytes
VWB2_ENTRY_SIZE  = 40          # bytes per tensor table entry
DTYPE_BD4        = 0           # BlockDialect FP4 encoded
DTYPE_FLOAT32    = 1           # raw float32


def fnv1a32(s: str) -> int:
    """FNV-1a 32-bit hash of a UTF-8 string.  Matches the C implementation in weight_blob.h."""
    h = 0x811C9DC5
    for byte in s.encode("utf-8"):
        h ^= byte
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


class TensorEntry:
    """Metadata for one tensor in a VWB2 blob."""
    __slots__ = ("name", "name_hash", "dtype", "tensor_offset", "tensor_bytes",
                 "n_elements", "shape")

    def __init__(self, name: str, dtype: int, tensor_offset: int,
                 tensor_bytes: int, n_elements: int, shape: Tuple[int, ...]) -> None:
        self.name          = name
        self.name_hash     = fnv1a32(name)
        self.dtype         = dtype
        self.tensor_offset = tensor_offset
        self.tensor_bytes  = tensor_bytes
        self.n_elements    = n_elements
        self.shape         = shape

    def __repr__(self) -> str:
        dtype_str = "BD4" if self.dtype == DTYPE_BD4 else "FLOAT32"
        return (f"TensorEntry({self.name!r}, dtype={dtype_str}, "
                f"shape={self.shape}, offset={self.tensor_offset:#x}, "
                f"bytes={self.tensor_bytes})")


def _pack_entry(entry: TensorEntry) -> bytes:
    """Pack a TensorEntry into its 40-byte binary representation."""
    shape4 = (list(entry.shape) + [0, 0, 0, 0])[:4]
    ndim = min(len(entry.shape), 4)
    raw = struct.pack("<IIIIII",
        entry.name_hash,
        entry.dtype,
        entry.tensor_offset,
        entry.tensor_bytes,
        entry.n_elements,
        ndim,
    ) + struct.pack("<IIII", *shape4)
    assert len(raw) == VWB2_ENTRY_SIZE, f"entry size {len(raw)} != {VWB2_ENTRY_SIZE}"
    return raw


def _unpack_entry(data: bytes, offset: int = 0) -> TensorEntry:
    """Parse a 40-byte entry from data[offset:]."""
    (name_hash, dtype, tensor_offset, tensor_bytes,
     n_elements, ndim) = struct.unpack_from("<IIIIII", data, offset)
    shape4 = struct.unpack_from("<IIII", data, offset + 24)
    shape = tuple(shape4[:ndim])
    entry = TensorEntry.__new__(TensorEntry)
    entry.name          = ""          # names not stored; caller resolves via hash
    entry.name_hash     = name_hash
    entry.dtype         = dtype
    entry.tensor_offset = tensor_offset
    entry.tensor_bytes  = tensor_bytes
    entry.n_elements    = n_elements
    entry.shape         = shape
    return entry


def write_weight_blob_v2(
    tensors: List[Tuple[str, int, np.ndarray]],
    output_path: str,
) -> bytes:
    """Write a VWB2 indexed weight blob.

    Args:
        tensors: list of (name, dtype, array) tuples.
                   name  : human-readable tensor name (e.g. "conv1.weight")
                   dtype : DTYPE_BD4 or DTYPE_FLOAT32
                   array : float32 ndarray (shape preserved in table)
        output_path: destination file path.

    Returns: the complete file contents as bytes.

    For DTYPE_BD4   tensors: the array is encoded via encode_tensor().
    For DTYPE_FLOAT32 tensors: the array is stored as raw float32.
    """
    # ---- encode each tensor into its payload bytes ----
    entries:  List[TensorEntry] = []
    payloads: List[bytes]       = []
    cursor = 0

    for name, dtype, arr in tensors:
        arr = np.asarray(arr, dtype=np.float32)
        shape = tuple(arr.shape)
        n_elements = int(arr.size)

        if dtype == DTYPE_BD4:
            payload = encode_tensor(arr.reshape(-1))
        elif dtype == DTYPE_FLOAT32:
            payload = arr.reshape(-1).astype(np.float32).tobytes()
        else:
            raise ValueError(f"Unknown dtype {dtype} for tensor {name!r}")

        # 4-byte align
        while len(payload) % 4 != 0:
            payload += b"\x00"

        entries.append(TensorEntry(name, dtype, cursor, len(payload), n_elements, shape))
        payloads.append(payload)
        cursor += len(payload)

    # ---- compute offsets ----
    n_tensors    = len(entries)
    table_offset = VWB2_HDR_SIZE
    # align data start to 16-byte boundary for firmware convenience
    raw_data_start = table_offset + n_tensors * VWB2_ENTRY_SIZE
    data_offset = (raw_data_start + 15) & ~15
    padding_before_data = data_offset - raw_data_start
    data_bytes = sum(len(p) for p in payloads)

    # ---- build file ----
    blob = bytearray()

    # header
    blob += struct.pack("<IIIIIIII",
        MAGIC_BD_V2,
        VWB2_VERSION,
        n_tensors,
        BLOCK_SIZE,
        table_offset,
        data_offset,
        data_bytes,
        0,  # reserved
    )
    assert len(blob) == VWB2_HDR_SIZE

    # tensor table
    for entry in entries:
        blob += _pack_entry(entry)

    # alignment padding before data
    blob += b"\x00" * padding_before_data
    assert len(blob) == data_offset

    # payload
    for p in payloads:
        blob += p

    assert len(blob) == data_offset + data_bytes

    with open(output_path, "wb") as f:
        f.write(blob)

    return bytes(blob)


def read_weight_blob_v2(input_path: str) -> Tuple[List[TensorEntry], bytes]:
    """Read a VWB2 file and return (entries, raw_data_bytes).

    The caller can decode individual tensors with decode_tensor_v2().
    """
    with open(input_path, "rb") as f:
        data = f.read()

    magic = struct.unpack_from("<I", data, 0)[0]
    if magic != MAGIC_BD_V2:
        raise ValueError(f"Bad magic 0x{magic:08X} (expected VWB2=0x{MAGIC_BD_V2:08X})")
    version = struct.unpack_from("<I", data, 4)[0]
    if version != VWB2_VERSION:
        raise ValueError(f"Unsupported VWB2 version {version}")
    n_tensors, block_size, table_offset, data_offset, data_bytes = (
        struct.unpack_from("<IIIII", data, 8))
    if block_size != BLOCK_SIZE:
        raise ValueError(f"BLOCK_SIZE mismatch: file has {block_size}, codec expects {BLOCK_SIZE}")

    entries: List[TensorEntry] = []
    pos = table_offset
    for _ in range(n_tensors):
        entries.append(_unpack_entry(data, pos))
        pos += VWB2_ENTRY_SIZE

    raw_payload = data[data_offset:data_offset + data_bytes]
    return entries, raw_payload


def decode_tensor_v2(entry: TensorEntry, raw_payload: bytes) -> np.ndarray:
    """Decode one tensor from a VWB2 payload slice.

    Args:
        entry      : TensorEntry from read_weight_blob_v2()
        raw_payload: the full raw_payload bytes returned by read_weight_blob_v2()

    Returns float32 ndarray with the original shape (or 1-D if shape unknown).
    """
    chunk = raw_payload[entry.tensor_offset:entry.tensor_offset + entry.tensor_bytes]

    if entry.dtype == DTYPE_BD4:
        flat, _ = decode_tensor(chunk, offset=0)
        flat = flat[:entry.n_elements]
    elif entry.dtype == DTYPE_FLOAT32:
        flat = np.frombuffer(chunk, dtype=np.float32)[:entry.n_elements]
    else:
        raise ValueError(f"Unknown dtype {entry.dtype}")

    if entry.shape:
        return flat.reshape(entry.shape)
    return flat


def lookup_tensor_v2(entries: List[TensorEntry], name: str) -> TensorEntry:
    """Find a TensorEntry by tensor name (via FNV-1a hash).

    Raises KeyError if not found.
    """
    h = fnv1a32(name)
    for e in entries:
        if e.name_hash == h:
            e.name = name  # resolve name lazily
            return e
    raise KeyError(f"Tensor {name!r} (hash=0x{h:08x}) not found in blob")
