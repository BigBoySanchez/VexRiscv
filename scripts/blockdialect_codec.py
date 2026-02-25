#!/usr/bin/env python3
"""
BlockDialect-Lite Codec — Encoder and Decoder for DialectFP4 weight quantization.

Implements a practical subset of BlockDialect (arXiv 2501.01144v5):
  - 16-dialect DialectFP4 formatbook
  - Per-block (32 elements) optimal dialect selection
  - 4-bit codes (1-bit sign + 3-bit index) per element
  - Per-block metadata: dialect_id (4 bits) + shared_exponent (5 bits)
  - Encoding: int8 → 4-bit DialectFP4
  - Decoding: 4-bit DialectFP4 → int8

Binary format per block (18 bytes):
  [0:2]   metadata: dialect_id (bits 15..12) | shared_exp (bits 11..7) | padding (bits 6..0)
  [2:18]  packed_codes: 32 × 4 bits = 16 bytes (each byte = high_nibble:elem[2i] | low_nibble:elem[2i+1])

The 0.5-granularity representable magnitudes are stored as integers 0..15.
Multiplication: real_value = sign * 0.5 * integer_value * 2^shared_exp
"""

import numpy as np
import struct
import math

# ============================================================================
# DialectFP4 Formatbook — 16 dialects
# ============================================================================
# Each dialect defines 8 unsigned magnitude levels (index 0..7).
# Values are in 0.5 granularity: the real magnitude = 0.5 * table[index].
# From the paper Figure 4, the 16 dialects come in 8 pairs.
# Each pair shares the same maximum and the 6 smallest values, differing
# in one "large magnitude" slot.
#
# Base FP4 E2M1 magnitudes (as 0.5-scaled integers): [0, 1, 2, 3, 4, 6, 8, 12]
# We construct 16 dialects that cover all max magnitudes from 4 to 15
# in pairs, following the paper's design principles.

BLOCK_SIZE = 32

# fmt: off
# Dialect table: DIALECTS[d][i] gives the unsigned integer magnitude for
# dialect d, index i.  Real value = 0.5 * DIALECTS[d][i].
# Sorted ascending; index 7 is always the maximum.
DIALECTS = [
    # Pair 0 (max = 4): compact range
    [0, 1, 2, 3, 4, 4, 4, 4],   # D0: max 4, duplicate large values
    [0, 1, 2, 3, 3, 3, 4, 4],   # D1: max 4, different mid
    # Pair 1 (max = 5)
    [0, 1, 2, 3, 4, 5, 5, 5],   # D2
    [0, 1, 2, 3, 3, 4, 5, 5],   # D3
    # Pair 2 (max = 6) — closest to base FP4 E2M1
    [0, 1, 2, 3, 4, 5, 6, 6],   # D4
    [0, 1, 2, 3, 4, 4, 6, 6],   # D5
    # Pair 3 (max = 7)
    [0, 1, 2, 3, 4, 5, 6, 7],   # D6
    [0, 1, 2, 3, 4, 5, 7, 7],   # D7
    # Pair 4 (max = 8) — matches FP4 E2M1 range
    [0, 1, 2, 3, 4, 6, 7, 8],   # D8
    [0, 1, 2, 3, 4, 6, 8, 8],   # D9
    # Pair 5 (max = 10)
    [0, 1, 2, 3, 4, 6, 8, 10],  # D10
    [0, 1, 2, 3, 4, 6, 10, 10], # D11
    # Pair 6 (max = 12) — standard FP4
    [0, 1, 2, 3, 4, 6, 10, 12], # D12
    [0, 1, 2, 3, 4, 6, 12, 12], # D13
    # Pair 7 (max = 15)
    [0, 1, 2, 3, 4, 6, 12, 15], # D14
    [0, 1, 2, 3, 4, 6, 13, 15], # D15
]
# fmt: on

# Precompute numpy arrays for vectorized operations
_DIALECT_ARRAYS = [np.array(d, dtype=np.int32) for d in DIALECTS]


def _compute_shared_exponent(block: np.ndarray) -> int:
    """Compute the shared exponent for a block of int8 values.

    The shared exponent shifts the block's max magnitude into the DialectFP4
    representable range [0, 15] (i.e., real magnitudes [0, 7.5]).

    Returns non-negative integer exponent e such that:
        max(|block|) / 2^e  is in [0, 15]  (as 0.5-scaled integers)
    """
    max_mag = int(np.max(np.abs(block.astype(np.int32))))
    if max_mag == 0:
        return 0
    # We want max_mag / 2^e <= 15, so e >= log2(max_mag / 15)
    # But we also want to preserve as much precision as possible,
    # so we use the smallest e that makes it fit.
    # max_mag_in_half_units = max_mag * 2 (since our table uses 0.5 granularity)
    # Actually: real_value = sign * 0.5 * table_val * 2^e
    # So |value| = 0.5 * table_val * 2^e
    # We want 0.5 * 15 * 2^e >= max_mag
    # => 2^e >= max_mag / 7.5
    # => e >= ceil(log2(max_mag / 7.5))
    if max_mag <= 7:
        return 0  # Fits without scaling (max_dialect_val=15 → real 7.5)
    e = math.ceil(math.log2(max_mag / 7.5))
    return max(0, e)


def _select_dialect(scaled_magnitudes: np.ndarray) -> int:
    """Select the optimal dialect for a block of scaled magnitudes (int, 0..15 range).

    Uses MSE-based selection (practical for offline weight quantization).
    For each dialect, quantize all values to nearest representable and compute MSE.
    """
    best_dialect = 0
    best_mse = float('inf')

    for d_idx, d_arr in enumerate(_DIALECT_ARRAYS):
        # For each value, find nearest representable in this dialect
        # d_arr is sorted ascending, shape (8,)
        # scaled_magnitudes shape (BLOCK_SIZE,)
        # Compute distance to each representable value
        diffs = np.abs(scaled_magnitudes[:, None] - d_arr[None, :])  # (N, 8)
        nearest_idx = np.argmin(diffs, axis=1)
        quantized = d_arr[nearest_idx]
        mse = np.mean((scaled_magnitudes - quantized) ** 2)
        if mse < best_mse:
            best_mse = mse
            best_dialect = d_idx

    return best_dialect


def _quantize_to_dialect(magnitude: int, dialect_idx: int) -> int:
    """Quantize a single unsigned magnitude to the nearest value in the dialect.

    Returns the 3-bit index (0..7) into the dialect's representable values.
    """
    d = DIALECTS[dialect_idx]
    best_idx = 0
    best_dist = abs(magnitude - d[0])
    for i in range(1, 8):
        dist = abs(magnitude - d[i])
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx


def encode_block(block: np.ndarray) -> tuple:
    """Encode a block of int8 values into BlockDialect-Lite format.

    Args:
        block: numpy array of int8 values, length BLOCK_SIZE (padded with 0 if shorter)

    Returns:
        (dialect_id, shared_exp, codes) where codes is array of 4-bit values
        (1-bit sign + 3-bit index), length BLOCK_SIZE.
    """
    assert len(block) == BLOCK_SIZE

    block_i32 = block.astype(np.int32)
    signs = (block_i32 < 0).astype(np.int32)
    magnitudes = np.abs(block_i32)

    # 1. Compute shared exponent
    shared_exp = _compute_shared_exponent(block)

    # 2. Scale magnitudes down by shared exponent
    # real_value = 0.5 * scaled_int * 2^shared_exp
    # => scaled_int = real_value / (0.5 * 2^shared_exp) = real_value * 2 / 2^shared_exp
    # => scaled_int = magnitude * 2 >> shared_exp  (with rounding)
    if shared_exp > 0:
        divisor = (1 << shared_exp)  # 2^e
        # scaled_int = round(magnitude * 2 / divisor)  -- the *2 is because table uses 0.5 units
        scaled = np.round(magnitudes * 2.0 / divisor).astype(np.int32)
    else:
        scaled = (magnitudes * 2).astype(np.int32)

    # Clamp to [0, 15]
    scaled = np.clip(scaled, 0, 15)

    # 3. Select optimal dialect
    dialect_id = _select_dialect(scaled.astype(np.float64))

    # 4. Quantize each element
    codes = np.zeros(BLOCK_SIZE, dtype=np.uint8)
    for i in range(BLOCK_SIZE):
        idx_3bit = _quantize_to_dialect(int(scaled[i]), dialect_id)
        codes[i] = (signs[i] << 3) | idx_3bit

    return dialect_id, shared_exp, codes


def decode_block(dialect_id: int, shared_exp: int, codes: np.ndarray) -> np.ndarray:
    """Decode a BlockDialect-Lite block back to int8 values.

    Args:
        dialect_id: 0..15
        shared_exp: 0..31
        codes: array of uint8, length BLOCK_SIZE (each is 4-bit: sign:1 | index:3)

    Returns:
        numpy array of int8 values, length BLOCK_SIZE
    """
    dialect = DIALECTS[dialect_id]
    result = np.zeros(BLOCK_SIZE, dtype=np.int32)

    for i in range(BLOCK_SIZE):
        code = int(codes[i]) & 0x0F
        sign = (code >> 3) & 1
        idx = code & 0x07
        magnitude_scaled = dialect[idx]  # 0.5-unit integer

        # Real magnitude = 0.5 * magnitude_scaled * 2^shared_exp
        #                = magnitude_scaled * 2^(shared_exp - 1)
        if shared_exp == 0:
            # real_mag = 0.5 * magnitude_scaled → round to nearest int
            real_mag = (magnitude_scaled + 1) // 2  # round 0.5 up
        else:
            real_mag = magnitude_scaled * (1 << (shared_exp - 1))

        # Clamp to int8 range
        real_mag = min(real_mag, 127)

        result[i] = -real_mag if sign else real_mag

    return result.astype(np.int8)


def pack_block(dialect_id: int, shared_exp: int, codes: np.ndarray) -> bytes:
    """Pack a single block into its binary representation (18 bytes).

    Format:
      [0:2]  metadata: dialect_id(4) | shared_exp(5) | padding(7), big-endian uint16
      [2:18] packed codes: 32 × 4 bits = 16 bytes
    """
    # Metadata: bits 15..12 = dialect_id, bits 11..7 = shared_exp, bits 6..0 = 0
    meta = ((dialect_id & 0xF) << 12) | ((shared_exp & 0x1F) << 7)
    data = struct.pack('>H', meta)

    # Pack codes: two 4-bit values per byte (high nibble = even index, low nibble = odd)
    packed = bytearray(16)
    for i in range(16):
        high = codes[2 * i] & 0x0F
        low = codes[2 * i + 1] & 0x0F
        packed[i] = (high << 4) | low

    return data + bytes(packed)


def unpack_block(data: bytes) -> tuple:
    """Unpack 18 bytes into (dialect_id, shared_exp, codes[32]).

    Returns:
        (dialect_id, shared_exp, codes) where codes is np.array of uint8, length 32
    """
    assert len(data) >= 18
    meta = struct.unpack('>H', data[0:2])[0]
    dialect_id = (meta >> 12) & 0xF
    shared_exp = (meta >> 7) & 0x1F

    codes = np.zeros(BLOCK_SIZE, dtype=np.uint8)
    for i in range(16):
        byte_val = data[2 + i]
        codes[2 * i] = (byte_val >> 4) & 0x0F
        codes[2 * i + 1] = byte_val & 0x0F

    return dialect_id, shared_exp, codes


# ============================================================================
# Tensor-level encode/decode
# ============================================================================

def encode_tensor(tensor: np.ndarray) -> bytes:
    """Encode an int8 tensor into BlockDialect-Lite binary format.

    The tensor is flattened and split into blocks of BLOCK_SIZE.
    Last block is zero-padded if needed.

    Returns:
        bytes: tensor_size(4) | num_blocks(4) | block_data...
    """
    flat = tensor.flatten().astype(np.int8)
    n_elements = len(flat)
    n_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Pad to full blocks
    padded_len = n_blocks * BLOCK_SIZE
    if padded_len > n_elements:
        flat = np.concatenate([flat, np.zeros(padded_len - n_elements, dtype=np.int8)])

    blob = struct.pack('<II', n_elements, n_blocks)

    for b in range(n_blocks):
        block = flat[b * BLOCK_SIZE:(b + 1) * BLOCK_SIZE]
        dialect_id, shared_exp, codes = encode_block(block)
        blob += pack_block(dialect_id, shared_exp, codes)

    return blob


def decode_tensor(data: bytes, offset: int = 0) -> tuple:
    """Decode a BlockDialect-Lite tensor from binary data.

    Args:
        data: raw bytes
        offset: start position in data

    Returns:
        (tensor_int8, bytes_consumed)
    """
    n_elements, n_blocks = struct.unpack_from('<II', data, offset)
    pos = offset + 8

    result = np.zeros(n_blocks * BLOCK_SIZE, dtype=np.int8)

    for b in range(n_blocks):
        dialect_id, shared_exp, codes = unpack_block(data[pos:pos + 18])
        result[b * BLOCK_SIZE:(b + 1) * BLOCK_SIZE] = decode_block(dialect_id, shared_exp, codes)
        pos += 18

    return result[:n_elements], pos - offset


# ============================================================================
# Weight blob helpers
# ============================================================================

MAGIC_BD = 0x56574231  # 'VWB1'

def write_weight_blob(tensors: list, output_path: str):
    """Write a list of encoded tensor blobs into a single weight file.

    Args:
        tensors: list of bytes (each from encode_tensor())
        output_path: file path
    """
    blob = bytearray()

    # Header
    blob += struct.pack('<I', MAGIC_BD)
    blob += struct.pack('<I', 0)          # placeholder: payload size
    blob += struct.pack('<I', BLOCK_SIZE)
    blob += struct.pack('<I', 0)          # reserved

    for t_blob in tensors:
        blob += t_blob
        # Align to 4 bytes
        while len(blob) % 4 != 0:
            blob += b'\x00'

    # Fill payload size
    struct.pack_into('<I', blob, 4, len(blob) - 16)

    with open(output_path, 'wb') as f:
        f.write(blob)

    return bytes(blob)


def read_weight_blob(input_path: str) -> list:
    """Read a BlockDialect weight blob and decode all tensors.

    Returns:
        list of np.ndarray (int8)
    """
    with open(input_path, 'rb') as f:
        data = f.read()

    magic = struct.unpack_from('<I', data, 0)[0]
    assert magic == MAGIC_BD, f"Bad magic: 0x{magic:08X} (expected 0x{MAGIC_BD:08X})"
    payload_size = struct.unpack_from('<I', data, 4)[0]
    block_size = struct.unpack_from('<I', data, 8)[0]
    assert block_size == BLOCK_SIZE

    tensors = []
    pos = 16
    end = 16 + payload_size

    while pos < end:
        tensor, consumed = decode_tensor(data, pos)
        tensors.append(tensor)
        pos += consumed
        # Skip alignment padding
        while pos < end and pos % 4 != 0:
            pos += 1

    return tensors
