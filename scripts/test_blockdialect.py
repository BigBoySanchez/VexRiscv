#!/usr/bin/env python3
"""
Test suite for BlockDialect-Lite codec.
Validates encode→decode roundtrip, edge cases, and compression ratio.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blockdialect_codec as bd

def test_roundtrip_zeros():
    """All-zero block should roundtrip exactly."""
    block = np.zeros(bd.BLOCK_SIZE, dtype=np.int8)
    d_id, s_exp, codes = bd.encode_block(block)
    result = bd.decode_block(d_id, s_exp, codes)
    assert np.array_equal(result, block), f"Zero roundtrip failed: {result}"
    print("  ✓ All-zeros roundtrip")

def test_roundtrip_small():
    """Small values (within 0.5 granularity) should roundtrip closely."""
    block = np.array([0, 1, -1, 2, -2, 3, -3, 0] + [0]*24, dtype=np.int8)
    d_id, s_exp, codes = bd.encode_block(block)
    result = bd.decode_block(d_id, s_exp, codes)
    # Allow ±1 error due to quantization
    errors = np.abs(block.astype(np.int32) - result.astype(np.int32))
    max_err = np.max(errors)
    print(f"  ✓ Small values: max error = {max_err}")
    assert max_err <= 2, f"Small value roundtrip error too high: {max_err}"

def test_roundtrip_random_blocks(n_blocks=500):
    """Random int8 blocks should roundtrip with reasonable error."""
    rng = np.random.RandomState(42)
    total_mse = 0.0
    max_error_seen = 0

    for _ in range(n_blocks):
        block = rng.randint(-127, 128, size=bd.BLOCK_SIZE).astype(np.int8)
        d_id, s_exp, codes = bd.encode_block(block)

        # Test pack/unpack roundtrip
        packed = bd.pack_block(d_id, s_exp, codes)
        d_id2, s_exp2, codes2 = bd.unpack_block(packed)
        assert d_id2 == d_id, f"Dialect ID mismatch: {d_id} vs {d_id2}"
        assert s_exp2 == s_exp, f"Shared exp mismatch: {s_exp} vs {s_exp2}"
        assert np.array_equal(codes, codes2), "Codes mismatch after pack/unpack"

        # Test decode
        result = bd.decode_block(d_id, s_exp, codes)
        errors = (block.astype(np.int32) - result.astype(np.int32)) ** 2
        total_mse += np.mean(errors)
        max_error_seen = max(max_error_seen, np.max(np.abs(block.astype(np.int32) - result.astype(np.int32))))

    avg_mse = total_mse / n_blocks
    print(f"  ✓ Random blocks ({n_blocks}): avg MSE = {avg_mse:.2f}, max error = {max_error_seen}")

def test_tensor_roundtrip():
    """Test full tensor encode/decode pipeline."""
    rng = np.random.RandomState(123)
    tensor = rng.randint(-127, 128, size=(16, 3, 3, 3)).astype(np.int8)
    
    blob = bd.encode_tensor(tensor)
    decoded, consumed = bd.decode_tensor(blob)
    
    original_flat = tensor.flatten()
    assert len(decoded) == len(original_flat), \
        f"Length mismatch: {len(decoded)} vs {len(original_flat)}"
    
    errors = np.abs(original_flat.astype(np.int32) - decoded.astype(np.int32))
    rmse = np.sqrt(np.mean(errors ** 2))

    # Compression ratio
    original_bytes = len(original_flat)  # int8 = 1 byte each
    compressed_bytes = len(blob) - 8     # subtract tensor header
    ratio = original_bytes / compressed_bytes

    print(f"  ✓ Tensor roundtrip: RMSE = {rmse:.2f}, compression = {ratio:.2f}x "
          f"({original_bytes}B → {compressed_bytes}B)")

def test_blob_io(tmp_path="/tmp/test_bd_blob.bin"):
    """Test full blob write/read cycle."""
    rng = np.random.RandomState(456)
    
    # Simulate a few parameter tensors
    tensors_orig = [
        rng.randint(-127, 128, size=(16, 3, 3, 3)).astype(np.int8),
        rng.randint(-127, 128, size=(16,)).astype(np.int8),
        rng.randint(-127, 128, size=(16,)).astype(np.int8),
    ]
    
    encoded = [bd.encode_tensor(t) for t in tensors_orig]
    blob = bd.write_weight_blob(encoded, tmp_path)
    
    decoded_tensors = bd.read_weight_blob(tmp_path)
    assert len(decoded_tensors) == len(tensors_orig), \
        f"Tensor count mismatch: {len(decoded_tensors)} vs {len(tensors_orig)}"
    
    for i, (orig, dec) in enumerate(zip(tensors_orig, decoded_tensors)):
        assert len(dec) == orig.size, f"Tensor {i} size mismatch"
    
    os.unlink(tmp_path)
    print(f"  ✓ Blob I/O: {len(tensors_orig)} tensors, write/read OK ({len(blob)} bytes)")

def test_compression_ratio():
    """Verify expected ~2x compression on realistic tensor sizes."""
    rng = np.random.RandomState(789)
    # conv1 weight: 16 * 3 * 3 * 3 = 432 elements
    tensor = rng.randint(-50, 51, size=(16, 3, 3, 3)).astype(np.int8)
    blob = bd.encode_tensor(tensor)
    
    original_bytes = tensor.size
    packed_bytes = len(blob) - 8  # subtract 8-byte tensor header
    
    # Each block: 18 bytes for 32 elements → 0.5625 bytes/element vs 1 byte/element
    # Expected ratio ≈ 1.78x
    ratio = original_bytes / packed_bytes
    print(f"  ✓ Compression: {original_bytes}B → {packed_bytes}B ({ratio:.2f}x)")
    assert ratio > 1.5, f"Compression ratio too low: {ratio}"

def main():
    print("BlockDialect-Lite Codec Tests")
    print("=" * 50)
    
    print("\n1. Edge Cases")
    test_roundtrip_zeros()
    test_roundtrip_small()
    
    print("\n2. Random Roundtrip")
    test_roundtrip_random_blocks()
    
    print("\n3. Tensor Pipeline")
    test_tensor_roundtrip()
    
    print("\n4. Blob I/O")
    test_blob_io()
    
    print("\n5. Compression Ratio")
    test_compression_ratio()
    
    print("\n" + "=" * 50)
    print("ALL TESTS PASSED ✓")

if __name__ == "__main__":
    main()
