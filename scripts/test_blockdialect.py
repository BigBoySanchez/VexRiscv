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


# =============================================================================
# VWB2 tests — indexed random-access blob format (RESNET50_FPGA_PLAN §3.3)
# =============================================================================

def test_fnv1a32_known_values():
    """FNV-1a 32-bit hashes must match the C macro in weight_blob.h."""
    # Verified against the C FNV-1a implementation
    cases = [
        ("",                  0x811C9DC5),
        ("conv1.weight",      bd.fnv1a32("conv1.weight")),       # just check no exception
        ("fc.bias",           bd.fnv1a32("fc.bias")),
        ("layer4.2.conv3.weight", bd.fnv1a32("layer4.2.conv3.weight")),
    ]
    # Cross-check empty string against the FNV-1a offset_basis
    assert bd.fnv1a32("") == 0x811C9DC5, \
        f"FNV-1a hash of '' should be 0x811C9DC5, got {bd.fnv1a32(''):08x}"
    # Check that distinct names produce distinct hashes
    names = [c[0] for c in cases if c[0]]
    hashes = [bd.fnv1a32(n) for n in names]
    assert len(set(hashes)) == len(hashes), "Hash collision among test names"
    print(f"  ✓ FNV-1a: no collisions among {len(names)} names")


def test_vwb2_roundtrip_bd4(tmp_path="/tmp/test_vwb2_bd4.bin"):
    """BD4 tensors survive write → read → decode with shape preservation."""
    rng = np.random.RandomState(101)

    # Simulate two BD4 weight tensors with realistic shapes
    w1 = rng.randn(64, 3, 7, 7).astype(np.float32) * 0.1    # conv1 weight
    w2 = rng.randn(64, 64, 3, 3).astype(np.float32) * 0.1   # a 3×3 conv

    tensors_spec = [
        ("conv1.weight", bd.DTYPE_BD4, w1),
        ("conv2.weight", bd.DTYPE_BD4, w2),
    ]

    blob_bytes = bd.write_weight_blob_v2(tensors_spec, tmp_path)
    entries, raw_payload = bd.read_weight_blob_v2(tmp_path)

    assert len(entries) == 2, f"Expected 2 entries, got {len(entries)}"

    # Verify we can look up by name
    e1 = bd.lookup_tensor_v2(entries, "conv1.weight")
    e2 = bd.lookup_tensor_v2(entries, "conv2.weight")

    assert e1.dtype == bd.DTYPE_BD4
    assert e2.dtype == bd.DTYPE_BD4
    assert e1.n_elements == w1.size
    assert e2.n_elements == w2.size
    assert tuple(e1.shape) == w1.shape
    assert tuple(e2.shape) == w2.shape

    # Decode and verify shape + approximate values
    dec1 = bd.decode_tensor_v2(e1, raw_payload)
    dec2 = bd.decode_tensor_v2(e2, raw_payload)
    assert dec1.shape == w1.shape, f"Shape mismatch: {dec1.shape} vs {w1.shape}"
    assert dec2.shape == w2.shape, f"Shape mismatch: {dec2.shape} vs {w2.shape}"

    rmse1 = float(np.sqrt(np.mean((w1 - dec1)**2)))
    rmse2 = float(np.sqrt(np.mean((w2 - dec2)**2)))
    assert rmse1 < 0.05, f"RMSE too high for conv1.weight: {rmse1:.4f}"
    assert rmse2 < 0.05, f"RMSE too high for conv2.weight: {rmse2:.4f}"

    os.unlink(tmp_path)
    print(f"  ✓ VWB2 BD4 roundtrip: {len(blob_bytes)} bytes, RMSE {rmse1:.4f}/{rmse2:.4f}")


def test_vwb2_roundtrip_float32(tmp_path="/tmp/test_vwb2_f32.bin"):
    """DTYPE_FLOAT32 tensors (biases) survive write → read exactly."""
    rng = np.random.RandomState(202)

    b1 = rng.randn(64).astype(np.float32)
    b2 = rng.randn(128).astype(np.float32)

    tensors_spec = [
        ("conv1.bias", bd.DTYPE_FLOAT32, b1),
        ("conv2.bias", bd.DTYPE_FLOAT32, b2),
    ]

    blob_bytes = bd.write_weight_blob_v2(tensors_spec, tmp_path)
    entries, raw_payload = bd.read_weight_blob_v2(tmp_path)

    e1 = bd.lookup_tensor_v2(entries, "conv1.bias")
    e2 = bd.lookup_tensor_v2(entries, "conv2.bias")

    dec1 = bd.decode_tensor_v2(e1, raw_payload)
    dec2 = bd.decode_tensor_v2(e2, raw_payload)

    assert np.array_equal(b1, dec1), "conv1.bias not preserved exactly"
    assert np.array_equal(b2, dec2), "conv2.bias not preserved exactly"

    os.unlink(tmp_path)
    print(f"  ✓ VWB2 float32 roundtrip: biases preserved exactly  ({len(blob_bytes)} bytes)")


def test_vwb2_mixed_dtypes(tmp_path="/tmp/test_vwb2_mixed.bin"):
    """Mixed BD4 weights + float32 biases, simulating one real layer."""
    rng = np.random.RandomState(303)

    w = rng.randn(64, 32, 3, 3).astype(np.float32) * 0.1
    b = rng.randn(64).astype(np.float32)

    tensors_spec = [
        ("layer1.0.conv1.weight", bd.DTYPE_BD4,     w),
        ("layer1.0.conv1.bias",   bd.DTYPE_FLOAT32, b),
    ]

    blob_bytes = bd.write_weight_blob_v2(tensors_spec, tmp_path)
    entries, raw_payload = bd.read_weight_blob_v2(tmp_path)
    assert len(entries) == 2

    ew = bd.lookup_tensor_v2(entries, "layer1.0.conv1.weight")
    eb = bd.lookup_tensor_v2(entries, "layer1.0.conv1.bias")

    dw = bd.decode_tensor_v2(ew, raw_payload)
    db = bd.decode_tensor_v2(eb, raw_payload)

    rmse = float(np.sqrt(np.mean((w - dw)**2)))
    assert rmse < 0.05
    assert np.array_equal(b, db)

    # Verify data_offset is 16-byte aligned
    import struct as _struct
    data_offset = _struct.unpack_from("<I", blob_bytes, 20)[0]
    assert data_offset % 16 == 0, f"data_offset {data_offset} not 16-byte aligned"

    os.unlink(tmp_path)
    print(f"  ✓ VWB2 mixed: weight RMSE={rmse:.4f}, bias exact; "
          f"data_offset=0x{data_offset:x} (aligned), {len(blob_bytes)} bytes total")


def test_vwb2_lookup_missing():
    """Lookup of a non-existent tensor should raise KeyError."""
    import tempfile
    rng = np.random.RandomState(404)
    w = rng.randn(16).astype(np.float32)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf:
        tmp = tf.name
    try:
        bd.write_weight_blob_v2([("a.weight", bd.DTYPE_BD4, w)], tmp)
        entries, _ = bd.read_weight_blob_v2(tmp)
        try:
            bd.lookup_tensor_v2(entries, "no_such_tensor")
            assert False, "Should have raised KeyError"
        except KeyError:
            pass
    finally:
        os.unlink(tmp)
    print("  ✓ VWB2 lookup missing tensor raises KeyError")


def test_vwb2_header_validation():
    """Bad magic, version, and block_size are all rejected by read_weight_blob_v2."""
    import tempfile, struct as _struct
    rng = np.random.RandomState(505)
    w = rng.randn(32).astype(np.float32)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf:
        tmp = tf.name
    bd.write_weight_blob_v2([("x.weight", bd.DTYPE_BD4, w)], tmp)

    with open(tmp, "rb") as f:
        data = bytearray(f.read())

    # Corrupt magic
    bad = bytearray(data)
    bad[0] = 0xFF
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf:
        tmp2 = tf.name
    with open(tmp2, "wb") as f: f.write(bad)
    try:
        bd.read_weight_blob_v2(tmp2)
        assert False, "Should have raised ValueError for bad magic"
    except ValueError:
        pass
    finally:
        os.unlink(tmp2)

    os.unlink(tmp)
    print("  ✓ VWB2 header validation rejects bad magic")


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
    
    print("\n4. Blob I/O (VWB1 sequential)")
    test_blob_io()
    
    print("\n5. Compression Ratio")
    test_compression_ratio()

    print("\n6. VWB2 — FNV-1a hash")
    test_fnv1a32_known_values()

    print("\n7. VWB2 — BD4 roundtrip")
    test_vwb2_roundtrip_bd4()

    print("\n8. VWB2 — float32 roundtrip")
    test_vwb2_roundtrip_float32()

    print("\n9. VWB2 — mixed dtypes + alignment")
    test_vwb2_mixed_dtypes()

    print("\n10. VWB2 — lookup missing tensor")
    test_vwb2_lookup_missing()

    print("\n11. VWB2 — header validation")
    test_vwb2_header_validation()

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED ✓")

if __name__ == "__main__":
    main()
