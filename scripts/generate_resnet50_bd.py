#!/usr/bin/env python3
import os
import sys

import numpy as np
import torch
import torchvision.models as models

# Import the *fixed* BlockDialect codec from this directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blockdialect_codec as bd


def main():
    print("Loading ResNet-50 (random init, weights=None)...")
    model = models.resnet50(weights=None)
    model.eval()

    print("Encoding weights to BlockDialect (DialectFP4, 4-bit indices + per-block metadata)...")
    tensors = []

    total_params = 0
    total_float_bytes = 0
    total_bd_bytes = 0

    # Encode parameters (weights + biases). If you want BN running stats too,
    # either (a) fuse BN into conv offline, or (b) iterate state_dict and handle
    # non-parameter buffers separately.
    for name, param in model.named_parameters():
        if not param.dtype.is_floating_point:
            continue

        arr = param.detach().cpu().numpy().astype(np.float32)
        total_params += arr.size
        total_float_bytes += arr.size * 4

        encoded = bd.encode_tensor(arr)
        tensors.append(encoded)
        total_bd_bytes += len(encoded)

        if total_params % 5_000_000 < arr.size:
            # periodic progress
            print(f"  encoded ~{total_params/1e6:.1f}M params...")

    print("\nEncoding complete:")
    print(f"Total parameters encoded: {total_params:,}")
    print(f"Float32 size: {total_float_bytes:,} bytes ({total_float_bytes/(1024*1024):.2f} MB)")
    print(f"BD payload size (sum of tensor blobs): {total_bd_bytes:,} bytes ({total_bd_bytes/(1024*1024):.2f} MB)")

    output_file = "resnet50_bd_weights.bin"
    print(f"\nWriting weight blob to: {output_file}")
    bd.write_weight_blob(tensors, output_file)

    file_size = os.path.getsize(output_file)
    print(f"Actual file size on disk: {file_size:,} bytes ({file_size/(1024*1024):.2f} MB)")

    float_size_mb = total_float_bytes / (1024 * 1024)
    file_size_mb = file_size / (1024 * 1024)
    if file_size_mb > 0:
        print(f"Compression ratio (Float32 -> BD file): {float_size_mb / file_size_mb:.2f}x")


if __name__ == "__main__":
    main()
