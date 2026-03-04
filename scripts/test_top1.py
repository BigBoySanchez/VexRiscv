#!/usr/bin/env python3
"""
Test ResNet-1202 top-1 classification result.

Runs the full Python reference inference and shows:
  - All 10 class logits
  - Top-1 prediction (class index + name)
  - Expected top-1 for FPGA comparison
"""

import numpy as np
import sys
sys.path.insert(0, 'scripts')
import quantized_reference as qr

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

print("="*70)
print("ResNet-1202 Top-1 Classification Test")
print("="*70)

# Load inputs
blob = open('scripts/resnet1202_artifacts/weights_bd.bin', 'rb').read()
inp = np.fromfile('scripts/resnet1202_artifacts/input_32x32.raw', dtype=np.int8)

print(f"\nInput: 3072 bytes (3×32×32 CIFAR-10 image)")
print(f"Blob:  {len(blob):,} bytes (VWB2 weight format)")

# Run full inference (this will take a few minutes)
print(f"\nRunning full ResNet-1202 inference...")
print(f"(This may take 5-10 minutes for all 600 blocks)\n")

try:
    result = qr.run_rn1202_bd4_inference(
        blob,
        inp.reshape(3, 32, 32),
        shift=7,
        verbose=True,
        has_proj=False,
        wide_add_relu=False
    )
except KeyboardInterrupt:
    print("\n\nInterrupted!")
    sys.exit(1)

print("\n" + "="*70)
print("RESULT")
print("="*70)

logits = result['logits']
top1 = result['top1']

print(f"\nAll class logits (int32):")
for i, (name, logit) in enumerate(zip(CIFAR10_CLASSES, logits)):
    marker = " ← TOP-1" if i == top1 else ""
    print(f"  [{i}] {name:12s}  logit = {int(logit):8d}{marker}")

print(f"\n{'='*70}")
print(f"Top-1 Prediction:  Class {top1} ({CIFAR10_CLASSES[top1]})")
print(f"Logit value:       {int(logits[top1])}")
print(f"{'='*70}")

print(f"\nTo verify against FPGA firmware:")
print(f"  Look for: '[fc] ... class {top1} ({CIFAR10_CLASSES[top1]})'")
print(f"  In UART output after full 600-block inference completes")

print(f"\nStage checksums (for debug):")
for stage_name, cksum in result['hashes'].items():
    print(f"  {stage_name:12s}  0x{cksum:08X}")
