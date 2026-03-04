# ResNet-1202 Model Artifacts

This directory is populated by running:

```bash
# Dry-run (estimate only, no model loading):
python3 scripts/gen_resnet1202_model.py --dry-run

# With a pre-trained checkpoint (Option A, recommended):
python3 scripts/gen_resnet1202_model.py --checkpoint /path/to/resnet1202_cifar10.pth

# Train from scratch (Option B, ~4h on GPU):
python3 scripts/gen_resnet1202_model.py --train --save-checkpoint resnet1202_trained.pth
```

## Generated files

| File | Description |
|------|-------------|
| `weights_bd.bin` | VWB2 weight blob (~10.9 MB of BD4 weights) |
| `model_constants.h` | C header: tensor offsets, blob size, topology constants |
| `input.h` | CIFAR-10 test image as int8 C array (3×32×32) |
| `input_32x32.raw` | Same image as raw int8 binary |
| `expected_fp32.h` | FP32 logits + top-1 class + SHA-256 + u32sum |
| `quantized_ref.h` | Stage-boundary u32sum hashes for firmware verification |
| `weight_budget.txt` | Param count, BD blocks, flash-fit check |

## Flash layout

```
Offset     Contents
0x000000   iCE40 bitstream
0x100000   Firmware ELF  (iceprog -o 0x100000 firmware.bin)
0x110000   VWB2 weight blob  (iceprog -o 0x110000 weights_bd.bin)
```

The weight blob offset matches `FLASH_WEIGHT_OFFSET = 0x100000` in `model_constants.h`
and `FLASH_OFFSET_BYTES` in `gen_resnet1202_model.py`.

See `RESNET1202_FPGA_PLAN.md §7` for the reflashing policy.
