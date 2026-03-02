#!/usr/bin/env python3
"""
Activation memory budget report for ResNet-50 on iCEBreaker (128 KiB SPRAM).

Prints per-layer activation sizes for:
  - int8    (1 byte/element)
  - BlockDialect A4 (4-bit codes + per-block meta, block size 32)
    BD A4 bytes = ceil(N/32) * 18  (2-byte meta + 16-byte packed codes per block)

Also identifies "spill points" where activations exceed SPRAM and reports
peak memory under a tiled execution strategy.

Usage:
    python3 scripts/activation_budget.py
"""

from __future__ import annotations
import math

# ── ResNet-50 topology ────────────────────────────────────────────────────────
# Each entry: (name, out_c, out_h, out_w, note)
# After conv1 + maxpool: 64 × 56 × 56
# layer1: 256 × 56 × 56   (3 bottleneck blocks)
# layer2: 512 × 28 × 28   (4 bottleneck blocks)
# layer3: 1024 × 14 × 14  (6 bottleneck blocks)
# layer4: 2048 × 7 × 7    (3 bottleneck blocks)

SPRAM_BYTES = 128 * 1024  # 131072 bytes
BD_BLOCK_SIZE = 32
BD_BLOCK_BYTES = 18  # 2-byte meta + 16-byte packed codes

def bd_a4_bytes(n_elements: int) -> int:
    """Compute BlockDialect A4 storage for n_elements."""
    n_blocks = (n_elements + BD_BLOCK_SIZE - 1) // BD_BLOCK_SIZE
    return n_blocks * BD_BLOCK_BYTES

# ── Layer definitions ─────────────────────────────────────────────────────────

# (layer_name, channels, height, width, description)
# These are the ACTIVATION tensors that must be stored at various points.

STEM_ACTS = [
    ("input",           3,   224, 224, "model input (int8 CHW)"),
    ("conv1_out",       64,  112, 112, "after conv1 (7×7, stride 2) + ReLU"),
    ("maxpool_out",     64,   56,  56, "after 3×3 maxpool stride 2"),
]

# Bottleneck block internal activations:
# For each block: input -> conv1(1×1) -> conv2(3×3) -> conv3(1×1) -> +skip -> ReLU
# Spatial dims stay same within a stage (except first block of stage 2/3/4 which halves)

def bottleneck_acts(stage: int, block: int, in_c: int, mid_c: int, out_c: int,
                    in_h: int, in_w: int, stride: int):
    """Return list of activation tensors for one bottleneck block."""
    pfx = f"layer{stage}.{block}"
    oh, ow = in_h // stride, in_w // stride
    has_ds = (stride > 1) or (in_c != out_c)
    acts = []

    # Input to block (needed for skip connection)
    acts.append((f"{pfx}.input", in_c, in_h, in_w, "block input (skip source)"))

    # conv1 output: 1×1, stride 1 -> mid_c × in_h × in_w (spatial unchanged)
    acts.append((f"{pfx}.conv1_out", mid_c, in_h, in_w, "1×1 squeeze + ReLU"))

    # conv2 output: 3×3, stride -> mid_c × oh × ow
    acts.append((f"{pfx}.conv2_out", mid_c, oh, ow, f"3×3 stride={stride} + ReLU"))

    # conv3 output: 1×1, stride 1 -> out_c × oh × ow (before + skip)
    acts.append((f"{pfx}.conv3_out", out_c, oh, ow, "1×1 expand (pre-add)"))

    if has_ds:
        acts.append((f"{pfx}.ds_out", out_c, oh, ow, "1×1 downsample skip"))

    # Final residual output
    acts.append((f"{pfx}.out", out_c, oh, ow, "residual + ReLU"))

    return acts


def all_activations():
    """Generate all activation tensors for ResNet-50."""
    acts = list(STEM_ACTS)

    # Stage configs: (n_blocks, in_c_first, mid_c, out_c, H, W, stride_first)
    stages = [
        (1, 3, [64, 64, 64],   [64, 64, 64],     [256, 256, 256],     56, 56, [1, 1, 1]),
        (2, 4, [256, 512, 512, 512], [128, 128, 128, 128], [512, 512, 512, 512], 56, 56, [2, 1, 1, 1]),
        (3, 6, [512]+[1024]*5,  [256]*6, [1024]*6, 28, 28, [2]+[1]*5),
        (4, 3, [1024]+[2048]*2, [512]*3, [2048]*3, 14, 14, [2]+[1]*2),
    ]

    # layer1: special — first block has in_c=64 (from maxpool), stride=1, but downsample 64→256
    layer1_cfg = [
        (0, 64,  64, 256, 56, 56, 1),
        (1, 256, 64, 256, 56, 56, 1),
        (2, 256, 64, 256, 56, 56, 1),
    ]
    for blk, in_c, mid_c, out_c, h, w, s in layer1_cfg:
        acts.extend(bottleneck_acts(1, blk, in_c, mid_c, out_c, h, w, s))

    # layer2
    layer2_cfg = [
        (0, 256, 128, 512, 56, 56, 2),
        (1, 512, 128, 512, 28, 28, 1),
        (2, 512, 128, 512, 28, 28, 1),
        (3, 512, 128, 512, 28, 28, 1),
    ]
    for blk, in_c, mid_c, out_c, h, w, s in layer2_cfg:
        acts.extend(bottleneck_acts(2, blk, in_c, mid_c, out_c, h, w, s))

    # layer3
    layer3_cfg = [(0, 512, 256, 1024, 28, 28, 2)] + \
                 [(i, 1024, 256, 1024, 14, 14, 1) for i in range(1, 6)]
    for blk, in_c, mid_c, out_c, h, w, s in layer3_cfg:
        acts.extend(bottleneck_acts(3, blk, in_c, mid_c, out_c, h, w, s))

    # layer4
    layer4_cfg = [(0, 1024, 512, 2048, 14, 14, 2)] + \
                 [(i, 2048, 512, 2048, 7, 7, 1) for i in range(1, 3)]
    for blk, in_c, mid_c, out_c, h, w, s in layer4_cfg:
        acts.extend(bottleneck_acts(4, blk, in_c, mid_c, out_c, h, w, s))

    # Final layers
    acts.append(("avgpool_out", 2048, 1, 1, "global average pool"))
    acts.append(("fc_out", 1000, 1, 1, "FC logits"))

    return acts


def main():
    acts = all_activations()

    print("=" * 100)
    print("ResNet-50 Activation Memory Budget (iCEBreaker: 128 KiB SPRAM)")
    print("=" * 100)
    print()
    print(f"{'Layer':<35s} {'Shape':>18s} {'Elements':>10s} "
          f"{'int8':>8s} {'BD_A4':>8s} {'Ratio':>6s} {'Fits?':>6s}")
    print("-" * 100)

    peak_int8 = 0
    peak_bd = 0
    spill_layers = []

    for name, c, h, w, desc in acts:
        n = c * h * w
        b_int8 = n
        b_bd = bd_a4_bytes(n)
        ratio = b_int8 / b_bd if b_bd > 0 else 0
        fits = "YES" if b_bd <= SPRAM_BYTES else "no"

        if b_int8 > SPRAM_BYTES:
            spill_layers.append((name, c, h, w, b_int8, b_bd))

        peak_int8 = max(peak_int8, b_int8)
        peak_bd = max(peak_bd, b_bd)

        shape_str = f"{c}×{h}×{w}"
        print(f"  {name:<33s} {shape_str:>18s} {n:>10,d} "
              f"{b_int8:>7,d}B {b_bd:>7,d}B {ratio:>5.2f}× {fits:>5s}")

    print("-" * 100)
    print(f"\n  Peak int8:  {peak_int8:>10,d} bytes ({peak_int8/1024:.1f} KiB)")
    print(f"  Peak BD A4: {peak_bd:>10,d} bytes ({peak_bd/1024:.1f} KiB)")
    print(f"  SPRAM:      {SPRAM_BYTES:>10,d} bytes ({SPRAM_BYTES/1024:.0f} KiB)")

    # ── Simultaneous memory analysis ──────────────────────────────────────────
    # During a bottleneck block, we need AT MINIMUM:
    #   - block input (for skip connection)
    #   - current working tensor (output of current conv)
    #   - if downsample: the projected skip
    # The worst case is the block with the largest (input + working + skip).

    print("\n" + "=" * 100)
    print("Simultaneous Memory: Bottleneck Block Worst-Case")
    print("=" * 100)
    print()
    print(f"{'Block':<25s} {'input_int8':>10s} {'input_BD':>10s} "
          f"{'work_int8':>10s} {'work_BD':>10s}  {'total_int8':>10s} {'total_BD':>10s}")
    print("-" * 100)

    # For each bottleneck block, compute simultaneous requirement
    block_configs = [
        # (name, in_c, mid_c, out_c, in_h, in_w, stride)
        ("layer1.0",   64,  64, 256, 56, 56, 1),
        ("layer1.1",  256,  64, 256, 56, 56, 1),
        ("layer1.2",  256,  64, 256, 56, 56, 1),
        ("layer2.0",  256, 128, 512, 56, 56, 2),
        ("layer2.1",  512, 128, 512, 28, 28, 1),
        ("layer2.2",  512, 128, 512, 28, 28, 1),
        ("layer2.3",  512, 128, 512, 28, 28, 1),
        ("layer3.0",  512, 256, 1024, 28, 28, 2),
        ("layer3.1", 1024, 256, 1024, 14, 14, 1),
        ("layer3.2", 1024, 256, 1024, 14, 14, 1),
        ("layer3.3", 1024, 256, 1024, 14, 14, 1),
        ("layer3.4", 1024, 256, 1024, 14, 14, 1),
        ("layer3.5", 1024, 256, 1024, 14, 14, 1),
        ("layer4.0", 1024, 512, 2048, 14, 14, 2),
        ("layer4.1", 2048, 512, 2048,  7,  7, 1),
        ("layer4.2", 2048, 512, 2048,  7,  7, 1),
    ]

    worst_total_int8 = 0
    worst_total_bd = 0
    worst_name = ""

    for name, in_c, mid_c, out_c, ih, iw, stride in block_configs:
        oh, ow = ih // stride, iw // stride

        # Input kept alive for skip: in_c × ih × iw
        input_n = in_c * ih * iw
        # Worst-case working tensor: max of (mid_c×ih×iw, mid_c×oh×ow, out_c×oh×ow)
        work_n = max(mid_c * ih * iw, mid_c * oh * ow, out_c * oh * ow)

        total_int8 = input_n + work_n
        total_bd = bd_a4_bytes(input_n) + work_n  # working tensor int8, skip in BD

        print(f"  {name:<23s} {input_n:>9,d}B {bd_a4_bytes(input_n):>9,d}B "
              f"{work_n:>9,d}B {bd_a4_bytes(work_n):>9,d}B  "
              f"{total_int8:>9,d}B {total_bd:>9,d}B")

        if total_bd > worst_total_bd:
            worst_total_bd = total_bd
            worst_total_int8 = total_int8
            worst_name = name

    print("-" * 100)
    print(f"\n  Worst-case block (BD skip + int8 work): {worst_name}")
    print(f"    int8 total: {worst_total_int8:>10,d} bytes ({worst_total_int8/1024:.1f} KiB)")
    print(f"    BD+int8:    {worst_total_bd:>10,d} bytes ({worst_total_bd/1024:.1f} KiB)")
    print(f"    SPRAM:      {SPRAM_BYTES:>10,d} bytes ({SPRAM_BYTES/1024:.0f} KiB)")
    fits = worst_total_bd <= SPRAM_BYTES
    print(f"    Fits? {'YES' if fits else 'NO — must tile spatially'}")

    # ── Spill layer summary ───────────────────────────────────────────────────
    if spill_layers:
        print(f"\n{'='*100}")
        print(f"Layers that EXCEED SPRAM as int8 ({len(spill_layers)} total):")
        print(f"{'='*100}")
        for name, c, h, w, b8, bbd in spill_layers:
            print(f"  {name:<33s} {c}×{h}×{w} = {b8:,d}B int8, {bbd:,d}B BD "
                  f"({'fits BD' if bbd <= SPRAM_BYTES else 'NEEDS TILE'})")

    # ── Tiling strategy recommendation ────────────────────────────────────────
    print(f"\n{'='*100}")
    print("Tiling Strategy Recommendation")
    print(f"{'='*100}")
    print("""
  1. Store skip-connection tensors in BD A4 format (~1.78× compression)
  2. Keep working activations (current conv input/output) in int8
  3. For blocks where even BD skip + int8 work > 128 KiB:
     - Tile spatially (e.g., process rows in strips)
     - conv1 (1×1): trivially tileable per-row
     - conv2 (3×3): need 1-row halo overlap
     - conv3 (1×1): trivially tileable per-row
  4. Spill points for BD A4 encoding:
     - After each bottleneck block output (skip sources)
     - Stage transition outputs
  5. Safe working buffer: ~64 KiB leaves room for stack + weight decode cache
""")


if __name__ == "__main__":
    main()
