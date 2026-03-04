---
name: bd-activations-fix
description: >
  Corrected plan for paper-faithful BlockDialect activation quantization.
  Ensures activations are stored as BD4 in SPRAM with no int8 intermediate,
  matching the ICML 2025 paper's dataflow.
---


The paper (§3.3) says activations are "requantized into 4-bit DialectFP4 format for the next matrix multiplication." There is **no int8 intermediate**. The dataflow is:

```
Layer accumulator → BD4-quantize → store BD4 → next layer reads BD4 → MAC(BD4 × BD4)
```

My previous plan was wrong in having int8 as the storage format with BD4 packing happening at MAC time. Here's the corrected plan:

---

## Corrected Plan: BD4 Activation Storage (Paper-Faithful)

### The Paper's Dataflow

```
conv output (int32 accum)
  → scale + BD4-quantize (two-stage selector)
  → store BD4-packed in SPRAM           ← activations live as BD4
  → next layer reads BD4 from SPRAM
  → hardware decodes BD4 → half-units   ← BDMac32 does this
  → MAC: w_hu × a_hu                    ← both operands from BD4
```

No int8 anywhere in the activation path. The only quantization step is accumulator → BD4 at each layer boundary.

### What Changes From Current Firmware

**Current** (wrong):
```
conv accum (int32) → clamp int8 → store int8 in SPRAM → read int8 → hu × int8
```

**Corrected** (paper-faithful):
```
conv accum (int32) → BD4-pack → store BD4 in SPRAM → read BD4 → BDMac32(BD4 × BD4)
```

### SPRAM Layout Change

Activation buffers shrink from 16 KB to 9 KB (stage1):

```
Current:   buf_A = 16 KB (int8)    buf_B = 16 KB (int8)     = 32 KB
Corrected: buf_A =  9 KB (BD4)     buf_B =  9 KB (BD4)      = 18 KB
```

This actually **saves** 14 KB of SPRAM, consistent with the plan's §0 "BD4 skip" variant.

### Step-by-Step Implementation

#### Step 1: New output quantizer — `conv_output_to_bd4()`

Instead of clamping the int32 accumulator to int8, BD4-pack directly from the accumulator values:

```c
/* Pack one spatial plane (out_c values at position y,x) is wrong —
   BD4 blocks are 32 contiguous elements in flattened CHW order.
   So we accumulate an entire output tensor in int32, THEN pack. */

/* After conv finishes writing int32 output: */
static void quantize_output_bd4(
    const int32_t *accum,      /* out_c × oh × ow int32 values */
    uint8_t       *bd_out,     /* BD4-packed output buffer */
    int            n_elements,
    int            out_shift   /* accumulator → real-value scaling */
) {
    int n_blocks = (n_elements + 31) / 32;
    for (int b = 0; b < n_blocks; b++) {
        int32_t tmp[32];
        int base = b * 32;
        for (int i = 0; i < 32; i++) {
            int idx = base + i;
            /* Scale accumulator to activation range, then pack */
            tmp[i] = (idx < n_elements) ? (accum[idx] >> out_shift) : 0;
        }
        bd_act_pack32_twostage(tmp, bd_out + b * BD_BLOCK_BYTES);
    }
}
```

The key difference: `bd_act_pack32_twostage` sees the **full accumulator precision** (shifted), not pre-clamped int8 values. The BD4 quantizer itself decides how to represent each value — no int8 bottleneck.

#### Step 2: Modify conv to accumulate into int32 buffer, output BD4

The current `conv3x3` writes `int8_t` output inline. Split into:

1. **Accumulate pass**: conv writes int32 accumulator results to a scratch buffer
2. **Quantize pass**: `quantize_output_bd4()` packs the int32 buffer to BD4

SPRAM scratch for the int32 accumulator: stage1 worst case = 16×32×32 = 16384 elements × 4 bytes = 64 KB. That's tight but fits — 64 KB accumulator + 9 KB BD4 input + 9 KB BD4 output = 82 KB < 128 KB.

For stages 2 and 3 (smaller tensors), it's easier: stage3 = 64×8×8 = 4096 × 4 = 16 KB.

#### Step 3: Conv inner loop reads BD4, feeds BDMac32

For **3×3 conv** — the hardest case due to spatial gather:

The input activation tensor is stored as BD4 in CHW order. To read individual values at `(ic, iy, ix)`, we must unpack the BD4 block containing that element. The unpacked values are already BD4-rounded (no additional quantization loss). After gathering 32 values for one weight block's receptive field, we BD4-pack them and feed to BDMac32.

**Critical insight: this gather-repack is nearly lossless.** The values are already on the BD4 representable grid. When repacked into a new block with a potentially different dialect/exponent, the quantizer finds the same or very close codes.

```c
/* Helper: read one activation value from BD4 storage */
static inline int8_t bd4_read_element(
    const uint8_t *bd_tensor,  /* BD4-packed tensor in SPRAM */
    int            flat_idx     /* CHW-order element index */
) {
    int block_idx   = flat_idx / 32;
    int elem_in_blk = flat_idx % 32;
    const uint8_t *blk = bd_tensor + block_idx * BD_BLOCK_BYTES;
    /* Decode just this one element (or cache the block) */
    int8_t vals[32];
    bd_act_unpack32(blk, vals);
    return vals[elem_in_blk];
}
```

This is expensive (unpack 32 to get 1). Optimization: **cache the last unpacked block** — many consecutive gather accesses hit the same block:

```c
static int     cached_block_idx = -1;
static int8_t  cached_vals[32];

static inline int8_t bd4_read_cached(const uint8_t *bd, int flat) {
    int bi = flat / 32;
    if (bi != cached_block_idx) {
        bd_act_unpack32(bd + bi * BD_BLOCK_BYTES, cached_vals);
        cached_block_idx = bi;
    }
    return cached_vals[flat % 32];
}
```

Then the conv inner loop builds a 32-element gather buffer, BD4-packs it, and calls BDMac32:

```c
for (int i = 0; i < count; i++) {
    int flat = elem_done + i;
    int ic = flat/9, ky = (flat%9)/3, kx = flat%9%3;
    int iy = y*stride - 1 + ky, ix = x*stride - 1 + kx;
    if (iy >= 0 && iy < h && ix >= 0 && ix < w)
        a_gather[i] = (int32_t)bd4_read_cached(input_bd4, ic*h*w + iy*w + ix);
    else
        a_gather[i] = 0;
}
bd_act_pack32_twostage(a_gather, a_block);
int es;
int32_t ps = bdmac32_mac_block(bp, a_block, &es);
```

For **1×1 conv and FC** — block-aligned, no gather needed:

The activation elements for one weight block are `input[ic..ic+31]` at position `(y,x)` — these are NOT contiguous in CHW BD4 storage (they're at stride `h*w` apart). So 1×1 also requires gathering. However, the FC layer's input is a flat 64-element vector, which IS contiguous and aligns directly with 2 BD4 blocks — those can be fed directly to BDMac32 without any unpack/repack.

#### Step 4: ReLU integration

The paper applies ReLU **before** BD4 quantization (it's part of the layer output). Since post-ReLU values are ≥ 0, all sign bits in the BD4 block will be 0. This is naturally handled: clamp negative accumulator values to 0 before calling `bd_act_pack32_twostage`.

```c
/* In quantize_output_bd4, if do_relu: */
tmp[i] = (accum[idx] >> out_shift);
if (do_relu && tmp[i] < 0) tmp[i] = 0;
```

#### Step 5: Residual add in accumulator domain

Currently: `add_relu_inplace(int8 dst, int8 src)`.

Paper-faithful: the residual add happens on the **BD4-decoded** values, then the sum is BD4-quantized:

```c
/* Unpack both BD4 tensors → int8, add, BD4-pack result */
static void add_relu_bd4(
    const uint8_t *bd_a,    /* BD4 conv output */
    const uint8_t *bd_skip, /* BD4 skip tensor */
    uint8_t       *bd_out,  /* BD4 result */
    int            n_elements
) {
    int n_blocks = (n_elements + 31) / 32;
    for (int b = 0; b < n_blocks; b++) {
        int8_t va[32], vs[32];
        bd_act_unpack32(bd_a    + b * BD_BLOCK_BYTES, va);
        bd_act_unpack32(bd_skip + b * BD_BLOCK_BYTES, vs);
        int32_t sum[32];
        for (int i = 0; i < 32; i++) {
            int32_t s = (int32_t)va[i] + (int32_t)vs[i];
            sum[i] = (s < 0) ? 0 : s;  /* ReLU */
        }
        bd_act_pack32_twostage(sum, bd_out + b * BD_BLOCK_BYTES);
    }
}
```

#### Step 6: SPRAM budget (revised)

```
int32 accum scratch:     65,536 B = 64 KB  (stage1: 16×32×32 × 4 bytes)
buf_A BD4:                9,216 B =  9 KB  (stage1: 512 blocks × 18)
buf_B BD4:                9,216 B =  9 KB
bd_skip BD4:              9,216 B =  9 KB
block cache (32 bytes):      32 B
stack:                    2,048 B =  2 KB
─────────────────────────────────────────────
Total:                  ~95,264 B = ~93 KB
Available SPRAM:        131,072 B = 128 KB
Headroom:               ~35,808 B =  35 KB  ✓
```

Tight but fits. For stages 2–3 the int32 scratch shrinks (32 KB → 16 KB), freeing more headroom.

#### Step 7: Python reference update

Update `quantized_reference.py` to simulate the paper-faithful path:
1. Each layer's output: accumulator → ReLU → BD4-quantize (no int8 intermediate)
2. Next layer reads BD4-decoded values as input
3. MAC computes in half-units² domain
4. Generate new expected hashes for `quantized_ref.h`

#### Step 8: Test extension

Add a test in test_blockdialect.py that verifies:
- int32 accumulator values → BD4-pack → BDMac32 golden model produces correct partial_sum
- Compare against the old int8-intermediate path to quantify the accuracy improvement

### What Does NOT Change

- **BDMac32.scala** — hardware already correct
- **DialectFP4DecodeCore.scala** — unchanged
- **SoC/bitstream** — no rebuild (BDMac32 already mapped at 0x40031000)
- **Weight blob** — no reflash
- **`bd_act_pack32_twostage`** — already takes int32 input, works as-is
- **`bd_act_unpack32`** — already outputs int8 reconstructed values

### Summary of Corrections

| Aspect | Previous (wrong) | Corrected (paper-faithful) |
|--------|-------------------|---------------------------|
| Activation storage | int8 in SPRAM | **BD4** in SPRAM |
| Layer output | accum → clamp int8 | accum → BD4-pack directly |
| MAC input | gather int8 → BD4-pack | gather from BD4 → BD4-pack (nearly lossless) |
| Quantization steps | Two (int8 then BD4) | **One** (BD4 only) |
| Skip tensor | BD4-packed already ✓ | BD4-packed ✓ (no change) |
| Residual add | int8 + int8 → int8 | BD4-decode + BD4-decode → BD4-pack |