# ResNet-1202 + BlockDialect on iCEBreaker — FPGA Plan

**Target:** ResNet-1202 (He et al., "Deep Residual Learning for Image Recognition", CIFAR-10
variant) with BlockDialect A4 compression on **both weights and activations**, running
end-to-end on the iCEBreaker iCE40UP5K via MuraxHyperRAM.

**Supersedes:** `RESNET50_FPGA_PLAN.md` — ResNet-50 is infeasible at any practical input
resolution due to its bottleneck topology creating activation tensors that pressure against
SPRAM even with BD4, and requiring a 10 MB+ weight blob that is only barely acceptable.
ResNet-1202 is the right target: CIFAR-10 input (32×32), BasicBlock topology, all activation
tensors provably fit in SPRAM without tiling, and weights are accessible from flash.

---

## 0 — Memory architecture (the single most important section)

Before anything else, the three memory regions must be understood precisely.

```
Address         Region    Size   What lives there
─────────────────────────────────────────────────────────────────
0x10000000      BRAM      12 KB  Firmware: .text, .data, .bss, stack
0x11000000      SPRAM    128 KB  Activation buffers A, B, residual
0x20000000      WeightStore (flash-backed, read-only window)
                           ~14 MB available  BD-compressed weights (~10.9 MB)
```

The linker script already enforces this split
(`src/main/c/murax/resnet50_phase0_smoke/src/linker_icebreaker.ld`,
`MEMORY { BRAM 12k; SPRAM 128k }`).

### The 12 KB / 128 KB distinction settles the activation-memory question

The "~12 KB dynamic RAM" concern applies only to **BRAM** (firmware `.text` + `.data`
+ `.rodata`).  BRAM is at `0x10000000`, 12 KB (confirmed in
`MuraxHyperRAM_iCEBreaker.scala` line 20: `onChipRamSize = 12 kB`).

Activation tensors **must** live in **SPRAM** at `0x11000000`, 128 KB — that is its
entire purpose in this SoC.  Do not store activations in BRAM.

**However**, the linker script places `.bss` and `._stack` sections in SPRAM too.
Stack is 2 KB (`_stack_size = 2048`).  So the effective SPRAM budget for activation
buffers is:

```
128 KB total SPRAM
 -  2 KB stack
 -  ~0.5 KB .bss overhead (misc globals, dialect table is in .rodata → BRAM)
─────────────────────────
 ≈ 125.5 KB available for activation buffers
```

### Activation size table for ResNet-1202 (32×32 input)

BD4 packed size = `ceil(elements/32) × 18` bytes (2-byte meta + 16-byte codes per
32-element block = 0.5625 bytes/element).  This corresponds to an effective bitwidth
of **4.5 bits/element**, compared to the paper's theoretical minimum of 4.28 bits
(9-bit metadata / 32 elements = 0.28 bits overhead; our implementation pads the
meta to 2 bytes for byte alignment).  Paper §I: "a total overhead of 9/block_size."

| Stage | Shape | Elements | int8 | BD4 packed (18B / 32 elem) |
|-------|-------|----------|------|----------------------------|
| conv1 output | 16 × 32 × 32 | 16,384 | 16,384 B = 16 KB | 512 blocks × 18 = **9,216 B ≈ 9 KB** |
| stage1 buffer | 16 × 32 × 32 | 16,384 | 16,384 B = 16 KB | 512 blocks × 18 = **9,216 B ≈ 9 KB** |
| stage2 buffer | 32 × 16 × 16 | 8,192 | 8,192 B = 8 KB | 256 blocks × 18 = **4,608 B ≈ 4.5 KB** |
| stage3 buffer | 64 × 8 × 8 | 4,096 | 4,096 B = 4 KB | 128 blocks × 18 = **2,304 B ≈ 2.3 KB** |

Note: the `bd_act_storage_bytes()` function in `bd_act.h` computes these values
correctly: `((n_elements + 31) / 32) * BD_BLOCK_BYTES` where `BD_BLOCK_BYTES = 18`.

### BasicBlock buffer analysis (critical: 3 buffers needed, not 2)

A ResNet BasicBlock with an identity skip connection looks like:

```
input (in SPRAM buf_A)
  → conv3×3, BN, ReLU → intermediate (write to buf_B)
  → conv3×3, BN       → result (write to buf_A... but wait, buf_A still has the skip!)
  + input (skip from buf_A)
  → ReLU → output
```

The problem: the second conv reads from buf_B and wants to write to buf_A, but buf_A
is still holding the original input for the residual add.  You **cannot** do the
residual in-place with only two ping-pong buffers.

**Three buffer strategies (worst case: stage1, 16×32×32):**

| Strategy | buf_A | buf_B | skip_buf | Total | Fits? |
|----------|-------|-------|----------|-------|-------|
| All int8 | 16 KB (in) | 16 KB (mid) | 16 KB (skip) | **48 KB** | ✓ (38% of 125.5 KB) |
| BD4 skip | 16 KB (in) | 16 KB (mid) | 9 KB (BD4) | **41 KB** | ✓ (33% of 125.5 KB) |

Both fit comfortably.  We process each conv with int8 spatial buffers (needed for
random-access 3×3 kernel addressing) and store the skip tensor either in int8 or
BD4 depending on headroom.

**Weight decode scratch** adds a negligible amount:
- Stage 1: largest kernel per OC = 16 × 3 × 3 = 144 elements → 288 bytes (int16 half-units)
- Stage 3: largest kernel per OC = 64 × 3 × 3 = 576 elements → 1,152 bytes

**Full worst-case SPRAM budget (stage1, BD4 skip):**
```
buf_A (int8 input/output):   16,384 bytes = 16.0 KB
buf_B (int8 intermediate):   16,384 bytes = 16.0 KB
skip_buf (BD4 packed):        9,216 bytes =  9.0 KB
Weight decode scratch:          288 bytes =  0.3 KB
Stack:                        2,048 bytes =  2.0 KB
.bss misc:                     ~256 bytes =  0.3 KB
───────────────────────────────────────────────────────
Total peak:                 ~44,576 bytes ≈ 43.5 KB
Available SPRAM:           131,072 bytes = 128.0 KB
Headroom:                  ~86,496 bytes ≈ 84.5 KB  ✓✓✓
```

**Conclusion: activations fit in SPRAM with massive (~85 KB) headroom, even with**
**the most conservative assumptions.  No spatial tiling of activations is needed.**

### Where tiling IS required: weight decoding

~10.9 MB of BD4 weights live in flash.  They cannot be decompressed all at once.
The firmware decodes one **output-channel slice** at a time from the WeightStore window,
processes it, then discards it.  This is already the discipline established in
`hyperram_phase_full`.

### Why BD4 activations are still worth doing

Even though SPRAM is large enough for three int8 buffers without BD4 activations
(48 KB < 125.5 KB available), BD4 activation storage has three practical benefits:

1. **Residual path compression**: the 16 KB int8 `skip` tensor for a stage1 block
   drops to 9 KB in BD4.  This saves 7 KB per block per stage — useful if you later
   add per-output-channel scaling metadata or accumulator buffers.

2. **Paper completeness**: the BD paper explicitly targets both weight and activation
   compression (§3.3: "requantized into 4-bit DialectFP4 format for the next matrix
   multiplication").  Running both lets you report a meaningful end-to-end compression
   ratio and validate the full hardware path.

3. **Hardware path validation**: using `bd_act_pack32()` on activations exercises the
   same dialect selection + packing path that a future hardware quantizer (paper §3.2
   two-stage approach, Table 6: 5-cycle quantization) would use.  Our software brute-
   force MSE selection in `bd_act.h` is functionally equivalent (paper Table 2 shows
   MSE vs. two-stage are within ~0.6% accuracy), but the two-stage approach should be
   implemented before Milestone 5 (`BDMac32`) for latency reasons.

The `bd_act.h` implementation in `resnet50_common` is already correct and reusable.

### Paper cross-reference: BD4 effective bitwidth

Paper Appendix I defines effective bitwidth as:
- MXFP4: `4 + 5/block_size` = 4.16 bits/element (block_size=32)
- BlockDialect: `4 + 9/block_size` = **4.28** bits/element (block_size=32; 4-bit dialect_id + 5-bit shared_exp)

Our implementation uses **4.5** bits/element (16-bit meta word instead of 9-bit) because
we pad the meta to a full 2 bytes for byte alignment.  This costs 0.22 bits/element
extra overhead (5.1% larger than the paper's theoretical format).  The padding is
irrelevant for memory fitting — it changes 8,762 bytes to 9,216 bytes per stage1
buffer — but should be noted when reporting compression ratios.

---

## 1 — ResNet-1202 architecture reference

**Model:** He et al. 2016, CIFAR-10 variant `n=200` (6n+2 = 1202 total layers).

```
Layer         Shape out       Params (BN folded)
────────────────────────────────────────────────
conv1          16 × 32 × 32        432   (3×3×3×16)
stage1         16 × 32 × 32    921,600   (200 BasicBlocks, stride=1)
stage2         32 × 16 × 16  3,687,424   (200 BasicBlocks, first stride=2, proj)
stage3         64 × 8 × 8   14,748,160   (200 BasicBlocks, first stride=2, proj)
avgpool        64 × 1 × 1          —
fc             10                 640
────────────────────────────────────────────────
Total params                ≈ 19.36 M
```

**BasicBlock (all non-projection blocks):**
```
input (C × H × W)
  → conv3×3(C, C, stride=1), BN, ReLU
  → conv3×3(C, C, stride=1), BN
  + identity skip
  → ReLU
  → output (C × H × W)
```

**Projection block (first block of stage2 and stage3):**
```
input
  → conv3×3(Cin, Cout, stride=2), BN, ReLU
  → conv3×3(Cout, Cout, stride=1), BN
  + conv1×1(Cin, Cout, stride=2), BN   ← projection shortcut
  → ReLU
```

### Weight budget (BD4)

```
19,360,256 params ÷ 32 params/block = 605,008 BD blocks
605,008 blocks × 18 bytes/block     = ~10.89 MB
Float32 biases (folded BN):          < 10 KB
────────────────────────────────────────────────────────
Total flash usage:                   ≈ 10.90 MB
iCEBreaker flash (16 MB):            ~14 MB available (excl. bitstream + firmware)
Status:                              FITS  ✓
```

Run `python3 scripts/gen_resnet1202_model.py --dry-run` for exact numbers after
BN folding.

---

## 2 — What already exists and is directly reusable

| Artifact | Status | Notes |
|----------|--------|-------|
| `scripts/blockdialect_codec.py` | ✅ correct | canonical BD encoder/decoder |
| `scripts/gen_resnet_model.py` | ✅ working | supports resnet20/110 (CIFAR); extend for 1202 |
| `scripts/quantized_reference.py` | ✅ working | patched for 96×96; re-patch for 32×32 |
| `src/main/scala/vexriscv/demo/BlockDialectDecoder.scala` | ✅ correct | APB peripheral, shared-table decoder |
| `resnet50_common/bd_act.h` | ✅ correct | BD4 activation pack/unpack (reuse as-is) |
| `resnet50_common/bd_decode_sw.h` | ✅ correct | software BD weight decoder (reuse as-is) |
| `resnet50_common/weight_blob.h` | ✅ correct | VWB2 indexed blob parser (reuse as-is) |
| `resnet50_phase0_smoke` firmware | ✅ built | smoke test + blob header validation |
| `resnet50_phase1_int8` firmware | ✅ built | int8 inference skeleton |

What needs to be created:
- `scripts/gen_resnet1202_model.py` — exporter for ResNet-1202 (CIFAR-10)
- `scripts/resnet1202_artifacts/` — model constants, input, expected hashes, weight budget
- `src/main/c/murax/resnet1202_common/` — topology headers adapted from resnet50_common
- `src/main/c/murax/resnet1202_phase0_smoke/` — board smoke test
- `src/main/c/murax/resnet1202_phase1_int8/` — CPU baseline
- `src/main/c/murax/resnet1202_phase2_bd_full/` — BD weights + BD activations, CPU decode
- `src/main/c/murax/resnet1202_phase3_hw_decode/` — use APB `BlockDialectDecoder` peripheral
- `src/main/scala/vexriscv/demo/DialectFP4DecodeCore.scala` — refactored decode core

---

## 3 — Python toolchain: gen_resnet1202_model.py

Create `scripts/gen_resnet1202_model.py` modelled on `gen_resnet50_model.py`.

### Hard-coded constants (do not drift)

```python
MODEL_VARIANT       = "resnet1202_cifar10"
INPUT_H             = 32
INPUT_W             = 32
INPUT_C             = 3
N_CLASSES           = 10
N_PER_STAGE         = 200          # n in 6n+2 = 1202
BLOCK_SIZE          = 32
FLASH_WEIGHT_BASE   = 0x20000000   # WeightStore window
FLASH_OFFSET_BYTES  = 0x100000     # 1 MiB (matches MuraxHyperRAM default)
```

### What it produces (in `scripts/resnet1202_artifacts/`)

| File | Content |
|------|---------|
| `weights_bd.bin` | VWB2 blob: all conv+FC weights in BD4, biases in float32 |
| `model_constants.h` | C header with topology + blob-size constants |
| `input.h` | CIFAR-10 test image, int8 CHW (3×32×32) |
| `input_32x32.raw` | same image as raw binary for `quantized_reference.py` |
| `expected_fp32.h` | FP32 logits + top-1 class + SHA-256 + u32sum |
| `quantized_ref.h` | Integer hashes at all stage boundaries |
| `weight_budget.txt` | Param count, BD block count, bytes, flash fit check |

### Model loading strategy

ResNet-1202 is not in torchvision.  Two options:

**Option A (preferred for reproductibility):** Load a pre-trained checkpoint from a
known source (e.g., the `pytorch-cifar` repo) and hard-code the SHA-256 of the
checkpoint file in `gen_resnet1202_model.py`.

**Option B:** Train from scratch with a fixed seed on CIFAR-10.  Slower, but fully
self-contained.  Add `--train` flag to the script.

Either way, BN folding and weight export to VWB2 format use the exact same code path
as `gen_resnet50_model.py`.

---

## 4 — Firmware common headers: resnet1202_common

Create `src/main/c/murax/resnet1202_common/` by copying and adapting the files from
`resnet50_common`.

### Files to create (changes from resnet50_common noted)

**`weight_blob.h`** — copy verbatim.  The VWB2 format is topology-agnostic.

**`bd_decode_sw.h`** — copy verbatim.  The BD dialect table is topology-agnostic.

**`bd_act.h`** — copy verbatim.  The activation pack/unpack is topology-agnostic.

**`resnet1202_conv.h`** — conv kernels (copy from `resnet50_conv.h`, no changes needed
if using the generic `conv3x3_int8` and `conv1x1_int8` signatures).

**`resnet1202_layers.h`** — new file.  Replace the `BOTTLENECK_BLOCKS` table with a
`BASICBLOCK_TABLE`:

```c
/* BasicBlock configuration for ResNet-1202 (CIFAR-10, n=200 per stage) */
typedef struct {
    uint8_t  stage;         /* 1, 2, or 3                          */
    uint8_t  block_idx;     /* 0..199                              */
    uint8_t  has_proj;      /* 1 if this block has projection skip */
    uint8_t  stride;        /* 1 (all blocks except stage 2/3 first) */
    uint16_t in_c;          /* input channels                      */
    uint16_t out_c;         /* output channels                     */
    uint16_t in_h, in_w;   /* spatial dims                        */
} BasicBlockConf;

/* Stage 1: 200 blocks, all 16→16, in_h=in_w=32, stride=1 */
/* Stage 2: 200 blocks, first is 16→32 stride=2, rest 32→32 stride=1 */
/* Stage 3: 200 blocks, first is 32→64 stride=2, rest 64→64 stride=1 */
```

Because there are 600 blocks (too large for a static array), the firmware
**generates block configurations on the fly** given (stage, block_idx):

```c
static inline BasicBlockConf rn1202_block_conf(int stage, int block_idx) {
    static const uint16_t CHANS[4]  = {0, 16, 32, 64};
    static const uint16_t SPATIAL[4] = {0, 32, 16, 8};
    BasicBlockConf c;
    c.stage     = stage;
    c.block_idx = block_idx;
    c.has_proj  = (block_idx == 0 && stage > 1) ? 1 : 0;
    c.stride    = (block_idx == 0 && stage > 1) ? 2 : 1;
    c.in_c      = (block_idx == 0 && stage > 1) ? CHANS[stage-1] : CHANS[stage];
    c.out_c     = CHANS[stage];
    c.in_h      = (block_idx == 0 && stage > 1) ? SPATIAL[stage-1] : SPATIAL[stage];
    c.in_w      = c.in_h;
    return c;
}
```

This saves BRAM: one function plus 4 constants vs. a 600-entry table.

**`model_constants.h`** — generated by `gen_resnet1202_model.py` into
`scripts/resnet1202_artifacts/`.  Include from there at build time.

---

## 5 — Firmware milestone ladder

### Milestone 0 — Board smoke test (resnet1202_phase0_smoke)

Clone `resnet50_phase0_smoke`, rename, update blob magic check and constant names.
The smoke test should:

1. Print firmware identifier + `rdcycle` baseline.
2. Read the VWB2 header at `0x20000000`.
3. Validate magic `'VWB2'`, version, block_size=32.
4. Walk the tensor table and print: tensor count, first 3 tensor name hashes, total blob bytes.
5. Run the BD decoder self-test (APB `BlockDialectDecoder`): decode a known
   `(dialect, idx)` pair and compare to the software table.
6. Print `PASS` or stop with the failing step number.

**Deliverable:** "FPGA boots and says: found ResNet-1202 blob, 10.9 MB, decoder OK."

---

### Milestone 1 — CPU baseline, int8 weights + int8 activations (resnet1202_phase1_int8)

Strictly sequential: write, verify passing Python hash, then move to next stage.

**Inner loop discipline (weight decode, one output channel at a time):**

```c
for (int oc = 0; oc < out_c; oc++) {
    /* 1. Decode the 32-element BD block(s) for this output channel's kernel */
    bd_decode_channel_sw(&blob, tensor_id, oc, w_buf, kernel_elems);

    /* 2. Run the inner conv loop writing to output[oc, :, :] */
    conv2d_row_oc(input_spram, output_spram, w_buf, ...);
    /* w_buf can be reused immediately — no accumulation across channels needed here */
}
```

**Activation buffer locations (in SPRAM):**

```c
/* Compile-time SPRAM layout for ResNet-1202 */
#define ACT_A_BASE  (0x11000000)          /* 16 KB  — ping buffer */
#define ACT_B_BASE  (0x11004000)          /* 16 KB  — pong buffer */
/* ACT_A and ACT_B alternate roles across blocks;
   for downsampling blocks, input lives in ACT_A, output in ACT_B, then swap. */
/* Remaining ~96 KB SPRAM is free for scratch / line buffers */
```

This is correct even for int8 (2 × 16 KB = 32 KB < 128 KB SPRAM).

**Verification tripwires:**

After each stage, compute a `u32sum` hash of the SPRAM activation buffer and compare
to the expected value from `quantized_ref.h`.  Fail fast on mismatch.  This is the
same pattern as `hyperram_phase_full`.

**Python reference:** re-run `quantized_reference.py` patched for 32×32 CIFAR input:

```bash
python3 scripts/quantized_reference.py --input scripts/resnet1202_artifacts/input_32x32.raw
```

The `BOTTLENECK_NAMES` / spatial dims must be updated for the BasicBlock topology and
32×16×8 spatial progression.

**Deliverable:** full CIFAR-10 ResNet-1202 inference in software, output hash matches
Python reference.  Cycle count printed.  Expected to be slow (days at 12 MHz) — that
is fine at this milestone; correctness is the only goal.

---

### Milestone 2 — BD activations (resnet1202_phase2_bd_full)

Add BD4 activation compression at the **stage boundaries and residual paths**.

**Why now:** Milestone 1 proves int8 activations are correct.  Milestone 2 replaces
the stored activation values with BD4 to validate the compression path and the pack/
unpack round-trip on real ResNet activations.

**What changes:**

1. After each 3×3 conv + BN + ReLU, call `bd_act_pack32()` on the output activation
   block before writing to SPRAM.  Use the SPRAM layout:

```c
/* BD4-compressed activation buffers (stage1: 8 KB each vs. 16 KB int8) */
#define ACT_BD_A_BASE  (0x11000000)   /* 8 KB */
#define ACT_BD_B_BASE  (0x11002000)   /* 8 KB */
/* 112 KB SPRAM headroom — ample for int32 accumulators and scratch */
```

2. The `bd_act_pack32()` / `bd_act_unpack32()` functions from `bd_act.h` are used
   directly.  No code change needed to those functions.

3. Residual path: for identity blocks, the skip tensor is also stored BD4-compressed
   in SPRAM.  Since it has the same shape as the input, you need at most 3 × 8 KB =
   24 KB simultaneously (input, output, skip) for stage1 — well within 128 KB.

4. Verify: after each stage, unpack the compressed activation and compare the
   `u32sum` hash.  Accept a small numeric difference (BD4 is lossy); the expected hash
   comes from the Python `quantized_reference.py` run with BD activation rounding.

**Deliverable:** end-to-end inference with all activations stored in BD4.  Report:
- Activation peak memory usage (should be ≤ 24 KB worst case)
- BD4 activation MSE per stage (from Python reference)
- Top-1 accuracy impact (run on 100 CIFAR-10 test images from Python, compare to int8 baseline)

---

### Milestone 3 — Hardware BD decoder (resnet1202_phase3_hw_decode)

Replace `bd_decode_sw.h` software decode with APB `BlockDialectDecoder` peripheral
calls for weight decoding.

**What changes:**

1. Firmware reads one BD block (18 bytes: meta16 + codes[16]) from the WeightStore window.
2. Writes `meta` and `codes` to the APB decoder peripheral registers.
3. Reads back 32 decoded int8 half-units (512 bits = 16 MMIO reads of 4 bytes each,
   or use the peripheral's burst-read register if implemented).
4. Applies the shared exponent scaling in software (one multiply per block, not per element).

**No SoC rebuild required** — the APB decoder peripheral is already in `MuraxHyperRAM.v`.
The FPGA bitstream does not need to change between Milestone 2 and Milestone 3.

**Deliverable:** same hash as Milestone 2, with measurably fewer CPU cycles for the
weight-decode loop.

---

### Milestone 4 — DialectFP4DecodeCore refactor (RTL cleanup, enabler for Milestone 5)

Refactor the APB `BlockDialectDecoder.scala` into two layers:

**`DialectFP4DecodeCore.scala`** (new file):
```
inputs:  dialect_id[3:0], idx[2:0]
outputs: mag[3:0]  (half-units 0..15, unsigned)
logic:   the existing case/MuxOH from BlockDialectDecoder, moved verbatim
```

**`BlockDialectDecoderAPB.scala`** (renamed wrapper):
- Instantiates `DialectFP4DecodeCore` for each of the 32 lanes
- Same APB register interface as today — no firmware changes

This split is a prerequisite for Milestone 5 (`BDMac32`).  It does not change
the SoC interface or any existing test.

---

### Milestone 5 — BDMac32: hardware 32-lane dot product accelerator

Implement `BDMac32.scala` as an APB peripheral at `0x40031000`:

```
Inputs (via MMIO):
  w_packed[127:0]      — 32 × 4-bit weight codes (sign|idx)
  w_meta[15:0]         — weight block meta (dialect_id, shared_exp)
  a_packed[127:0]      — 32 × 4-bit activation codes (if using BD activations)
  a_meta[15:0]         — activation block meta
  start                — write 1 to start

Outputs:
  partial_sum[31:0]    — signed accumulator result for this 32-element block
  exp_sum[5:0]         — w_exp + a_exp (for scaling in firmware)
  done                 — poll until 1
```

**Inner computation per element i:**
```
w_sign  = w_packed[i*4+3]
w_idx   = w_packed[i*4+2:i*4]
w_mag   = DialectFP4DecodeCore(w_dialect, w_idx)   // 4-bit unsigned

a_sign  = a_packed[i*4+3]
a_idx   = a_packed[i*4+2:i*4]
a_mag   = DialectFP4DecodeCore(a_dialect, a_idx)   // 4-bit unsigned

product_mag = w_mag * a_mag                         // 4b×4b → 8b
product_sign = w_sign XOR a_sign
product = product_sign ? -product_mag : product_mag
```

**Accumulator:** signed 24-bit (8-bit products × 32 elements → max 255 × 32 = 8160;
24 bits is safe).

**Scaling rule (paper §3.4):** after all blocks in a dot product, firmware applies:
```c
int64_t scaled = (int64_t)partial_sum * (1 << (w_exp + a_exp));
scaled >>= 2;   /* account for 0.5 * 0.5 = 1/4 per half-unit multiply */
```

**Deliverable:** a verified `BDMac32` module with a Python golden test
(`test_blockdialect.py` extension).  Measurable speedup for 1×1 conv inner loop.

---

## 6 — Activation BD quantization: the full-paper path

Once Milestone 3 works (correctness confirmed), add online activation quantization:

### Paper's two-stage dialect selection for activations

1. **Stage 1:** given block max_abs, compute `block_maxhu` (the max half-unit magnitude).
   Round `block_maxhu` to the nearest integer.  Pick the dialect *pair* that has this
   rounded value as their `maxhu` column (dialects come in pairs sharing `maxhu`).

2. **Stage 2:** count how many of the 32 elements fall in the "beneficial range" of
   dialect A vs. dialect B.  The paper shows this reduces to counting elements in
   a specific magnitude interval — implementable as a popcount + compare.

`bd_act.h` currently uses brute-force MSE (correct, but O(16 × 32)).  For a
hardware-accelerated path:

- Add `bd_act_pack32_twostage()` alongside the existing brute-force version.
- Validate both produce the same dialect selection for random blocks.
- Switch the firmware to `_twostage` before building `BDMac32`.

---

## 7 — MINIMIZE REFLASHING

The weights blob (~10.9 MB) takes ~1 hour to flash.  Strict rules:

1. **Lock the blob format first.** Only reflash when the VWB2 blob format OR the model
   weights change.  Firmware bugs and topology bugs do not require reflashing.

2. **Smoke-test before flashing.** Always run `gen_resnet1202_model.py` and verify
   `weight_budget.txt` before writing to flash.

3. **Flash layout** (2 regions — firmware lives in BRAM, not flash):

```
Offset      Size         Contents
0x000000    ≤1 MB        iCE40 bitstream  (BRAM firmware baked in via onChipRamHexFile)
0x100000    ~11 MB       VWB2 weight blob
```

`flashOffset = 0x100000` in the SoC → CPU `0x20000000` = flash `0x100000` (weights).  
`WEIGHT_BLOB_ADDR = 0x20000000` in firmware.

4. **How to update each region:**

| What changed | Command | Time |
|---|---|---|
| Firmware only | `sbt "runMain vexriscv.demo.MuraxHyperRAM_iCEBreaker"` then `iceprog bin/toplevel.bin` | ~2 min |
| Weights only | `iceprog -o 1M weights_bd.bin` | ~1 hour |
| Both | Flash weights first, then bitstream | ~1 hour |

Updating firmware = rebuilding the bitstream (firmware is baked into BRAM via `onChipRamHexFile` in `MuraxHyperRAM_iCEBreaker.scala`). Weights blob only needs reflashing when the model changes.
This is the normal development cycle for all Milestones 1–5.

5. Reflash the full image only when:
   - The bitstream changes (new SoC rebuild, e.g., for Milestone 4/5)
   - The weight blob changes (model variant changed)

---

## 8 — Definition of done

The iCEBreaker boot log must show all of the following:

```
[0] VWB2 blob OK: 10.9 MB, 605008 BD blocks, 160 tensors
[1] BD decoder self-test: dialect14 idx7 → mag=8 ✓
[2] ResNet-1202 inference: 32×32 CIFAR-10 test image
[3] Stage1 hash: 0x________  expected: 0x________  ✓
[4] Stage2 hash: 0x________  expected: 0x________  ✓
[5] Stage3 hash: 0x________  expected: 0x________  ✓
[6] logits u32sum: 0x________  expected: 0x________  ✓
[7] top-1 class: N (matches Python reference)
[8] total rdcycles: ___________
[9] PASS
```

---

## 9 — Milestone sequence and dependencies

```
Milestone 0: smoke test (clone resnet50_phase0_smoke, rename, no reflash)
    ↓
Milestone 1: int8 baseline (new gen_resnet1202_model.py + resnet1202_phase1_int8)
    ↓ (reflash weights once here)
Milestone 2: BD4 activation compression (firmware-only change, no reflash)
    ↓
Milestone 3: APB HW decoder for weights (firmware-only, no SoC rebuild, no reflash)
    ↓
Milestone 4: DialectFP4DecodeCore RTL refactor (SoC rebuild → reflash bitstream)
    ↓
Milestone 5: BDMac32 accelerator (SoC rebuild → reflash bitstream)
```

No milestone requires reflashing the weight blob after the initial Milestone 1 flash.

---

## FINISH LINE

Key deliverables: A fully functional ResNet1202 run on the iCEBreaker quantized with **BlockDialect** on both weights and activations; implementing the paper to the "T", a clean repo with documentation of what I accomplished, a short demo video that A) shows it's possible to have a >=1m parameter model on an FPGA without external RAM b) shows the power of BlockDialect by presenting our accurate implementation c) be a substitute for a paper, showing the methods and background behind the work
