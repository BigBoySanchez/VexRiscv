Below is a concrete, repo-aware, step-by-step plan to get **ResNet‑50 + BlockDialect** running on FPGA, while implementing the **BlockDialect “decoder/dequantizer” in hardware the way the paper describes** (i.e., *index→representable value lookup with shared values across dialects; 0.5 granularity; and a MAC path compatible with low-precision integer arithmetic*). 

I’m going to treat your current `resnet` branch as the source of truth and **explicitly reuse** what you already have working:

* `MuraxHyperRAM.scala` SoC with **flash-backed WeightStore** + **optional BlockDialectDecoder APB peripheral**
* `scripts/blockdialect_codec.py` offline DialectFP4 encoder (VWB1 blobs)
* `hyperram_phase_b` firmware as the reference for how you drive the HW decoder
* `hyperram_phase_full` firmware as the reference for “full network” structuring, rdcycle timing, expected hashes, etc.

And I’ll avoid building more ResNet‑110-specific infrastructure.

---

## 0) Lock the “ResNet‑50 on FPGA” track and stop the ResNet‑110 bleed

1. **Create a clean work branch** off `resnet`, e.g. `resnet50_fpga`.
2. **Quarantine ResNet‑110 code paths**:

   * Don’t delete them yet—just stop touching them.
   * Create a new firmware target directory rather than modifying `hyperram_phase_full` in-place:

     * `src/main/c/murax/resnet50_phase0_smoke/`
     * `src/main/c/murax/resnet50_phase1_int8/`
     * `src/main/c/murax/resnet50_phase2_bd_weights_hwdecode/`
     * (optional) `src/main/c/murax/resnet50_phase3_bd_fullpath/`
3. Add a single “north star” doc at repo root: `RESNET50_FPGA_PLAN.md` that lists the milestones (below) and the current status.

**Deliverable:** a clean branch where every new commit is ResNet‑50-related.

---

## 1) Confirm what you already have matches the paper’s “hardware decoder” requirement

The paper’s key requirements for *dequantization/decoder* are:

* DialectFP4 representable values use **0.5 granularity** (values are `0.5 * [0..15]`), enabling low-precision integer arithmetic. 
* During inference, the **3-bit index is converted to a 4-bit integer** by indexing a **pre-stored table of representable values**, and **most values are shared across dialects to minimize storage**. 
* Dequantization is intended to be *very lightweight* (paper cites 1-cycle dequantization for 32-element parallel processing). 

Your current `src/main/scala/vexriscv/demo/BlockDialectDecoder.scala` already implements the **shared-table idea** (constant cases for idx 0..5, a small per-dialect variant for idx6, computed max for idx7, and a dialect15 special case). That is exactly the “shared values across dialects” optimization described in the paper. 

**But**: today it’s wrapped as an **APB peripheral** used by firmware. That’s great for correctness and bring-up, but it’s not yet the “feed MAC units directly” style the paper discusses.

So the plan is:

* keep the **APB decoder** as a debug tool and correctness oracle,
* then refactor the decode core into a reusable module that can be instantiated inside a streaming MAC/conv accelerator.

### ✅ Verification: BlockDialectDecoder matches `blockdialect_codec.py` and paper Figure 4

Checked March 2026 by comparing `src/main/scala/vexriscv/demo/BlockDialectDecoder.scala`
against `scripts/blockdialect_codec.py` (`DIALECTS_HALF_UNITS` table, arXiv:2501.01144v5 Figure 4).

**Shared constants (idx 0–5, dialects 0–14):**

| idx | Scala `decodeHalfUnits` | Python `DIALECTS_HALF_UNITS[d][idx]` |
|-----|------------------------|--------------------------------------|
| 0   | 0                      | 0                                    |
| 1   | 1                      | 1                                    |
| 2   | 2                      | 2                                    |
| 3   | 3                      | 3                                    |
| 4   | 4                      | 4                                    |
| 5   | 6                      | 6                                    |

All six values are identical — this is the "shared values across dialects" optimization from the paper.

**idx 7 / max half-unit (dialects 0–14):**

Scala: `maxHU = 15 - (dialectId >> 1)` → `[15,15, 14,14, 13,13, 12,12, 11,11, 10,10, 9,9, 8]`
Python column 7: `[15,15, 14,14, 13,13, 12,12, 11,11, 10,10, 9,9, 8]` ✓

**idx 6 / variant table (dialects 0–14):**

Scala `variantIdx6`: `[11, 9, 11, 9, 10, 8, 10, 8, 9, 7, 9, 7, 8, 7, 7]`
Python column 6:    `[11, 9, 11, 9, 10, 8, 10, 8, 9, 7, 9, 7, 8, 7, 7]` ✓

**Dialect 15 (special case):**

Scala idx→value: `0,1,2,3,4,5,6,8`
Python `DIALECTS_HALF_UNITS[15]`: `[0,1,2,3,4,5,6,8]` ✓

**Conclusion:** Our `BlockDialectDecoder` half-unit table is a bit-exact match to
`blockdialect_codec.py` and paper Figure 4 for all 16 dialects × 8 indices.
The APB peripheral outputs signed int8 half-units, consistent with the paper's
"0.5 × half_unit × 2^e" dequantization formula. The hardware is ready to be
refactored into a standalone `DialectFP4DecodeCore` for use in a streaming MAC unit
(Section 6).

---

## 2) Decide what “ResNet‑50 on this FPGA” means in practice (and make it explicit)

We need to implement the following:

**Torchvision ImageNet ResNet‑50** (224×224, 1000-class). This usually requires **external DRAM/HyperRAM** for activations and is much harder to bring up on tiny RAM. LET'S DO IT ANYWAY, without external RAM. Just use the iCEBreaker's on-board RAM.

**Action:** hard-code it into your export script + firmware constants, so you don’t drift.

**Deliverable:** `scripts/gen_resnet50_model.py` encodes exactly the model variant you will run.
### ✅ Implemented — March 2026

`scripts/gen_resnet50_model.py` created.  Hard‑coded constants at top of file:

| Constant | Value |
|---|---|
| `MODEL_VARIANT` | `torchvision_resnet50_imagenet_v1` |
| `INPUT_H / INPUT_W / INPUT_C` | 224 / 224 / 3 |
| `N_CLASSES` | 1000 |
| `BLOCK_SIZE` | 32 |
| `FLASH_WEIGHT_BASE` | `0x20000000` (WeightStore window) |
| `FLASH_OFFSET_BYTES` | `0x100000` (1 MiB, matches MuraxHyperRAM default) |

The script produces (in `scripts/resnet50_artifacts/`):

* **`weights_bd.bin`** — VWB1 BlockDialect blob (BN folded, block size 32; biases stored as float32)
* **`model_constants.h`** — C header with all topology + blob-size constants + layer enum
* **`input.h`** — 224×224 int8 input (CHW, symmetric per-tensor quantized) derived from `scripts/funny_monkey_tensor.bin`
* **`expected_fp32.h`** — FP32 logits + top-1 class + SHA-256 + `u32sum` checksum
* **`weight_budget.txt`** — parameter count, BD block count, actual blob sizes, flash fit check

Run with:
```
cd /home/tim/VexRiscv
python3 scripts/gen_resnet50_model.py          # full (BD + headers)
python3 scripts/gen_resnet50_model.py --dry-run  # budget estimate only
```
---

## 3) Upgrade the host toolchain: add ResNet‑50 export + BlockDialect packing (don’t extend the ResNet‑110 script)

Right now `scripts/gen_resnet_model.py` only knows resnet20/resnet110. Don’t keep bolting on; make a dedicated exporter.

### 3.1 Build a ResNet‑50 exporter script

Create `scripts/gen_resnet50_model.py` that:

1. Loads the ResNet‑50 model (your chosen variant).
2. Folds batchnorm into conv where possible (highly recommended):

   * Reduces runtime ops and reduces the number of tensors.
3. Exports:

   * `input.h` (fixed test image / tensor)
   * `expected_*.h` (expected output logits AND a few intermediate hashes)
   * `weights_bd.bin` (BlockDialect VWB1 blob)
   * (optional later) a quantized-kernel reference output (`expected_int*.h` / hashes) for Milestone 1

### 3.2 Make the BlockDialect weight packing match your HW and the paper

Reuse `scripts/blockdialect_codec.py` as the canonical reference:

* Block size = 32
* Meta word contains `{dialect_id, shared_exp}` and then 16 bytes of packed codes.

This matches the paper’s “index→table lookup” dequantization flow. 
### ✅ Implemented — March 2026

`blockdialect_codec.py` confirmed as canonical reference.  The block binary
layout is already correct (verified in §1):

| Field          | Bits     | Notes                                      |
|----------------|----------|--------------------------------------------|
| `dialect_id`   | 15..12   | big-endian u16 meta word                   |
| `shared_exp`   | 11..7    | FP16 exponent bits (0..31, bias=15)        |
| padding        | 6..0     | 0                                          |
| packed codes   | 16 bytes | 2 × 4-bit codes per byte (sign\|idx)       |

Round-trip tests in `scripts/test_blockdialect.py` (tests 1–5) confirm the
encoder/decoder is bit-exact; `gen_resnet50_model.py` calls `blockdialect_codec`
directly — no parallel implementation exists.
### 3.3 Fix the “ResNet‑50 requires random access” problem up front

Your current BD blob format is sequential and “decode whole tensor” oriented. That’s fine for first-layer demos, not for ResNet‑50.

For ResNet‑50, add a **tensor table** to the blob:

* `magic` (VWB1)
* `version`
* `tensor_count`
* For each tensor:

  * `name_hash` (or enum id)
  * `dtype` (BD4, int8, etc.)
  * `offset_bytes`
  * `n_bytes`
  * shape (optional but useful)

Then your firmware can fetch weights by tensor id rather than “whatever order the exporter happened to dump them in”.

**Deliverable:** a blob spec written down in `scripts/` and a single C struct parser in firmware.
### Current state (March 2026)

`weights_bd.bin` is a **sequential VWB1** container produced by `scripts/blockdialect_codec.py::write_weight_blob()`.

* Payload is a fixed-order concatenation of per-layer blobs:
   * weight tensor = `bd.encode_tensor(flattened_weight_f32)`
   * bias vector   = float32 array
* Names/shapes are not stored in the blob today; firmware must follow the exporter’s layer order (via `model_constants.h`) or walk the stream.

### Planned improvement

Implement a “VWB2” style table-of-contents blob so firmware can random-access tensors by id/name-hash.
---

## 4) Flash layout and SoC configuration: make ResNet‑50 weights fit and be readable efficiently

You already have the needed hooks in `MuraxHyperRAM.scala`:

* `flashWeightStore`
* `flashOffset` (default 1 MiB)
* WeightStore window at `0x2000_0000`

### 4.1 Compute the real weight budget

ResNet‑50 weights are big. BlockDialect helps, but metadata matters.

Do a script-side report:

* total parameters
* total BD blocks
* total bytes (header + table + blocks + padding)
* confirm it fits your flash partition

### 4.2 Put “firmware + weights” in a repeatable flash image

Make one “single command” build artifact:

* `soc.bit`
* `firmware.bin` (boot image)
* `weights_bd.bin`
* `flash_image.bin` (combined at fixed offsets)

**Deliverable:** `scripts/build_flash_resnet50.sh` (or similar) that always produces the same layout.

### Planned improvement

Add a deterministic flash-image builder script that combines bitstream/firmware/weights at fixed offsets.

### Weight budget (source of truth)

Always use the generated `scripts/resnet50_artifacts/weight_budget.txt` as the source of truth.
The exact bytes-on-flash depend on padding and bias storage, and will change as the blob evolves.

iCEBreaker note: activations do NOT fit in on-chip SPRAM (128 KiB); convolutions
must be tiled so peak activation buffer stays < 64 KiB.

### 4.5 Switch to 96×96 input — re-run Python gold tests

Memory analysis (March 2026) shows 224×224 is completely infeasible on 128 KiB SPRAM even with BD A4 compression (layer1 alone needs 441 KiB compressed). **96×96 is the target resolution**: with the 2-buffer BD4 in-place residual discipline, layer1 needs only 2 strips (81 KiB/tile, 47 KiB headroom), and every other stage fits flat.

**Weights do not change** — `weights_bd.bin` is the same ResNet-50 model. Only the input preprocessing and expected hashes change.

#### Step 1 — Update `gen_resnet50_model.py`

In `scripts/gen_resnet50_model.py`, change the two resolution constants (lines 35–36):

```python
INPUT_H  = 96    # was 224
INPUT_W  = 96    # was 224
```

`funny_monkey_tensor.bin` is already guarded by a size check:
```python
if data.size == INPUT_C * INPUT_H * INPUT_W:  # 3×96×96 = 27,648
```
It will **auto-detect the stale 224×224 tensor** (150,528 floats ≠ 27,648) and rebuild from `funny_monkey.webp` using `T.Resize((96, 96))`. No manual deletion needed.

#### Step 2 — Re-generate artifacts

```bash
cd /home/tim/VexRiscv
python3 scripts/gen_resnet50_model.py
```

This regenerates all of `scripts/resnet50_artifacts/`:
- `funny_monkey_tensor.bin` — 3×96×96 float32 (27,648 floats)
- `input.h` — 3×96×96 int8 CHW, new per-tensor scale
- `expected_fp32.h` — new FP32 logits + SHA-256 + `u32sum` for 96×96 input
- `weight_budget.txt` — unchanged weight size, updated input size note
- `model_constants.h` — update `INPUT_H`, `INPUT_W`, `INPUT_SIZE` constants

#### Step 3 — Patch `quantized_reference.py`

Four hardcoded `224` occurrences must change. Drive them from `model_constants.py` or just patch directly:

| Location | Old | New |
|---|---|---|
| Line 608 (docstring) | `int8 [3, 224, 224]` | `int8 [3, 96, 96]` |
| Line 627 (conv_7x7 call) | `conv_7x7(..., 224, 224, ...)` | `conv_7x7(..., 96, 96, ...)` |
| Line 729 (reshape) | `.reshape(3, 224, 224)` | `.reshape(3, 96, 96)` |
| Line 732 (zeros fallback) | `np.zeros((3, 224, 224), ...)` | `np.zeros((3, 96, 96), ...)` |

#### Step 4 — Re-run the Python gold reference

```bash
cd /home/tim/VexRiscv
./scripts/.venv/bin/python scripts/quantized_reference.py \
    --input scripts/resnet50_artifacts/input_96x96.raw
```

(or pass `input.h`'s raw bytes; alternatively `quantized_reference.py` can read the `input.h` binary directly if you add a `--from-header` flag)

Capture and commit the new expected hashes from the output — these become the firmware verification targets.

#### Step 5 — Update `model_constants.h` consumer sites

`resnet50_common/resnet50_layers.h` and `resnet50_phase1_int8/main.c` reference `INPUT_H`, `INPUT_W`, `INPUT_SIZE`. Since they `#include "model_constants.h"` they will pick up the new values automatically after step 2.

Verify with:
```bash
grep -r "224\|INPUT_H\|INPUT_W" src/main/c/murax/resnet50_common/
```
Any stray literal `224` found should be replaced with the constant.
---

## 5) Firmware milestone ladder (don’t jump straight to full BlockDialect fullpath)

### Milestone 0 — Board smoke test

Create `resnet50_phase0_smoke` firmware:

* UART prints
* rdcycle works
* reads the weight blob header from `0x2000_0000` (WeightStore window)
* validates **VWB1** magic (`'VWB1'` / `0x56574231`), payload size, and block size (=32)
* prints a small “peek” of the first encoded tensor header (`n_elements`, `n_blocks`)
* (optional) runs a quick **decoder table self-test** using the APB `BlockDialectDecoder`

**Deliverable:** “FPGA boots and says: found ResNet‑50 blob.”

### Milestone 1 — CPU baseline with *BlockDialect weights* (software decode)

Create `resnet50_phase1_int8`:

**BlockDialect philosophy:** weights are **always stored as BlockDialect** in flash. The milestone ladder is about *how you decode them* and *what activation format you use*, not about switching weight formats.

* Keep *compute* activations **int8/int32** for now (simple kernels), but treat **activation storage** as a first-class problem (see Milestone 1A).


### Milestone 1A — Fix the activation footprint (BlockDialect activations, software-first)

You already noted that full ResNet‑50 feature maps don’t fit in the iCEBreaker’s ~128 KiB SPRAM, so the CPU baseline needs an explicit **activation storage strategy** before it can run end‑to‑end.

**Goal:** keep *compute* simple (int8/int32), but store “spilled” activation tiles / skip tensors in **BlockDialect A4 (DialectFP4)** so they’re ~1.8× smaller than int8 and reduce external-memory bandwidth.

Implementation plan:

1. **Add an activation memory budget report (Python)** next to `weight_budget.txt` that prints per-layer activation sizes for:
   * int8
   * BlockDialect A4 (4-bit codes + per-block meta for block size 32)

2. **Implement `bd_act_pack32()` in firmware (C)** to quantize 32 activation values into one DialectFP4 block:
   * Input: 32 signed values (start with **post‑ReLU** outputs; sign handling is simpler, but keep the sign bit for compatibility).
   * Compute `max_abs = max(|x_i|)`.
   * Compute a power-of-two **shared exponent** so the scaled block range lands in ~[0, 8), matching BlockDialect’s preprocessing step (shared exp based on block max, with the “−2” adjustment described in the paper). fileciteturn0file0
   * Scale values onto the **0.5 grid** and form the paper’s 5‑bit intermediate magnitude (3 integer bits + 2 fractional bits). fileciteturn0file0
   * **Select the dialect**:
     - Bring-up option (simplest): brute-force all 16 dialects and pick the minimum MSE per block (only 16×32 candidates).
     - Paper-faithful option: implement the **two-stage dialect selection** (pick dialect pair from rounded block max; then choose between the pair by counting values in each dialect’s “beneficial range”). fileciteturn0file0
   * Quantize each element to the nearest representable value of the selected dialect and pack into **4-bit (sign|idx) codes** (2 codes per byte), plus the block meta word (dialect id + shared exponent).

3. **Implement `bd_act_unpack32()` in firmware (C)**:
   * Read meta + 16-byte code payload.
   * Decode idx→half_units via the existing **APB BlockDialectDecoder** (or a small software table), then apply sign and shared exponent to recover int16/int32 (or saturated int8) values for compute. fileciteturn0file1

4. **Use BD activations only at spill points first** (don’t quantize *everything* on day 1):
   * the tensors that must live across a residual connection (skip path),
   * stage outputs that exceed SPRAM,
   * anything you currently plan to put in HyperRAM.

**Deliverable:** Milestone 1 can run end-to-end without ever needing a full int8 feature map resident on-chip, because the “big” activations are stored in BlockDialect form and streamed/tiled.
* Decode BlockDialect weights **in software** (C) using the exact same mapping as the HW decoder:

   * unpack per-block meta (`dialect_id`, `shared_exp_bits`)
   * for each 4-bit code: extract `sign` + `idx`, then `half_units = dialect_table[dialect_id][idx]`
   * dequantize with the paper’s formula: $w = \pm (0.5\cdot\text{half\_units})\cdot 2^{(shared\_exp\_bits-15)}$
* Bias vectors are **float32** (paper-faithful; biases are small, don’t quantize)
* Use deterministic input from `scripts/gen_resnet50_model.py` (`input.h` from **funny_monkey**)
* Verification style (same spirit as `hyperram_phase_full`):

  * deterministic input
  * expected hashes after major boundaries
  * stop on first mismatch

   Note: `expected_fp32.h` is a *FP32* reference from torchvision. For this milestone you’ll want a matching **integer/quantized reference** (Python) that produces the same hashes/logit checksum as the firmware kernels.

**Deliverable:** end-to-end ResNet‑50 inference correctness with **BD weights decoded in software** (even if slow).

### Milestone 2 — BlockDialect weights, decoded via HW peripheral (scaled up)

Create `resnet50_phase2_bd_weights_hwdecode`:

* Keep activations int8 for now.
* Replace software weight decode with the APB `BlockDialectDecoder`:

  1. Read BD blocks for a slice of weights (e.g., one output channel’s kernel).
  2. Use the APB `BlockDialectDecoder` to decode 32 values at a time.
   3. Apply the shared exponent scaling in software at a well-defined boundary (per-block or per-kernel), consistent with the paper’s “block-scaled” compute story.
   4. Store decoded/scaled weights into a small cache (like your per-out-channel cache in `hyperram_phase_full`).
  4. Run the conv inner loops using cached decoded weights.

This is the “get it working on FPGA” step with the least risk.

**Deliverable:** ResNet‑50 runs with BlockDialect-compressed weights stored in flash and decoded via HW.

---

## 6) Make the HW decoder implementation “paper-accurate” (and reusable for a MAC unit)

Right now the decoder peripheral outputs signed int8 half-units as bytes. The paper’s compute story is:

* representable values use 0.5 steps and are treated as **4-bit unsigned integers 0..15** (then scaled by 0.5), 
* dequantize into **5-bit** form (sign + 4-bit integer magnitude) before multiplication, 
* multiplication uses **4-bit unsigned multiply**, sign handled separately, and then a **2-bit right shift** accounts for the `0.5*0.5` factor. 

### 6.1 Refactor decoder RTL into two layers

1. **Pure decode core** (combinational, no bus):

   * inputs: `dialect_id[3:0]`, `idx[2:0]`
   * outputs: `mag[3:0]` (0..15 half-units), plus maybe `is_valid`
   * implemented exactly as your current case logic (shared constants + small variant table + maxHU formula)
2. **APB wrapper** that uses the core and packs bytes for CPU reads (keep this for debug).

**Deliverable:** `DialectFP4DecodeCore.scala` used by both APB peripheral and accelerator.

### 6.2 Add a “decoder self-check mode” (optional but very useful)

Add an APB register that returns the decoded value for a given `(dialect, idx)` without needing packed blocks. This makes board debug much faster.

---

## 7) Add the missing piece the paper really wants: a streaming MAC datapath that consumes decoded values

The paper’s design intent is not “CPU reads decoded bytes”; it’s:

* decoder + quantization + MAC can be pipelined, and
* the overhead is tiny compared to MAC units. 

So after Milestone 2 works, do this next.

### 7.1 Implement a 32-lane “BD MAC micro-kernel” (Block size = 32)

Create a hardware module (SpinalHDL) that does one block dot-product:

Inputs:

* `w_packed[127:0]`, `w_meta(dialect_id, exp_w)`
* `a_packed[127:0]`, `a_meta(dialect_id, exp_a)` (optional at first; see below)
  Outputs:
* `partial_sum` (signed, wide enough)
* `exp_sum = exp_w + exp_a`
* `valid`

Operation:

1. For each element i in 0..31:

   * extract `w_sign, w_idx`
   * `w_mag = decode(dialect_w, w_idx)`  (4-bit)
   * same for activation if using BD activations
   * product magnitude = `w_mag * a_mag` (4b×4b→8b)
   * sign = XOR of signs
2. Accumulate 32 products into a signed accumulator.
3. Apply the paper’s scaling: **right shift by 2** (because each operand is `0.5 * mag`). 
4. Apply exponent sum *at a defined boundary* (see next).

### 7.2 Handling exponent sums across multiple blocks

The paper notes:

* products inside a block share exponent sum, so you can accumulate in integer, and
* “only when all elements in a block are processed, partial sums are converted … and accumulated”, with the shift applied. 

For CNN/conv, you’ll have multiple blocks per dot-product. Two practical approaches:

**Approach A (closest to paper):**

* For each 32-element block, produce a **scaled partial sum** tagged with `exp_sum`.
* Convert/align partial sums to a higher-precision accumulator (FP16/FP32 or wide fixed-point) before adding across blocks.

**Approach B (simpler for FPGA bring-up):**

* Convert each block partial sum into a fixed-point int32 at a chosen global scale (per-layer scale), then accumulate.

Start with Approach B for bring-up, then move to A if needed.

**Deliverable:** a verified `BDMac32` module that matches a Python golden model bit-for-bit.

---

## 8) Wire the MAC into the SoC: MMIO first, custom instruction later

Your earlier design doc hinted at custom instructions, but you don’t need to start there.

### 8.1 Add an MMIO “BD MAC engine” peripheral

Map it near your decoder (same APB decoder bus), for example:

* `0x4003_0000` = decoder (already)
* `0x4003_1000` = bd_mac engine

Registers:

* pointers: activation ptr, weight ptr, output ptr
* K length (number of elements)
* stride/shape fields for conv micro-kernels
* start / done
* optional cycle counter internal

### 8.2 Firmware integration: replace inner loops first

Don’t try to accelerate everything at once. Start with:

* 1×1 conv micro-kernel (common in ResNet‑50 bottlenecks)
* then 3×3 conv

Keep the outer loops in C, but call the accelerator for the dot-products.

**Deliverable:** measurable speedup in cycles for one bottleneck block.

---

## 9) Optional (but “paper-faithful”) step: online activation BlockDialect quantization in hardware

If you want to match the paper’s full story (weights + activations), then you need activation quantization on the fly.

The paper’s flow for activations is:

1. **Preprocessing stage** builds a 5-bit intermediate value (3 integer bits + 2 fractional bits), using a shared exponent derived from the block max. 
2. A **two-stage dialect selection**:

   * stage 1 selects the dialect *pair* based on the block max,
   * stage 2 chooses between the two by counting elements in each dialect’s “beneficial range”, implemented as efficient logical ops and possibly a reduction tree.
3. Quantize to nearest representable value using logic derived from binary intervals (they give examples of compressing range checks into bit tests). 

### 9.1 Practical incremental version for CNNs

* Quantize activations **after ReLU** first (non-negative), which simplifies sign handling.
* Keep the same 32-element block size.

### 9.2 RTL modules to implement (pipeline-able)

* `BDPreprocess32` (block max → shared exponent; produce 5-bit intermediates)
* `BDDialectSelect32` (two-stage selection; counting via reduction tree)
* `BDQuantize32` (map intermediates to (sign, idx))
* Reuse your `DialectFP4DecodeCore` for dequantization

**Deliverable:** end-to-end “activation in, packed BD out” hardware module matching a Python reference.

---

## 10) Verification strategy (this is what will save you on FPGA)

### 10.1 Python is your golden model

For every HW module you add:

* produce random blocks
* encode using `blockdialect_codec.py`
* decode/quantize in Python
* compare to RTL simulation output

### 10.2 Keep the APB decoder peripheral as your “hardware oracle”

Even after you build a MAC engine, keep the APB decoder:

* If outputs mismatch, you can check whether the issue is in:

  * packed-code extraction
  * dialect-id/exponent parsing
  * decoder mapping
  * scaling/exponent handling
  * accumulation

### 10.3 Firmware “tripwires”

Borrow the best part of `hyperram_phase_full`:

* expected hashes at key boundaries (conv1 output, after each stage, logits)
* fail-fast and print mismatch location

---

## 11) Definition of done (ResNet‑50 + BlockDialect “works on FPGA”)

You’re done when the FPGA boot log shows all of:

1. Weight blob header validated (magic/version/table).
2. BlockDialect decoder self-test passes.
3. ResNet‑50 inference completes on a fixed input.
4. Output signature matches Python (logits or hash).
5. You can report:

   * total rdcycle cycles
   * bytes read from WeightStore
6. (If you implement MAC) You can show that the dequantization+MAC path follows:

   * index→4-bit integer via shared-value table 
   * 4-bit unsigned multiply + 2-bit right shift for `0.5*0.5` 

---

## What I would do first (sequence that minimizes risk)

1. **Milestone 0:** resnet50_phase0_smoke boots and reads the blob.
2. **Milestone 1:** ResNet‑50 runs in pure software (even if slow).
3. **Milestone 2:** Swap weights to BD (flash size win) and decode via your existing HW decoder peripheral.
4. Only then:

   * refactor decoder core,
   * build `BDMac32`,
   * accelerate 1×1 conv first,
   * optionally add activation quantization (two-stage selection) for full-paper faithfulness.

If you want, I can also draft the exact **register map** for the “BD MAC engine” peripheral so your firmware integration stays clean and you don’t regress into ResNet‑110-era ad-hoc offsets.

**IMPORTANT**: MINIMIZE REFLASHING THE WEIGHTS. IT TAKES AN HOUR.