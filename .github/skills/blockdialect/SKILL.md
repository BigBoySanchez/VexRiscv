---
name: blockdialect
description: >
  BlockDialect / DialectFP4 quantization format — paper theory, binary encoding,
  dialect table, two-stage activation selector, MAC arithmetic, VWB2 blob format,
  Python codec, C firmware decoder, SpinalHDL RTL decoder and BDMac32 peripheral.
  Use when encoding/decoding BD4 blocks, writing or reading weight blobs, modifying
  the DialectFP4DecodeCore or BDMac32 hardware, implementing activation packing,
  or debugging quantization mismatches.
---

# BlockDialect — Complete Reference

Paper: Jang & Tambe, "BlockDialect: Block-wise Fine-grained Mixed Format
Quantization for Energy-Efficient LLM Inference", ICML 2025 (arXiv:2501.01144v5).

---

## 1. Core Concept

BlockDialect is a **post-training, block-wise, mixed-format** 4-bit quantization
scheme.  A tensor is split into contiguous 1-D blocks of 32 elements.  Each block
gets its own:

- **shared exponent** (5 bits) — a power-of-two scaling factor (the FP16 exponent
  field of the block's max magnitude, adjusted by −2).
- **dialect_id** (4 bits) — selects one of 16 DialectFP4 number formats from a
  fixed "formatbook".

Each element is then encoded as a 4-bit code: 1-bit sign + 3-bit index into the
chosen dialect's 8 representable unsigned magnitudes.

### Key insight (paper §1)

> "If a group of numbers deserves its own scaling factor, why not a number format?"

Instead of one FP4 format with a shared exponent (MXFP4), BlockDialect assigns a
**per-block optimal format variant** ("dialect") that better captures the block's
magnitude distribution — especially where the block max doesn't align well with the
fixed FP4 E2M1 representable values.

---

## 2. The DialectFP4 Formatbook (Paper Figure 4)

All 16 dialects share **six common magnitudes** (indices 0–5) and differ only in
the two **largest** magnitudes (indices 6 and 7).  Values are multiples of 0.5,
stored as unsigned integer "half-units" (0–15), so `real_magnitude = 0.5 × half_units`.

```
Dialect  idx0  idx1  idx2  idx3  idx4  idx5  idx6  idx7   (half-units)
───────  ────  ────  ────  ────  ────  ────  ────  ────
  0       0     1     2     3     4     6     11    15     ← maxhu=15 (mag 7.5)
  1       0     1     2     3     4     6      9    15     ← maxhu=15 (mag 7.5)
  2       0     1     2     3     4     6     11    14     ← maxhu=14 (mag 7.0)
  3       0     1     2     3     4     6      9    14
  4       0     1     2     3     4     6     10    13     ← maxhu=13 (mag 6.5)
  5       0     1     2     3     4     6      8    13
  6       0     1     2     3     4     6     10    12     ← maxhu=12 (mag 6.0)
  7       0     1     2     3     4     6      8    12
  8       0     1     2     3     4     6      9    11     ← maxhu=11 (mag 5.5)
  9       0     1     2     3     4     6      7    11
 10       0     1     2     3     4     6      9    10     ← maxhu=10 (mag 5.0)
 11       0     1     2     3     4     6      7    10
 12       0     1     2     3     4     6      8     9     ← maxhu=9  (mag 4.5)
 13       0     1     2     3     4     6      7     9
 14       0     1     2     3     4     6      7     8     ← maxhu=8  (mag 4.0)
 15       0     1     2     3     4     5      6     8     ← SPECIAL: idx5=5 not 6
```

### Structural patterns

- **Dialects form 8 pairs** sharing the same `maxhu` (index 7): (0,1), (2,3), …, (14,15).
  Within each pair, the even dialect ("A") has a **larger** idx6 value than the odd
  dialect ("B").

- **Indices 0–4 are identical** across all 16 dialects: 0, 1, 2, 3, 4.

- **Index 5 = 6 for dialects 0–14**, and = 5 for dialect 15.

- **maxhu = 15 − floor(dialect_id / 2)** for all 16 dialects.
  (Pair 0: maxhu=15, pair 1: maxhu=14, …, pair 7: maxhu=8.)

- **Dialect 15 is special**: it's the only dialect where idx5 ≠ 6.  Its magnitudes
  in real values are [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0] — the most compressed
  dynamic range.

### Three design criteria (paper §3.1)

1. **Minimize wasted/underestimated range** — dialects with different max magnitudes
   cover blocks whose max doesn't align with FP4's max of 6.
2. **Prioritize representation of larger magnitudes** — larger values contribute more
   after multiplication; pairs differ only in their second-largest value.
3. **Hardware efficiency** — 0.5 granularity means all magnitudes are integers in
   half-unit space, compatible with 4-bit unsigned integer arithmetic.

---

## 3. Block Binary Format — 18 Bytes

Each BD4 block encodes 32 elements in exactly 18 bytes:

```
Byte offset  Content
──────────────────────────────────────────────────────────
[0:2]        metadata — big-endian uint16:
               bits 15..12 : dialect_id  (0..15)
               bits 11..7  : shared_exp  (0..31, FP16 exponent bits)
               bits  6..0  : reserved (0)

[2:18]       packed_codes — 16 bytes = 32 × 4-bit codes:
               byte i = (code[2i] << 4) | code[2i+1]
               each code = (sign << 3) | idx
                 sign : 1 = negative, 0 = positive
                 idx  : 3-bit index into the dialect's 8 magnitudes
```

### Effective bitwidth

Per element: 4 bits (sign + idx).
Per block overhead: 9 bits (4-bit dialect_id + 5-bit shared_exp).
Effective bits/element = 4 + 9/32 = **4.28 bits** (paper theoretical).

Our implementation pads the 9-bit metadata to 16 bits (2 bytes) for byte alignment,
giving 4 + 16/32 = **4.5 bits/element** — 5.1% larger than theoretical.

### Storage formula

```
storage_bytes = ceil(n_elements / 32) × 18
```

C helper: `bd_act_storage_bytes(n)` in `bd_act.h`.
Python: `blockdialect_codec.py::encode_tensor()`.

---

## 4. Decoding Formula

Given a 4-bit code and per-block metadata:

```
sign       = code[3]           (1 bit)
idx        = code[2:0]         (3 bits)
half_units = DIALECT_TABLE[dialect_id][idx]   (0..15, unsigned)
e          = shared_exp_bits - 15             (unbiased FP16 exponent)
value      = (-1)^sign × 0.5 × half_units × 2^e
```

Equivalently in integer/fixed-point (no FP hardware needed):

```
value_fixedpt = (-1)^sign × half_units × 2^(shared_exp_bits - 16)
```

The `−16` combines the FP16 bias (−15) and the 0.5 factor (−1).

---

## 5. Encoding — Weight Path (Offline, MSE-Based)

For weights, all data is known at export time.  Per block:

1. **Compute shared exponent** — `shared_exp_bits = FP16_exp(max|block|) − 2`,
   clamped to [0, 31].  This maps the block max into ~[4, 8) in half-unit space.

2. **Scale to half-units** — `scaled_hu[i] = round(|x[i]| × 2^(1 − e))` where
   `e = shared_exp_bits − 15`.  Clamp to [0, 15].

3. **Select dialect (brute-force MSE)** — For each of the 16 dialects, quantize
   every element to its nearest representable value, compute MSE.  Pick the dialect
   with the lowest MSE.

4. **Quantize** — For each element, find the nearest index in the chosen dialect.
   Set `code = (sign << 3) | nearest_idx`.

5. **Pack** — Build the 18-byte block record.

Implementation:
- Python: `blockdialect_codec.py::encode_block()`
- C: `bd_act.h::bd_act_pack32()` (uses same MSE approach for activations)

---

## 6. Encoding — Activation Path (Online, Two-Stage Selection)

For activations, dialect selection must be done in real-time.  The paper's two-stage
algorithm (§3.2) runs in O(N) instead of O(16N):

### Preprocessing

Compute shared exponent and scale all 32 elements to 5-bit intermediate values
(3-bit integer + 2-bit fractional, range [0.0, 7.5] with 0.25 granularity).
The extra fractional bit enables accurate rounding.

### Stage 1 — Pair selection from block max

Round the block's max half-unit value to determine which dialect **pair** to use:

```
block_maxhu = max(scaled_hu[0..31])

 block_maxhu ≥ 15 → pair 0 (dialects 0, 1)    maxhu = 15
 block_maxhu ≥ 14 → pair 1 (dialects 2, 3)    maxhu = 14
 block_maxhu ≥ 13 → pair 2 (dialects 4, 5)    maxhu = 13
 block_maxhu ≥ 12 → pair 3 (dialects 6, 7)    maxhu = 12
 block_maxhu ≥ 11 → pair 4 (dialects 8, 9)    maxhu = 11
 block_maxhu ≥ 10 → pair 5 (dialects 10, 11)  maxhu = 10
 block_maxhu ≥  9 → pair 6 (dialects 12, 13)  maxhu =  9
 otherwise        → pair 7 (dialects 14, 15)  maxhu =  8
```

This ensures the chosen dialect's max representable value matches the block's
actual range, avoiding wasted or underestimated dynamic range.

### Stage 2 — Choose A or B within the pair

The two dialects in a pair differ only at **index 6** (the second-largest value).
Dialect A (even) has a larger idx6 than dialect B (odd).  The "beneficial range"
for dialect A is the interval where A's idx6 value would produce less quantization
error than B's:

```
beneficial_lo = (A.idx6 + B.idx6) / 2      (midpoint of the two differing values)
beneficial_hi = (A.idx6 + maxhu) / 2        (midpoint between A.idx6 and maxhu)
```

Count how many of the 32 elements fall in [beneficial_lo, beneficial_hi).  If the
count is ≥ 16 (majority), select dialect A; otherwise select dialect B.

To avoid fractional arithmetic in hardware, the codec uses doubled bounds:

```python
# From blockdialect_codec.py
_BENEFICIAL_LO_X2 = [20, 20, 18, 18, 16, 16, 15, 13]
_BENEFICIAL_HI_X2 = [26, 25, 23, 22, 20, 19, 17, 15]

# Check: lo_x2 <= 2*scaled_hu < hi_x2
count_a = sum(lo_x2 <= 2*hu < hi_x2 for hu in scaled_hu)
dialect = pair*2 if count_a*2 >= 32 else pair*2 + 1
```

### Paper hardware cost (Table 7)

The two-stage selector synthesizes to:
- 5 clock cycles, 0.7 mW, 42,834 µm² (130nm)
- vs. MSE-based: 8 cycles, 6.9 mW, 399,409 µm² (9.3× larger, 9.9× more power)

Implementation: `blockdialect_codec.py::_select_dialect_twostage()`,
`blockdialect_codec.py::encode_block_twostage()`.

---

## 7. MAC Arithmetic (Paper §3.3)

Because all magnitudes are multiples of 0.5, they can be represented as 4-bit
unsigned integers (half-units 0–15).  Multiplication becomes:

```
product_halfunits² = w_hu × a_hu     (4-bit × 4-bit unsigned → 8-bit, max 15×15=225)
product_sign       = w_sign XOR a_sign
```

For a 32-element dot product:

```
partial_sum = Σ (±1)^(w_sign[i] XOR a_sign[i]) × w_hu[i] × a_hu[i]
```

This is a signed 24-bit accumulator (max magnitude: 225 × 32 = 7,200, fits in 13 bits,
but 24 bits provides headroom for multi-block accumulation).

### Shared-exponent scaling (applied once per block, not per element)

```
scaled_result = partial_sum × 2^(w_shared_exp + a_shared_exp)
scaled_result >>= 2    // compensate for 0.5 × 0.5 = 0.25 per product
```

Or equivalently:

```
scaled_result = partial_sum × 2^(w_exp + a_exp − 2)
```

The `−2` accounts for each operand being in half-units (factor of 0.5 each).

### Paper MAC unit comparison (Table 5, 45nm @ 0.5 GHz)

| Type     | Area (µm²) | Power (µW) |
|----------|-------------|------------|
| INT4     | 207         | 104        |
| FP4      | 247         | 129        |
| **Ours** | **248**     | **135**    |
| FP6      | 382         | 214        |
| INT8     | 554         | 331        |

BlockDialect MAC unit has FP4-like efficiency while achieving significantly better
quantization accuracy.

---

## 8. Implementation Inventory

### Python codec: `scripts/blockdialect_codec.py`

Canonical reference implementation.  Key functions:

| Function | Purpose |
|----------|---------|
| `encode_block(block)` | Encode 32 floats → (dialect_id, shared_exp, codes[32]) via MSE |
| `encode_block_twostage(block)` | Same but uses two-stage selector (for activations) |
| `decode_block(dialect_id, shared_exp, codes)` | Decode to float32[32] |
| `pack_block(d, e, codes)` → 18 bytes | Serialize one block |
| `unpack_block(data)` → (d, e, codes) | Deserialize one block |
| `encode_tensor(tensor)` → bytes | Encode full tensor (with 8-byte header) |
| `decode_tensor(data)` → float32[] | Decode full tensor |
| `write_weight_blob_v2(tensors, path)` | Write VWB2 indexed blob |
| `read_weight_blob_v2(path)` | Read VWB2 blob |
| `fnv1a32(name)` | FNV-1a hash for tensor lookup |

Constants: `BLOCK_SIZE = 32`, `FP16_EXP_BIAS = 15`, `DIALECTS_HALF_UNITS[16][8]`.

### C firmware headers

**`bd_decode_sw.h`** — Software BD4 block decoder (topology-agnostic, reusable).
- `BD_DIALECT_TABLE[16][8]` — same table as Python
- `bd_decode_block_hu(block, out_hu, &did, &seb)` — decode 18 bytes → 32 signed half-units
- `bd_decode_block_fixedpt(block, out_i32, frac_bits)` — decode with exponent scaling
- `bd_decode_tensor_to_i8(blocks, n, out, n_elem, frac)` — full tensor decode to int8
- `bd_dot_hu_i8(w_hu, act, count)` — dot product of half-units × int8

**`bd_act.h`** — Activation BD4 pack/unpack (topology-agnostic, reusable).
- `bd_act_compute_exp(max_abs)` → shared_exp_bits
- `bd_act_scale_hu(abs_val, seb)` → scaled half-unit (0..15)
- `bd_act_nearest_idx(target_hu, dialect_id)` → 3-bit index
- `bd_act_pack32(vals[32], block_out[18])` — pack 32 int values to BD4 block
- `bd_act_unpack32(block[18], out_i8[32])` — unpack BD4 block to int8
- `bd_act_pack_tensor(input, n, bd_out)` — pack full tensor
- `bd_act_unpack_tensor(blocks, n_blocks, out, n_elem)` — unpack full tensor
- `bd_act_storage_bytes(n)` — compute storage: `ceil(n/32) × 18`

**`weight_blob.h`** — VWB2 blob parser for firmware (topology-agnostic).
- `vwb2_verify_header(hdr)`, `vwb2_find_tensor(hdr, name)`, `vwb2_tensor_data(hdr, entry)`
- FNV-1a: `fnv1a32(s)` — matches Python codec exactly

### SpinalHDL RTL (Scala → Verilog)

**`DialectFP4DecodeCore.scala`** — Pure combinational lookup.
- Inputs: `dialect_id[3:0]`, `idx[2:0]`
- Output: `mag[3:0]` (unsigned half-units)
- Logic: idx 0–4 → identity; idx 5 → 6 (or 5 for dialect 15); idx 6 → variant table;
  idx 7 → `15 − floor(dialect_id/2)`
- Variant table for idx 6: `[11, 9, 11, 9, 10, 8, 10, 8, 9, 7, 9, 7, 8, 7, 7]`

**`BlockDialectDecoderAPB.scala`** — 32-lane APB3 weight decoder.
- Instantiates 32 `DialectFP4DecodeCore` instances (combinational).
- Register map (base e.g. `0x40030000`):
  - `0x00` META (W): dialect_id[15:12] | shared_exp[11:7]
  - `0x04–0x10` PACKED0–3 (W): 16 bytes of packed codes
  - `0x20–0x3C` DECODED0–7 (R): 32 decoded int8 half-units (4/word, LE)
  - `0x40` STATUS (R): always 1
  - `0x44` SHARED_EXP (R): shared_exp_bits[4:0]

**`BlockDialectDecoder.scala`** — Compatibility shim extending `BlockDialectDecoderAPB`.

**`BDMac32.scala`** — Sequential 32-element fused MAC peripheral.
- **Sequential** design: processes 1 element/cycle, 32 cycles total.
  Uses 2 `DialectFP4DecodeCore` instances + 1 multiplier + 24-bit accumulator.
  Fits within iCE40UP5K LC budget (~5280 LCs).
- Register map (base `0x40031000`):

  | Offset | Name | Dir | Description |
  |--------|------|-----|-------------|
  | 0x00 | W_PACKED0 | W | w_packed[31:0] — elements 0..7 |
  | 0x04 | W_PACKED1 | W | w_packed[63:32] — elements 8..15 |
  | 0x08 | W_PACKED2 | W | w_packed[95:64] — elements 16..23 |
  | 0x0C | W_PACKED3 | W | w_packed[127:96] — elements 24..31 |
  | 0x10 | W_META | W | dialect_id[15:12] \| shared_exp[11:7] |
  | 0x14 | A_PACKED0 | W | a_packed[31:0] |
  | 0x18 | A_PACKED1 | W | a_packed[63:32] |
  | 0x1C | A_PACKED2 | W | a_packed[95:64] |
  | 0x20 | A_PACKED3 | W | a_packed[127:96] |
  | 0x24 | A_META | W | dialect_id[15:12] \| shared_exp[11:7] |
  | 0x28 | CTRL | W | Write 1 → start; clears DONE |
  | 0x30 | PARTIAL_SUM | R | Signed 32-bit result (half-units²) |
  | 0x34 | EXP_SUM | R | w_shared_exp + a_shared_exp [5:0] |
  | 0x38 | DONE | R | 0 = computing, 1 = result valid |

- Firmware scaling after reading PARTIAL_SUM:
  ```c
  int64_t scaled = (int64_t)partial_sum << exp_sum;
  scaled >>= 2;  // compensate 0.5 × 0.5 = 0.25
  ```

- Inner computation per element:
  ```
  product_mag  = DialectFP4DecodeCore(w_dialect, w_idx) × DialectFP4DecodeCore(a_dialect, a_idx)
  product_sign = w_sign XOR a_sign
  accum       += product_sign ? −product_mag : +product_mag
  ```

### Tests: `scripts/test_blockdialect.py`

Covers: dialect table correctness, encode/decode round-trip, VWB2 read/write,
two-stage vs MSE dialect selection agreement, BDMac32 golden vectors.

---

## 9. VWB2 Blob Format Summary

VWB2 is the indexed weight blob format for FPGA deployment.

```
Offset   Content
────────────────────────────────
0        Header (32 bytes)
           magic     = 0x56574232 ('VWB2')
           version   = 1
           tensor_count, block_size=32
           table_offset=32, data_offset, data_bytes
32       Tensor table (tensor_count × 40 bytes each)
           name_hash (FNV-1a 32), dtype (0=BD4, 1=F32)
           tensor_offset, tensor_bytes, n_elements
           shape_ndim, shape[0..3]
aligned  Payload section
           BD4 tensors: 8-byte sub-header + N×18-byte blocks
           F32 tensors: raw float32 bytes (for biases)
```

Firmware looks up tensors by FNV-1a hash of their name string.
See `scripts/VWB2_SPEC.md` for the full specification.

---

## 10. Shared Exponent Computation Details

The shared exponent maps a block's dynamic range so scaled values fall in [0, ~15]
half-units (i.e. [0, 7.5] real magnitudes):

**Paper method (§3.2):**
```
shared_exp_bits = floor(log2(block_max_abs)) − 2
```
Then scale: `scaled = |x| / (0.5 × 2^(shared_exp_bits − 15))`
         = `|x| × 2^(16 − shared_exp_bits)`

**Python implementation** (`blockdialect_codec.py`):
```python
emax_bits = FP16_exponent_bits(max_abs)   # 0..31
shared_exp_bits = clamp(emax_bits - 2, 0, 31)
scaled_hu = round(ldexp(|x|, 1 - e))     # where e = shared_exp_bits - 15
```

**C firmware** (`bd_act.h::bd_act_compute_exp`):
```c
// For integer activations:
int log2_val = floor(log2(max_abs));        // via shift loop
int seb = log2_val + 12;                    // +12 ≈ −2 + FP16_BIAS(15) − 1
return clamp(seb, 0, 31);
```

The `+12` constant is `FP16_BIAS(15) − 2 − 1 = 12`, where the extra `−1` accounts
for the integer→half-unit mapping difference vs the float path.

---

## 11. Packed Code Byte Layout

Codes are packed two per byte, high nibble first:

```
byte[i] = (code[2*i] << 4) | (code[2*i + 1] & 0x0F)
```

So the 16 packed bytes encode elements 0–31:
```
byte 0:  elem 0 (high nibble) | elem 1  (low nibble)
byte 1:  elem 2 (high nibble) | elem 3  (low nibble)
...
byte 15: elem 30 (high nibble) | elem 31 (low nibble)
```

Each code's 4 bits: `sign[3] | idx[2:0]`.

---

## 12. Critical Numeric Details for Correctness

### Metadata encoding (big-endian uint16)

```
bit 15 14 13 12 | 11 10  9  8  7 |  6  5  4  3  2  1  0
    ─── dialect ─  ───── exp ────    ──── reserved (0) ──
```

Extraction in C:
```c
uint8_t dialect_id = (byte0 >> 4) & 0x0F;
uint8_t shared_exp = ((byte0 & 0x0F) << 1) | ((byte1 >> 7) & 0x01);
```

### FP16 exponent bias

Always 15.  The unbiased exponent: `e = shared_exp_bits − 15`.

### Half-unit to real conversion

`real_value = 0.5 × half_units × 2^e = half_units × 2^(e − 1)`

### Saturating decode to int8

```c
int shift = shared_exp_bits - 16;
int32_t v = (shift >= 0) ? (hu << shift) : (hu >> (-shift));
out = clamp(v, -128, 127);
```

### Zero handling

A block of all zeros gets `shared_exp_bits = 0`, all codes = `0x00` (sign=0, idx=0,
half_units=0).  Decodes to zero regardless of dialect.

---

## 13. Practical Numbers

| Metric | Value |
|--------|-------|
| Block size | 32 elements |
| Block storage | 18 bytes (2 meta + 16 codes) |
| Effective bits/element (theoretical) | 4.28 |
| Effective bits/element (padded meta) | 4.50 |
| Compression vs int8 | 1.78× |
| Compression vs float32 | 7.11× |
| Number of dialects | 16 |
| Dialect ID bits | 4 |
| Shared exponent bits | 5 |
| Code bits per element | 4 (1 sign + 3 index) |
| Max representable magnitude | 7.5 (half-units = 15) |
| Min nonzero magnitude | 0.5 (half-units = 1) |
| Granularity | 0.5 |

---

## 14. File Locations

| File | Purpose |
|------|---------|
| `scripts/blockdialect_codec.py` | Python reference codec (encode/decode/VWB2) |
| `scripts/test_blockdialect.py` | Test suite (25 tests) |
| `scripts/VWB2_SPEC.md` | VWB2 binary format specification |
| `src/main/c/murax/resnet50_common/bd_decode_sw.h` | C software decoder |
| `src/main/c/murax/resnet50_common/bd_act.h` | C activation pack/unpack |
| `src/main/c/murax/resnet50_common/weight_blob.h` | C VWB2 blob parser |
| `src/main/c/murax/resnet1202_common/bd_decode_sw.h` | Copy of above for ResNet-1202 |
| `src/main/c/murax/resnet1202_common/bd_act.h` | Extended activation codec for ResNet-1202 |
| `src/main/scala/vexriscv/demo/DialectFP4DecodeCore.scala` | RTL: combinational lookup |
| `src/main/scala/vexriscv/demo/BlockDialectDecoderAPB.scala` | RTL: 32-lane APB decoder |
| `src/main/scala/vexriscv/demo/BlockDialectDecoder.scala` | RTL: compatibility shim |
| `src/main/scala/vexriscv/demo/BDMac32.scala` | RTL: sequential MAC peripheral |
| `BDMac32.v` | Generated Verilog for BDMac32 |
| `BlockDialectDecoder.v` | Generated Verilog for decoder |