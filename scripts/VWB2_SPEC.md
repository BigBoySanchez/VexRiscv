# VWB2 — Indexed BlockDialect Weight Blob Format

**Version 1** — March 2026

VWB2 is the weight-blob format for ResNet-50 on FPGA.  It extends the original
sequential VWB1 format with a **tensor index table** so firmware can seek
directly to any layer's weights or biases without scanning the entire blob.

See also: `blockdialect_codec.py` (canonical Python writer/reader),
`src/main/c/murax/resnet50_common/weight_blob.h` (C firmware parser).

---

## Motivation

The original VWB1 format concatenated all weight blobs sequentially with no
index.  Firmware had to know the byte offsets at compile time (hard-coded from
the export script) and could not recover gracefully if the model changed.

VWB2 stores a self-describing **tensor table** so:
- Firmware finds any tensor by its **FNV-1a 32-bit name hash** — no hard-coded
  offsets required at build time.
- Tool changes (re-quantizing one layer, reordering layers) do not silently
  corrupt firmware reads; the hash mismatch is caught at runtime.
- The table doubles as metadata: element count, shape, dtype, and byte range
  are all stored per-tensor.

---

## Binary Layout

All integers are **little-endian**.

### File Header — 32 bytes

| Offset | Size | Field          | Value / Notes                                             |
|--------|------|----------------|-----------------------------------------------------------|
| 0      | u32  | `magic`        | `0x56574232` = `'VWB2'` in ASCII (LE)                    |
| 4      | u32  | `version`      | `1`                                                       |
| 8      | u32  | `tensor_count` | Number of tensors in the table                            |
| 12     | u32  | `block_size`   | BlockDialect block size in elements (always 32)           |
| 16     | u32  | `table_offset` | Byte offset of tensor table from file start (always 32)   |
| 20     | u32  | `data_offset`  | Byte offset of payload section from file start            |
| 24     | u32  | `data_bytes`   | Total bytes of payload section                            |
| 28     | u32  | `reserved`     | 0                                                         |

`data_offset` is always aligned to a 16-byte boundary:

```
data_offset = align16(32 + tensor_count × 40)
```

### Tensor Table — `tensor_count × 40` bytes

Immediately follows the file header at `table_offset = 32`.

Each entry is 40 bytes:

| Offset | Size | Field            | Notes                                                    |
|--------|------|------------------|----------------------------------------------------------|
| 0      | u32  | `name_hash`      | FNV-1a 32-bit hash of tensor name string (UTF-8)        |
| 4      | u32  | `dtype`          | `0` = DTYPE_BD4, `1` = DTYPE_FLOAT32                    |
| 8      | u32  | `tensor_offset`  | Byte offset of payload from `data_offset`                |
| 12     | u32  | `tensor_bytes`   | Byte count of this tensor's payload (4-byte aligned)     |
| 16     | u32  | `n_elements`     | Number of logical elements (original, before padding)    |
| 20     | u32  | `shape_ndim`     | Number of valid shape dimensions (0 = scalar, max 4)     |
| 24     | u32  | `shape[0]`       | First dimension (0 if ndim < 1)                          |
| 28     | u32  | `shape[1]`       | Second dimension (0 if ndim < 2)                         |
| 32     | u32  | `shape[2]`       | Third dimension (0 if ndim < 3)                          |
| 36     | u32  | `shape[3]`       | Fourth dimension (0 if ndim < 4)                         |

### Payload Section

Starts at `data_offset`.  Tensors are packed back-to-back, each 4-byte aligned.

Each tensor's payload format depends on `dtype`:

#### DTYPE_BD4 = 0 — BlockDialect FP4

The `encode_tensor` binary format:

```
[0:4]   n_elements  u32 LE   (same as table entry n_elements)
[4:8]   n_blocks    u32 LE   (= ceil(n_elements / 32))
[8:]    n_blocks × 18-byte block records
```

Each 18-byte block record:

```
[0:2]   meta        big-endian u16:
            bits 15..12 : dialect_id (0..15)
            bits 11..7  : shared_exp (0..31, FP16 exponent bits)
            bits  6..0  : 0 (reserved)
[2:18]  packed_codes: 32 × 4-bit codes, two per byte
            byte i = (code[2i] << 4) | code[2i+1]
            code = (sign << 3) | idx_3bit
```

#### DTYPE_FLOAT32 = 1 — Raw float32

```
n_elements × 4 bytes (IEEE-754 single-precision, little-endian)
```

Used for biases (the paper never quantizes biases).

---

## Tensor Naming Convention

For ResNet-50, each layer `<name>` produces two tensor entries:

| Tensor name        | dtype        | Contents                        |
|--------------------|--------------|----------------------------------|
| `<name>.weight`    | DTYPE_BD4    | BN-folded conv/fc weights       |
| `<name>.bias`      | DTYPE_FLOAT32| BN-folded conv/fc biases        |

Example layer names (as produced by `gen_resnet50_model.py`):

```
conv1
layer1.0.conv1
layer1.0.conv2
layer1.0.conv3
layer1.0.downsample
layer1.1.conv1
...
layer4.2.conv3
fc
```

---

## FNV-1a 32-bit Hash

```c
static inline uint32_t fnv1a32(const char *s) {
    uint32_t h = 0x811C9DC5u;
    while (*s) {
        h ^= (uint8_t)*s++;
        h *= 0x01000193u;
    }
    return h;
}
```

The Python implementation in `blockdialect_codec.py::fnv1a32()` is identical.

---

## Firmware Integration Summary

```c
#include "weight_blob.h"

// 1. Map blob into address space
const vwb2_header_t *hdr = (const vwb2_header_t *)WEIGHT_BLOB_ADDR;
vwb2_verify_header(hdr);            // checks magic, version, block_size

// 2. Find a tensor by name
const vwb2_entry_t *e = vwb2_find_tensor(hdr, "conv1.weight");

// 3. Get a pointer to the tensor payload
const uint8_t *data = vwb2_tensor_data(hdr, e);

// 4. For DTYPE_BD4, parse blocks manually or via the APB decoder peripheral
```

See `weight_blob.h` for the complete C API.

---

## File size formula

```
file_bytes = 32                                      (header)
           + tensor_count × 40                       (table)
           + alignment padding (0–15 bytes)
           + Σ tensor_bytes[i]                       (payload)
```

For ResNet-50 BN-folded (53 layers × 2 tensors = 106 entries):

```
header    :       32 bytes
table     :  106 × 40 = 4240 bytes
padding   : 0..15 bytes (aligned to 16)
payload   : actual encoded tensor data
```

Total is dominated by the BD4 payload (~25 MiB for ResNet-50 weights).
See `scripts/resnet50_artifacts/weight_budget.txt` for the exact measurement.
