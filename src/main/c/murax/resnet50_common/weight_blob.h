/* weight_blob.h — VWB2 indexed BlockDialect weight blob parser
 *
 * AUTO-GENERATED specification: see scripts/VWB2_SPEC.md
 * Python writer/reader     : scripts/blockdialect_codec.py (write_weight_blob_v2)
 * Export script            : scripts/gen_resnet50_model.py
 *
 * Usage:
 *   #include "weight_blob.h"
 *
 *   const vwb2_header_t *hdr = (const vwb2_header_t *)WEIGHT_BLOB_ADDR;
 *   if (vwb2_verify_header(hdr) != WB_OK) { uart_puts("bad blob\n"); }
 *
 *   const vwb2_entry_t  *e   = vwb2_find_tensor(hdr, "conv1.weight");
 *   const uint8_t       *ptr = vwb2_tensor_data(hdr, e);
 *   // ptr now points to the BD4 encode_tensor payload for conv1.weight
 *
 * All structures are packed; the blob MUST be 4-byte aligned in memory.
 * For iCEBreaker/MuraxHyperRAM the WeightStore window is already word-aligned.
 */
#ifndef WEIGHT_BLOB_H
#define WEIGHT_BLOB_H

#include <stdint.h>

/* -------------------------------------------------------------------------
 * Magic and version
 * ---------------------------------------------------------------------- */
#define VWB2_MAGIC      0x56574232u  /* 'VWB2' little-endian */
#define VWB2_VERSION    1u
#define VWB2_BLOCK_SIZE 32u

/* -------------------------------------------------------------------------
 * Dtype constants
 * ---------------------------------------------------------------------- */
#define VWB2_DTYPE_BD4     0u  /* BlockDialect FP4 (encode_tensor format) */
#define VWB2_DTYPE_FLOAT32 1u  /* Raw float32 array (biases)               */

/* -------------------------------------------------------------------------
 * Error codes
 * ---------------------------------------------------------------------- */
typedef enum {
    WB_OK           = 0,
    WB_ERR_MAGIC    = 1,  /* bad magic word    */
    WB_ERR_VERSION  = 2,  /* unsupported version */
    WB_ERR_BLKSIZE  = 3,  /* block_size != 32  */
    WB_ERR_NOT_FOUND = 4, /* tensor hash not in table */
} wb_err_t;

/* -------------------------------------------------------------------------
 * File header — 32 bytes, all fields little-endian u32
 * ---------------------------------------------------------------------- */
typedef struct {
    uint32_t magic;         /* 0x56574232 = 'VWB2'                       */
    uint32_t version;       /* 1                                          */
    uint32_t tensor_count;  /* number of entries in the tensor table      */
    uint32_t block_size;    /* BlockDialect block size (always 32)        */
    uint32_t table_offset;  /* byte offset of tensor table from file start */
    uint32_t data_offset;   /* byte offset of payload section             */
    uint32_t data_bytes;    /* total bytes of payload section             */
    uint32_t reserved;      /* 0                                          */
} vwb2_header_t;            /* sizeof = 32 bytes                          */

/* -------------------------------------------------------------------------
 * Tensor table entry — 40 bytes, all fields little-endian u32
 * ---------------------------------------------------------------------- */
typedef struct {
    uint32_t name_hash;      /* FNV-1a 32-bit hash of tensor name (UTF-8) */
    uint32_t dtype;          /* VWB2_DTYPE_BD4 or VWB2_DTYPE_FLOAT32      */
    uint32_t tensor_offset;  /* byte offset of payload from data_offset   */
    uint32_t tensor_bytes;   /* byte count of this tensor's payload        */
    uint32_t n_elements;     /* logical element count (before padding)     */
    uint32_t shape_ndim;     /* valid entries in shape[] (0..4)            */
    uint32_t shape[4];       /* dimensions; unused dims are 0              */
} vwb2_entry_t;             /* sizeof = 40 bytes                          */

/* -------------------------------------------------------------------------
 * BD4 per-tensor sub-header (first 8 bytes of a DTYPE_BD4 payload)
 * ---------------------------------------------------------------------- */
typedef struct {
    uint32_t n_elements;   /* element count (LE)                          */
    uint32_t n_blocks;     /* block count  (LE)                           */
    /* Followed by n_blocks × 18-byte block records (see VWB2_SPEC.md)   */
} vwb2_bd4_hdr_t;          /* sizeof = 8 bytes                            */

/* -------------------------------------------------------------------------
 * BD4 block record — 18 bytes
 * Layout:
 *   [0:2]  meta (big-endian u16):
 *              bits 15..12 : dialect_id  (0..15)
 *              bits 11..7  : shared_exp  (0..31, FP16 exponent bits)
 *              bits  6..0  : 0 (reserved)
 *   [2:18] packed_codes: 32 × 4-bit codes, two per byte
 *              byte i = (code[2i] << 4) | code[2i+1]
 *              code   = (sign_bit << 3) | idx_3bit
 * ---------------------------------------------------------------------- */
#define VWB2_BLOCK_BYTES 18u  /* 2-byte meta + 16-byte packed codes */

/* Extract fields from a big-endian 2-byte meta word.
 * Pass the two raw bytes (meta_hi, meta_lo). */
#define VWB2_META_DIALECT(hi, lo)   (((hi) >> 4) & 0x0Fu)
#define VWB2_META_EXP(hi, lo)       ((((hi) & 0x0Fu) << 1) | (((lo) >> 7) & 0x01u))
#define VWB2_CODE_SIGN(code)        (((code) >> 3) & 0x01u)
#define VWB2_CODE_IDX(code)         ((code) & 0x07u)

/* -------------------------------------------------------------------------
 * FNV-1a 32-bit hash  (must match blockdialect_codec.fnv1a32)
 * ---------------------------------------------------------------------- */
static inline uint32_t vwb2_fnv1a32(const char *s) {
    uint32_t h = 0x811C9DC5u;
    while (*s) {
        h ^= (uint8_t)(*s++);
        h *= 0x01000193u;
    }
    return h;
}

/* -------------------------------------------------------------------------
 * Verify file header.  Returns WB_OK or an error code.
 * ---------------------------------------------------------------------- */
static inline wb_err_t vwb2_verify_header(const vwb2_header_t *hdr) {
    if (hdr->magic      != VWB2_MAGIC)      return WB_ERR_MAGIC;
    if (hdr->version    != VWB2_VERSION)    return WB_ERR_VERSION;
    if (hdr->block_size != VWB2_BLOCK_SIZE) return WB_ERR_BLKSIZE;
    return WB_OK;
}

/* -------------------------------------------------------------------------
 * Return a pointer to the tensor table (first entry).
 * ---------------------------------------------------------------------- */
static inline const vwb2_entry_t *vwb2_table(const vwb2_header_t *hdr) {
    return (const vwb2_entry_t *)((const uint8_t *)hdr + hdr->table_offset);
}

/* -------------------------------------------------------------------------
 * Find a tensor entry by name.  Returns NULL if not found.
 * ---------------------------------------------------------------------- */
static inline const vwb2_entry_t *vwb2_find_tensor(const vwb2_header_t *hdr,
                                                    const char *name) {
    uint32_t hash = vwb2_fnv1a32(name);
    const vwb2_entry_t *tbl = vwb2_table(hdr);
    uint32_t i;
    for (i = 0; i < hdr->tensor_count; i++) {
        if (tbl[i].name_hash == hash)
            return &tbl[i];
    }
    return (const vwb2_entry_t *)0;
}

/* -------------------------------------------------------------------------
 * Get a pointer to the raw payload bytes for a tensor entry.
 * ---------------------------------------------------------------------- */
static inline const uint8_t *vwb2_tensor_data(const vwb2_header_t *hdr,
                                               const vwb2_entry_t *entry) {
    return (const uint8_t *)hdr + hdr->data_offset + entry->tensor_offset;
}

/* -------------------------------------------------------------------------
 * Convenience: get a pointer to the BD4 sub-header for a DTYPE_BD4 tensor.
 * ---------------------------------------------------------------------- */
static inline const vwb2_bd4_hdr_t *vwb2_bd4_header(const vwb2_header_t *hdr,
                                                      const vwb2_entry_t *entry) {
    return (const vwb2_bd4_hdr_t *)vwb2_tensor_data(hdr, entry);
}

/* -------------------------------------------------------------------------
 * Convenience: get a pointer to the first BD4 block record (skips 8-byte hdr).
 * ---------------------------------------------------------------------- */
static inline const uint8_t *vwb2_bd4_blocks(const vwb2_header_t *hdr,
                                              const vwb2_entry_t *entry) {
    return vwb2_tensor_data(hdr, entry) + 8u;
}

/* -------------------------------------------------------------------------
 * Convenience: get a pointer to a float32 array for a DTYPE_FLOAT32 tensor.
 * ---------------------------------------------------------------------- */
static inline const float *vwb2_float32_data(const vwb2_header_t *hdr,
                                              const vwb2_entry_t *entry) {
    return (const float *)vwb2_tensor_data(hdr, entry);
}

#endif /* WEIGHT_BLOB_H */
