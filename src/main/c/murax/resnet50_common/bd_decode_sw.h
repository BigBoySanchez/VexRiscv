/* bd_decode_sw.h — Software BlockDialect DialectFP4 decoder
 *
 * Bit-exact match to:
 *   - scripts/blockdialect_codec.py  (DIALECTS_HALF_UNITS table)
 *   - BlockDialectDecoder.scala      (APB hardware decoder)
 *   - arXiv:2501.01144v5 Figure 4
 *
 * Decoding formula (paper Section 3.2):
 *   half_units = DIALECT_TABLE[dialect_id][idx]   (0..15)
 *   value = sign * (0.5 * half_units) * 2^(shared_exp_bits - 15)
 *
 * For integer-only compute (no FP):
 *   We decode to a fixed-point int16 representation:
 *     decoded_i16 = sign * half_units   (in half-unit scale)
 *   Then accumulate dot-products in int32 and apply the shared exponent
 *   and 0.5 factor at block boundaries.
 *
 * Usage:
 *   #include "bd_decode_sw.h"
 *
 *   // Decode one 18-byte block into 32 int16 values (half-unit scale)
 *   int16_t out[32];
 *   int     dialect_id, shared_exp;
 *   bd_decode_block_hu(block_ptr, out, &dialect_id, &shared_exp);
 *
 *   // Or decode to int32 with exponent scaling applied:
 *   int32_t out32[32];
 *   bd_decode_block_scaled(block_ptr, out32, 8);  // shift=8 for Q8 fixed-point
 */
#ifndef BD_DECODE_SW_H
#define BD_DECODE_SW_H

#include <stdint.h>

/* ── DialectFP4 formatbook: 16 dialects × 8 half-unit magnitudes ──────── */
/* half_units[d][idx]: real magnitude = 0.5 * half_units[d][idx]           */
/* Matches blockdialect_codec.py DIALECTS_HALF_UNITS exactly.              */

static const uint8_t BD_DIALECT_TABLE[16][8] = {
    { 0, 1, 2, 3, 4, 6, 11, 15 },  /* dialect  0 */
    { 0, 1, 2, 3, 4, 6,  9, 15 },  /* dialect  1 */
    { 0, 1, 2, 3, 4, 6, 11, 14 },  /* dialect  2 */
    { 0, 1, 2, 3, 4, 6,  9, 14 },  /* dialect  3 */
    { 0, 1, 2, 3, 4, 6, 10, 13 },  /* dialect  4 */
    { 0, 1, 2, 3, 4, 6,  8, 13 },  /* dialect  5 */
    { 0, 1, 2, 3, 4, 6, 10, 12 },  /* dialect  6 */
    { 0, 1, 2, 3, 4, 6,  8, 12 },  /* dialect  7 */
    { 0, 1, 2, 3, 4, 6,  9, 11 },  /* dialect  8 */
    { 0, 1, 2, 3, 4, 6,  7, 11 },  /* dialect  9 */
    { 0, 1, 2, 3, 4, 6,  9, 10 },  /* dialect 10 */
    { 0, 1, 2, 3, 4, 6,  7, 10 },  /* dialect 11 */
    { 0, 1, 2, 3, 4, 6,  8,  9 },  /* dialect 12 */
    { 0, 1, 2, 3, 4, 6,  7,  9 },  /* dialect 13 */
    { 0, 1, 2, 3, 4, 6,  7,  8 },  /* dialect 14 */
    { 0, 1, 2, 3, 4, 5,  6,  8 },  /* dialect 15 */
};

/* ── Block layout constants ──────────────────────────────────────────── */
#define BD_BLOCK_ELEMS  32
#define BD_BLOCK_BYTES  18  /* 2-byte meta + 16-byte packed codes */
#define BD_FP16_BIAS    15

/* ── Extract meta fields from 2-byte big-endian meta word ──────────── */
static inline uint8_t bd_meta_dialect(uint8_t hi, uint8_t lo) {
    return (hi >> 4) & 0x0Fu;
}

static inline uint8_t bd_meta_exp(uint8_t hi, uint8_t lo) {
    return ((hi & 0x0Fu) << 1) | ((lo >> 7) & 0x01u);
}

/* ── Decode one 4-bit code to signed half-units ─────────────────────── */
static inline int16_t bd_decode_code(uint8_t dialect_id, uint8_t code) {
    uint8_t sign = (code >> 3) & 1;
    uint8_t idx  = code & 0x07;
    uint8_t hu   = BD_DIALECT_TABLE[dialect_id][idx];
    return sign ? -(int16_t)hu : (int16_t)hu;
}

/* ── Decode one 18-byte block to 32 signed half-unit int16 values ──── */
/*    Also returns dialect_id and shared_exp_bits for caller's use.      */
static inline void bd_decode_block_hu(
    const uint8_t *block,      /* 18-byte block record               */
    int16_t       *out,        /* output: 32 signed half-unit values  */
    int           *dialect_id, /* output: dialect id (0..15)          */
    int           *shared_exp  /* output: shared exponent bits (0..31)*/
) {
    uint8_t mhi = block[0];
    uint8_t mlo = block[1];
    uint8_t did = bd_meta_dialect(mhi, mlo);
    uint8_t seb = bd_meta_exp(mhi, mlo);

    if (dialect_id) *dialect_id = did;
    if (shared_exp) *shared_exp = seb;

    const uint8_t *codes = block + 2;  /* 16 bytes of packed codes */
    for (int i = 0; i < 16; i++) {
        uint8_t byte = codes[i];
        uint8_t code_hi = (byte >> 4) & 0x0F;
        uint8_t code_lo = byte & 0x0F;
        out[2 * i]     = bd_decode_code(did, code_hi);
        out[2 * i + 1] = bd_decode_code(did, code_lo);
    }
}

/* ── Decode one block and apply shared exponent → int32 values ─────── */
/*                                                                       */
/* The exact dequantization is:                                          */
/*   value_f = sign * (0.5 * half_units) * 2^(shared_exp - 15)          */
/*                                                                       */
/* For integer compute we want a fixed-point result with 'frac_bits'     */
/* fractional bits.  We compute:                                         */
/*   value_fp = sign * half_units * 2^(shared_exp - 15 - 1 + frac_bits) */
/*            = sign * half_units << (shared_exp - 16 + frac_bits)       */
/*                                                                       */
/* If the shift is negative, we right-shift (rounding toward zero).      */
static inline void bd_decode_block_fixedpt(
    const uint8_t *block,      /* 18-byte block record       */
    int32_t       *out,        /* output: 32 fixed-point i32 */
    int            frac_bits   /* number of fractional bits   */
) {
    int did, seb;
    int16_t hu[32];
    bd_decode_block_hu(block, hu, &did, &seb);

    /* shift = shared_exp - 16 + frac_bits
     * (the -16 combines the -15 FP16 bias and -1 for the 0.5 factor) */
    int shift = seb - 16 + frac_bits;

    if (shift >= 0) {
        for (int i = 0; i < 32; i++)
            out[i] = (int32_t)hu[i] << shift;
    } else {
        int rshift = -shift;
        for (int i = 0; i < 32; i++)
            out[i] = (int32_t)hu[i] >> rshift;
    }
}

/* ── Decode a full BD4 tensor to int8 (saturating) ─────────────────── */
/* This is the simplest path: decode each block to fixed-point, then     */
/* saturate to int8.  'w_scale_shift' controls the overall magnitude.    */
/*                                                                       */
/* For weight decode, a typical usage:                                   */
/*   bd_decode_tensor_to_i8(blocks, n_blocks, output, n_elements, 0);   */
/*                                                                       */
/* This produces output ≈ sign * half_units * 2^(shared_exp - 16),       */
/* saturated to [-128, +127].                                            */
static inline void bd_decode_tensor_to_i8(
    const uint8_t *blocks,     /* pointer to first block record      */
    uint32_t       n_blocks,   /* number of 18-byte blocks           */
    int8_t        *out,        /* output: decoded int8 values        */
    uint32_t       n_elements, /* actual element count (may < 32*n)  */
    int            frac_bits   /* fractional bits for fixed-point    */
) {
    int32_t tmp[32];
    uint32_t pos = 0;
    for (uint32_t b = 0; b < n_blocks && pos < n_elements; b++) {
        bd_decode_block_fixedpt(blocks + b * BD_BLOCK_BYTES, tmp, frac_bits);
        uint32_t count = n_elements - pos;
        if (count > 32) count = 32;
        for (uint32_t i = 0; i < count; i++) {
            int32_t v = tmp[i];
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            out[pos + i] = (int8_t)v;
        }
        pos += count;
    }
}

/* ── Compute a dot product of two half-unit blocks → int32 ──────────── */
/*                                                                        */
/* For BD weight × int8 activation:                                       */
/*   sum += w_half_units[i] * act[i]                                      */
/*   (caller scales by 2^(shared_exp - 16) after accumulation)            */
static inline int32_t bd_dot_hu_i8(
    const int16_t *w_hu,   /* 32 signed half-unit weight values */
    const int8_t  *act,    /* 32 int8 activation values         */
    int            count   /* number of elements (≤ 32)         */
) {
    int32_t sum = 0;
    for (int i = 0; i < count; i++) {
        sum += (int32_t)w_hu[i] * (int32_t)act[i];
    }
    return sum;
}

#endif /* BD_DECODE_SW_H */
