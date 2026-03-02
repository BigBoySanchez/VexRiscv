/* bd_act.h — BlockDialect A4 activation packing/unpacking (software)
 *
 * Implements Milestone 1A: store "spilled" activations in BlockDialect A4
 * format (~1.78× smaller than int8) to fit in 128 KiB SPRAM.
 *
 * Packing (bd_act_pack32):
 *   1. Compute max_abs of 32 values
 *   2. Compute shared exponent (power-of-two so scaled range ≈ [0, 8))
 *   3. Scale to half-units (0..15) on a 0.5 grid
 *   4. Select dialect via brute-force MSE (16 dialects × 32 candidates)
 *   5. Quantize to nearest representable value → 4-bit (sign|idx) codes
 *   6. Pack into 18-byte block: 2-byte meta + 16-byte packed codes
 *
 * Unpacking (bd_act_unpack32):
 *   Standard BD decode: meta → dialect_id + shared_exp,
 *   then idx → half_units via dialect table, apply sign + exponent.
 *
 * This uses the exact same block format as weight encoding, so the
 * hardware APB decoder could also be used for unpacking.
 *
 * Reference: arXiv:2501.01144v5, Section 3.2/3.3
 */
#ifndef BD_ACT_H
#define BD_ACT_H

#include <stdint.h>
#include "bd_decode_sw.h"  /* BD_DIALECT_TABLE, bd_decode_block_hu, etc. */

/* ── Shared exponent computation for activations ─────────────────────── */
/* Paper: shared_exp based on block max, with "-2" adjustment.             */
/* We want: scaled = |x| / (0.5 * 2^e) to land in [0, ~15].              */
/* e = floor(log2(max_abs)) - 2  (biased: shared_exp_bits = e + 15)       */
/*                                                                         */
/* This maps max_abs into roughly [4, 8) half-units, matching the paper.   */
static inline uint8_t bd_act_compute_exp(int32_t max_abs) {
    if (max_abs == 0) return 0;

    /* Find floor(log2(max_abs)) using CLZ */
    int log2_val = 0;
    {
        uint32_t v = (uint32_t)max_abs;
        /* Simple log2 for embedded (no __builtin_clz on all targets) */
        while (v > 1) { v >>= 1; log2_val++; }
    }

    /* shared_exp = log2(max_abs) - 2 + FP16_BIAS
     * But we need to also account for the 0.5 factor in half-units:
     *   scaled_hu = |x| / (0.5 * 2^(seb - 15)) = |x| * 2^(16 - seb)
     * We want max_abs * 2^(16 - seb) ≈ 8..15 (high half-units)
     * So: 2^(16-seb) ≈ 12/max_abs → seb ≈ 16 - log2(12/max_abs)
     *   ≈ 16 - log2(12) + log2(max_abs) ≈ log2(max_abs) + 12.4
     *
     * Simpler: seb = log2(max_abs) + 12, then clamp to [0, 31].
     * This puts max_abs at scaled_hu ≈ 8..16, which is in the sweet spot.
     */
    int seb = log2_val + 12;
    if (seb < 0) seb = 0;
    if (seb > 31) seb = 31;
    return (uint8_t)seb;
}

/* ── Scale |value| to half-units given shared_exp_bits ───────────────── */
/* scaled_hu = round(|x| * 2^(16 - seb)), clamped to [0, 15]             */
static inline uint8_t bd_act_scale_hu(int32_t abs_val, uint8_t seb) {
    /* shift = 16 - seb; positive → left shift, negative → right shift */
    int shift = 16 - (int)seb;
    int32_t scaled;

    if (shift >= 0) {
        /* Safety: if shift is too large, the result overflows.
         * For practical seb values (≤31) and |val| < 2^20, this is fine. */
        scaled = abs_val << shift;
    } else {
        /* Right shift with rounding */
        int rsh = -shift;
        scaled = (abs_val + (1 << (rsh - 1))) >> rsh;
    }

    /* Clamp to [0, 15] */
    if (scaled > 15) scaled = 15;
    if (scaled < 0) scaled = 0;
    return (uint8_t)scaled;
}

/* ── Find nearest index in a dialect for a target half-unit value ──── */
static inline uint8_t bd_act_nearest_idx(uint8_t target_hu, uint8_t dialect_id) {
    const uint8_t *d = BD_DIALECT_TABLE[dialect_id];
    uint8_t best_i = 0;
    int best_dist = (int)target_hu - (int)d[0];
    if (best_dist < 0) best_dist = -best_dist;

    for (uint8_t i = 1; i < 8; i++) {
        int dist = (int)target_hu - (int)d[i];
        if (dist < 0) dist = -dist;
        if (dist < best_dist) {
            best_dist = dist;
            best_i = i;
        }
    }
    return best_i;
}

/* ── Pack 32 signed int values into one BlockDialect A4 block ──────── */
/* Uses brute-force MSE dialect selection (simplest, 16×32 candidates).   */
/*                                                                         */
/* Input:  vals[32]      — signed values (e.g., post-ReLU int8/int16)     */
/* Output: block_out[18] — packed 18-byte block                           */
/*                                                                         */
/* Returns: shared_exp_bits (for caller's reference)                       */
static inline uint8_t bd_act_pack32(
    const int32_t *vals,    /* 32 signed activation values */
    uint8_t       *block_out /* 18-byte output block       */
) {
    /* Step 1: compute signs and max_abs */
    uint8_t signs[32];
    uint8_t scaled_hu[32];
    int32_t max_abs = 0;

    for (int i = 0; i < 32; i++) {
        int32_t v = vals[i];
        if (v < 0) {
            signs[i] = 1;
            v = -v;
        } else {
            signs[i] = 0;
        }
        if (v > max_abs) max_abs = v;
    }

    /* Step 2: compute shared exponent */
    uint8_t seb = bd_act_compute_exp(max_abs);

    /* Step 3: scale all values to half-units */
    for (int i = 0; i < 32; i++) {
        int32_t v = vals[i];
        if (v < 0) v = -v;
        scaled_hu[i] = bd_act_scale_hu(v, seb);
    }

    /* Step 4: brute-force dialect selection (MSE over all 16 dialects) */
    uint8_t best_dialect = 0;
    uint32_t best_mse = 0xFFFFFFFFu;

    for (uint8_t d = 0; d < 16; d++) {
        uint32_t mse = 0;
        for (int i = 0; i < 32; i++) {
            /* find nearest representable value in dialect d */
            uint8_t idx = bd_act_nearest_idx(scaled_hu[i], d);
            int diff = (int)scaled_hu[i] - (int)BD_DIALECT_TABLE[d][idx];
            mse += (uint32_t)(diff * diff);
        }
        if (mse < best_mse) {
            best_mse = mse;
            best_dialect = d;
        }
    }

    /* Step 5: quantize each element to nearest in chosen dialect */
    uint8_t codes[32];
    for (int i = 0; i < 32; i++) {
        uint8_t idx = bd_act_nearest_idx(scaled_hu[i], best_dialect);
        codes[i] = (signs[i] << 3) | (idx & 0x07);
    }

    /* Step 6: pack into 18-byte block */
    /* Meta word: dialect_id[15:12] | shared_exp[11:7] | padding[6:0] */
    uint16_t meta = ((uint16_t)(best_dialect & 0xF) << 12) |
                    ((uint16_t)(seb & 0x1F) << 7);
    block_out[0] = (uint8_t)(meta >> 8);    /* big-endian high byte */
    block_out[1] = (uint8_t)(meta & 0xFF);  /* big-endian low byte  */

    /* Packed codes: 2 codes per byte */
    for (int i = 0; i < 16; i++) {
        block_out[2 + i] = (codes[2 * i] << 4) | (codes[2 * i + 1] & 0x0F);
    }

    return seb;
}

/* ── Unpack one BD A4 block → 32 int8 values (saturating) ────────────── */
/* The activation is recovered as:                                          */
/*   value = sign * half_units * 2^(seb - 16)                              */
/* Then saturated to [-128, +127].                                          */
/*                                                                          */
/* For post-ReLU activations, values should be non-negative, but we handle  */
/* signed values for generality (as the plan specifies).                    */
static inline void bd_act_unpack32(
    const uint8_t *block,   /* 18-byte packed block   */
    int8_t        *out      /* 32 int8 output values  */
) {
    int did, seb;
    int16_t hu[32];
    bd_decode_block_hu(block, hu, &did, &seb);

    /* Reconstruct: value = hu[i] * 2^(seb - 16) (int, saturated to i8) */
    int shift = seb - 16;
    for (int i = 0; i < 32; i++) {
        int32_t v;
        if (shift >= 0) {
            v = (int32_t)hu[i] << shift;
        } else {
            v = (int32_t)hu[i] >> (-shift);
        }
        if (v > 127) v = 127;
        if (v < -128) v = -128;
        out[i] = (int8_t)v;
    }
}

/* ── Pack a full activation tensor (arbitrary length) ─────────────────── */
/* Zero-pads the last block if n_elements is not a multiple of 32.          */
/*                                                                          */
/* Returns: number of 18-byte blocks written.                               */
static inline uint32_t bd_act_pack_tensor(
    const int8_t *input,      /* int8 activation tensor       */
    uint32_t      n_elements, /* number of elements           */
    uint8_t      *bd_out      /* output: packed BD blocks     */
) {
    uint32_t n_blocks = (n_elements + 31) / 32;
    int32_t tmp[32];

    for (uint32_t b = 0; b < n_blocks; b++) {
        uint32_t base = b * 32;
        uint32_t count = n_elements - base;
        if (count > 32) count = 32;

        /* Copy to int32 tmp, zero-pad if needed */
        for (uint32_t i = 0; i < 32; i++) {
            tmp[i] = (i < count) ? (int32_t)input[base + i] : 0;
        }

        bd_act_pack32(tmp, bd_out + b * BD_BLOCK_BYTES);
    }
    return n_blocks;
}

/* ── Unpack a full BD activation tensor → int8 ────────────────────────── */
static inline void bd_act_unpack_tensor(
    const uint8_t *bd_blocks,  /* packed BD blocks          */
    uint32_t       n_blocks,   /* number of 18-byte blocks  */
    int8_t        *output,     /* output: int8 values       */
    uint32_t       n_elements  /* actual element count       */
) {
    int8_t tmp[32];
    uint32_t pos = 0;

    for (uint32_t b = 0; b < n_blocks && pos < n_elements; b++) {
        bd_act_unpack32(bd_blocks + b * BD_BLOCK_BYTES, tmp);
        uint32_t count = n_elements - pos;
        if (count > 32) count = 32;
        for (uint32_t i = 0; i < count; i++) {
            output[pos + i] = tmp[i];
        }
        pos += count;
    }
}

/* ── BD A4 storage size calculation ──────────────────────────────────── */
static inline uint32_t bd_act_storage_bytes(uint32_t n_elements) {
    uint32_t n_blocks = (n_elements + 31) / 32;
    return n_blocks * BD_BLOCK_BYTES;
}

#endif /* BD_ACT_H */
