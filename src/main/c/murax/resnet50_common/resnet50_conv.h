/* resnet50_conv.h — Tiled convolution engine for ResNet-50
 *
 * Key design choices:
 *   - Weights decoded from BD format per output channel (streamed from flash)
 *   - Working activations in int8, tiled spatially for large feature maps
 *   - Skip connections in BD A4 format when they exceed the budget
 *   - All accumulation in int32, quantized to int8 with bias + shift
 *
 * Convolution types needed:
 *   1. conv_7x7_s2 — stem layer (3→64)
 *   2. conv_1x1    — bottleneck squeeze/expand and downsample projections
 *   3. conv_3x3    — bottleneck spatial layer
 *   4. maxpool_3x3_s2 — after stem
 *   5. avgpool_7x7 — final global pool
 *   6. fc_linear   — final classifier
 *
 * Memory layout:
 *   We use a simple double-buffer strategy with tile processing.
 *   The SPRAM budget allows ~60 KiB for activation buffers (leaving room
 *   for stack, weight decode cache, and BD spill buffers).
 */
#ifndef RESNET50_CONV_H
#define RESNET50_CONV_H

#include <stdint.h>
#include "bd_decode_sw.h"
#include "bd_act.h"
#include "resnet50_layers.h"

/* ── Configuration ───────────────────────────────────────────────────── */
/* Weight decode scratch: per output channel.
 * Largest kernel per-OC: conv1 = 3*7*7=147, conv_3x3 = mid_c*3*3.
 * layer4.0.conv2: 512*3*3=4608 elements → need 4608 * 2 bytes = 9216 bytes
 * We decode to half-units (int16) and store shared exponents per block. */
#define W_DECODE_MAX_ELEMS  4608  /* 512 in_c × 3 × 3 (worst case 3x3) */
#define W_DECODE_MAX_BLOCKS ((W_DECODE_MAX_ELEMS + 31) / 32)  /* 144 blocks */

/* Bias is float32 — we convert to int32 with appropriate scaling.
 * For now, we use a simple fixed-point bias: bias_i32 = (int32_t)(bias_f32 * scale)
 * where scale accounts for the weight and activation quantization. */

/* ── Quantization: accumulator → int8 output ─────────────────────────── */
/* After a convolution, each output element is:                            */
/*   acc = Σ (w_hu[i] * act_i8[i])                                        */
/* where w_hu is in half-units.  The actual weight value is:               */
/*   w = 0.5 * w_hu * 2^(seb - 15)                                        */
/* So the true conv result is:                                             */
/*   result = acc * 0.5 * 2^(seb-15) + bias                               */
/*                                                                          */
/* For multi-block kernels, each block has its own seb, so we must         */
/* apply per-block scaling before accumulating across blocks.               */
/* Simplified approach: decode weights to ~int8 scale and accumulate       */
/* with a global per-layer right-shift.                                    */

/* Per-layer quantization parameters (precomputed by Python golden model) */
/* For Milestone 1, we use a simple approach:                              */
/*   1. Decode BD weights → half-units (int16)                             */
/*   2. Scale to int8-ish range using shared_exp                           */
/*   3. Accumulate: sum += w_scaled * act_i8                               */
/*   4. Apply output_shift: out_i8 = saturate(sum >> output_shift)         */
/*   5. Add float32 bias (converted to int32 at appropriate scale)          */

/* ── Conv 1×1 (the workhorse for bottleneck blocks) ──────────────────── */
/* 1×1 conv is essentially a matrix multiply: no spatial kernel.            */
/* For each output pixel (y,x) and output channel oc:                      */
/*   out[oc][y][x] = sum_over_ic( weight[oc][ic] * input[ic][y][x] ) + bias */
/*                                                                          */
/* We process per output channel to keep weight decode cache small.         */
/* For large in_c (up to 2048), the weight vector per OC is 2048 elements. */

static void conv_1x1(
    const int8_t  *input,     /* CHW layout, in_c × h × w        */
    int8_t        *output,    /* CHW layout, out_c × oh × ow     */
    const uint8_t *w_blocks,  /* BD weight blocks for this layer  */
    const float   *bias_f32,  /* float32 bias[out_c]              */
    int in_c, int out_c,
    int h, int w,
    int stride,               /* 1 or 2 */
    int output_shift          /* right-shift for accumulator→int8 */
) {
    int oh = h / stride;
    int ow = w / stride;
    int kernel_elems = in_c;  /* 1×1 kernel: in_c elements per OC */
    int n_blocks_per_oc = (kernel_elems + 31) / 32;

    /* Weight pointer advances per output channel */
    const uint8_t *w_ptr = w_blocks;

    for (int oc = 0; oc < out_c; oc++) {
        /* Decode this OC's weights to int8 (simplified: pick frac_bits to
         * bring half-units into int8 range) */
        /* For now, decode to half-units and accumulate with per-block scaling */

        /* Simpler approach: decode weights per-block and do scaled dot products.
         * For each block of 32 weights, decode → hu[32] + seb, then:
         *   partial_sum += dot(hu, act_slice) * 2^(seb - 16) */

        /* Fixed-point bias */
        int32_t bias_i32 = 0;
        {
            /* Convert float32 bias to fixed-point at the accumulator's scale.
             * The accumulator holds: sum of (hu * act) where hu is half-units
             * and act is int8.  The real value would be:
             *   sum * 0.5 * 2^(seb-15) * act_scale
             * For simplicity, just convert bias to int32 with a rough scale. */
            float bf = bias_f32[oc];
            /* We'll apply output_shift to the accumulator, so bias should be
             * at the same scale: bias_i32 = bias_f * 2^output_shift / weight_scale
             * For Milestone 1, use a heuristic: bias_i32 = (int32_t)(bf * 128) */
            bias_i32 = (int32_t)(bf * (float)(1 << output_shift));
        }

        for (int y = 0; y < oh; y++) {
            for (int x = 0; x < ow; x++) {
                int32_t acc = 0;
                int iy = y * stride;
                int ix = x * stride;

                /* Process weight blocks for this OC */
                const uint8_t *bptr = w_ptr;
                int elem_done = 0;

                for (int b = 0; b < n_blocks_per_oc; b++) {
                    int16_t hu[32];
                    int did, seb;
                    bd_decode_block_hu(bptr, hu, &did, &seb);
                    bptr += BD_BLOCK_BYTES;

                    int count = kernel_elems - elem_done;
                    if (count > 32) count = 32;

                    /* shift = seb - 16; this converts hu to int value
                     * where value = hu * 2^(seb-16) */
                    int shift = seb - 16;

                    int32_t block_sum = 0;
                    for (int i = 0; i < count; i++) {
                        int ic = elem_done + i;
                        int8_t a = input[ic * h * w + iy * w + ix];
                        block_sum += (int32_t)hu[i] * (int32_t)a;
                    }

                    /* Apply per-block exponent scaling */
                    if (shift >= 0)
                        acc += block_sum << shift;
                    else
                        acc += block_sum >> (-shift);

                    elem_done += count;
                }

                /* Add bias and quantize to int8 */
                acc += bias_i32;
                int32_t out_val = acc >> output_shift;
                if (out_val > 127) out_val = 127;
                if (out_val < -128) out_val = -128;

                output[oc * oh * ow + y * ow + x] = (int8_t)out_val;
            }
        }

        /* Advance weight pointer to next output channel */
        w_ptr += n_blocks_per_oc * BD_BLOCK_BYTES;
    }
}

/* ── Conv 3×3 ────────────────────────────────────────────────────────── */
static void conv_3x3(
    const int8_t  *input,
    int8_t        *output,
    const uint8_t *w_blocks,
    const float   *bias_f32,
    int in_c, int out_c,
    int h, int w,
    int stride,
    int padding,
    int output_shift
) {
    int oh = (h + 2 * padding - 3) / stride + 1;
    int ow = (w + 2 * padding - 3) / stride + 1;
    int kernel_elems = in_c * 9;  /* 3×3 kernel */
    int n_blocks_per_oc = (kernel_elems + 31) / 32;

    const uint8_t *w_ptr = w_blocks;

    for (int oc = 0; oc < out_c; oc++) {
        int32_t bias_i32 = (int32_t)(bias_f32[oc] * (float)(1 << output_shift));

        for (int y = 0; y < oh; y++) {
            for (int x = 0; x < ow; x++) {
                int32_t acc = 0;
                const uint8_t *bptr = w_ptr;
                int elem_done = 0;

                for (int b = 0; b < n_blocks_per_oc; b++) {
                    int16_t hu[32];
                    int did, seb;
                    bd_decode_block_hu(bptr, hu, &did, &seb);
                    bptr += BD_BLOCK_BYTES;

                    int count = kernel_elems - elem_done;
                    if (count > 32) count = 32;

                    int shift = seb - 16;
                    int32_t block_sum = 0;

                    for (int i = 0; i < count; i++) {
                        int flat_idx = elem_done + i;
                        int ic = flat_idx / 9;
                        int k_rem = flat_idx % 9;
                        int ky = k_rem / 3;
                        int kx = k_rem % 3;

                        int iy = y * stride - padding + ky;
                        int ix = x * stride - padding + kx;

                        int8_t a = 0;
                        if (iy >= 0 && iy < h && ix >= 0 && ix < w) {
                            a = input[ic * h * w + iy * w + ix];
                        }
                        block_sum += (int32_t)hu[i] * (int32_t)a;
                    }

                    if (shift >= 0)
                        acc += block_sum << shift;
                    else
                        acc += block_sum >> (-shift);

                    elem_done += count;
                }

                acc += bias_i32;
                int32_t out_val = acc >> output_shift;
                if (out_val > 127) out_val = 127;
                if (out_val < -128) out_val = -128;

                output[oc * oh * ow + y * ow + x] = (int8_t)out_val;
            }
        }

        w_ptr += n_blocks_per_oc * BD_BLOCK_BYTES;
    }
}

/* ── Conv 7×7 (stem layer only) ──────────────────────────────────────── */
static void conv_7x7(
    const int8_t  *input,
    int8_t        *output,
    const uint8_t *w_blocks,
    const float   *bias_f32,
    int in_c, int out_c,
    int h, int w,
    int stride,
    int padding,
    int output_shift
) {
    int oh = (h + 2 * padding - 7) / stride + 1;
    int ow = (w + 2 * padding - 7) / stride + 1;
    int kernel_elems = in_c * 49;  /* 7×7 kernel */
    int n_blocks_per_oc = (kernel_elems + 31) / 32;

    const uint8_t *w_ptr = w_blocks;

    for (int oc = 0; oc < out_c; oc++) {
        int32_t bias_i32 = (int32_t)(bias_f32[oc] * (float)(1 << output_shift));

        for (int y = 0; y < oh; y++) {
            for (int x = 0; x < ow; x++) {
                int32_t acc = 0;
                const uint8_t *bptr = w_ptr;
                int elem_done = 0;

                for (int b = 0; b < n_blocks_per_oc; b++) {
                    int16_t hu[32];
                    int did, seb;
                    bd_decode_block_hu(bptr, hu, &did, &seb);
                    bptr += BD_BLOCK_BYTES;

                    int count = kernel_elems - elem_done;
                    if (count > 32) count = 32;

                    int shift = seb - 16;
                    int32_t block_sum = 0;

                    for (int i = 0; i < count; i++) {
                        int flat_idx = elem_done + i;
                        int ic = flat_idx / 49;
                        int k_rem = flat_idx % 49;
                        int ky = k_rem / 7;
                        int kx = k_rem % 7;

                        int iy = y * stride - padding + ky;
                        int ix = x * stride - padding + kx;

                        int8_t a = 0;
                        if (iy >= 0 && iy < h && ix >= 0 && ix < w) {
                            a = input[ic * h * w + iy * w + ix];
                        }
                        block_sum += (int32_t)hu[i] * (int32_t)a;
                    }

                    if (shift >= 0)
                        acc += block_sum << shift;
                    else
                        acc += block_sum >> (-shift);

                    elem_done += count;
                }

                acc += bias_i32;
                int32_t out_val = acc >> output_shift;
                if (out_val > 127) out_val = 127;
                if (out_val < -128) out_val = -128;

                output[oc * oh * ow + y * ow + x] = (int8_t)out_val;
            }
        }

        w_ptr += n_blocks_per_oc * BD_BLOCK_BYTES;
    }
}

/* ── ReLU in-place ───────────────────────────────────────────────────── */
static inline void relu_inplace(int8_t *data, int n) {
    for (int i = 0; i < n; i++) {
        if (data[i] < 0) data[i] = 0;
    }
}

/* ── Add + ReLU: dst = ReLU(dst + src) ───────────────────────────────── */
static inline void add_relu(int8_t *dst, const int8_t *src, int n) {
    for (int i = 0; i < n; i++) {
        int32_t val = (int32_t)dst[i] + (int32_t)src[i];
        if (val < 0) val = 0;
        if (val > 127) val = 127;
        dst[i] = (int8_t)val;
    }
}

/* ── MaxPool 3×3 stride 2, padding 1 ────────────────────────────────── */
static void maxpool_3x3_s2(
    const int8_t *input,   /* C × H × W */
    int8_t       *output,  /* C × OH × OW */
    int c, int h, int w
) {
    int oh = (h + 2 * 1 - 3) / 2 + 1;
    int ow = (w + 2 * 1 - 3) / 2 + 1;

    for (int ch = 0; ch < c; ch++) {
        for (int y = 0; y < oh; y++) {
            for (int x = 0; x < ow; x++) {
                int8_t max_val = -128;
                for (int ky = 0; ky < 3; ky++) {
                    int iy = y * 2 - 1 + ky;
                    for (int kx = 0; kx < 3; kx++) {
                        int ix = x * 2 - 1 + kx;
                        int8_t v = -128;
                        if (iy >= 0 && iy < h && ix >= 0 && ix < w) {
                            v = input[ch * h * w + iy * w + ix];
                        }
                        if (v > max_val) max_val = v;
                    }
                }
                output[ch * oh * ow + y * ow + x] = max_val;
            }
        }
    }
}

/* ── Global Average Pool 7×7 ────────────────────────────────────────── */
static void avgpool_global(
    const int8_t *input,   /* C × 7 × 7 */
    int8_t       *output,  /* C × 1 × 1 */
    int c
) {
    for (int ch = 0; ch < c; ch++) {
        int32_t sum = 0;
        for (int i = 0; i < 49; i++) {
            sum += (int32_t)input[ch * 49 + i];
        }
        /* Divide by 49; approximate as (sum * 1311) >> 16 ≈ sum/49.85 */
        /* Or simply: sum / 49 (integer division) */
        int32_t avg = sum / 49;
        if (avg > 127) avg = 127;
        if (avg < -128) avg = -128;
        output[ch] = (int8_t)avg;
    }
}

/* ── FC layer (weight from BD, bias from float32) ────────────────────── */
static void fc_linear(
    const int8_t  *input,      /* 2048 int8 values */
    int32_t       *logits,     /* 1000 int32 output logits */
    const uint8_t *w_blocks,   /* BD weight blocks */
    const float   *bias_f32,   /* float32 bias[1000] */
    int in_c,                  /* 2048 */
    int out_c                  /* 1000 */
) {
    int kernel_elems = in_c;
    int n_blocks_per_oc = (kernel_elems + 31) / 32;

    const uint8_t *w_ptr = w_blocks;

    for (int oc = 0; oc < out_c; oc++) {
        int32_t acc = 0;
        const uint8_t *bptr = w_ptr;
        int elem_done = 0;

        for (int b = 0; b < n_blocks_per_oc; b++) {
            int16_t hu[32];
            int did, seb;
            bd_decode_block_hu(bptr, hu, &did, &seb);
            bptr += BD_BLOCK_BYTES;

            int count = kernel_elems - elem_done;
            if (count > 32) count = 32;

            int shift = seb - 16;
            int32_t block_sum = 0;

            for (int i = 0; i < count; i++) {
                int8_t a = input[elem_done + i];
                block_sum += (int32_t)hu[i] * (int32_t)a;
            }

            if (shift >= 0)
                acc += block_sum << shift;
            else
                acc += block_sum >> (-shift);

            elem_done += count;
        }

        /* FC output: keep as int32 (don't quantize to int8) */
        /* Add bias in a rough fixed-point way */
        acc += (int32_t)(bias_f32[oc] * 128.0f);
        logits[oc] = acc;

        w_ptr += n_blocks_per_oc * BD_BLOCK_BYTES;
    }
}

#endif /* RESNET50_CONV_H */
