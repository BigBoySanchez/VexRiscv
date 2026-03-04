/* resnet1202_conv.h — Convolution engine for ResNet-1202 (CIFAR-10 BasicBlock variant)
 *
 * §1 + §4 of RESNET1202_FPGA_PLAN.md
 *
 * Convolution types:
 *   1. conv3x3        — generic 3×3 CHW conv, any stride, padding=1
 *   2. conv1x1_proj   — 1×1 projection shortcut conv, stride=2
 *   3. relu_inplace   — in-place ReLU, int8
 *   4. add_relu_inplace — dst = ReLU(dst + src), int8 saturating
 *   5. avgpool_hw     — global average pool over H×W → scalar per channel
 *   6. fc_linear      — FC layer: 64→10 (no bias for phase1; added in phase2)
 *
 * Design:
 *   - Weights decoded on-the-fly from BD4 blocks, one output-channel at a time.
 *   - Accumulation in int32, quantized back to int8 with a per-layer shift.
 *   - For all ResNet-1202 stages, full activation tensors fit in 128 KB SPRAM
 *     without tiling (worst case stage1: 16×32×32 = 16 KB << 128 KB).
 *   - CHW (channels-first) layout throughout.
 *
 * Weight pointer convention:
 *   The caller passes w_blocks pointing to the first BD4 block of the tensor.
 *   For a conv3×3(in_c, out_c), the tensor has out_c groups of
 *   ceil(in_c*9/32) BD blocks each.  Blocks advance by BD_BLOCK_BYTES=18 bytes.
 */
#ifndef RESNET1202_CONV_H
#define RESNET1202_CONV_H

#include <stdint.h>
#include "bd_decode_sw.h"   /* BD_DIALECT_TABLE, bd_decode_block_hu, BD_BLOCK_BYTES */
#include "bd_act.h"         /* bd_act_pack_tensor_twostage, bd_act_unpack_tensor      */
#include "weight_blob.h"    /* vwb2_* types and helpers                              */
#include "resnet1202_layers.h"  /* BasicBlockConf, rn1202_block_conf                */

/* ── bias_scale: (int32_t)(bias * 2^shift) without __mulsf3 or __fixsfsi ──
 *
 * Reads the IEEE 754 representation directly and adjusts the exponent,
 * avoiding any soft-float library calls.  Saves ~1108 B (mulsf3 + fixsfsi).
 */
static int32_t bias_scale(const float *fp, int shift)
{
    union { float f; uint32_t b; } u;
    u.f = *fp;
    int exp      = (int)((u.b >> 23) & 0xFFu) - 127;
    if (exp == -127) return 0;              /* zero / denormal */
    uint32_t man = (u.b & 0x7FFFFFu) | 0x800000u;
    int sign     = (u.b >> 31) ? -1 : 1;
    int total    = exp - 23 + shift;        /* man * 2^total = result magnitude */
    int32_t mag;
    if (total >= 31)  return sign > 0 ? 0x7FFFFFFF : (int32_t)0x80000001u;
    if (total >= 0)   mag = (int32_t)(man << total);
    else if (-total < 24) mag = (int32_t)(man >> (-total));
    else              return 0;
    return sign > 0 ? mag : -mag;
}

/* ── In-place ReLU ──────────────────────────────────────────────────── */
static inline void relu_inplace(int8_t *buf, int n)
{
    for (int i = 0; i < n; i++)
        if (buf[i] < 0) buf[i] = 0;
}

/* ── Saturating add + ReLU: dst[i] = clamp(dst[i] + src[i], 0, 127) ── */
static inline void add_relu_inplace(int8_t *dst, const int8_t *src, int n)
{
    for (int i = 0; i < n; i++) {
        int32_t v = (int32_t)dst[i] + (int32_t)src[i];
        if (v < 0)   v = 0;
        if (v > 127) v = 127;
        dst[i] = (int8_t)v;
    }
}

/* ── Option A zero-pad stride-2 skip: in_c×H×W → out_c×(H/2)×(W/2) ──
 *
 * Implements He et al. Option A shortcut for the first block of each
 * downsampling stage.  There are no learnable weights; the skip tensor
 * is formed by:
 *   1. Spatially subsampling by stride 2 (pick every 2nd position)
 *   2. Padding the channel dimension with zeros: channels [in_c, out_c)
 *      are zero-filled.  For ResNet-1202: 16→32 and 32→64.
 *
 * Called when conf->stride == 2 && conf->has_proj == 0.
 */
static void zero_pad_stride2(
    const int8_t *input,   /* in_c × H × W                       */
    int8_t       *output,  /* out_c × (H/2) × (W/2), zeroed first */
    int in_c, int out_c,
    int h, int w
) {
    int oh = h / 2;
    int ow = w / 2;
    int out_spatial = oh * ow;

    /* Zero the whole output (covers the padded channels) */
    for (int i = 0; i < out_c * out_spatial; i++)
        output[i] = 0;

    /* Copy stride-2 sampled channels 0..in_c-1 */
    for (int ic = 0; ic < in_c; ic++) {
        for (int y = 0; y < oh; y++) {
            for (int x = 0; x < ow; x++) {
                output[ic * out_spatial + y * ow + x] =
                    input[ic * h * w + (y * 2) * w + (x * 2)];
            }
        }
    }
}

/* ── Compute uint32 verification hash (sum of int8-cast-to-int32) ───── */
static inline uint32_t act_hash(const int8_t *buf, int n)
{
    uint32_t sum = 0;
    for (int i = 0; i < n; i++)
        sum += (uint32_t)(int32_t)buf[i];
    return sum;
}

/* ── conv3x3: generic 3×3 convolution, padding=1, any stride ───────── */
/*
 * Computes:
 *   out[oc, y, x] = sum_{ic,ky,kx} w[oc, ic, ky, kx] * in[ic, y*s+ky-1, x*s+kx-1]
 *                   + bias[oc]
 * then optionally apply ReLU and quantize to int8.
 *
 * Parameters:
 *   input      — int8 CHW, in_c × h × w
 *   output     — int8 CHW, out_c × oh × ow  (oh = h/stride for stride=2,  or h for stride=1)
 *   w_blocks   — BD4 weight blocks, out_c × ceil(in_c*9/32) blocks
 *   bias       — float32 bias[out_c] (BN-folded), may be NULL for zero bias
 *   in_c, out_c — channel counts
 *   h, w       — input spatial dims
 *   stride     — 1 (identity) or 2 (downsampling)
 *   out_shift  — right-shift for int32 accumulator → int8 output
 *   do_relu    — 1 to apply ReLU after quantization, 0 to skip
 */
static void conv3x3(
    const int8_t  *input,
    int8_t        *output,
    const uint8_t *w_blocks,
    const float   *bias,
    int in_c, int out_c,
    int h, int w,
    int stride,
    int out_shift,
    int do_relu
) {
    int oh = h / stride;
    int ow = w / stride;
    int kernel_elems    = in_c * 9;   /* 3×3 kernel per OC */
    int n_blocks_per_oc = (kernel_elems + 31) / 32;

    const uint8_t *w_ptr = w_blocks;

    for (int oc = 0; oc < out_c; oc++) {
        int32_t bias_i32 = bias ? (int32_t)(bias[oc] * (float)(1 << out_shift)) : 0;

        for (int y = 0; y < oh; y++) {
            for (int x = 0; x < ow; x++) {
                int32_t acc       = 0;
                const uint8_t *bp = w_ptr;
                int elem_done     = 0;

                for (int b = 0; b < n_blocks_per_oc; b++) {
                    int16_t hu[32];
                    int did, seb;
                    bd_decode_block_hu(bp, hu, &did, &seb);
                    bp += BD_BLOCK_BYTES;

                    int count = kernel_elems - elem_done;
                    if (count > 32) count = 32;

                    int shift      = seb - 16;
                    int32_t bsum   = 0;

                    for (int i = 0; i < count; i++) {
                        /* flat weight index → (ic, ky, kx) */
                        int flat = elem_done + i;
                        int ic   = flat / 9;
                        int krem = flat % 9;
                        int ky   = krem / 3;
                        int kx   = krem % 3;

                        int iy   = y * stride - 1 + ky;
                        int ix   = x * stride - 1 + kx;

                        int8_t a = 0;
                        if (iy >= 0 && iy < h && ix >= 0 && ix < w)
                            a = input[ic * h * w + iy * w + ix];

                        bsum += (int32_t)hu[i] * (int32_t)a;
                    }

                    if (shift >= 0) acc += bsum << shift;
                    else            acc += bsum >> (-shift);

                    elem_done += count;
                }

                acc += bias_i32;
                int32_t v = acc >> out_shift;
                if (do_relu && v < 0) v = 0;
                if (v >  127) v =  127;
                if (v < -128) v = -128;

                output[oc * oh * ow + y * ow + x] = (int8_t)v;
            }
        }

        w_ptr += n_blocks_per_oc * BD_BLOCK_BYTES;
    }
}

/* ── conv1x1: 1×1 projection shortcut, stride=2 ────────────────────── */
/*
 * Used for projection skip connections (Cin→Cout, stride=2).
 * Like conv3x3 but kernel=1×1, no padding; out spatial = in/stride.
 */
static void conv1x1(
    const int8_t  *input,
    int8_t        *output,
    const uint8_t *w_blocks,
    const float   *bias,
    int in_c, int out_c,
    int h, int w,
    int stride,
    int out_shift,
    int do_relu
) {
    int oh = h / stride;
    int ow = w / stride;
    int kernel_elems    = in_c;   /* 1×1 kernel per OC */
    int n_blocks_per_oc = (kernel_elems + 31) / 32;

    const uint8_t *w_ptr = w_blocks;

    for (int oc = 0; oc < out_c; oc++) {
        int32_t bias_i32 = bias ? (int32_t)(bias[oc] * (float)(1 << out_shift)) : 0;

        for (int y = 0; y < oh; y++) {
            for (int x = 0; x < ow; x++) {
                int32_t acc       = 0;
                const uint8_t *bp = w_ptr;
                int elem_done     = 0;

                for (int b = 0; b < n_blocks_per_oc; b++) {
                    int16_t hu[32];
                    int did, seb;
                    bd_decode_block_hu(bp, hu, &did, &seb);
                    bp += BD_BLOCK_BYTES;

                    int count = kernel_elems - elem_done;
                    if (count > 32) count = 32;

                    int shift    = seb - 16;
                    int32_t bsum = 0;

                    for (int i = 0; i < count; i++) {
                        int ic  = elem_done + i;
                        int iy  = y * stride;
                        int ix_ = x * stride;
                        int8_t a = input[ic * h * w + iy * w + ix_];
                        bsum += (int32_t)hu[i] * (int32_t)a;
                    }

                    if (shift >= 0) acc += bsum << shift;
                    else            acc += bsum >> (-shift);

                    elem_done += count;
                }

                acc += bias_i32;
                int32_t v = acc >> out_shift;
                if (do_relu && v < 0) v = 0;
                if (v >  127) v =  127;
                if (v < -128) v = -128;

                output[oc * oh * ow + y * ow + x] = (int8_t)v;
            }
        }

        w_ptr += n_blocks_per_oc * BD_BLOCK_BYTES;
    }
}

/* ── Global average pool: C×H×W → C×1×1 (stored as flat C array) ───── */
static void global_avgpool(const int8_t *input, int8_t *output, int c, int h, int w)
{
    int spatial = h * w;
    for (int ch = 0; ch < c; ch++) {
        int32_t sum = 0;
        const int8_t *base = input + ch * spatial;
        for (int i = 0; i < spatial; i++)
            sum += (int32_t)base[i];
        int32_t avg = sum / spatial;
        if (avg >  127) avg =  127;
        if (avg < -128) avg = -128;
        output[ch] = (int8_t)avg;
    }
}

/* ── FC layer: in_c → out_c, weights from BD4, bias float32 → int32 ── */
/*
 * For ResNet-1202: 64→10 classifier.
 * Output is int32 logits (not quantized to int8).
 */
static void fc_linear(
    const int8_t  *input,     /* in_c int8 values  */
    int32_t       *logits,    /* out_c int32 output */
    const uint8_t *w_blocks,  /* BD4 weight blocks */
    const float   *bias,      /* float32 bias[out_c], may be NULL */
    int in_c,
    int out_c
) {
    int n_blocks_per_oc = (in_c + 31) / 32;
    const uint8_t *w_ptr = w_blocks;

    for (int oc = 0; oc < out_c; oc++) {
        int32_t acc       = 0;
        const uint8_t *bp = w_ptr;
        int elem_done     = 0;

        for (int b = 0; b < n_blocks_per_oc; b++) {
            int16_t hu[32];
            int did, seb;
            bd_decode_block_hu(bp, hu, &did, &seb);
            bp += BD_BLOCK_BYTES;

            int count = in_c - elem_done;
            if (count > 32) count = 32;

            int shift    = seb - 16;
            int32_t bsum = 0;
            for (int i = 0; i < count; i++)
                bsum += (int32_t)hu[i] * (int32_t)input[elem_done + i];

            if (shift >= 0) acc += bsum << shift;
            else            acc += bsum >> (-shift);

            elem_done += count;
        }

        if (bias)
            acc += bias_scale(bias + oc, 7);  /* no __mulsf3 */
        logits[oc] = acc;

        w_ptr += n_blocks_per_oc * BD_BLOCK_BYTES;
    }
}

/* ── Helper: get pointer to raw BD weight blocks for a tensor ───────── */
/* Each weight layer occupies 2 consecutive VWB2 entries: [weight, bias]. */
/* tensor_id counts weight layers (0 .. RN1202_TOTAL_TENSORS-1), so the  */
/* VWB2 table index is tensor_id * 2 (weight) / tensor_id * 2 + 1 (bias).*/
static inline const uint8_t *rn1202_weight_blocks(const vwb2_header_t *hdr,
                                                    uint16_t tensor_id)
{
    const vwb2_entry_t *tbl = vwb2_table(hdr);
    const vwb2_entry_t *e   = &tbl[(uint32_t)tensor_id * 2u];
    return vwb2_bd4_blocks(hdr, e);
}

/* ── Helper: get pointer to float32 bias array for a tensor ─────────── */
/* Bias tensor is the entry immediately after the weight tensor.          */
static inline const float *rn1202_bias_f32(const vwb2_header_t *hdr,
                                            uint16_t tensor_id)
{
    const vwb2_entry_t *tbl    = vwb2_table(hdr);
    const vwb2_entry_t *bias_e = &tbl[(uint32_t)tensor_id * 2u + 1u];
    return vwb2_float32_data(hdr, bias_e);
}

/* ── conv3x3_bd4: conv3x3 with BD4-packed output (paper-faithful) ─────── */
/*                                                                            */
/* Same arithmetic as conv3x3 but output is BD4-packed rather than int8.     */
/* Accumulation is written into caller-supplied int32 scratch, then the full  */
/* output tensor is packed to BD4 in one pass via quantize_output_bd4().      */
/* This eliminates the int8 intermediate present in conv3x3.                  */
/*                                                                            */
/* Parameters:                                                                */
/*   input      — int8 CHW, in_c × h × w                                    */
/*   bd_out     — BD4 output, must hold ceil(out_c*oh*ow/32)*BD_BLOCK_BYTES  */
/*   accum_tmp  — caller-supplied int32 scratch, must hold out_c×oh×ow elems */
/*   w_blocks   — BD4 weight blocks, out_c × ceil(in_c*9/32) blocks          */
/*   bias       — float32 bias[out_c] (BN-folded), may be NULL               */
/*   in_c, out_c — channel counts                                             */
/*   h, w       — input spatial dims                                          */
/*   stride     — 1 (identity) or 2 (downsampling)                           */
/*   out_shift  — right-shift for int32 accumulator → BD4 quantization        */
/*   do_relu    — 1 to clamp negatives to 0 before BD4-packing, 0 to skip    */
static void conv3x3_bd4(
    const int8_t  *input,
    uint8_t       *bd_out,
    int32_t       *accum_tmp,
    const uint8_t *w_blocks,
    const float   *bias,
    int in_c, int out_c,
    int h, int w,
    int stride,
    int out_shift,
    int do_relu
) {
    int oh = h / stride;
    int ow = w / stride;
    int out_elems       = out_c * oh * ow;
    int kernel_elems    = in_c * 9;
    int n_blocks_per_oc = (kernel_elems + 31) / 32;

    /* Zero accumulator */
    for (int i = 0; i < out_elems; i++) accum_tmp[i] = 0;

    const uint8_t *w_ptr = w_blocks;

    for (int oc = 0; oc < out_c; oc++) {
        int32_t bias_i32 = bias ? bias_scale(bias + oc, out_shift) : 0;

        for (int y = 0; y < oh; y++) {
            for (int x = 0; x < ow; x++) {
                int32_t acc       = 0;
                const uint8_t *bp = w_ptr;
                int elem_done     = 0;

                for (int b = 0; b < n_blocks_per_oc; b++) {
                    int16_t hu[32];
                    int did, seb;
                    bd_decode_block_hu(bp, hu, &did, &seb);
                    bp += BD_BLOCK_BYTES;

                    int count = kernel_elems - elem_done;
                    if (count > 32) count = 32;

                    int shift    = seb - 16;
                    int32_t bsum = 0;

                    for (int i = 0; i < count; i++) {
                        int flat = elem_done + i;
                        int ic   = flat / 9;
                        int krem = flat % 9;
                        int ky   = krem / 3;
                        int kx   = krem % 3;

                        int iy  = y * stride - 1 + ky;
                        int ix_ = x * stride - 1 + kx;

                        int8_t a = 0;
                        if (iy >= 0 && iy < h && ix_ >= 0 && ix_ < w)
                            a = input[ic * h * w + iy * w + ix_];

                        bsum += (int32_t)hu[i] * (int32_t)a;
                    }

                    if (shift >= 0) acc += bsum << shift;
                    else            acc += bsum >> (-shift);

                    elem_done += count;
                }

                accum_tmp[oc * oh * ow + y * ow + x] = acc + bias_i32;
            }
        }

        w_ptr += n_blocks_per_oc * BD_BLOCK_BYTES;
    }

    /* Pack entire int32 accumulator output tensor → BD4 (no int8 intermediate) */
    quantize_output_bd4(accum_tmp, bd_out, (uint32_t)out_elems, out_shift, do_relu);
}

/* ── conv1x1_bd4: 1×1 projection shortcut with BD4-packed output ──────── */
/*                                                                            */
/* Same as conv1x1 but output is BD4-packed via quantize_output_bd4().       */
/* accum_tmp must hold out_c × oh × ow int32 elements.                       */
static void conv1x1_bd4(
    const int8_t  *input,
    uint8_t       *bd_out,
    int32_t       *accum_tmp,
    const uint8_t *w_blocks,
    const float   *bias,
    int in_c, int out_c,
    int h, int w,
    int stride,
    int out_shift,
    int do_relu
) {
    int oh = h / stride;
    int ow = w / stride;
    int out_elems       = out_c * oh * ow;
    int kernel_elems    = in_c;
    int n_blocks_per_oc = (kernel_elems + 31) / 32;

    /* Zero accumulator */
    for (int i = 0; i < out_elems; i++) accum_tmp[i] = 0;

    const uint8_t *w_ptr = w_blocks;

    for (int oc = 0; oc < out_c; oc++) {
        int32_t bias_i32 = bias ? bias_scale(bias + oc, out_shift) : 0;

        for (int y = 0; y < oh; y++) {
            for (int x = 0; x < ow; x++) {
                int32_t acc       = 0;
                const uint8_t *bp = w_ptr;
                int elem_done     = 0;

                for (int b = 0; b < n_blocks_per_oc; b++) {
                    int16_t hu[32];
                    int did, seb;
                    bd_decode_block_hu(bp, hu, &did, &seb);
                    bp += BD_BLOCK_BYTES;

                    int count = kernel_elems - elem_done;
                    if (count > 32) count = 32;

                    int shift    = seb - 16;
                    int32_t bsum = 0;

                    for (int i = 0; i < count; i++) {
                        int ic  = elem_done + i;
                        int iy  = y * stride;
                        int ix_ = x * stride;
                        int8_t a = input[ic * h * w + iy * w + ix_];
                        bsum += (int32_t)hu[i] * (int32_t)a;
                    }

                    if (shift >= 0) acc += bsum << shift;
                    else            acc += bsum >> (-shift);

                    elem_done += count;
                }

                accum_tmp[oc * oh * ow + y * ow + x] = acc + bias_i32;
            }
        }

        w_ptr += n_blocks_per_oc * BD_BLOCK_BYTES;
    }

    /* Pack entire int32 accumulator output tensor → BD4 (no int8 intermediate) */
    quantize_output_bd4(accum_tmp, bd_out, (uint32_t)out_elems, out_shift, do_relu);
}

/* ── Run one BasicBlock (identity or projection) ────────────────────── */
/*
 * Phase 1 (int8 activations):
 *
 *   buf_in   → conv_a(3×3 s=stride, BN, ReLU) → buf_mid
 *   buf_mid  → conv_b(3×3 s=1, BN) → buf_out
 *   [if projection] buf_skip → conv1x1(s=stride, BN) → proj_buf (reuse buf_mid)
 *   buf_out += skip (add + ReLU) → buf_out is the final output
 *
 * Buffer ownership rules:
 *   buf_in   = input activation, will be preserved for skip connection
 *   buf_mid  = temporary intermediate (conv_a output)
 *   buf_out  = final block output (add+ReLU result)
 *   skip_buf = copy of input saved before conv_a overwrites buf_in
 *              (used when we need to reclaim buf_in as buf_mid output)
 *
 * For identity blocks: skip = original input (saved in skip_buf).
 * For projection blocks: skip = conv1x1(original input, proj weight).
 *
 * The caller must ensure:
 *   - buf_in  has in_c × in_h × in_w elements
 *   - buf_mid has max(conv_a output, proj output) elements
 *   - buf_out has out_c × (in_h/stride) × (in_w/stride) elements
 *   - skip_buf has in_c × in_h × in_w elements
 *
 * out_shift is per-stage and kept fixed for Milestone 1.
 */
static void run_basic_block(
    const BasicBlockConf   *conf,
    const vwb2_header_t    *hdr,
    const int8_t           *buf_in,    /* input: in_c × in_h × in_w   */
    int8_t                 *buf_mid,   /* scratch: max(conv_a, proj)   */
    int8_t                 *buf_out,   /* output: out_c × oh × ow      */
    int8_t                 *skip_buf,  /* saved input for residual      */
    int                     out_shift  /* int32 → int8 right-shift     */
) {
    int in_c  = conf->in_c;
    int out_c = conf->out_c;
    int ih    = conf->in_h;
    int iw    = conf->in_w;
    int s     = conf->stride;
    int oh    = ih / s;
    int ow    = iw / s;

    int in_elems  = in_c * ih * iw;
    int out_elems = out_c * oh * ow;

    /* Step 0: save input to skip_buf (always needed for skip connection) */
    {
        const int8_t *src = buf_in;
        int8_t *dst = skip_buf;
        for (int i = 0; i < in_elems; i++) dst[i] = src[i];
    }

    /* Step 1: conv_a  (3×3, stride, BN, ReLU) → buf_mid */
    const uint8_t *wa = rn1202_weight_blocks(hdr, conf->tid_conv_a);
    const float   *ba = rn1202_bias_f32(hdr,      conf->tid_conv_a);
    conv3x3(buf_in, buf_mid, wa, ba,
            in_c, out_c, ih, iw, s, out_shift, /*do_relu=*/1);

    /* Step 2: conv_b  (3×3, stride=1, BN, no ReLU) → buf_out */
    const uint8_t *wb = rn1202_weight_blocks(hdr, conf->tid_conv_b);
    const float   *bb = rn1202_bias_f32(hdr,      conf->tid_conv_b);
    conv3x3(buf_mid, buf_out, wb, bb,
            out_c, out_c, oh, ow, 1, out_shift, /*do_relu=*/0);

    /* Step 3: compute skip tensor (add into buf_out then relu) */
    if (conf->has_proj) {
        /* Option B: skip = conv1×1(original_input, proj_weight, stride=2)    */
        /* Write result into buf_mid (reuse; conv_a output no longer needed) */
        const uint8_t *wp = rn1202_weight_blocks(hdr, conf->tid_proj);
        const float   *bp = rn1202_bias_f32(hdr,      conf->tid_proj);
        conv1x1(skip_buf, buf_mid, wp, bp,
                in_c, out_c, ih, iw, s, out_shift, /*do_relu=*/0);

        /* buf_out += buf_mid (projection skip), then ReLU */
        add_relu_inplace(buf_out, buf_mid, out_elems);
    } else if (s == 2) {
        /* Option A downsampling block: zero-pad-stride-2 skip into buf_mid  */
        /* (buf_mid output from conv_a is no longer needed at this point)    */
        zero_pad_stride2(skip_buf, buf_mid, in_c, out_c, ih, iw);
        add_relu_inplace(buf_out, buf_mid, out_elems);
    } else {
        /* Identity skip: same spatial and channel dims, add directly */
        add_relu_inplace(buf_out, skip_buf, out_elems);
    }
    /* buf_out is now the block's output after add+ReLU */
}

/* ── Phase 2 extension: BD4 skip_buf ────────────────────────────────── */
/*
 * Phase 2 replaces the int8 skip_buf copy with a BD4-packed version to
 * save memory and exercise the bd_act round-trip on real skip tensors.
 *
 * The caller passes bd_skip_buf (a uint8_t[] of BD_BLOCK_BYTES-sized blocks)
 * and the same int8 skip_buf for unpacking back before the residual add.
 */
static void run_basic_block_bd_skip(
    const BasicBlockConf   *conf,
    const vwb2_header_t    *hdr,
    const int8_t           *buf_in,
    int8_t                 *buf_mid,
    int8_t                 *buf_out,
    int8_t                 *skip_buf_i8,   /* temp for unpack */
    uint8_t                *bd_skip_buf,   /* BD4 storage for skip  */
    int                     out_shift
) {
    int in_c  = conf->in_c;
    int out_c = conf->out_c;
    int ih    = conf->in_h;
    int iw    = conf->in_w;
    int s     = conf->stride;
    int oh    = ih / s;
    int ow    = iw / s;
    int in_elems  = in_c * ih * iw;
    int out_elems = out_c * oh * ow;

    /* Step 0: pack input into BD4 skip buffer (two-stage paper algorithm §3.2) */
    uint32_t n_skip_blocks = bd_act_pack_tensor_twostage(buf_in, (uint32_t)in_elems, bd_skip_buf);

    /* Steps 1+2: same as phase1 */
    const uint8_t *wa = rn1202_weight_blocks(hdr, conf->tid_conv_a);
    const float   *ba = rn1202_bias_f32(hdr,      conf->tid_conv_a);
    conv3x3(buf_in, buf_mid, wa, ba,
            in_c, out_c, ih, iw, s, out_shift, 1);

    const uint8_t *wb = rn1202_weight_blocks(hdr, conf->tid_conv_b);
    const float   *bb = rn1202_bias_f32(hdr,      conf->tid_conv_b);
    conv3x3(buf_mid, buf_out, wb, bb,
            out_c, out_c, oh, ow, 1, out_shift, 0);

    /* Step 3: unpack skip, compute residual */
    if (conf->has_proj) {
        /* Option B: unpack BD4 skip, then conv1×1 projection */
        bd_act_unpack_tensor(bd_skip_buf, n_skip_blocks, skip_buf_i8, (uint32_t)in_elems);

        const uint8_t *wp = rn1202_weight_blocks(hdr, conf->tid_proj);
        const float   *bp = rn1202_bias_f32(hdr,      conf->tid_proj);
        conv1x1(skip_buf_i8, buf_mid, wp, bp,
                in_c, out_c, ih, iw, s, out_shift, 0);

        add_relu_inplace(buf_out, buf_mid, out_elems);
    } else if (s == 2) {
        /* Option A downsampling: unpack BD4 skip then zero-pad-stride-2     */
        bd_act_unpack_tensor(bd_skip_buf, n_skip_blocks, skip_buf_i8, (uint32_t)in_elems);
        zero_pad_stride2(skip_buf_i8, buf_mid, in_c, out_c, ih, iw);
        add_relu_inplace(buf_out, buf_mid, out_elems);
    } else {
        /* Identity skip: unpack BD4 directly and add */
        bd_act_unpack_tensor(bd_skip_buf, n_skip_blocks, skip_buf_i8, (uint32_t)in_elems);
        add_relu_inplace(buf_out, skip_buf_i8, out_elems);
    }
}

/* ══════════════════════════════════════════════════════════════════════════ */
/* Step 3 (bd-activations-fix §3): BD4 activation input + BDMac32 dot product */
/*                                                                            */
/* The following variants replace the int8 activation read path with a        */
/* BD4-aware gather (via bd4_read_cached) and feed both weight and activation  */
/* blocks directly to bdmac32_mac_block() for hardware-accelerated dot        */
/* products.  They are compiled ONLY when bd_decode_hw.h has been included    */
/* first (providing bdmac32_mac_block).                                        */
#ifdef BD_DECODE_HW_H

/* ── bd4_read_cached: single-element read from a BD4-packed tensor ────── */
/*                                                                            */
/* Maintains a per-tensor block cache: on cache miss the 18-byte block is    */
/* unpacked once into a 32-int8 buffer; on hit the buffered value is         */
/* returned directly.  Switching tensors (different bd pointer) flushes the  */
/* cache automatically.                                                        */
static __attribute__((unused)) inline int8_t bd4_read_cached(const uint8_t *bd, int flat_idx)
{
    static int            s_blk = -1;
    static const uint8_t *s_ptr = (const uint8_t *)0;
    static int8_t         s_buf[32];

    int bi = flat_idx / 32;
    if (bi != s_blk || bd != s_ptr) {
        bd_act_unpack32(bd + (unsigned)bi * BD_BLOCK_BYTES, s_buf);
        s_blk = bi;
        s_ptr = bd;
    }
    return s_buf[flat_idx % 32];
}

/* ── conv3x3_bd4_hwmac: 3×3 conv, BD4 input, BDMac32 dot product ───────── */
/*                                                                            */
/* Same accumulation as conv3x3_bd4 but activations come from a BD4-packed   */
/* tensor.  For each weight block the corresponding activation elements are   */
/* gathered element-by-element via bd4_read_cached(), assembled into a 32-   */
/* element int32 buffer, BD4-packed, and fed to bdmac32_mac_block together   */
/* with the weight block.                                                      */
/*                                                                            */
/* Scaling:  acc += ps * 2^(exp_sum - 32)                                    */
/*           The accumulator domain matches conv3x3_bd4 (real dot product).  */
/*            After looping: quantize_output_bd4(accum_tmp, …, out_shift).   */
/*                                                                            */
/* Parameters: identical to conv3x3_bd4 except input_bd4 replaces input.    */
static inline void conv3x3_bd4_hwmac(
    const uint8_t *input_bd4,   /* BD4-packed input CHW, in_c × h × w    */
    uint8_t       *bd_out,      /* BD4 output, ceil(out_c*oh*ow/32)*18 B  */
    int32_t       *accum_tmp,   /* int32 scratch, out_c × oh × ow elems  */
    const uint8_t *w_blocks,    /* BD4 weight blocks                      */
    const float   *bias,
    int in_c, int out_c,
    int h, int w,
    int stride,
    int out_shift,
    int do_relu
) {
    int oh = h / stride;
    int ow = w / stride;
    int out_elems       = out_c * oh * ow;
    int kernel_elems    = in_c * 9;
    int n_blocks_per_oc = (kernel_elems + 31) / 32;

    for (int i = 0; i < out_elems; i++) accum_tmp[i] = 0;

    const uint8_t *w_ptr = w_blocks;

    for (int oc = 0; oc < out_c; oc++) {
        int32_t bias_i32 = bias ? bias_scale(bias + oc, out_shift) : 0;

        for (int y = 0; y < oh; y++) {
            for (int x = 0; x < ow; x++) {
                int32_t acc       = 0;
                const uint8_t *bp = w_ptr;
                int elem_done     = 0;

                for (int b = 0; b < n_blocks_per_oc; b++) {
                    int count = kernel_elems - elem_done;
                    if (count > 32) count = 32;

                    /* Gather activation elements into int32 buffer */
                    int32_t a_gather[32];
                    for (int i = 0; i < 32; i++) {
                        if (i < count) {
                            int flat = elem_done + i;
                            int ic   = flat / 9;
                            int krem = flat % 9;
                            int ky   = krem / 3;
                            int kx_  = krem % 3;
                            int iy   = y * stride - 1 + ky;
                            int ix_  = x * stride - 1 + kx_;
                            if (iy >= 0 && iy < h && ix_ >= 0 && ix_ < w)
                                a_gather[i] = (int32_t)bd4_read_cached(
                                    input_bd4, ic * h * w + iy * w + ix_);
                            else
                                a_gather[i] = 0;
                        } else {
                            a_gather[i] = 0;  /* zero-pad partial block */
                        }
                    }

                    /* Pack gathered activations into BD4 block */
                    uint8_t a_block[BD_BLOCK_BYTES];
                    bd_act_pack32_twostage(a_gather, a_block);

                    /* Hardware dot product: ps in half-units², es = w_seb+a_seb */
                    int es;
                    int32_t ps = bdmac32_mac_block(bp, a_block, &es);

                    /* Scale: real_dp = ps * 2^(es-32); accumulate in same domain */
                    int shift = es - 32;
                    if (shift >= 0) acc += ps <<  shift;
                    else            acc += ps >> -shift;

                    bp += BD_BLOCK_BYTES;
                    elem_done += count;
                }

                accum_tmp[oc * oh * ow + y * ow + x] = acc + bias_i32;
            }
        }

        w_ptr += n_blocks_per_oc * BD_BLOCK_BYTES;
    }

    quantize_output_bd4(accum_tmp, bd_out, (uint32_t)out_elems, out_shift, do_relu);
}

/* ── conv1x1_bd4_hwmac: 1×1 projection conv, BD4 input, BDMac32 ─────────── */
/*                                                                             */
/* Same structure as conv3x3_bd4_hwmac but 1×1 kernel (no ky/kx loop).       */
/* Each activation element is at input[ic, y*stride, x*stride] in CHW order, */
/* gathered via bd4_read_cached from the BD4-packed input tensor.             */
static inline void conv1x1_bd4_hwmac(
    const uint8_t *input_bd4,
    uint8_t       *bd_out,
    int32_t       *accum_tmp,
    const uint8_t *w_blocks,
    const float   *bias,
    int in_c, int out_c,
    int h, int w,
    int stride,
    int out_shift,
    int do_relu
) {
    int oh = h / stride;
    int ow = w / stride;
    int out_elems       = out_c * oh * ow;
    int kernel_elems    = in_c;
    int n_blocks_per_oc = (kernel_elems + 31) / 32;

    for (int i = 0; i < out_elems; i++) accum_tmp[i] = 0;

    const uint8_t *w_ptr = w_blocks;

    for (int oc = 0; oc < out_c; oc++) {
        int32_t bias_i32 = bias ? bias_scale(bias + oc, out_shift) : 0;

        for (int y = 0; y < oh; y++) {
            for (int x = 0; x < ow; x++) {
                int32_t acc       = 0;
                const uint8_t *bp = w_ptr;
                int elem_done     = 0;

                for (int b = 0; b < n_blocks_per_oc; b++) {
                    int count = kernel_elems - elem_done;
                    if (count > 32) count = 32;

                    int32_t a_gather[32];
                    for (int i = 0; i < 32; i++) {
                        if (i < count) {
                            int ic  = elem_done + i;
                            int iy  = y * stride;
                            int ix_ = x * stride;
                            a_gather[i] = (int32_t)bd4_read_cached(
                                input_bd4, ic * h * w + iy * w + ix_);
                        } else {
                            a_gather[i] = 0;
                        }
                    }

                    uint8_t a_block[BD_BLOCK_BYTES];
                    bd_act_pack32_twostage(a_gather, a_block);

                    int es;
                    int32_t ps = bdmac32_mac_block(bp, a_block, &es);

                    int shift = es - 32;
                    if (shift >= 0) acc += ps <<  shift;
                    else            acc += ps >> -shift;

                    bp += BD_BLOCK_BYTES;
                    elem_done += count;
                }

                accum_tmp[oc * oh * ow + y * ow + x] = acc + bias_i32;
            }
        }

        w_ptr += n_blocks_per_oc * BD_BLOCK_BYTES;
    }

    quantize_output_bd4(accum_tmp, bd_out, (uint32_t)out_elems, out_shift, do_relu);
}

/* ── fc_linear_bd4_hwmac: FC layer, BD4 input, BDMac32 dot product ──────── */
/*                                                                             */
/* Same as fc_linear but the flat input vector is stored BD4-packed.          */
/* Each weight block's activation elements are consecutive in the flat input  */
/* (offset = elem_done .. elem_done+count-1), fetched via bd4_read_cached.    */
/* Output is int32 logits (no quantization).                                  */
static inline void fc_linear_bd4_hwmac(
    const uint8_t *input_bd4,   /* BD4-packed flat input, in_c elements  */
    int32_t       *logits,
    const uint8_t *w_blocks,
    const float   *bias,
    int in_c,
    int out_c
) {
    int n_blocks_per_oc = (in_c + 31) / 32;
    const uint8_t *w_ptr = w_blocks;

    for (int oc = 0; oc < out_c; oc++) {
        int32_t acc       = 0;
        const uint8_t *bp = w_ptr;
        int elem_done     = 0;

        for (int b = 0; b < n_blocks_per_oc; b++) {
            int count = in_c - elem_done;
            if (count > 32) count = 32;

            /* Flat-input gather: elements are consecutive, cache always hits */
            int32_t a_gather[32];
            for (int i = 0; i < 32; i++) {
                a_gather[i] = (i < count)
                    ? (int32_t)bd4_read_cached(input_bd4, elem_done + i)
                    : 0;
            }

            uint8_t a_block[BD_BLOCK_BYTES];
            bd_act_pack32_twostage(a_gather, a_block);

            int es;
            int32_t ps = bdmac32_mac_block(bp, a_block, &es);

            int shift = es - 32;
            if (shift >= 0) acc += ps <<  shift;
            else            acc += ps >> -shift;

            bp += BD_BLOCK_BYTES;
            elem_done += count;
        }

        if (bias)
            acc += (int32_t)(bias[oc] * 128.0f);
        logits[oc] = acc;

        w_ptr += n_blocks_per_oc * BD_BLOCK_BYTES;
    }
}

/* ── add_relu_bd4: BD4 residual add + ReLU (paper-faithful, no int8 intermediate) ──
 *
 * Unpacks both input blocks, adds element-wise, applies ReLU, repacks to BD4.
 * bd_out may alias bd_a or bd_b (block-at-a-time processing is safe).
 */
static void add_relu_bd4(
    const uint8_t *bd_a,
    const uint8_t *bd_b,
    uint8_t       *bd_out,
    uint32_t       n_elements
) {
    uint32_t n_blocks = (n_elements + 31u) / 32u;
    int8_t  va[32], vb[32];
    int32_t tmp[32];
    for (uint32_t b = 0; b < n_blocks; b++) {
        bd_act_unpack32(bd_a + b * BD_BLOCK_BYTES, va);
        bd_act_unpack32(bd_b + b * BD_BLOCK_BYTES, vb);
        for (int i = 0; i < 32; i++) {
            int32_t s = (int32_t)va[i] + (int32_t)vb[i];
            tmp[i] = (s < 0) ? 0 : s;  /* ReLU */
        }
        bd_act_pack32_twostage(tmp, bd_out + b * BD_BLOCK_BYTES);
    }
}

/* ── bd4_zero_pad_stride2: stride-2 skip from BD4 input, no int8 intermediate ──
 *
 * Subsamples bd_in by stride=2, zero-pads channels [in_c, out_c), and packs
 * result into bd_out.  Uses accum_tmp (int32, out_c*oh*ow elements) as scratch.
 * Reads via bd4_read_cached so no full unpack is needed.
 */
static void bd4_zero_pad_stride2(
    const uint8_t *bd_in,
    uint8_t       *bd_out,
    int32_t       *tmp,
    int in_c, int out_c, int h, int w
) {
    int oh = h / 2, ow = w / 2;
    int out_elems = out_c * oh * ow;
    for (int i = 0; i < out_elems; i++) tmp[i] = 0;
    for (int ic = 0; ic < in_c; ic++)
        for (int y = 0; y < oh; y++)
            for (int x = 0; x < ow; x++)
                tmp[ic * oh * ow + y * ow + x] =
                    (int32_t)bd4_read_cached(bd_in, ic * h * w + y * 2 * w + x * 2);
    uint32_t nb = ((uint32_t)out_elems + 31u) / 32u;
    for (uint32_t b = 0; b < nb; b++)
        bd_act_pack32_twostage(tmp + (int)(b * 32u), bd_out + b * BD_BLOCK_BYTES);
}

/* ── run_basic_block_bd4: BD4 in → BD4 out, no int8 anywhere ──────────────
 *
 * Paper-faithful BasicBlock with BULK UNPACK optimisation:
 *   Instead of reading activations element-by-element from BD4 via
 *   bd4_read_cached (massive cache thrashing), we bulk-unpack the BD4
 *   input to int8 once, then call conv3x3_bd4 (int8-input path).
 *   This eliminates both the repeated unpack overhead AND the per-pixel
 *   BD4 repack needed by conv3x3_bd4_hwmac.  ~5× faster.
 *
 * Dataflow:
 *   bd_in (BD4) → unpack → int8 → conv3x3_bd4 → bd_mid (BD4)
 *   bd_mid (BD4) → unpack → int8 → conv3x3_bd4 → bd_out (BD4)
 *   skip:  identity  → add_relu_bd4(bd_out, bd_in,   bd_out)
 *          stride=2  → unpack bd_in, zero-pad, pack → add_relu_bd4
 *          has_proj  → unpack bd_in → conv1x1_bd4   → add_relu_bd4
 *
 * unpack_buf must hold ACT_MAX_INT8_SIZE (16384) bytes.
 */
static void run_basic_block_bd4(
    const BasicBlockConf *conf,
    const vwb2_header_t  *hdr,
    const uint8_t        *bd_in,
    uint8_t              *bd_mid,   /* scratch for conv_a output */
    uint8_t              *bd_out,   /* final block output        */
    uint8_t              *bd_skip,  /* scratch for skip tensor (may == bd_mid) */
    int32_t              *accum_tmp,
    int                   out_shift, /* legacy: unused, prefer conf->shift_a/b */
    int8_t               *unpack_buf /* int8 bulk-unpack cache */
) {
    int in_c  = conf->in_c,  out_c = conf->out_c;
    int ih    = conf->in_h,  iw    = conf->in_w;
    int s     = conf->stride;
    int oh    = ih / s,      ow    = iw / s;
    int in_elems  = in_c * ih * iw;
    int out_elems = out_c * oh * ow;

    /* Per-conv shifts from BasicBlockConf (calibrated by quantized_reference.py). */
    int sa = (int)conf->shift_a;
    int sb = (int)conf->shift_b;
    int sp = (int)conf->shift_proj;
    (void)out_shift;

    /* ── Bulk-unpack bd_in → int8 (one-time, cheap) ──────────────────── */
    uint32_t n_in_blks = ((uint32_t)in_elems + 31u) / 32u;
    bd_act_unpack_tensor(bd_in, n_in_blks, unpack_buf, (uint32_t)in_elems);

    /* conv_a: int8 input → BD4 mid, ReLU=1 */
    const uint8_t *wa = rn1202_weight_blocks(hdr, conf->tid_conv_a);
    const float   *ba = rn1202_bias_f32(hdr, conf->tid_conv_a);
    conv3x3_bd4(unpack_buf, bd_mid, accum_tmp, wa, ba,
                in_c, out_c, ih, iw, s, sa, 1);

    /* ── Bulk-unpack bd_mid → int8 for conv_b ─────────────────────────── */
    uint32_t n_mid_blks = ((uint32_t)out_elems + 31u) / 32u;
    bd_act_unpack_tensor(bd_mid, n_mid_blks, unpack_buf, (uint32_t)out_elems);

    /* conv_b: int8 input → BD4 out, ReLU=0 */
    const uint8_t *wb = rn1202_weight_blocks(hdr, conf->tid_conv_b);
    const float   *bb = rn1202_bias_f32(hdr, conf->tid_conv_b);
    conv3x3_bd4(unpack_buf, bd_out, accum_tmp, wb, bb,
                out_c, out_c, oh, ow, 1, sb, 0);

    /* ── Skip connection + ReLU ────────────────────────────────────────── */
    if (conf->has_proj) {
        /* Option B: unpack bd_in again for projection conv1×1 */
        bd_act_unpack_tensor(bd_in, n_in_blks, unpack_buf, (uint32_t)in_elems);
        const uint8_t *wp = rn1202_weight_blocks(hdr, conf->tid_proj);
        const float   *bp = rn1202_bias_f32(hdr, conf->tid_proj);
        conv1x1_bd4(unpack_buf, bd_skip, accum_tmp, wp, bp,
                    in_c, out_c, ih, iw, s, sp, 0);
        add_relu_bd4(bd_out, bd_skip, bd_out, (uint32_t)out_elems);
    } else if (s == 2) {
        /* Option A: unpack bd_in, zero-pad stride-2, pack BD4 skip */
        bd_act_unpack_tensor(bd_in, n_in_blks, unpack_buf, (uint32_t)in_elems);
        int out_spatial = oh * ow;
        for (int i = 0; i < out_elems; i++) accum_tmp[i] = 0;
        for (int ic = 0; ic < in_c; ic++)
            for (int y = 0; y < oh; y++)
                for (int x = 0; x < ow; x++)
                    accum_tmp[ic * out_spatial + y * ow + x] =
                        (int32_t)unpack_buf[ic * ih * iw + y * 2 * iw + x * 2];
        uint32_t nb = ((uint32_t)out_elems + 31u) / 32u;
        for (uint32_t b = 0; b < nb; b++)
            bd_act_pack32_twostage(accum_tmp + (int)(b * 32u),
                                   bd_skip + b * BD_BLOCK_BYTES);
        add_relu_bd4(bd_out, bd_skip, bd_out, (uint32_t)out_elems);
    } else {
        /* Identity: bd_in same shape as bd_out */
        add_relu_bd4(bd_out, bd_in, bd_out, (uint32_t)out_elems);
    }
}

/* ── global_avgpool_bd4: BD4 C×H×W → int8 C (one value per channel) ──── */
static void global_avgpool_bd4(const uint8_t *bd_in, int8_t *output, int c, int h, int w)
{
    int spatial = h * w;
    for (int ch = 0; ch < c; ch++) {
        int32_t sum = 0;
        int base = ch * spatial;
        for (int i = 0; i < spatial; i++)
            sum += (int32_t)bd4_read_cached(bd_in, base + i);
        int32_t avg = sum / spatial;
        if (avg >  127) avg =  127;
        if (avg < -128) avg = -128;
        output[ch] = (int8_t)avg;
    }
}

/* ══════════════════════════════════════════════════════════════════════════
 * §5  CHW → block-aligned tiling (paper §3.2, per-tap channel blocking)
 *
 * BlockDialect quantises each operand "along its respective multiplication
 * dimension" (arXiv:2501.01144v5 §3).  For a 3×3 conv the reduction chunk
 * for one output pixel is (IC, KY, KX).  Tiling with a flat [IC*9] layout
 * mixes spatial taps inside one BD block, forcing a scatter-gather of 32
 * values from up to 9 different spatial positions (cache-hostile).
 *
 * TAP-BLOCKED layout instead maps every 32-element BD block to a *single
 * (KY,KX) tap* over consecutive input channels:
 *   weight block [oc, tap=ky*3+kx, cb] covers  ic = cb*32 .. cb*32+31
 *
 * Combined with the HWCB activation layout (§4 of bd_act.h), each MAC pair
 *   bdmac32_mac_block(w[oc,tap,cb],  a[iy,ix,cb])
 * is a "clean" single-tap channel-block dot product — both operands cover
 * the same 32 input channels at the same spatial tap, exactly as intended
 * by the paper.  Activation blocks for the 9 taps of one output row are
 * sequential in memory as the output x-coordinate advances.
 *
 * Weight storage order per output channel (oc):
 *   tap 0 (ky=0,kx=0), cb=0 .. n_cb-1
 *   tap 1 (ky=0,kx=1), cb=0 .. n_cb-1
 *   ...
 *   tap 8 (ky=2,kx=2), cb=0 .. n_cb-1
 *   ──────────────────────────────────
 *   total n_blocks_per_oc = 9 * ceil(in_c / 32)
 *
 * Python export: gen_resnet1202_model.py --tap-blocked permutes conv3×3
 * weights from [OC,IC,KY,KX] to [OC,KY,KX,IC] before BD4 encoding.
 *
 * conv1×1 (projection) and FC weights are unaffected (no spatial kernel).
 * ══════════════════════════════════════════════════════════════════════════ */

/* ── conv3x3_bd4_tap: int8-input, HWCB-output, tap-blocked weights ────── */
/*
 * Entry point for layers whose input is still int8 (stem conv) or when a
 * caller has already unpacked activations to int8.  Uses software weight
 * decode (bd_decode_block_hu).  Output is packed directly into HWCB BD4
 * format, one spatial position at a time, with no large accumulator scratch:
 * only a stack array of `out_c` int32 elements (≤ 256 B) is needed.
 *
 * Weight layout (tap-major, channel-inner):
 *   w_blocks[oc][tap][cb]  each block = bd_decode_block_hu target
 *   n_blocks_per_oc = 9 * ceil(in_c/32)
 */
static void conv3x3_bd4_tap(
    const int8_t  *input,      /* int8 CHW, in_c × h × w                   */
    uint8_t       *bd_out,     /* HWCB BD4 output, out_c × oh × ow          */
    const uint8_t *w_blocks,   /* tap-blocked weights (see layout above)     */
    const float   *bias,
    int in_c, int out_c,
    int h, int w,
    int stride,
    int out_shift,
    int do_relu
) {
    int oh = h / stride, ow = w / stride;
    int n_cb_in  = hwcb_n_cb(in_c);
    int n_cb_out = hwcb_n_cb(out_c);
    int n_wblocks_per_oc = 9 * n_cb_in;

    int32_t accum_row[64]; /* max out_c = 64; only 256 B stack */
    int32_t tmp[32];

    /* Pre-compute bias once — loop-invariant w.r.t. (y,x) */
    int32_t bias_init[64];
    for (int oc = 0; oc < out_c; oc++)
        bias_init[oc] = bias ? bias_scale(bias + oc, out_shift) : 0;

    for (int y = 0; y < oh; y++) {
        for (int x = 0; x < ow; x++) {

            /* ── Per-output-channel dot product (bias pre-loaded from bias_init) ── */
            for (int oc = 0; oc < out_c; oc++) {
                int32_t acc = bias_init[oc];
                const uint8_t *w_oc = w_blocks +
                    (uint32_t)(oc * n_wblocks_per_oc) * BD_BLOCK_BYTES;

                for (int tap = 0; tap < 9; tap++) {
                    int ky = tap / 3, kx = tap % 3;
                    int iy = y * stride - 1 + ky;
                    int ix = x * stride - 1 + kx;
                    /* Boundary check: padding = 0, skip (contributes nothing) */
                    if (iy < 0 || iy >= h || ix < 0 || ix >= w) continue;

                    const uint8_t *wb = w_oc + (uint32_t)(tap * n_cb_in) * BD_BLOCK_BYTES;

                    for (int cb = 0; cb < n_cb_in; cb++) {
                        int16_t hu[32];
                        int did, seb;
                        bd_decode_block_hu(wb + (uint32_t)cb * BD_BLOCK_BYTES,
                                          hu, &did, &seb);
                        int ic_base = cb * 32;
                        int count   = in_c - ic_base;
                        if (count > 32) count = 32;
                        int shift  = seb - 16;
                        int32_t bsum = 0;
                        for (int i = 0; i < count; i++)
                            bsum += (int32_t)hu[i] *
                                    (int32_t)input[(ic_base + i) * h * w + iy * w + ix];
                        if (shift >= 0) acc += bsum <<  shift;
                        else            acc += bsum >> -shift;
                    }
                }
                accum_row[oc] = acc;
            }

            /* ── Pack HWCB output blocks for (y,x) ── */
            for (int cb = 0; cb < n_cb_out; cb++) {
                int ic_base = cb * 32;
                for (int i = 0; i < 32; i++) {
                    int oc = ic_base + i;
                    if (oc < out_c) {
                        int32_t v = accum_row[oc] >> out_shift;
                        if (do_relu && v < 0) v = 0;
                        tmp[i] = v;
                    } else {
                        tmp[i] = 0;
                    }
                }
                bd_act_pack32_twostage(tmp,
                    hwcb_block_ptr(bd_out, y, x, ow, n_cb_out, cb));
            }
        }
    }
}

/* ── conv3x3_bd4_tap_hwmac: HWCB-input, HWCB-output, BDMac32 ───────────── */
/*
 * Fully hardware-accelerated 3×3 conv with tap-blocked weights and HWCB
 * activations.  For each output pixel (y,x):
 *
 *   1. Prefetch pointers to the 9×n_cb activation blocks for this pixel
 *      (no copy — just record the addresses in the HWCB buffer).
 *
 *   2. For each output channel oc:
 *      For each tap t, channel-block cb:
 *        bdmac32_mac_block(w[oc,t,cb],  act[iy,ix,cb])  → int32 partial sum
 *      Accumulate → accum_row[oc].
 *
 *   3. Pack accum_row[0..out_c-1] into the n_cb_out HWCB output blocks for
 *      spatial (y,x) in a single pass.
 *
 * Memory traffic:
 *   Activation blocks for the 9 output-row taps are sequential in HWCB
 *   memory as x advances → linear read pattern for the BD-MAC unit.
 *   Each activation block for this pixel is shared across all out_c
 *   channels → loaded 1×, used out_c times.
 *
 * No large int32 accumulator scratch; stack usage: out_c×4 + tap_ptrs stack.
 */
static inline void conv3x3_bd4_tap_hwmac(
    const uint8_t *bd_in,     /* HWCB BD4 input, in_c × h × w              */
    uint8_t       *bd_out,    /* HWCB BD4 output, out_c × oh × ow           */
    const uint8_t *w_blocks,  /* tap-blocked BD4 weights                    */
    const float   *bias,
    int in_c, int out_c,
    int h, int w,
    int stride,
    int out_shift,
    int do_relu
) {
    int oh = h / stride, ow = w / stride;
    int n_cb_in  = hwcb_n_cb(in_c);
    int n_cb_out = hwcb_n_cb(out_c);
    int n_wblocks_per_oc = 9 * n_cb_in;

    /* Static all-zeros BD block for out-of-bounds (padding) taps */
    static const uint8_t s_zero_block[BD_BLOCK_BYTES];  /* zero-initialised */

    /* Stack: activation block pointers for all 9 taps × n_cb_in channel blocks.
     * max n_cb_in = 2 (stage3, 64ch), so 9*2 = 18 pointers = 72 B stack.  */
    const uint8_t *tap_act[9][2]; /* [tap][cb]: max 9 taps, 2 chan-blocks */

    int32_t accum_row[64]; /* max out_c = 64 → 256 B stack */
    int32_t tmp[32];

    /* Pre-compute bias once — loop-invariant w.r.t. (y,x) */
    int32_t bias_init[64];
    for (int oc = 0; oc < out_c; oc++)
        bias_init[oc] = bias ? bias_scale(bias + oc, out_shift) : 0;

    for (int y = 0; y < oh; y++) {
        for (int x = 0; x < ow; x++) {

            /* ── Step 1: prefetch activation block ptrs for all 9 taps ── */
            for (int tap = 0; tap < 9; tap++) {
                int ky = tap / 3, kx = tap % 3;
                int iy = y * stride - 1 + ky;
                int ix = x * stride - 1 + kx;
                int in_bounds = (iy >= 0 && iy < h && ix >= 0 && ix < w);
                for (int cb = 0; cb < n_cb_in; cb++) {
                    tap_act[tap][cb] = in_bounds
                        ? hwcb_block_ptr_r(bd_in, iy, ix, w, n_cb_in, cb)
                        : s_zero_block;
                }
            }

            /* ── Step 2+3: per-channel dot products via BDMac32 (bias pre-loaded) ── */
            for (int oc = 0; oc < out_c; oc++) {
                int32_t acc = bias_init[oc];
                const uint8_t *w_oc = w_blocks +
                    (uint32_t)(oc * n_wblocks_per_oc) * BD_BLOCK_BYTES;

                for (int tap = 0; tap < 9; tap++) {
                    for (int cb = 0; cb < n_cb_in; cb++) {
                        int es;
                        int32_t ps = bdmac32_mac_block(
                            w_oc + (uint32_t)(tap * n_cb_in + cb) * BD_BLOCK_BYTES,
                            tap_act[tap][cb], &es);
                        int shift = es - 32;
                        if (shift >= 0) acc += ps <<  shift;
                        else            acc += ps >> -shift;
                    }
                }
                accum_row[oc] = acc;
            }

            /* ── Step 4: pack HWCB output for (y,x) ── */
            for (int cb = 0; cb < n_cb_out; cb++) {
                int ic_base = cb * 32;
                for (int i = 0; i < 32; i++) {
                    int oc = ic_base + i;
                    if (oc < out_c) {
                        int32_t v = accum_row[oc] >> out_shift;
                        if (do_relu && v < 0) v = 0;
                        tmp[i] = v;
                    } else {
                        tmp[i] = 0;
                    }
                }
                bd_act_pack32_twostage(tmp,
                    hwcb_block_ptr(bd_out, y, x, ow, n_cb_out, cb));
            }
        }
    }
}

/* ── conv1x1_bd4_tap_hwmac: 1×1 projection, HWCB-input, BDMac32 ─────────── */
/*
 * 1×1 convolution with stride for projection shortcuts.
 * Weight layout: [oc][cb=0..n_cb_in-1] — no tap dimension (single tap).
 * This is equivalent to the flat layout for 1×1 kernels; tap permutation
 * has no effect, so existing conv1x1 weight tensors work unchanged.
 */
static inline void conv1x1_bd4_tap_hwmac(
    const uint8_t *bd_in,
    uint8_t       *bd_out,
    const uint8_t *w_blocks,
    const float   *bias,
    int in_c, int out_c,
    int h, int w,
    int stride,
    int out_shift,
    int do_relu
) {
    int oh = h / stride, ow = w / stride;
    int n_cb_in  = hwcb_n_cb(in_c);
    int n_cb_out = hwcb_n_cb(out_c);

    int32_t accum_row[64];
    int32_t tmp[32];

    /* Pre-compute bias once — loop-invariant w.r.t. (y,x) */
    int32_t bias_init[64];
    for (int oc = 0; oc < out_c; oc++)
        bias_init[oc] = bias ? bias_scale(bias + oc, out_shift) : 0;

    for (int y = 0; y < oh; y++) {
        for (int x = 0; x < ow; x++) {
            int iy = y * stride, ix = x * stride;

            for (int oc = 0; oc < out_c; oc++) {
                int32_t acc = bias_init[oc];
                const uint8_t *w_oc = w_blocks +
                    (uint32_t)(oc * n_cb_in) * BD_BLOCK_BYTES;
                for (int cb = 0; cb < n_cb_in; cb++) {
                    const uint8_t *ab =
                        hwcb_block_ptr_r(bd_in, iy, ix, w, n_cb_in, cb);
                    int es;
                    int32_t ps = bdmac32_mac_block(
                        w_oc + (uint32_t)cb * BD_BLOCK_BYTES, ab, &es);
                    int shift = es - 32;
                    if (shift >= 0) acc += ps <<  shift;
                    else            acc += ps >> -shift;
                }
                accum_row[oc] = acc;
            }

            for (int cb = 0; cb < n_cb_out; cb++) {
                int ic_base = cb * 32;
                for (int i = 0; i < 32; i++) {
                    int oc = ic_base + i;
                    if (oc < out_c) {
                        int32_t v = accum_row[oc] >> out_shift;
                        if (do_relu && v < 0) v = 0;
                        tmp[i] = v;
                    } else {
                        tmp[i] = 0;
                    }
                }
                bd_act_pack32_twostage(tmp,
                    hwcb_block_ptr(bd_out, y, x, ow, n_cb_out, cb));
            }
        }
    }
}

/* ── bd4_zero_pad_stride2_hwcb: Option-A stride-2 skip in HWCB format ──── */
/*
 * Implements He et al. Option A shortcut for the first downsampling block:
 *   - Spatial sub-sample bd_in by stride 2: (oy,ox) ← (2oy, 2ox)
 *   - Channel zero-pad: in_c input channels, zero-filled beyond in_c
 *
 * Works channel-block-by-channel-block:
 *   cb < n_cb_in  → copy HWCB block from (2oy, 2ox) verbatim
 *                   (positions ic > in_c-1 within the last block are already
 *                    zero because packing always zero-pads partial blocks)
 *   cb ≥ n_cb_in  → write all-zero BD block (meta=0, codes=0 → all values 0)
 *
 * ResNet-1202 cases:
 *   in_c=16, out_c=32: n_cb_in=1, n_cb_out=1 → one block copied per pixel ✓
 *   in_c=32, out_c=64: n_cb_in=1, n_cb_out=2 → block 0 copied, block 1 zeroed ✓
 */
static void bd4_zero_pad_stride2_hwcb(
    const uint8_t *bd_in,   /* HWCB in_c × h × w                     */
    uint8_t       *bd_out,  /* HWCB out_c × (h/2) × (w/2)            */
    int in_c, int out_c, int h, int w
) {
    int oh = h / 2, ow = w / 2;
    int n_cb_in  = hwcb_n_cb(in_c);
    int n_cb_out = hwcb_n_cb(out_c);

    for (int oy = 0; oy < oh; oy++) {
        for (int ox = 0; ox < ow; ox++) {
            for (int cb = 0; cb < n_cb_out; cb++) {
                uint8_t *dst = hwcb_block_ptr(bd_out, oy, ox, ow, n_cb_out, cb);
                if (cb < n_cb_in) {
                    const uint8_t *src =
                        hwcb_block_ptr_r(bd_in, oy * 2, ox * 2, w, n_cb_in, cb);
                    for (int b = 0; b < BD_BLOCK_BYTES; b++) dst[b] = src[b];
                } else {
                    /* zero BD block: meta=0 (dialect 0, seb 0), all codes 0 */
                    for (int b = 0; b < BD_BLOCK_BYTES; b++) dst[b] = 0;
                }
            }
        }
    }
}

/* ── global_avgpool_hwcb: HWCB BD4 C×H×W → int8 C ────────────────────── */
/*
 * Iterates spatially-major (y,x), unpacking all n_cb channel-blocks per
 * spatial position.  Accumulates channel sums in a stack array, then divides
 * by H*W.  No full-tensor unpack to int8 is needed.
 */
static void global_avgpool_hwcb(const uint8_t *bd_in, int8_t *output,
                                int c, int h, int w)
{
    int n_cb   = hwcb_n_cb(c);
    int spatial = h * w;
    int8_t  tmp[32];
    int32_t sum[64];  /* max c = 64 */

    for (int i = 0; i < c; i++) sum[i] = 0;

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            for (int cb = 0; cb < n_cb; cb++) {
                const uint8_t *blk =
                    hwcb_block_ptr_r(bd_in, y, x, w, n_cb, cb);
                bd_act_unpack32(blk, tmp);
                int ic_base = cb * 32;
                for (int i = 0; i < 32; i++) {
                    int ch = ic_base + i;
                    if (ch < c) sum[ch] += (int32_t)tmp[i];
                }
            }
        }
    }

    for (int ch = 0; ch < c; ch++) {
        int32_t avg = sum[ch] / spatial;
        if (avg >  127) avg =  127;
        if (avg < -128) avg = -128;
        output[ch] = (int8_t)avg;
    }
}

/* ── add_relu_bd4_hwcb: element-wise add + ReLU for HWCB BD4 tensors ─── */
/*
 * Iterates over all h*w*n_cb HWCB blocks.  bd_out may alias bd_a or bd_b
 * (each block is unpacked, merged, and repacked independently — safe).
 */
static inline void add_relu_bd4_hwcb(
    const uint8_t *bd_a,
    const uint8_t *bd_b,
    uint8_t       *bd_out,
    int c, int h, int w
) {
    int n_cb = hwcb_n_cb(c);
    uint32_t n_blocks = (uint32_t)(h * w * n_cb);
    int8_t  va[32], vb[32];
    int32_t tmp[32];
    for (uint32_t b = 0; b < n_blocks; b++) {
        const uint8_t *pa = bd_a   + b * BD_BLOCK_BYTES;
        const uint8_t *pb = bd_b   + b * BD_BLOCK_BYTES;
              uint8_t *po = bd_out + b * BD_BLOCK_BYTES;
        bd_act_unpack32(pa, va);
        bd_act_unpack32(pb, vb);
        for (int i = 0; i < 32; i++) {
            int32_t s = (int32_t)va[i] + (int32_t)vb[i];
            tmp[i] = s < 0 ? 0 : s;  /* ReLU */
        }
        bd_act_pack32_twostage(tmp, po);
    }
}

/* ── run_basic_block_bd4_tap: HWCB BD4 → HWCB BD4 via tap-blocked convs ──
 *
 * Paper-faithful BasicBlock using tap-blocked weights and HWCB activations.
 * No large int32 accumulator scratch and no int8 unpack buffer required;
 * all per-pixel accumulation uses an on-stack array (≤ 256 B).
 *
 * Dataflow:
 *   bd_in  (HWCB) → conv3x3_bd4_tap_hwmac → bd_mid (HWCB)  relu=1
 *   bd_mid (HWCB) → conv3x3_bd4_tap_hwmac → bd_out (HWCB)  relu=0
 *   Skip (all in HWCB):
 *     identity  → add_relu_bd4_hwcb(bd_out, bd_in,   bd_out)
 *     stride=2  → bd4_zero_pad_stride2_hwcb → bd_mid → add_relu_bd4_hwcb
 *     has_proj  → conv1x1_bd4_tap_hwmac    → bd_mid → add_relu_bd4_hwcb
 *
 * Shifts per conv are read from conf->shift_a / shift_b / shift_proj.
 * bd_mid and bd_skip may be the same buffer: the skip computation always
 * follows the conv_b write, so no aliasing hazard exists.
 */
static void run_basic_block_bd4_tap(
    const BasicBlockConf *conf,
    const vwb2_header_t  *hdr,
    const uint8_t        *bd_in,   /* HWCB BD4 input,  in_c × ih × iw */
    uint8_t              *bd_mid,  /* HWCB BD4 scratch, conv_a output  */
    uint8_t              *bd_out,  /* HWCB BD4 block output            */
    uint8_t              *bd_skip  /* HWCB BD4 scratch, skip tensor    */
) {
    int in_c  = conf->in_c,  out_c = conf->out_c;
    int ih    = conf->in_h,  iw    = conf->in_w;
    int s     = conf->stride;
    int oh    = ih / s,      ow    = iw / s;
    int sa    = (int)conf->shift_a;
    int sb    = (int)conf->shift_b;
    int sp    = (int)conf->shift_proj;

    /* conv_a: HWCB in → HWCB mid,  ReLU=1 */
    const uint8_t *wa = rn1202_weight_blocks(hdr, conf->tid_conv_a);
    const float   *ba = rn1202_bias_f32(hdr, conf->tid_conv_a);
    conv3x3_bd4_tap_hwmac(bd_in, bd_mid, wa, ba,
                          in_c, out_c, ih, iw, s, sa, /*relu=*/1);

    /* conv_b: HWCB mid → HWCB out, ReLU=0 */
    const uint8_t *wb = rn1202_weight_blocks(hdr, conf->tid_conv_b);
    const float   *bb = rn1202_bias_f32(hdr, conf->tid_conv_b);
    conv3x3_bd4_tap_hwmac(bd_mid, bd_out, wb, bb,
                          out_c, out_c, oh, ow, /*stride=*/1, sb, /*relu=*/0);

    /* Skip connection + ReLU */
#if RN1202_HAS_PROJ
    if (conf->has_proj) {
        /* Option B: conv1×1 projection, HWCB in → HWCB skip */
        const uint8_t *wp = rn1202_weight_blocks(hdr, conf->tid_proj);
        const float   *bp = rn1202_bias_f32(hdr, conf->tid_proj);
        conv1x1_bd4_tap_hwmac(bd_in, bd_skip, wp, bp,
                              in_c, out_c, ih, iw, s, sp, /*relu=*/0);
        add_relu_bd4_hwcb(bd_out, bd_skip, bd_out, out_c, oh, ow);
    } else
#endif
    if (s == 2) {
        /* Option A: zero-pad-stride-2 skip directly in HWCB format */
        bd4_zero_pad_stride2_hwcb(bd_in, bd_skip, in_c, out_c, ih, iw);
        add_relu_bd4_hwcb(bd_out, bd_skip, bd_out, out_c, oh, ow);
    } else {
        /* Identity: in_c == out_c, same spatial dims */
        add_relu_bd4_hwcb(bd_out, bd_in, bd_out, out_c, ih, iw);
    }
}

#endif /* BD_DECODE_HW_H */

#endif /* RESNET1202_CONV_H */

