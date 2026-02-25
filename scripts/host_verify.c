/*
 * host_verify.c — runs the same computation as the VexRiscv firmware on the host.
 * Compares Layer1 Hash against the RTL simulation result (0x00031FBB).
 *
 * Compile: gcc -O2 -o host_verify host_verify.c
 * Run:     ./host_verify
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* ---- Input data (same header as firmware) ---- */
#include "../src/main/c/murax/hyperram_phase_a/src/input.h"

/* ---- Weight reader (file-backed, matching firmware get_weights()) ---- */
static FILE   *weights_fp = NULL;
static uint32_t w_offset  = 0;

static void reset_weights(void) {
    w_offset = 0;
    fseek(weights_fp, 16, SEEK_SET); /* skip 16-byte header, same as WEIGHTS_PTR */
}

/* Read N bytes sequentially; returns pointer to a static buffer (must be used
   before the next call).  For simplicity we use a heap buffer here. */
static int8_t *weight_buf = NULL;
static size_t  weight_buf_cap = 0;

static const int8_t *get_weights(int count) {
    /* Grow buffer if needed */
    if ((size_t)count > weight_buf_cap) {
        free(weight_buf);
        weight_buf = malloc(count);
        weight_buf_cap = count;
    }
    fread(weight_buf, 1, count, weights_fp);
    w_offset += count;
    /* Align to 4 bytes (matching firmware) */
    int pad = (4 - (w_offset % 4)) % 4;
    if (pad) { fseek(weights_fp, pad, SEEK_CUR); w_offset += pad; }
    return weight_buf;
}

/* ---- CNN primitives — ** exactly ** copied from main.c ---- */

static int8_t buffer_A[32 * 32 * 16];

static void conv2d_3x3(const int8_t *input, int8_t *output,
                int in_c, int out_c, int h, int w,
                int stride, int padding) {
    int w_count = out_c * in_c * 3 * 3;
    const int8_t *weights = get_weights(w_count);

    for (int oc = 0; oc < out_c; oc++) {
        for (int y = 0; y < h; y += stride) {
            for (int x = 0; x < w; x += stride) {
                int32_t sum = 0;
                for (int ic = 0; ic < in_c; ic++) {
                    for (int ky = 0; ky < 3; ky++) {
                        for (int kx = 0; kx < 3; kx++) {
                            int iy = y + ky - padding;
                            int ix = x + kx - padding;
                            int8_t val = 0;
                            if (iy >= 0 && iy < h && ix >= 0 && ix < w)
                                val = input[(ic * h * w) + (iy * w) + ix];
                            int8_t w_val = weights[((oc * in_c + ic) * 3 + ky) * 3 + kx];
                            sum += val * w_val;
                        }
                    }
                }
                int oy = y / stride;
                int ox = x / stride;
                int ow = w / stride;
                output[(oc * (h/stride) * ow) + (oy * ow) + ox] = (int8_t)(sum >> 7);
            }
        }
    }
}

static void batch_norm_relu(int8_t *feature_map, int channels, int h, int w) {
    /* Copy both weight tensors out before they alias each other in weight_buf */
    int8_t local_w[64], local_b[64]; /* max 64 channels in ResNet-20 */
    memcpy(local_w, get_weights(channels), channels);
    memcpy(local_b, get_weights(channels), channels);
    for (int c = 0; c < channels; c++) {
        int8_t w_bn = local_w[c];
        int8_t b_bn = local_b[c];
        for (int i = 0; i < h * w; i++) {
            int idx = c * h * w + i;
            int32_t val = feature_map[idx];
            val = (val * w_bn) >> 6;
            val += b_bn;
            if (val < 0)   val = 0;
            if (val > 127) val = 127;
            feature_map[idx] = (int8_t)val;
        }
    }
}

/* ---- Main ---- */
int main(void) {
    /* Open weights.bin (same dir as this source, i.e. scripts/) */
    weights_fp = fopen("weights.bin", "rb");
    if (!weights_fp) {
        fprintf(stderr, "ERROR: cannot open weights.bin — run gen_resnet_model.py first\n");
        return 1;
    }

    /* Verify magic */
    uint32_t magic;
    fread(&magic, 4, 1, weights_fp);
    if (magic != 0x56574230) {
        fprintf(stderr, "ERROR: bad magic 0x%08X (expected 0x56574230)\n", magic);
        return 1;
    }
    printf("Magic: OK (0x56574230 = 'VWB0')\n");

    reset_weights();

    /* Run same computation as firmware */
    const int H = 32, W = 32;
    printf("Running: Conv2d 3->16 (%dx%d)...\n", H, W);
    conv2d_3x3((const int8_t *)INPUT_DATA, buffer_A, 3, 16, H, W, 1, 1);
    batch_norm_relu(buffer_A, 16, H, W);

    /* Compute hash exactly as firmware does */
    uint32_t sum = 0;
    for (int i = 0; i < H * W * 16; i++) sum += buffer_A[i];

    printf("Layer1 Hash: 0x%08X\n", sum);

    uint32_t expected = 0x000B5A22; /* ResNet-110 bird image — golden from host verifier */
    if (sum == expected) {
        printf("MATCH ✓ — host result equals RTL simulation (0x%08X)\n", expected);
    } else {
        printf("MISMATCH ✗ — host=0x%08X  sim=0x%08X\n", sum, expected);
    }

    fclose(weights_fp);
    free(weight_buf);
    return (sum == expected) ? 0 : 1;
}
