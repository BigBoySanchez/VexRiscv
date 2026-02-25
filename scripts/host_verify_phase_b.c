/*
 * host_verify_phase_b.c — Host-side verifier for Phase B (BlockDialect-Lite).
 * Runs the same decode + conv computation as the VexRiscv Phase B firmware.
 * Compares Layer1 Hash against Phase A golden value to document delta.
 *
 * Compile: gcc -O2 -o host_verify_b host_verify_phase_b.c
 * Run:     ./host_verify_b
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* ---- Input data (same header as firmware) ---- */
#include "../src/main/c/murax/hyperram_phase_a/src/input.h"

/* ---- BlockDialect Constants (must match firmware exactly) ---- */
#define BD_MAGIC        0x56574231
#define BD_BLOCK_SIZE   32
#define BD_BLOCK_BYTES  18
#define BD_NUM_DIALECTS 16

static const uint8_t DIALECT_LUT[BD_NUM_DIALECTS][8] = {
    {0, 1, 2, 3, 4, 4, 4, 4},
    {0, 1, 2, 3, 3, 3, 4, 4},
    {0, 1, 2, 3, 4, 5, 5, 5},
    {0, 1, 2, 3, 3, 4, 5, 5},
    {0, 1, 2, 3, 4, 5, 6, 6},
    {0, 1, 2, 3, 4, 4, 6, 6},
    {0, 1, 2, 3, 4, 5, 6, 7},
    {0, 1, 2, 3, 4, 5, 7, 7},
    {0, 1, 2, 3, 4, 6, 7, 8},
    {0, 1, 2, 3, 4, 6, 8, 8},
    {0, 1, 2, 3, 4, 6, 8, 10},
    {0, 1, 2, 3, 4, 6, 10, 10},
    {0, 1, 2, 3, 4, 6, 10, 12},
    {0, 1, 2, 3, 4, 6, 12, 12},
    {0, 1, 2, 3, 4, 6, 12, 15},
    {0, 1, 2, 3, 4, 6, 13, 15},
};

/* ---- Weight Reader ---- */
static FILE     *weights_fp = NULL;
static uint32_t bd_offset = 0;
static uint32_t bytes_read_total = 0;

static void reset_weights(void) {
    bd_offset = 0;
    bytes_read_total = 0;
    fseek(weights_fp, 16, SEEK_SET); /* skip file header */
}

static void decode_block(const uint8_t *block_data, int8_t *out) {
    uint16_t meta = ((uint16_t)block_data[0] << 8) | block_data[1];
    uint8_t dialect_id = (meta >> 12) & 0xF;
    uint8_t shared_exp = (meta >> 7) & 0x1F;
    const uint8_t *packed = block_data + 2;

    for (int i = 0; i < 16; i++) {
        uint8_t byte_val = packed[i];
        uint8_t code_hi = (byte_val >> 4) & 0x0F;
        uint8_t code_lo = byte_val & 0x0F;

        /* Decode high nibble */
        {
            uint8_t sign = (code_hi >> 3) & 1;
            uint8_t idx  = code_hi & 0x07;
            int32_t mag_scaled = DIALECT_LUT[dialect_id][idx];
            int32_t real_mag;
            if (shared_exp == 0)
                real_mag = (mag_scaled + 1) >> 1;
            else
                real_mag = mag_scaled << (shared_exp - 1);
            if (real_mag > 127) real_mag = 127;
            out[2*i] = sign ? (int8_t)(-real_mag) : (int8_t)(real_mag);
        }

        /* Decode low nibble */
        {
            uint8_t sign = (code_lo >> 3) & 1;
            uint8_t idx  = code_lo & 0x07;
            int32_t mag_scaled = DIALECT_LUT[dialect_id][idx];
            int32_t real_mag;
            if (shared_exp == 0)
                real_mag = (mag_scaled + 1) >> 1;
            else
                real_mag = mag_scaled << (shared_exp - 1);
            if (real_mag > 127) real_mag = 127;
            out[2*i + 1] = sign ? (int8_t)(-real_mag) : (int8_t)(real_mag);
        }
    }
}

#define DECODE_BUF_SIZE 512
static int8_t decode_buf[DECODE_BUF_SIZE];

static const int8_t *get_weights(int count) {
    /* Read tensor header */
    uint32_t n_elements, n_blocks;
    fread(&n_elements, 4, 1, weights_fp);
    fread(&n_blocks, 4, 1, weights_fp);
    bd_offset += 8;
    bytes_read_total += 8;

    /* Decode blocks */
    for (uint32_t b = 0; b < n_blocks; b++) {
        uint8_t block_data[BD_BLOCK_BYTES];
        fread(block_data, 1, BD_BLOCK_BYTES, weights_fp);
        if ((b + 1) * BD_BLOCK_SIZE <= DECODE_BUF_SIZE)
            decode_block(block_data, decode_buf + b * BD_BLOCK_SIZE);
        bd_offset += BD_BLOCK_BYTES;
        bytes_read_total += BD_BLOCK_BYTES;
    }

    /* Align to 4 bytes */
    int pad = (4 - (bd_offset % 4)) % 4;
    if (pad) { fseek(weights_fp, pad, SEEK_CUR); bd_offset += pad; }

    return decode_buf;
}

/* ---- CNN primitives (same as Phase A/B firmware) ---- */
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

#define MAX_CHANNELS 64

static void batch_norm_relu(int8_t *feature_map, int channels, int h, int w) {
    int8_t local_w[MAX_CHANNELS], local_b[MAX_CHANNELS];
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
    weights_fp = fopen("weights_bd.bin", "rb");
    if (!weights_fp) {
        fprintf(stderr, "ERROR: cannot open weights_bd.bin — run gen_resnet_model.py --phase b first\n");
        return 1;
    }

    /* Verify magic */
    uint32_t magic;
    fread(&magic, 4, 1, weights_fp);
    if (magic != BD_MAGIC) {
        fprintf(stderr, "ERROR: bad magic 0x%08X (expected 0x%08X)\n", magic, BD_MAGIC);
        return 1;
    }
    printf("Magic: OK (0x%08X = 'VWB1')\n", BD_MAGIC);

    reset_weights();

    /* Run computation */
    const int H = 32, W = 32;
    printf("Running: Conv2d 3->16 (%dx%d) [BlockDialect decode]...\n", H, W);
    conv2d_3x3((const int8_t *)INPUT_DATA, buffer_A, 3, 16, H, W, 1, 1);
    batch_norm_relu(buffer_A, 16, H, W);

    /* Compute hash */
    uint32_t sum = 0;
    for (int i = 0; i < H * W * 16; i++) sum += buffer_A[i];

    printf("Layer1 Hash: 0x%08X\n", sum);
    printf("Bytes Read:  %u\n", bytes_read_total);

    /* Phase A golden hash for comparison */
    uint32_t phase_a_hash = 0x000B5A22;
    printf("\nPhase A Hash: 0x%08X\n", phase_a_hash);
    printf("Phase B Hash: 0x%08X\n", sum);
    if (sum == phase_a_hash) {
        printf("MATCH ✓ — Phase A and Phase B produce identical output\n");
    } else {
        printf("DELTA — expected due to lossy 4-bit quantization\n");
        printf("  (This is normal; the important metric is bytes read reduction)\n");
    }

    fclose(weights_fp);
    return 0;
}
