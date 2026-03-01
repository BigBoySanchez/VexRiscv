#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "imagenet_classes.h"

/* ---- BlockDialect Constants ---- */
#define BD_MAGIC        0x56574231
#define BD_BLOCK_SIZE   32
#define BD_BLOCK_BYTES  18
#define BD_NUM_DIALECTS 16

static const uint8_t DIALECT_LUT[BD_NUM_DIALECTS][8] = {
    {0, 1, 2, 3, 4, 6, 11, 15},  /* 0:  7.5, 5.5, 3, 2, 1.5, 1, 0.5, 0 */
    {0, 1, 2, 3, 4, 6,  9, 15},  /* 1:  7.5, 4.5, 3, 2, 1.5, 1, 0.5, 0 */
    {0, 1, 2, 3, 4, 6, 11, 14},  /* 2:  7.0, 5.5, ... */
    {0, 1, 2, 3, 4, 6,  9, 14},  /* 3:  7.0, 4.5, ... */
    {0, 1, 2, 3, 4, 6, 10, 13},  /* 4:  6.5, 5.0, ... */
    {0, 1, 2, 3, 4, 6,  8, 13},  /* 5:  6.5, 4.0, ... */
    {0, 1, 2, 3, 4, 6, 10, 12},  /* 6:  6.0, 5.0, ... */
    {0, 1, 2, 3, 4, 6,  8, 12},  /* 7:  6.0, 4.0, ... */
    {0, 1, 2, 3, 4, 6,  9, 11},  /* 8:  5.5, 4.5, ... */
    {0, 1, 2, 3, 4, 6,  7, 11},  /* 9:  5.5, 3.5, ... */
    {0, 1, 2, 3, 4, 6,  9, 10},  /* 10: 5.0, 4.5, ... */
    {0, 1, 2, 3, 4, 6,  7, 10},  /* 11: 5.0, 3.5, ... */
    {0, 1, 2, 3, 4, 6,  8,  9},  /* 12: 4.5, 4.0, ... */
    {0, 1, 2, 3, 4, 6,  7,  9},  /* 13: 4.5, 3.5, ... */
    {0, 1, 2, 3, 4, 6,  7,  8},  /* 14: 4.0, 3.5, ... */
    {0, 1, 2, 3, 4, 5,  6,  8},  /* 15: 4.0, 3.0, 2.5, 2.0, ... */
};

static FILE *weights_fp = NULL;

static void decode_block_float(const uint8_t *block_data, float *out) {
    uint16_t meta = ((uint16_t)block_data[0] << 8) | block_data[1];
    uint8_t dialect_id = (meta >> 12) & 0xF;
    uint8_t shared_exp = (meta >> 7) & 0x1F;
    const uint8_t *packed = block_data + 2;

    int e = (int)shared_exp - 15; // FP16 bias
    float scale = ldexpf(0.5f, e);

    for (int i = 0; i < 16; i++) {
        uint8_t byte_val = packed[i];
        
        // High nibble
        uint8_t code_hi = (byte_val >> 4) & 0x0F;
        uint8_t sign_hi = (code_hi >> 3) & 1;
        uint8_t idx_hi  = code_hi & 0x07;
        float mag_hi = (float)DIALECT_LUT[dialect_id][idx_hi] * scale;
        out[2*i] = sign_hi ? -mag_hi : mag_hi;

        // Low nibble
        uint8_t code_lo = byte_val & 0x0F;
        uint8_t sign_lo = (code_lo >> 3) & 1;
        uint8_t idx_lo  = code_lo & 0x07;
        float mag_lo = (float)DIALECT_LUT[dialect_id][idx_lo] * scale;
        out[2*i + 1] = sign_lo ? -mag_lo : mag_lo;
    }
}

static float* decode_tensor_float(int n_elements) {
    uint32_t file_n_elems, n_blocks;
    fread(&file_n_elems, 4, 1, weights_fp);
    fread(&n_blocks, 4, 1, weights_fp);
    
    if (file_n_elems != n_elements) {
        fprintf(stderr, "Weight mismatch! Expected %d, got %d\n", n_elements, file_n_elems);
        exit(1);
    }

    float* buf = malloc(n_blocks * BD_BLOCK_SIZE * sizeof(float));
    uint8_t block_data[BD_BLOCK_BYTES];

    for (uint32_t b = 0; b < n_blocks; b++) {
        fread(block_data, 1, BD_BLOCK_BYTES, weights_fp);
        decode_block_float(block_data, buf + b * BD_BLOCK_SIZE);
    }

    long pos = ftell(weights_fp);
    int pad = (4 - (pos % 4)) % 4;
    if (pad) fseek(weights_fp, pad, SEEK_CUR);

    return buf;
}

// Global ping-pong buffers for activations
// Max size needed: float32: 64 * 112 * 112 = 3.2MB for ResNet50
static float buffer_A[64 * 112 * 112 * 4]; // overprovision just in case 256*56*56 = bounds
static float buffer_B[64 * 112 * 112 * 4]; // x4

void conv2d(const float *in, float *out, 
            int in_c, int out_c, int h, int w, 
            int k, int stride, int padding,
            const float *weights, const float *bias) {
    int out_h = (h + 2 * padding - k) / stride + 1;
    int out_w = (w + 2 * padding - k) / stride + 1;

    for (int oc = 0; oc < out_c; oc++) {
        for (int y = 0; y < out_h; y++) {
            for (int x = 0; x < out_w; x++) {
                float sum = bias[oc];
                for (int ic = 0; ic < in_c; ic++) {
                    for (int ky = 0; ky < k; ky++) {
                        for (int kx = 0; kx < k; kx++) {
                            int iy = y * stride + ky - padding;
                            int ix = x * stride + kx - padding;
                            if (iy >= 0 && iy < h && ix >= 0 && ix < w) {
                                float val = in[(ic * h * w) + (iy * w) + ix];
                                float w_val = weights[((oc * in_c + ic) * k + ky) * k + kx];
                                sum += val * w_val;
                            }
                        }
                    }
                }
                out[(oc * out_h * out_w) + (y * out_w) + x] = sum;
            }
        }
    }
}

void relu(float *x, int size) {
    for (int i = 0; i < size; i++) if (x[i] < 0) x[i] = 0;
}

void add_tensor(const float *a, float *out, int size) {
    for (int i = 0; i < size; i++) out[i] += a[i];
}

void maxpool2d_3x3_s2(const float *in, float *out, int c, int h, int w) {
    int padding = 1;
    int k = 3;
    int stride = 2;
    int out_h = (h + 2 * padding - k) / stride + 1;
    int out_w = (w + 2 * padding - k) / stride + 1;

    for (int tc = 0; tc < c; tc++) {
        for (int y = 0; y < out_h; y++) {
            for (int x = 0; x < out_w; x++) {
                float max_val = -INFINITY;
                for (int ky = 0; ky < k; ky++) {
                    for (int kx = 0; kx < k; kx++) {
                        int iy = y * stride + ky - padding;
                        int ix = x * stride + kx - padding;
                        float val = -INFINITY;
                        if (iy >= 0 && iy < h && ix >= 0 && ix < w) {
                            val = in[(tc * h * w) + (iy * w) + ix];
                        }
                        if (val > max_val) max_val = val;
                    }
                }
                out[(tc * out_h * out_w) + (y * out_w) + x] = max_val;
            }
        }
    }
}

void avgpool2d(const float *in, float *out, int c, int h, int w) {
    for (int tc = 0; tc < c; tc++) {
        float sum = 0;
        for (int i = 0; i < h * w; i++) sum += in[tc * h * w + i];
        out[tc] = sum / (h * w);
    }
}

void linear(const float *in, float *out, int in_features, int out_features, const float *weights, const float *bias) {
    for (int oc = 0; oc < out_features; oc++) {
        float sum = bias[oc];
        for (int ic = 0; ic < in_features; ic++) {
            sum += in[ic] * weights[oc * in_features + ic];
        }
        out[oc] = sum;
    }
}

// ---- Macro to do Conv + ReLU step automatically reading weights ----
// Assumes input is in A, puts output in B
#define DO_CONV_RELU(in, out, ic, oc, k, s, p, h, w) \
    do { \
        float *wts = decode_tensor_float(oc * ic * k * k); \
        float *bs = decode_tensor_float(oc); \
        conv2d(in, out, ic, oc, h, w, k, s, p, wts, bs); \
        free(wts); free(bs); \
        relu(out, oc * ((h + 2*p - k)/s + 1) * ((w + 2*p - k)/s + 1)); \
    } while(0)

#define DO_CONV(in, out, ic, oc, k, s, p, h, w) \
    do { \
        float *wts = decode_tensor_float(oc * ic * k * k); \
        float *bs = decode_tensor_float(oc); \
        conv2d(in, out, ic, oc, h, w, k, s, p, wts, bs); \
        free(wts); free(bs); \
    } while(0)

// Helper for Bottleneck
// returns the dimensions needed for next step ptr

// Helper for Bottleneck

// Helper for Bottleneck

// Helper for Bottleneck
void bottleneck(float* in_ptr, float* out_ptr, float* residual_ptr, 
                int in_c, int mid_c, int out_c, 
                int h, int w, int stride, int downsample) {

    // PRE-READ weights EXACTLY matching Python test_shape.py sequence
    
    // conv1.weight, conv1.bias
    float *w1 = decode_tensor_float(mid_c * in_c);
    float *b1 = decode_tensor_float(mid_c);
    
    // conv2.weight, conv2.bias
    float *w2 = decode_tensor_float(mid_c * mid_c * 3 * 3);
    float *b2 = decode_tensor_float(mid_c);
    
    // conv3.weight, conv3.bias
    float *w3 = decode_tensor_float(out_c * mid_c);
    float *b3 = decode_tensor_float(out_c);
    
    // downsample.0.weight, downsample.0.bias
    float *wd = NULL, *bd = NULL;
    if (downsample) {
        wd = decode_tensor_float(out_c * in_c);
        bd = decode_tensor_float(out_c);
    }
    
    // Evaluate!
    // Conv1
    conv2d(in_ptr, out_ptr, in_c, mid_c, h, w, 1, 1, 0, w1, b1);
    relu(out_ptr, mid_c * h * w);
    free(w1); free(b1);
    
    // Conv2
    int oh2 = (h - 1) / stride + 1;
    int ow2 = (w - 1) / stride + 1;
    conv2d(out_ptr, in_ptr, mid_c, mid_c, h, w, 3, stride, 1, w2, b2); // out in in_ptr
    relu(in_ptr, mid_c * oh2 * ow2);
    free(w2); free(b2);
    
    // Conv3
    conv2d(in_ptr, out_ptr, mid_c, out_c, oh2, ow2, 1, 1, 0, w3, b3); // out in out_ptr
    free(w3); free(b3);
    
    // Downsample & Residual
    if (downsample) {
        conv2d(residual_ptr, in_ptr, in_c, out_c, h, w, 1, stride, 0, wd, bd);
        free(wd); free(bd);
        add_tensor(in_ptr, out_ptr, out_c * oh2 * ow2);
    } else {
        add_tensor(residual_ptr, out_ptr, out_c * oh2 * ow2);
    }
    
    relu(out_ptr, out_c * oh2 * ow2);
}
int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <input_tensor.bin>\n", argv[0]);
        return 1;
    }

    /* Load input tensor */
    FILE *in_fp = fopen(argv[1], "rb");
    if (!in_fp) {
        fprintf(stderr, "Cannot open %s\n", argv[1]);
        return 1;
    }
    fread(buffer_A, 4, 3 * 224 * 224, in_fp);
    fclose(in_fp);

    /* Open weights */
    weights_fp = fopen("resnet50_bd_fused_weights.bin", "rb");
    if (!weights_fp) {
        fprintf(stderr, "Cannot open resnet50_bd_fused_weights.bin\n");
        return 1;
    }

    uint32_t magic, payload, bsize, resv;
    fread(&magic, 4, 1, weights_fp);
    fread(&payload, 4, 1, weights_fp);
    fread(&bsize, 4, 1, weights_fp);
    fread(&resv, 4, 1, weights_fp);
    if (magic != BD_MAGIC || bsize != BD_BLOCK_SIZE) {
        fprintf(stderr, "Bad weights file magic or block size\n");
        return 1;
    }

    printf("Running ResNet50...\n");
    // Layer 0: Conv1 -> 64x112x112
    printf("Conv1...\n");
    DO_CONV_RELU(buffer_A, buffer_B, 3, 64, 7, 2, 3, 224, 224);
    
    // MaxPool -> 64x56x56
    printf("MaxPool...\n");
    maxpool2d_3x3_s2(buffer_B, buffer_A, 64, 112, 112); // Output in A

    // Need to dynamically allocate a residual copy buffer since bottleneck destroys A
    float *residual = malloc(2048 * 56 * 56 * 4); 

    // Layer 1 (3 blocks)
    printf("Layer 1...\n");
    // block 0 (downsample=1)
    memcpy(residual, buffer_A, 64 * 56 * 56 * 4);
    bottleneck(buffer_A, buffer_B, residual, 64, 64, 256, 56, 56, 1, 1);
    // block 1 (downsample=0)
    memcpy(residual, buffer_B, 256 * 56 * 56 * 4);
    bottleneck(buffer_B, buffer_A, residual, 256, 64, 256, 56, 56, 1, 0); // Output in A
    // block 2 (downsample=0)
    memcpy(residual, buffer_A, 256 * 56 * 56 * 4);
    bottleneck(buffer_A, buffer_B, residual, 256, 64, 256, 56, 56, 1, 0); // Output in B

    // Layer 2 (4 blocks)
    printf("Layer 2...\n");
    // block 0
    memcpy(residual, buffer_B, 256 * 56 * 56 * 4);
    bottleneck(buffer_B, buffer_A, residual, 256, 128, 512, 56, 56, 2, 1); // Output in A, 28x28
    // block 1
    memcpy(residual, buffer_A, 512 * 28 * 28 * 4);
    bottleneck(buffer_A, buffer_B, residual, 512, 128, 512, 28, 28, 1, 0);
    // block 2
    memcpy(residual, buffer_B, 512 * 28 * 28 * 4);
    bottleneck(buffer_B, buffer_A, residual, 512, 128, 512, 28, 28, 1, 0);
    // block 3
    memcpy(residual, buffer_A, 512 * 28 * 28 * 4);
    bottleneck(buffer_A, buffer_B, residual, 512, 128, 512, 28, 28, 1, 0); // Output in B

    // Layer 3 (6 blocks)
    printf("Layer 3...\n");
    // block 0
    memcpy(residual, buffer_B, 512 * 28 * 28 * 4);
    bottleneck(buffer_B, buffer_A, residual, 512, 256, 1024, 28, 28, 2, 1); // out: 14x14
    for(int i=1; i<6; i++) {
        float* cur_in = (i%2==1) ? buffer_A : buffer_B;
        float* cur_out = (i%2==1) ? buffer_B : buffer_A;
        memcpy(residual, cur_in, 1024 * 14 * 14 * 4);
        bottleneck(cur_in, cur_out, residual, 1024, 256, 1024, 14, 14, 1, 0);
    }
    // layer3 ends in cur_out=buffer_A... check later: i=5 (loop runs 1, 2, 3, 4, 5).
    // i=1: A->B
    // i=2: B->A
    // i=3: A->B
    // i=4: B->A
    // i=5: A->B
    // output in buffer_B

    // Layer 4 (3 blocks)
    printf("Layer 4...\n");
    // block 0
    memcpy(residual, buffer_B, 1024 * 14 * 14 * 4);
    bottleneck(buffer_B, buffer_A, residual, 1024, 512, 2048, 14, 14, 2, 1); // out: 7x7
    // block 1
    memcpy(residual, buffer_A, 2048 * 7 * 7 * 4);
    bottleneck(buffer_A, buffer_B, residual, 2048, 512, 2048, 7, 7, 1, 0);
    // block 2
    memcpy(residual, buffer_B, 2048 * 7 * 7 * 4);
    bottleneck(buffer_B, buffer_A, residual, 2048, 512, 2048, 7, 7, 1, 0); // Output in A

    // AvgPool
    printf("AvgPool + Linear...\n");
    float *pooled = malloc(2048 * 4);
    avgpool2d(buffer_A, pooled, 2048, 7, 7);

    // Linear
    float *logits = malloc(1000 * 4);
    float *fc_wts = decode_tensor_float(1000 * 2048);
    float *fc_bs = decode_tensor_float(1000);

    linear(pooled, logits, 2048, 1000, fc_wts, fc_bs);

    // Argmax & final output
    int best_id = -1;
    float best_val = -INFINITY;
    for (int i = 0; i < 1000; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best_id = i;
        }
    }


    // Softmax to get probability
    float max_logit = -INFINITY;
    for (int i = 0; i < 1000; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }
    float sum_exp = 0;
    for (int i = 0; i < 1000; i++) {
        sum_exp += expf(logits[i] - max_logit);
    }
    float prob = expf(best_val - max_logit) / sum_exp;
    
    printf("\nFinal Prediction:\n");
    printf("ID: %d\n", best_id);
    printf("String: %s\n", IMAGENET_CLASSES[best_id]);
    printf("Confidence: %.4f\n", prob);
    fclose(weights_fp);
    return 0;
}
