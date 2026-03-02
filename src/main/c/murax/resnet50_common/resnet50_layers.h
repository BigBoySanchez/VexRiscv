/* resnet50_layers.h — ResNet-50 topology definitions and conv kernels
 *
 * Layer configuration for torchvision_resnet50_imagenet_v1:
 *   conv1  7×7 stride 2, pad 3, 3→64
 *   maxpool 3×3 stride 2, pad 1
 *   layer1: 3 bottleneck blocks (64→256, stride 1)
 *   layer2: 4 bottleneck blocks (256→512, first stride 2)
 *   layer3: 6 bottleneck blocks (512→1024, first stride 2)
 *   layer4: 3 bottleneck blocks (1024→2048, first stride 2)
 *   avgpool 7×7
 *   fc 2048→1000
 *
 * Memory strategy:
 *   - Weights decoded from BD on-the-fly (per output channel)
 *   - Skip connections stored in BD A4 format for large tensors
 *   - Working activations in int8, tiled spatially when needed
 *   - All conv compute in int32 accumulators, quantized to int8 output
 */
#ifndef RESNET50_LAYERS_H
#define RESNET50_LAYERS_H

#include <stdint.h>
#include "weight_blob.h"
#include "model_constants.h"
#include "bd_decode_sw.h"
#include "bd_act.h"

/* ── Layer configuration struct ──────────────────────────────────────── */
typedef struct {
    uint8_t  layer_id;     /* Resnet50LayerId enum value    */
    uint8_t  kernel_size;  /* 1, 3, or 7                    */
    uint8_t  stride;       /* 1 or 2                        */
    uint8_t  padding;      /* 0, 1, or 3                    */
    uint16_t in_c;         /* input channels                */
    uint16_t out_c;        /* output channels               */
    uint16_t in_h;         /* input spatial height           */
    uint16_t in_w;         /* input spatial width            */
} LayerConf;

/* ── Bottleneck block configuration ──────────────────────────────────── */
typedef struct {
    uint8_t  layer_id_conv1;  /* 1×1 squeeze    */
    uint8_t  layer_id_conv2;  /* 3×3 spatial     */
    uint8_t  layer_id_conv3;  /* 1×1 expand      */
    int8_t   layer_id_ds;     /* 1×1 downsample (-1 if none) */
    uint16_t in_c;            /* block input channels  */
    uint16_t mid_c;           /* bottleneck channels   */
    uint16_t out_c;           /* block output channels */
    uint16_t in_h, in_w;     /* block input spatial   */
    uint8_t  stride;          /* stride for conv2 and downsample */
} BottleneckConf;

/* ── Stage definitions ───────────────────────────────────────────────── */
/* layer1: 3 blocks, layer2: 4 blocks, layer3: 6 blocks, layer4: 3 blocks */

static const BottleneckConf BOTTLENECK_BLOCKS[16] = {
    /* layer1 */
    {  1,  2,  3,   4,   64,  64,  256, 56, 56, 1 }, /* layer1.0 (has ds) */
    {  5,  6,  7,  -1,  256,  64,  256, 56, 56, 1 }, /* layer1.1 */
    {  8,  9, 10,  -1,  256,  64,  256, 56, 56, 1 }, /* layer1.2 */

    /* layer2 */
    { 11, 12, 13,  14,  256, 128,  512, 56, 56, 2 }, /* layer2.0 (has ds, stride 2) */
    { 15, 16, 17,  -1,  512, 128,  512, 28, 28, 1 }, /* layer2.1 */
    { 18, 19, 20,  -1,  512, 128,  512, 28, 28, 1 }, /* layer2.2 */
    { 21, 22, 23,  -1,  512, 128,  512, 28, 28, 1 }, /* layer2.3 */

    /* layer3 */
    { 24, 25, 26,  27,  512, 256, 1024, 28, 28, 2 }, /* layer3.0 (has ds, stride 2) */
    { 28, 29, 30,  -1, 1024, 256, 1024, 14, 14, 1 }, /* layer3.1 */
    { 31, 32, 33,  -1, 1024, 256, 1024, 14, 14, 1 }, /* layer3.2 */
    { 34, 35, 36,  -1, 1024, 256, 1024, 14, 14, 1 }, /* layer3.3 */
    { 37, 38, 39,  -1, 1024, 256, 1024, 14, 14, 1 }, /* layer3.4 */
    { 40, 41, 42,  -1, 1024, 256, 1024, 14, 14, 1 }, /* layer3.5 */

    /* layer4 */
    { 43, 44, 45,  46, 1024, 512, 2048, 14, 14, 2 }, /* layer4.0 (has ds, stride 2) */
    { 47, 48, 49,  -1, 2048, 512, 2048,  7,  7, 1 }, /* layer4.1 */
    { 50, 51, 52,  -1, 2048, 512, 2048,  7,  7, 1 }, /* layer4.2 */
};

/* Stage boundaries in BOTTLENECK_BLOCKS[] */
#define STAGE1_START 0
#define STAGE1_COUNT 3
#define STAGE2_START 3
#define STAGE2_COUNT 4
#define STAGE3_START 7
#define STAGE3_COUNT 6
#define STAGE4_START 13
#define STAGE4_COUNT 3

/* ── Weight decode helpers ───────────────────────────────────────────── */

/* Get pointer to the BD blocks for a weight tensor (skips 8-byte BD4 header) */
static inline const uint8_t *get_weight_blocks(
    const vwb2_header_t *hdr,
    uint8_t              layer_id
) {
    const Resnet50LayerInfo *info = &RESNET50_LAYERS[layer_id];
    /* weight_offset is from WEIGHT_BLOB_ADDR; the VWB2 data_offset is from
     * the header. We use the pre-computed absolute offsets in RESNET50_LAYERS. */
    const uint8_t *base = (const uint8_t *)hdr;
    return base + info->weight_offset + 8;  /* skip BD4 sub-header (n_elem, n_blocks) */
}

/* Get pointer to float32 bias array for a layer */
static inline const float *get_bias_f32(
    const vwb2_header_t *hdr,
    uint8_t              layer_id
) {
    const Resnet50LayerInfo *info = &RESNET50_LAYERS[layer_id];
    const uint8_t *base = (const uint8_t *)hdr;
    return (const float *)(base + info->bias_offset);
}

/* Get the number of BD blocks for a weight tensor */
static inline uint32_t get_weight_n_blocks(uint8_t layer_id) {
    return RESNET50_LAYERS[layer_id].w_bd_blocks;
}

/* ── Decode one output channel's weights from BD to int16 half-units ── */
/* Returns shared_exp_bits for the last block (for simplicity, we handle  */
/* multi-block channels by decoding to int8 directly).                    */
/*                                                                         */
/* For a conv with kernel_size K and in_c input channels, one output      */
/* channel has in_c * K * K elements.                                     */

/* Decode 'n_elements' from BD blocks at 'block_ptr' into 'out_i16' as   */
/* signed half-units.  Returns number of blocks consumed.                 */
static inline uint32_t bd_decode_elements_hu(
    const uint8_t *block_ptr,
    int16_t       *out_hu,
    int32_t       *out_exp,       /* per-block shared exponent array */
    uint32_t       n_elements
) {
    uint32_t n_blocks = (n_elements + 31) / 32;
    uint32_t pos = 0;

    for (uint32_t b = 0; b < n_blocks; b++) {
        int did, seb;
        int16_t hu[32];
        bd_decode_block_hu(block_ptr + b * BD_BLOCK_BYTES, hu, &did, &seb);
        out_exp[b] = seb;

        uint32_t count = n_elements - pos;
        if (count > 32) count = 32;
        for (uint32_t i = 0; i < count; i++) {
            out_hu[pos + i] = hu[i];
        }
        pos += count;
    }
    return n_blocks;
}

#endif /* RESNET50_LAYERS_H */
