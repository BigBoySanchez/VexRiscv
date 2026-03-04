/* resnet1202_layers.h — ResNet-1202 topology definitions
 *
 * §1 of RESNET1202_FPGA_PLAN.md: Architecture reference
 *
 * Model: He et al. 2016, CIFAR-10 variant, n=200 (6n+2 = 1202 layers).
 *
 *   Layer           Shape out        Params (BN folded)
 *   ──────────────────────────────────────────────────────────────────
 *   conv1            16 × 32 × 32         432   (3×3×3×16)
 *   stage1           16 × 32 × 32     921,600   (200 BasicBlocks, stride=1)
 *   stage2           32 × 16 × 16   3,687,424   (200 BasicBlocks, first stride=2)
 *   stage3           64 ×  8 ×  8  14,748,160   (200 BasicBlocks, first stride=2)
 *   avgpool          64 ×  1 ×  1           —
 *   fc               10                   640   (64→10)
 *   ──────────────────────────────────────────────────────────────────
 *   Total params                     ≈ 19.36 M
 *
 * BasicBlock topology (non-projection blocks):
 *   input → conv3×3, BN, ReLU → conv3×3, BN → + identity skip → ReLU
 *
 * Projection block (first block of stage2 and stage3):
 *   input → conv3×3(Cin→Cout, stride=2), BN, ReLU
 *         → conv3×3(Cout→Cout, stride=1), BN
 *         + conv1×1(Cin→Cout, stride=2), BN   ← projection shortcut
 *   → ReLU
 *
 * Weight tensor layout in VWB2 blob (gen_resnet1202_model.py):
 *   [0]          conv1 stem (3×3×3→16)
 *   [1   .. 400] stage1 blocks 0..199, conv_a then conv_b each block
 *   [401 .. 801] stage2: block0={conv_a,conv_b,proj}, blocks1..199={conv_a,conv_b}
 *   [802 ..1202] stage3: same pattern as stage2
 *   [1203]       fc classifier (64→10)
 *
 * Design note: with 600 blocks the configuration is generated on-the-fly
 * from (stage, block_idx) rather than stored as a 600-entry table, saving
 * ~7 KB of BRAM.  See rn1202_block_conf().
 */
#ifndef RESNET1202_LAYERS_H
#define RESNET1202_LAYERS_H

#include <stdint.h>

/* ── §1: Architecture constants ──────────────────────────────────────── */

#define RN1202_N_PER_STAGE  200          /* n in 6n+2 formula               */
#define RN1202_N_STAGES       3          /* stage1, stage2, stage3          */
#define RN1202_TOTAL_BLOCKS 600          /* 3 × 200                         */
#define RN1202_BLOCK_SIZE    32          /* BD4 block size (elements/block)  */
#define RN1202_N_CLASSES     10          /* CIFAR-10                        */

/* Input image dimensions (CIFAR-10) */
#define RN1202_INPUT_H  32
#define RN1202_INPUT_W  32
#define RN1202_INPUT_C   3

/* Channel counts per stage */
#define RN1202_STAGE1_C  16u
#define RN1202_STAGE2_C  32u
#define RN1202_STAGE3_C  64u

/* Spatial dimensions per stage (output of each stage's first block) */
#define RN1202_STAGE1_H  32u
#define RN1202_STAGE1_W  32u
#define RN1202_STAGE2_H  16u
#define RN1202_STAGE2_W  16u
#define RN1202_STAGE3_H   8u
#define RN1202_STAGE3_W   8u

/* Total parameter counts per layer group (BN folded):                     */
/*   conv1:  3 × 3 × 3 × 16 =       432                                   */
/*   stage1: 200 × 2 × (3×3×16×16)  = 200 × 2 × 2304  =   921,600        */
/*   stage2: first block 2×(3×3×16×32)+(1×1×16×32) + 199×2×(3×3×32×32)  */
/*         = 2×4608+512 + 199×18432 = 9728 + 3,668,768 = 3,678,496       */
/*           (proj block adds 512 for conv1×1) → 3,687,424 with BN bias   */
/*   stage3: similarly 14,748,160                                          */
/*   fc:     64 × 10 = 640                                                 */

/* ── §1: VWB2 tensor ID layout constants ────────────────────────────── */

/* ── §4: Projection shortcut flag ──────────────────────────────────
 * RN1202_HAS_PROJ controls whether the first block of stage 2 and 3 has
 * a learnable conv1×1 projection shortcut (Option B) or a zero-pad skip
 * (Option A, the akamaster hub model default).
 *
 * Default: 0 (Option A — akamaster pretrained, load_model_hub() path).
 * Set to 1 only when using a checkpoint trained from scratch (--train).
 *
 * This flag also controls tensor ID layout because projection blocks
 * occupy 3 tensor slots (conv_a, conv_b, proj) whereas identity/zero-pad
 * blocks occupy 2 (conv_a, conv_b).
 */
#ifndef RN1202_HAS_PROJ
#  define RN1202_HAS_PROJ  0   /* 0 = Option A (default); 1 = Option B (--train) */
#endif

/* ── §1: VWB2 tensor ID layout constants ────────────────────────────
 * RN1202_B0_TENSORS: slots used by block 0 of stage 2 and 3.
 *   Option A (no proj): 2  (conv_a, conv_b)
 *   Option B (proj):    3  (conv_a, conv_b, proj)
 * All other blocks use exactly 2 slots.
 */
#define RN1202_B0_TENSORS        (2 + RN1202_HAS_PROJ)

/* First tensor ID for each group */
#define RN1202_TID_CONV1         0      /* stem conv (single tensor)               */
#define RN1202_TID_STAGE1_FIRST  1      /* stage1 block 0, conv_a (1 + 200*2 = 401 slots follow) */
/* stage2 starts after conv1 + 200*2 stage1 tensors = 401 */
#define RN1202_TID_STAGE2_FIRST  401    /* stage2 block 0, conv_a                  */
/* stage3 starts after stage2: RN1202_B0_TENSORS + 199*2 tensors = B0+398 */
#define RN1202_TID_STAGE3_FIRST  (401 + RN1202_B0_TENSORS + 398)
/* FC after stage3: same pattern */
#define RN1202_TID_FC            (RN1202_TID_STAGE3_FIRST + RN1202_B0_TENSORS + 398)

/* Total weight layer count (each layer has weight + bias tensors in the blob) */
#define RN1202_TOTAL_TENSORS     (RN1202_TID_FC + 1)

/* ── §1: BlockDialect block count per weight tensor ────────────────── */
/* n_bd_blocks = ceil(n_elements / 32)                                     */
/*   conv1:         3×3×3 = 27 elements per OC, 16 OC → 27*16=432 → 14 bd */
/*   stage1 conv3×3: 3×3×16 = 144 elements per OC, 16 OC → 2304 → 72 bd  */
/*   stage2 proj1×1: 1×1×16 =  16 elements per OC, 32 OC →  512 → 16 bd  */
/*   stage2 conv3×3(16→32): 3×3×16=144 per OC, 32 OC → 4608 → 144 bd     */
/*   stage2 conv3×3(32→32): 3×3×32=288 per OC, 32 OC → 9216 → 288 bd     */
/*   stage3 conv3×3(32→64): 3×3×32=288 per OC, 64 OC → 18432 → 576 bd    */
/*   stage3 conv3×3(64→64): 3×3×64=576 per OC, 64 OC → 36864 → 1152 bd   */
/*   fc: 64 elements, 10 OC → 640 → 20 bd                                 */

/* ── §1: BasicBlock configuration struct ────────────────────────────── */

typedef struct {
    uint8_t  stage;      /* 1, 2, or 3                                      */
    uint8_t  stride;     /* 1 = identity block; 2 = projection block        */
    uint8_t  has_proj;   /* 1 if conv1×1 projection shortcut present        */
    uint8_t  shift_a;    /* output right-shift for conv_a (calibrated)      */
    uint8_t  shift_b;    /* output right-shift for conv_b (calibrated)      */
    uint8_t  shift_proj; /* output right-shift for proj conv1×1 (calibrated)*/
    uint16_t in_c;       /* block input channels                             */
    uint16_t out_c;      /* block output channels (= in_c for identity)     */
    uint16_t in_h;       /* block input spatial height                       */
    uint16_t in_w;       /* block input spatial width                        */
    /* Tensor IDs for the weights in this block */
    uint16_t tid_conv_a; /* first  conv3×3                                  */
    uint16_t tid_conv_b; /* second conv3×3                                  */
    uint16_t tid_proj;   /* projection conv1×1 (valid only when has_proj=1) */
} BasicBlockConf;

/* ── Per-stage calibrated output shifts ─────────────────────────────── */
/* Produced by quantized_reference.py calibrate_rn1202_shifts().          */
/* Index by stage (1..3); index 0 is unused padding.                     */
#ifndef RN1202_SHIFT_STAGE1
#  define RN1202_SHIFT_STAGE1  0
#endif
#ifndef RN1202_SHIFT_STAGE2
#  define RN1202_SHIFT_STAGE2  0
#endif
#ifndef RN1202_SHIFT_STAGE3
#  define RN1202_SHIFT_STAGE3  0
#endif

/* ── §1: On-the-fly block configuration generator ──────────────────── */
/*
 * rn1202_block_conf() computes the full BasicBlockConf for any
 * (stage in [1..3], block_idx in [0..N_PER_STAGE-1]) pair without a
 * static table, saving ~7 KB of BRAM.  This is the approach prescribed
 * in §4 of RESNET1202_FPGA_PLAN.md.
 *
 * Channel progression:
 *   CHANS[1]=16 → CHANS[2]=32 → CHANS[3]=64
 *
 * Spatial progression (stride-2 first blocks halve H and W):
 *   stage1: 32×32 (no downsampling)
 *   stage2: 32×32 → 16×16 (first block stride=2)
 *   stage3: 16×16 →  8×8  (first block stride=2)
 *
 * Tensor IDs are computed from the offsets defined above.
 */
static inline BasicBlockConf rn1202_block_conf(int stage, int block_idx)
{
    /* Channel count indexed by stage (index 0 unused) */
    static const uint16_t CHANS[4] = { 0u, 16u, 32u, 64u };

    /* Input spatial dimensions indexed by stage (index 0 unused) */
    static const uint16_t IN_H[4]  = { 0u, 32u, 32u, 16u };
    static const uint16_t IN_W[4]  = { 0u, 32u, 32u, 16u };

    BasicBlockConf c;
    c.stage    = (uint8_t)stage;

    /* Per-stage calibrated shifts (index 0 unused) */
    static const uint8_t SHIFTS[4] = { 0u,
        RN1202_SHIFT_STAGE1, RN1202_SHIFT_STAGE2, RN1202_SHIFT_STAGE3 };
    c.shift_a    = SHIFTS[stage];
    c.shift_b    = SHIFTS[stage];
    c.shift_proj = SHIFTS[stage];

    /* Downsampling (stride=2) occurs at the first block of stage 2 and 3.
     * Whether that block also has a learnable projection shortcut depends on
     * the training option: Option B (RN1202_HAS_PROJ=1) adds a conv1×1;
     * Option A (RN1202_HAS_PROJ=0, default akamaster model) uses zero-pad. */
    int is_downsample = (stage >= 2 && block_idx == 0);
    c.has_proj = (uint8_t)(RN1202_HAS_PROJ && is_downsample);
    c.stride   = is_downsample ? 2u : 1u;

    /* Input channels: previous stage's channel count for block 0,
     * or current stage's channel count for all subsequent blocks.
     *
     *   stage1 block 0: in_c = CHANS[1] = 16  (from conv1 stem)
     *   stage2 block 0: in_c = CHANS[1] = 16  (from stage1 output)
     *   stage3 block 0: in_c = CHANS[2] = 32  (from stage2 output)
     *   any other block: in_c = out_c = CHANS[stage]
     */
    if (block_idx == 0 && stage >= 2) {
        c.in_c  = CHANS[stage - 1];
    } else {
        c.in_c  = CHANS[stage];
    }
    c.out_c = CHANS[stage];

    /* Input spatial dimensions:
     * For projection blocks the input is at the pre-downsampling size;
     * for identity blocks it is the current stage's output size.
     * Stage 1: always 32×32.
     * Stage 2: block 0 input is 32×32, rest are 16×16.
     * Stage 3: block 0 input is 16×16, rest are 8×8.
     */
    c.in_h = IN_H[stage];
    c.in_w = IN_W[stage];
    /* Projection blocks use the pre-strided spatial size (same as IN_H/W
     * since the stride is applied inside the block, not to the input). */

    /* ── Tensor IDs ─── */
    /* stage1: each block uses 2 consecutive IDs starting at offset 1 */
    /* stage2: block0 uses 3 IDs (conv_a, conv_b, proj); rest use 2   */
    /* stage3: same pattern                                            */
    switch (stage) {
        case 1: {
            int base     = RN1202_TID_STAGE1_FIRST + block_idx * 2;
            c.tid_conv_a = (uint16_t)(base);
            c.tid_conv_b = (uint16_t)(base + 1);
            c.tid_proj   = 0xFFFFu;   /* unused/invalid */
            break;
        }
        case 2: {
            int base;
            if (block_idx == 0) {
                base = RN1202_TID_STAGE2_FIRST;
                c.tid_conv_a = (uint16_t)(base);
                c.tid_conv_b = (uint16_t)(base + 1);
                /* proj tensor exists only for Option B */
                c.tid_proj   = RN1202_HAS_PROJ ? (uint16_t)(base + 2) : 0xFFFFu;
            } else {
                /* Non-zero blocks follow block 0 which uses RN1202_B0_TENSORS slots */
                base = RN1202_TID_STAGE2_FIRST + RN1202_B0_TENSORS + (block_idx - 1) * 2;
                c.tid_conv_a = (uint16_t)(base);
                c.tid_conv_b = (uint16_t)(base + 1);
                c.tid_proj   = 0xFFFFu;
            }
            break;
        }
        case 3: {
            int base;
            if (block_idx == 0) {
                base = RN1202_TID_STAGE3_FIRST;
                c.tid_conv_a = (uint16_t)(base);
                c.tid_conv_b = (uint16_t)(base + 1);
                c.tid_proj   = RN1202_HAS_PROJ ? (uint16_t)(base + 2) : 0xFFFFu;
            } else {
                base = RN1202_TID_STAGE3_FIRST + RN1202_B0_TENSORS + (block_idx - 1) * 2;
                c.tid_conv_a = (uint16_t)(base);
                c.tid_conv_b = (uint16_t)(base + 1);
                c.tid_proj   = 0xFFFFu;
            }
            break;
        }
        default:
            c.tid_conv_a = 0xFFFFu;
            c.tid_conv_b = 0xFFFFu;
            c.tid_proj   = 0xFFFFu;
            break;
    }

    return c;
}

/* ── §1: Tensor ID helpers for stem and classifier ───────────────────── */

/* Returns the VWB2 tensor ID for the conv1 stem (3→16, 3×3). */
static inline uint16_t rn1202_tid_conv1(void)
{
    return RN1202_TID_CONV1;
}

/* Returns the VWB2 tensor ID for the final fully-connected layer (64→10). */
static inline uint16_t rn1202_tid_fc(void)
{
    return RN1202_TID_FC;
}

/* ── §1: BD block count per weight tensor ───────────────────────────── */
/*
 * Returns the number of BD4 blocks for a weight tensor, given the
 * number of input and output channels plus the kernel size.
 * n_bd_blocks = ceil((kH * kW * in_c * out_c) / 32)
 * Note: n_elements is the TOTAL elements in the tensor, not per-OC.
 */
static inline uint32_t rn1202_bd_blocks_for_tensor(
    uint16_t in_c, uint16_t out_c, uint8_t kH, uint8_t kW)
{
    uint32_t n_elem = (uint32_t)in_c * out_c * kH * kW;
    return (n_elem + 31u) / 32u;
}

/* Returns BD block count for the weight tensor at tensor_id.
 * This mirrors gen_resnet1202_model.py's per-tensor layout.
 *
 * For a BasicBlock's conv3×3:
 *   identity block: in_c == out_c, kH = kW = 3 → n/32 blocks
 *   projection conv_a: in_c = CHANS[s-1], out_c = CHANS[s], kH=kW=3
 *   projection proj:   in_c = CHANS[s-1], out_c = CHANS[s], kH=kW=1
 *
 * Callers may also obtain per-OC block counts by dividing by out_c.
 */
static inline uint32_t rn1202_tensor_bd_blocks(
    int stage, int block_idx, int conv_id)
{
    static const uint16_t CHANS[4] = { 0u, 16u, 32u, 64u };

    uint16_t in_c, out_c;
    uint8_t  kH = 3u, kW = 3u;   /* default: 3×3 */

    out_c = CHANS[stage];

    if (block_idx == 0 && stage >= 2) {
        in_c = CHANS[stage - 1];   /* first block: previous stage channels */
    } else {
        in_c = CHANS[stage];
    }

    if (conv_id == 2) {
        /* Projection conv1×1 */
        kH = 1u; kW = 1u;
        /* in_c already set correctly above for block_idx==0 */
    }

    return rn1202_bd_blocks_for_tensor(in_c, out_c, kH, kW);
}

/* ── §1: Weight decode scratch size needed for a given block ────────── */
/*
 * Returns the maximum number of int16 half-unit slots needed in the
 * weight decode scratch buffer for a single output channel of the
 * largest kernel in this block (worst case: conv3×3 of the projection block).
 *
 * Stage 3: in_c=64 for identity blocks → 64×3×3=576 elements per OC.
 * This matches W_SCRATCH_HU_MAX in spram_layout.h.
 */
static inline uint32_t rn1202_scratch_hu_per_oc(int stage, int block_idx, int conv_id)
{
    static const uint16_t CHANS[4] = { 0u, 16u, 32u, 64u };

    uint16_t in_c;
    uint8_t  k;

    if (conv_id == 2) {
        /* Projection conv1×1 */
        k    = 1u;
        in_c = CHANS[stage - 1];   /* always projection block → in_c from prev stage */
    } else {
        k    = 3u;
        if (block_idx == 0 && stage >= 2) {
            in_c = CHANS[stage - 1];
        } else {
            in_c = CHANS[stage];
        }
    }

    return (uint32_t)in_c * k * k;
}

/* ── §1: Stage-loop iteration helpers ───────────────────────────────── */

/* Total number of blocks in a stage (always N_PER_STAGE = 200). */
static inline int rn1202_stage_n_blocks(int stage)
{
    (void)stage;
    return RN1202_N_PER_STAGE;
}

/* Output spatial height after a stage.
 * Stage 1 output: 32.  Stage 2 output: 16.  Stage 3 output: 8.
 */
static inline uint16_t rn1202_stage_out_h(int stage)
{
    static const uint16_t OUT_H[4] = { 0u, 32u, 16u, 8u };
    return OUT_H[stage];
}

static inline uint16_t rn1202_stage_out_w(int stage)
{
    return rn1202_stage_out_h(stage);   /* CIFAR: square spatial dims */
}

/* ── §1: Compile-time topology self-check ───────────────────────────── */
/*
 * Expected tensor counts:
 *   Option A (RN1202_HAS_PROJ=0): 1202 weight layers  (TID_FC = 1201)
 *   Option B (RN1202_HAS_PROJ=1): 1204 weight layers  (TID_FC = 1203)
 */
#if RN1202_HAS_PROJ
_Static_assert(RN1202_TID_FC == 1203,
    "Option B: FC tensor ID must be 1203");
_Static_assert(RN1202_TOTAL_TENSORS == 1204,
    "Option B: Total tensor count must be 1204");
#else
_Static_assert(RN1202_TID_FC == 1201,
    "Option A: FC tensor ID must be 1201 (no proj tensors)");
_Static_assert(RN1202_TOTAL_TENSORS == 1202,
    "Option A: Total tensor count must be 1202 (no proj tensors)");
#endif

/* Scratch buffer must accommodate the largest per-OC weight slice */
/* stage3 identity conv3×3: in_c=64, 3×3 → 576 elements per OC     */
#define RN1202_SCRATCH_HU_NEEDED 576u   /* elements */
/* Forward-reference: DEFINE_ACT_BUFFERS() must provide w_decode_hu[576] */

#endif /* RESNET1202_LAYERS_H */
