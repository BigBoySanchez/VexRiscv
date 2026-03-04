/* spram_layout.h — Compile-time SPRAM activation buffer layout for ResNet-1202
 *
 * §0 of RESNET1202_FPGA_PLAN.md: Memory architecture
 *
 * Physical SPRAM: 128 KB at 0x11000000–0x1101FFFF
 *
 * Budget summary (worst case: stage1, int8 skip = most conservative):
 *   buf_A int8  (16×32×32)  16,384 B = 16.0 KB   0x11000000 – 0x11003FFF
 *   buf_B int8  (16×32×32)  16,384 B = 16.0 KB   0x11004000 – 0x11007FFF
 *   skip_buf int8 (16,384B) 16,384 B = 16.0 KB   0x11008000 – 0x1100BFFF
 *   weight decode scratch     1,224 B =  1.2 KB   0x1100C000 – 0x1100C4C7
 *                                                  ──────────────────────
 *   .act_buffers total       ~50,376 B = ~49.2 KB  (< 52 KB reserved)
 *   .bss + .noinit + stack    ~3,072 B =  3.0 KB   at ~0x1100D000
 *                                                  ──────────────────────
 *   Peak SPRAM used           ~53,448 B ≈  52.2 KB
 *   Available:               131,072 B = 128.0 KB
 *   Headroom:                ~77,624 B ≈  75.8 KB  ✓✓✓
 *
 * BD4 skip variant (phase2+): skip_buf holds 9,216 B instead of 16,384 B.
 * The .act_buffers reserved allocation stays the same; the space simply
 * goes unused, with headroom increasing accordingly.
 *
 * How to use these buffers in firmware:
 *
 *   // In exactly one .c file (e.g. main.c):
 *   DEFINE_ACT_BUFFERS();
 *
 *   // Everywhere else, or in the same file after DEFINE_ACT_BUFFERS:
 *   #include "spram_layout.h"
 *   // act_A, act_B, skip_buf, w_decode_scratch are then available.
 *
 * All arrays are placed in ".act_buffers" section so the linker puts them
 * at 0x11000000, BEFORE .bss — matching the hardcoded ACT_*_BASE constants.
 */
#ifndef SPRAM_LAYOUT_H
#define SPRAM_LAYOUT_H

#include <stdint.h>

/* ── §0: Hardcoded SPRAM region base addresses ──────────────────────── */

/* int8 ping/pong activation buffers (used during conv passes) */
#define ACT_A_BASE    0x11000000u   /* 16 KB ping buffer */
#define ACT_B_BASE    0x11004000u   /* 16 KB pong buffer */

/* Residual/skip connection tensor.
 * Stage 1 worst case: 16×32×32 = 16,384 elements.
 * Stored as BD4 (space-saving) or int8 (simpler) depending on phase:
 *   phase1_int8:  SKIP_BUF_SIZE_INT8  = 16,384 bytes
 *   phase2_bd_full: SKIP_BUF_SIZE_BD4 =  9,216 bytes */
#define SKIP_BUF_BASE 0x11008000u

/* Weight decode scratch: one output-channel slice at a time.
 * Stage 3 worst case per OC: 64 × 3 × 3 = 576 elements.
 * int16 half-units (2 bytes each) + int32 shared_exp array.
 * 576 * 2 = 1,152 B (hu) + ceil(576/32)*4 = 72 B (exp) = 1,224 B.
 * Placed after 3 × 16 KB int8 buffers: 0x11000000 + 3×0x4000 = 0x1100C000. */
#define W_SCRATCH_BASE 0x1100C000u

/* ── §0: Buffer sizes per stage ──────────────────────────────────────── */

/* Maximum int8 activation buffer needed across all stages (stage1) */
#define ACT_MAX_INT8_SIZE  16384u   /* 16×32×32 */

/* Per-stage activation buffer sizes (int8) */
#define ACT_STAGE1_SIZE_INT8  16384u   /* 16 × 32 × 32 */
#define ACT_STAGE2_SIZE_INT8   8192u   /* 32 × 16 × 16 */
#define ACT_STAGE3_SIZE_INT8   4096u   /* 64 ×  8 ×  8 */

/* Per-stage activation buffer sizes (BD4 packed: ceil(n/32)*18 bytes) */
#define ACT_STAGE1_SIZE_BD4    9216u   /* 512 blocks × 18 B */
#define ACT_STAGE2_SIZE_BD4    4608u   /* 256 blocks × 18 B */
#define ACT_STAGE3_SIZE_BD4    2304u   /* 128 blocks × 18 B */

/* Residual skip buffer sizes */
#define SKIP_BUF_SIZE_INT8    16384u   /* worst case: stage1 int8 */
#define SKIP_BUF_SIZE_BD4      9216u   /* worst case: stage1 BD4  */

/* Weight decode scratch sizes */
#define W_SCRATCH_HU_MAX        576u   /* elements per OC, stage3 worst case */
#define W_SCRATCH_HU_BYTES     1152u   /* × 2 bytes per int16 */
#define W_SCRATCH_EXP_BYTES      72u   /* ceil(576/32)=18 blocks × 4 bytes */
#define W_SCRATCH_TOTAL_BYTES  1224u

/* ── §0: Alternative BD4 layout (used from phase2 onward) ───────────── */

/* When activation buffers hold BD4 data instead of int8, the same base
 * addresses are used but the logical size is smaller.  phase2 firmware
 * MAY choose to use these aliases for clarity. */
#define ACT_BD_A_BASE  ACT_A_BASE   /* reuse same physical address */
#define ACT_BD_B_BASE  ACT_B_BASE

/* ── §0: Section attribute for activation arrays ─────────────────────── */

/* All activation arrays must be tagged with this attribute so the linker
 * places them at 0x11000000 (before .bss), matching the *_BASE constants. */
#define ACT_SECTION __attribute__((section(".act_buffers")))

/* ── §0: Convenience macro to define all activation buffers in one TU ── */
/*
 * Place DEFINE_ACT_BUFFERS() in exactly ONE .c file.
 * The arrays will be placed in .act_buffers, starting at ACT_A_BASE.
 * Their order in the section is the declaration order below, matching
 * the address constants above.
 */
#define ACT_A_SIZE    ACT_MAX_INT8_SIZE   /* 16 KB  (int8 or BD4: same slot) */
#define ACT_B_SIZE    ACT_MAX_INT8_SIZE   /* 16 KB  (int8 or BD4: same slot) */
#define SKIP_BUF_SIZE SKIP_BUF_SIZE_INT8  /* 16 KB  (worst case; BD4 uses 9 KB) */

#define DEFINE_ACT_BUFFERS()                                            \
    int8_t act_A[ACT_A_SIZE]    ACT_SECTION;                           \
    int8_t act_B[ACT_B_SIZE]    ACT_SECTION;                           \
    int8_t skip_buf[SKIP_BUF_SIZE] ACT_SECTION;                        \
    int16_t w_decode_hu[W_SCRATCH_HU_MAX] ACT_SECTION;                 \
    int32_t w_decode_exp[18]    ACT_SECTION  /* 18 = ceil(576/32) */

/* Extern declarations for use in other translation units */
extern int8_t  act_A[];
extern int8_t  act_B[];
extern int8_t  skip_buf[];
extern int16_t w_decode_hu[];
extern int32_t w_decode_exp[];

/* ── §0: BD4 activation layout (phase4+, paper-faithful) ─────────────── */
/*                                                                           */
/* When activations are stored as BD4 (no int8 intermediate), the int32      */
/* accumulator scratch dominates SPRAM.  Layout in physical SPRAM:           */
/*   accum_scratch int32 (16×32×32×4 = 64 KB)  0x11000000 – 0x1100FFFF     */
/*   act_A BD4 (512 blocks × 18 B = 9 KB)       0x11010000 – 0x11012400     */
/*   act_B BD4 (9 KB)                            0x11012400 – 0x11014800     */
/*   bd_skip BD4 (9 KB)                          0x11014800 – 0x11016C00     */
/*   stack / bss (~2 KB)                         above 0x11016C00            */
/*   Total: ~93 KB < 128 KB ✓                                                */
/*                                                                           */
/* The int32 scratch buffer occupies the FIRST 64 KB of SPRAM, followed by  */
/* the three 9 KB BD4 buffers.  The caller passes the scratch pointer to     */
/* conv3x3_bd4 / conv1x1_bd4 directly; no additional base address needed.   */

/* Stage1 worst-case int32 accumulator scratch (16×32×32 × 4 = 64 KB) */
#define ACT_STAGE1_ACCUM_ELEMS  16384u
#define ACT_STAGE1_ACCUM_BYTES  65536u
/* Smaller stages for reference */
#define ACT_STAGE2_ACCUM_ELEMS   8192u   /* 32 × 16 × 16 */
#define ACT_STAGE2_ACCUM_BYTES  32768u
#define ACT_STAGE3_ACCUM_ELEMS   4096u   /* 64 ×  8 ×  8 */
#define ACT_STAGE3_ACCUM_BYTES  16384u

/* Maximum int32 accumulator size across all stages (stage1) */
#define ACT_MAX_ACCUM_ELEMS  ACT_STAGE1_ACCUM_ELEMS
#define ACT_MAX_ACCUM_BYTES  ACT_STAGE1_ACCUM_BYTES

/* SPRAM base addresses for the BD4 ping/pong and skip buffers in the new layout.
 * accum scratch sits at ACT_A_BASE (0x11000000) and occupies 64 KB. */
#define ACT_BD4_PING_BASE   0x11010000u   /* 9 KB BD4 ping buffer */
#define ACT_BD4_PONG_BASE   0x11012400u   /* 9 KB BD4 pong buffer */
#define ACT_BD4_SKIP_BASE   0x11014800u   /* 9 KB BD4 skip buffer */

/*
 * DEFINE_ACT_BUFFERS_BD4() — allocate BD4 activation layout in SPRAM.
 *
 * Place in exactly ONE .c file (typically main.c of the phase4 firmware).
 * Allocates (in .act_buffers section, order matters for SPRAM layout):
 *   1. int32 accum_scratch[16384] = 64 KB  — int32 conv accumulator
 *   2. uint8_t bd_act_A[9216]     =  9 KB  — BD4 ping buffer
 *   3. uint8_t bd_act_B[9216]     =  9 KB  — BD4 pong buffer
 *   4. uint8_t bd_skip_bd4[9216]  =  9 KB  — BD4 skip connection buffer
 *   5. int8_t act_unpack_i8[16384]= 16 KB  — bulk BD4→int8 unpack cache
 *   6. int16_t w_decode_hu[576]   =  1 KB  — weight decode scratch
 *   7. int32_t w_decode_exp[18]   = 72 B
 *
 * Total: ~108 KB < 128 KB.  The int8 unpack cache eliminates the
 * per-element bd4_read_cached overhead in conv loops (~5× speedup).
 */
#define DEFINE_ACT_BUFFERS_BD4()                                             \
    int32_t accum_scratch[ACT_MAX_ACCUM_ELEMS] ACT_SECTION;                  \
    uint8_t bd_act_A[ACT_STAGE1_SIZE_BD4]      ACT_SECTION;                  \
    uint8_t bd_act_B[ACT_STAGE1_SIZE_BD4]      ACT_SECTION;                  \
    uint8_t bd_skip_bd4[SKIP_BUF_SIZE_BD4]     ACT_SECTION;                  \
    int8_t  act_unpack_i8[ACT_MAX_INT8_SIZE]   ACT_SECTION;                  \
    int16_t w_decode_hu[W_SCRATCH_HU_MAX]      ACT_SECTION;                  \
    int32_t w_decode_exp[18]                   ACT_SECTION

extern int32_t accum_scratch[];
extern uint8_t bd_act_A[];
extern uint8_t bd_act_B[];
extern uint8_t bd_skip_bd4[];
extern int8_t  act_unpack_i8[];

/* ── §0: Static assertions on address alignment ──────────────────────── */
/*
 * These fire at compile time if the reserved region overflows SPRAM.
 * The top of .act_buffers must not exceed 0x11020000 (SPRAM end).
 *
 * sizeof(all arrays) = 16384 + 16384 + 16384 + 1152 + 72 = 50,376 B
 * 0x11000000 + 50,376 = 0x1100C488 << 0x11020000  ✓
 */
#define _ACT_TOTAL_BYTES (ACT_A_SIZE + ACT_B_SIZE + SKIP_BUF_SIZE \
                          + W_SCRATCH_HU_BYTES + W_SCRATCH_EXP_BYTES)

_Static_assert(_ACT_TOTAL_BYTES <= (128u * 1024u - 2048u - 512u),
    "SPRAM activation buffers exceed available SPRAM (128 KB - stack - bss)");

/* BD4 layout static assert (accum + 3×BD4 buffers + int8 unpack + scratch) */
#define _ACT_BD4_TOTAL_BYTES (ACT_MAX_ACCUM_BYTES          \
    + ACT_STAGE1_SIZE_BD4 * 3u + ACT_MAX_INT8_SIZE         \
    + W_SCRATCH_HU_BYTES + W_SCRATCH_EXP_BYTES)

_Static_assert(_ACT_BD4_TOTAL_BYTES <= (128u * 1024u - 2048u - 512u),
    "BD4 SPRAM buffers exceed available SPRAM (128 KB - stack - bss)");

/* ── §0: HWCB (tap-blocked) activation buffer sizes ────────────────────
 *
 * When using the HWCB (HWC-Channel-Blocked) activation layout for
 * paper-faithful per-tap channel blocking, each BD block covers exactly
 * 32 channels at one (y,x) spatial position.  Buffer sizes differ from
 * the flat CHW layout for stage1 because ceil(C/32)*H*W blocks are
 * needed instead of ceil(C*H*W/32).
 *
 *   Stage1 (16ch, 32×32): n_cb=1, 32*32*1*18 = 18 432 B  (2× CHW)
 *   Stage2 (32ch, 16×16): n_cb=1, 16*16*1*18 =  4 608 B  (same as CHW)
 *   Stage3 (64ch,  8× 8): n_cb=2,  8* 8*2*18 =  2 304 B  (same as CHW)
 *
 * Since we size for the worst case (stage1), max HWCB = 18 432 B.
 *
 * SPRAM budget for DEFINE_ACT_BUFFERS_BD4_TAP (no large accum scratch):
 *   bd_act_A  uint8[18432] = 18 KB
 *   bd_act_B  uint8[18432] = 18 KB
 *   bd_skip   uint8[18432] = 18 KB
 *   w_decode  int16[576]   =  1 KB
 *   ─────────────────────────────────
 *   Total:             ≈ 55 KB  << 128 KB  ✓✓✓
 *
 * The large int32 accum_scratch (64 KB) is not needed because tap-blocked
 * convolution accumulates one output spatial-pixel at a time using a small
 * stack-local array (at most out_c=64 int32 values = 256 B).
 */
#define ACT_STAGE1_SIZE_BD4_HWCB  18432u   /* 1024 blocks × 18 B */
#define ACT_STAGE2_SIZE_BD4_HWCB   4608u   /* 256 blocks × 18 B  (same as CHW) */
#define ACT_STAGE3_SIZE_BD4_HWCB   2304u   /* 128 blocks × 18 B  (same as CHW) */
#define ACT_MAX_SIZE_BD4_HWCB      ACT_STAGE1_SIZE_BD4_HWCB

/*
 * DEFINE_ACT_BUFFERS_BD4_TAP() — allocate HWCB BD4 activation layout.
 *
 * Place in exactly ONE .c file (typically main.c when USE_TAP_BLOCKED=1).
 * Do NOT combine with DEFINE_ACT_BUFFERS_BD4() in the same translation
 * unit — they define the same symbol names at different sizes.
 *
 * Allocates (in order, placed in .act_buffers at 0x11000000):
 *   1. uint8_t bd_act_A[18432]    = 18 KB  — HWCB BD4 ping buffer
 *   2. uint8_t bd_act_B[18432]    = 18 KB  — HWCB BD4 pong buffer
 *   3. uint8_t bd_skip_bd4[18432] = 18 KB  — HWCB BD4 skip buffer
 *   4. int16_t w_decode_hu[576]   =  1 KB  — weight decode scratch
 *   5. int32_t w_decode_exp[18]   = 72 B
 *
 * Total: ~55 KB.  The int32 accum_scratch present in DEFINE_ACT_BUFFERS_BD4
 * is intentionally absent; tap-blocked convolutions use a small stack array.
 */
#define DEFINE_ACT_BUFFERS_BD4_TAP()                                         \
    uint8_t bd_act_A[ACT_MAX_SIZE_BD4_HWCB]    ACT_SECTION;                  \
    uint8_t bd_act_B[ACT_MAX_SIZE_BD4_HWCB]    ACT_SECTION;                  \
    uint8_t bd_skip_bd4[ACT_MAX_SIZE_BD4_HWCB] ACT_SECTION;                  \
    int16_t w_decode_hu[W_SCRATCH_HU_MAX]       ACT_SECTION;                  \
    int32_t w_decode_exp[18]                    ACT_SECTION

/* Static assert: HWCB tap layout must fit in SPRAM */
#define _ACT_BD4_TAP_TOTAL_BYTES (ACT_MAX_SIZE_BD4_HWCB * 3u \
    + W_SCRATCH_HU_BYTES + W_SCRATCH_EXP_BYTES)

_Static_assert(_ACT_BD4_TAP_TOTAL_BYTES <= (128u * 1024u - 2048u - 512u),
    "BD4-TAP SPRAM buffers exceed available SPRAM (128 KB - stack - bss)");

#endif /* SPRAM_LAYOUT_H */
