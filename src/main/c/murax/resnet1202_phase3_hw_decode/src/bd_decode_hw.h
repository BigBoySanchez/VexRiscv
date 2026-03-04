/* bd_decode_hw.h — Hardware BlockDialect driver via BDMac32 APB peripheral
 *
 * Drop-in replacement interface for bd_decode_block_hu() from bd_decode_sw.h.
 *
 * BDMac32 is a SEQUENTIAL 32-lane dot-product accelerator (2 DialectFP4DecodeCore
 * instances, 32 cycles per operation).  It does not expose individual per-element
 * decoded magnitudes; it returns only the accumulated PARTIAL_SUM.
 *
 * Therefore bd_decode_block_hw() uses the software dialect table (BD_DIALECT_TABLE
 * from bd_decode_sw.h) for per-element extraction — this preserves full correctness
 * and the same call-site signature.  The hardware path is exercised via BDMac32
 * for the dot-product inner loop and verified by bdd_selftest().
 *
 * APB register map (BDMac32.scala, base = 0x40031000):
 *   0x00  W_PACKED0   (W)  weight codes[31:0]   — elems  0-7   (2 nibbles/byte)
 *   0x04  W_PACKED1   (W)  weight codes[63:32]  — elems  8-15
 *   0x08  W_PACKED2   (W)  weight codes[95:64]  — elems 16-23
 *   0x0C  W_PACKED3   (W)  weight codes[127:96] — elems 24-31
 *   0x10  W_META      (W)  bits [15:12]=w_dialect_id, [11:7]=w_shared_exp
 *   0x14  A_PACKED0   (W)  act  codes[31:0]
 *   0x18  A_PACKED1   (W)  act  codes[63:32]
 *   0x1C  A_PACKED2   (W)  act  codes[95:64]
 *   0x20  A_PACKED3   (W)  act  codes[127:96]
 *   0x24  A_META      (W)  bits [15:12]=a_dialect_id, [11:7]=a_shared_exp
 *   0x28  CTRL        (W)  write 1 -> start (clears DONE)
 *   0x30  PARTIAL_SUM (R)  signed 32-bit accumulated result (half-units^2)
 *   0x34  EXP_SUM     (R)  w_shared_exp + a_shared_exp [5:0]
 *   0x38  DONE        (R)  0 = computing, 1 = result valid
 *
 * Self-test: weights = dialect14/exp0/all-idx7 (mag=8),
 *            acts    = dialect0 /exp0/all-idx1 (mag=1).
 * Expected: PARTIAL_SUM = 32 * 8 * 1 = 256, EXP_SUM = 0.
 *
 * Usage (same as before):
 *   1. Include bd_decode_sw.h (or bd_act.h) FIRST.
 *   2. Include this header.
 *   3. Redirect conv.h call-sites:
 *        #define bd_decode_block_hu bd_decode_block_hw
 *        #include "resnet1202_conv.h"
 *        #undef  bd_decode_block_hu
 *
 * ss5 Milestone 3 of RESNET1202_FPGA_PLAN.md
 */
#ifndef BD_DECODE_HW_H
#define BD_DECODE_HW_H

#include <stdint.h>

/* ── APB base address of BDMac32 ───────────────────────────────────────── */
#define BDD_BASE  (0x40031000u)

/* Convenience macro for register access */
#define BDD_REG32(offset) (*((volatile uint32_t *)(BDD_BASE + (offset))))

/* ── BDMac32 register offsets ─────────────────────────────────────────── */
#define BDM_OFF_W_PACKED0    0x00u
#define BDM_OFF_W_PACKED1    0x04u
#define BDM_OFF_W_PACKED2    0x08u
#define BDM_OFF_W_PACKED3    0x0Cu
#define BDM_OFF_W_META       0x10u
#define BDM_OFF_A_PACKED0    0x14u
#define BDM_OFF_A_PACKED1    0x18u
#define BDM_OFF_A_PACKED2    0x1Cu
#define BDM_OFF_A_PACKED3    0x20u
#define BDM_OFF_A_META       0x24u
#define BDM_OFF_CTRL         0x28u
#define BDM_OFF_PARTIAL_SUM  0x30u
#define BDM_OFF_EXP_SUM      0x34u
#define BDM_OFF_DONE         0x38u

/* ── Block layout constants (guarded) ──────────────────────────────────── */
#ifndef BD_BLOCK_BYTES
#define BD_BLOCK_ELEMS  32
#define BD_BLOCK_BYTES  18
#define BD_FP16_BIAS    15
#endif

/* ── bd_decode_block_hw ─────────────────────────────────────────────────── */
/* Same signature as bd_decode_block_hu (SW version).                        */
/* Uses BD_DIALECT_TABLE for per-element magnitude extraction (BDMac32 only  */
/* exposes PARTIAL_SUM, not individual decoded values).                       */
/* Implementation in bd_decode_hw.c                                           */
void bd_decode_block_hw(
    const uint8_t *block,      /* 18-byte BD block record            */
    int16_t       *out,        /* output: 32 signed half-unit int16  */
    int           *dialect_id, /* output: dialect id  (0..15)        */
    int           *shared_exp  /* output: shared_exp bits (0..31)    */
);

/* ── Self-test ─────────────────────────────────────────────────────────── */
/* Tests BDMac32: w=dialect14/exp0/all-idx7, a=dialect0/exp0/all-idx1.      */
/* Expected: PARTIAL_SUM=256, EXP_SUM=0.                                    */
/* Returns 1 on PASS, 0 on FAIL.                                             */
int bdd_selftest(void);

/* ── bdmac32_mac_block: hardware BD4×BD4 dot product ────────────────────── */
/* Computes PARTIAL_SUM = sum(w_hu[i] * a_hu[i]) for i in 0..31 via APB hw.  */
/* *exp_sum is set to w_shared_exp + a_shared_exp (EXP_SUM register [5:0]).   */
/* Scaling: real dot product = PARTIAL_SUM * 2^(*exp_sum - 32).               */
static inline int32_t bdmac32_mac_block(const uint8_t *wb, const uint8_t *ab, int *exp_sum)
{
    BDD_REG32(BDM_OFF_W_META) = ((uint32_t)wb[0] << 8) | (uint32_t)wb[1];
    const uint8_t *wc = wb + 2;
    BDD_REG32(BDM_OFF_W_PACKED0) = (uint32_t)wc[0]|(uint32_t)wc[1]<<8|(uint32_t)wc[2]<<16|(uint32_t)wc[3]<<24;
    BDD_REG32(BDM_OFF_W_PACKED1) = (uint32_t)wc[4]|(uint32_t)wc[5]<<8|(uint32_t)wc[6]<<16|(uint32_t)wc[7]<<24;
    BDD_REG32(BDM_OFF_W_PACKED2) = (uint32_t)wc[8]|(uint32_t)wc[9]<<8|(uint32_t)wc[10]<<16|(uint32_t)wc[11]<<24;
    BDD_REG32(BDM_OFF_W_PACKED3) = (uint32_t)wc[12]|(uint32_t)wc[13]<<8|(uint32_t)wc[14]<<16|(uint32_t)wc[15]<<24;
    BDD_REG32(BDM_OFF_A_META) = ((uint32_t)ab[0] << 8) | (uint32_t)ab[1];
    const uint8_t *ac = ab + 2;
    BDD_REG32(BDM_OFF_A_PACKED0) = (uint32_t)ac[0]|(uint32_t)ac[1]<<8|(uint32_t)ac[2]<<16|(uint32_t)ac[3]<<24;
    BDD_REG32(BDM_OFF_A_PACKED1) = (uint32_t)ac[4]|(uint32_t)ac[5]<<8|(uint32_t)ac[6]<<16|(uint32_t)ac[7]<<24;
    BDD_REG32(BDM_OFF_A_PACKED2) = (uint32_t)ac[8]|(uint32_t)ac[9]<<8|(uint32_t)ac[10]<<16|(uint32_t)ac[11]<<24;
    BDD_REG32(BDM_OFF_A_PACKED3) = (uint32_t)ac[12]|(uint32_t)ac[13]<<8|(uint32_t)ac[14]<<16|(uint32_t)ac[15]<<24;
    BDD_REG32(BDM_OFF_CTRL) = 1u;
    volatile int t = 256;
    while (!BDD_REG32(BDM_OFF_DONE) && --t) ;
    if (exp_sum) *exp_sum = (int)(BDD_REG32(BDM_OFF_EXP_SUM) & 0x3Fu);
    return (int32_t)BDD_REG32(BDM_OFF_PARTIAL_SUM);
}

#endif /* BD_DECODE_HW_H */
