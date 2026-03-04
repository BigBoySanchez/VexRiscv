/* bd_decode_hw.c — BDMac32 APB peripheral driver
 *
 * bd_decode_block_hw : delegates to bd_decode_block_hu (SW table) — BDMac32
 *   exposes only PARTIAL_SUM, not per-element decoded magnitudes.
 *
 * bdd_selftest : verifies BDMac32 at 0x40031000 by computing a known MAC:
 *   w = dialect14/exp0/all-idx7 (mag=8),  a = dialect0/exp0/all-idx1 (mag=1)
 *   Expected: PARTIAL_SUM=256, EXP_SUM=0.
 *
 * ss5 Milestone 3 of RESNET1202_FPGA_PLAN.md
 */
#include <stdint.h>
#include "bd_decode_sw.h"
#include "bd_decode_hw.h"

/* ── BD_DIALECT_TABLE: single definition (extern in bd_decode_sw.h) ─── */
const uint8_t BD_DIALECT_TABLE[16][8] = {
    { 0, 1, 2, 3, 4, 6, 11, 15 },  /* dialect  0 */
    { 0, 1, 2, 3, 4, 6,  9, 15 },  /* dialect  1 */
    { 0, 1, 2, 3, 4, 6, 11, 14 },  /* dialect  2 */
    { 0, 1, 2, 3, 4, 6,  9, 14 },  /* dialect  3 */
    { 0, 1, 2, 3, 4, 6, 10, 13 },  /* dialect  4 */
    { 0, 1, 2, 3, 4, 6,  8, 13 },  /* dialect  5 */
    { 0, 1, 2, 3, 4, 6, 10, 12 },  /* dialect  6 */
    { 0, 1, 2, 3, 4, 6,  8, 12 },  /* dialect  7 */
    { 0, 1, 2, 3, 4, 6,  9, 11 },  /* dialect  8 */
    { 0, 1, 2, 3, 4, 6,  7, 11 },  /* dialect  9 */
    { 0, 1, 2, 3, 4, 6,  9, 10 },  /* dialect 10 */
    { 0, 1, 2, 3, 4, 6,  7, 10 },  /* dialect 11 */
    { 0, 1, 2, 3, 4, 6,  8,  9 },  /* dialect 12 */
    { 0, 1, 2, 3, 4, 6,  7,  9 },  /* dialect 13 */
    { 0, 1, 2, 3, 4, 6,  7,  8 },  /* dialect 14 */
    { 0, 1, 2, 3, 4, 5,  6,  8 },  /* dialect 15 */
};

/* ── bd_decode_block_hw ─────────────────────────────────────────────────── */
void bd_decode_block_hw(const uint8_t *block, int16_t *out,
                        int *dialect_id, int *shared_exp)
{
    /* BDMac32 does not expose per-element decoded values; use SW table. */
    bd_decode_block_hu(block, out, dialect_id, shared_exp);
}

/* ── bdd_selftest ─────────────────────────────────────────────────────── */
/* w = dialect14/exp0/all-idx7 (mag=8),  a = dialect0/exp0/all-idx1 (mag=1)
 * product per element = 8*1 = 8, all positive  →  sum = 32*8 = 256, exp=0  */
int bdd_selftest(void)
{
    uint8_t wb[BD_BLOCK_BYTES], ab[BD_BLOCK_BYTES];
    int i;
    wb[0]=0xE0u; wb[1]=0x00u;
    for(i=2;i<BD_BLOCK_BYTES;i++) wb[i]=0x77u;
    ab[0]=0x00u; ab[1]=0x00u;
    for(i=2;i<BD_BLOCK_BYTES;i++) ab[i]=0x11u;
    int es=-1;
    int32_t ps = bdmac32_mac_block(wb, ab, &es);
    return (ps==256 && es==0) ? 1 : 0;
}
