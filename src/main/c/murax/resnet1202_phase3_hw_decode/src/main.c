/* resnet1202_phase3_hw_decode — Milestone 3 + Step 4 (bd-activations-fix)
 *
 * Step 4: paper-faithful BD4 activation path.  There is NO int8 intermediate
 * anywhere in the inference pipeline:
 *
 *   stem int8 input → conv3x3_bd4 → BD4 acts
 *   each BasicBlock  → run_basic_block_bd4  (conv_a + conv_b + skip, all BD4)
 *   avgpool          → global_avgpool_bd4   → int8[64]  (only 64 bytes, for FC)
 *   FC 64→10         → fc_linear            → int32 logits
 *
 * Skip connections:
 *   identity  — add_relu_bd4(bd_out, bd_in, bd_out)      no copy needed
 *   stride=2  — bd4_zero_pad_stride2 → BD4 skip          no int8 unpack
 *   has_proj  — conv1x1_bd4_hwmac   → BD4 skip
 *
 * SPRAM layout (DEFINE_ACT_BUFFERS_BD4):
 *   accum_scratch int32[16384]  64 KB   0x11000000
 *   bd_act_A      uint8[9216]    9 KB   0x11010000
 *   bd_act_B      uint8[9216]    9 KB   0x11012400
 *   bd_skip_bd4   uint8[9216]    9 KB   0x11014800
 *   w_decode_hu   int16[576]     1 KB   ~0x11016C00
 *   Total: ~93 KB < 128 KB
 */

#include <stdint.h>
#include <stddef.h>
#include "murax.h"
#include "weight_blob.h"
#include "bd_act.h"
#include "bd_decode_hw.h"
#include "spram_layout.h"
#include "resnet1202_layers.h"
#define bd_decode_block_hu bd_decode_block_hw
#include "resnet1202_conv.h"
#undef  bd_decode_block_hu
#include "model_constants.h"

/* ── MuraxHyperRAM overrides ─────────────────────────────────────────── */
#undef  UART
#undef  GPIO_A
#define UART    ((Uart_Reg*)  (0x40010000))
#define GPIO_A  ((Gpio_Reg*)  (0x40000000))

/* ── Blob header pointer ─────────────────────────────────────────────── */
#define BLOB_HDR  ((const vwb2_header_t *)WEIGHT_BLOB_ADDR)

/* ─────────────────────────────────────────────────────────────────────────
 * USE_TAP_BLOCKED  — set to 1 to activate the per-tap channel-blocked HWCB
 * inference path (requires weights regenerated with --tap-blocked flag).
 * Set to 0 to stay on the flat BD4 bulk-unpack path (Step 4 baseline).
 * ───────────────────────────────────────────────────────────────────────── */
#define USE_TAP_BLOCKED 1

/* ── BD4 activation buffers ─────────────────────────────────────────── */
#if USE_TAP_BLOCKED
DEFINE_ACT_BUFFERS_BD4_TAP();   /* 3 × 18 KB HWCB — no large accum_scratch */
#else
DEFINE_ACT_BUFFERS_BD4();       /* 64 KB accum_scratch + 3 × 9 KB flat BD4 */
#endif

/* ── Minimal libc stubs ──────────────────────────────────────────────── */
void *memcpy(void *dst, const void *src, unsigned int n) {
    uint8_t *d=(uint8_t*)dst; const uint8_t*s=(const uint8_t*)src;
    while(n--)*d++=*s++; return dst;
}
void *memset(void *dst, int c, unsigned int n) {
    uint8_t *d=(uint8_t*)dst; while(n--)*d++=(uint8_t)c; return dst;
}

/* ── UART helpers ────────────────────────────────────────────────────── */
static void print(const char *s)  { while(*s) uart_write(UART,*s++); }
static void print_nl(void)        { uart_write(UART,'\r'); uart_write(UART,'\n'); }
static void print_hex(uint32_t v, int digits) {
    for(int i=(digits-1)*4;i>=0;i-=4){int d=(v>>i)&0xF;uart_write(UART,d<10?'0'+d:'A'+d-10);}
}
static void print_dec(uint32_t v) {
    if(!v){uart_write(UART,'0');return;}
    char b[12];int i=0;while(v){b[i++]='0'+(v%10);v/=10;}while(i>0)uart_write(UART,b[--i]);
}
static void tag(const char *t) { print("["); print(t); print("] "); }

/* ── BD4 buffer checksum (raw byte sum — reproducibility fingerprint) ── */
static void print_bd4_cksum(const char *name, const uint8_t *buf, int n_bytes) {
    uint32_t s = 0;
    for (int i = 0; i < n_bytes; i++) s += buf[i];
    tag("cksum"); print(name); print(": 0x"); print_hex(s, 8); print_nl();
}

/* ── rdcycle ─────────────────────────────────────────────────────────── */
static inline uint32_t rdcycle_csr(void) {
    uint32_t v; __asm__ volatile("csrr %0, mcycle":"=r"(v)); return v;
}

/* ── Weight blob spot-check ──────────────────────────────────────────── */
static void weight_spot_check(const vwb2_header_t *hdr) {
    const uint8_t *p = (const uint8_t *)hdr;
    uint32_t s = 0;
    for (int i = 0; i < 32; i++) s += p[i];
    const uint8_t *data = (const uint8_t *)hdr + hdr->data_offset;
    for (int i = 0; i < 18; i++) s += data[i];
    tag("weights");
    print("spot-check u32sum=0x"); print_hex(s, 8);
    print("  tc="); print_dec(hdr->tensor_count);
    print("  db="); print_dec(hdr->data_bytes);
    print_nl();
}

/* Per-stage output shifts — calibrated by quantized_reference.py
 * calibrate_rn1202_shifts(target_bits=7).  shift = max(0, floor(log2(max_abs)) - 6).
 *
 * The raw accumulator range for 1200/1201 block convolutions is < 64, so shift=0
 * (no right-shift).  Only the stem conv1 (3→16, 3×3, int8 input with 0–127 range)
 * produces accumulators up to ~277, requiring shift=2.
 *
 * IMPORTANT: shift=7 was the old bring-up default and ZEROED every layer's
 * output (dividing [-64,64] by 128).  Do not revert without re-calibrating.
 */
#define CONV_SHIFT_STEM    2
#define CONV_SHIFT_STAGE1  0
#define CONV_SHIFT_STAGE2  0
#define CONV_SHIFT_STAGE3  0

/* ── IRQ stub ────────────────────────────────────────────────────────── */
void irqCallback(void) { while(1); }

/* ── main ────────────────────────────────────────────────────────────── */
void main(void) {
    uint32_t t_start = rdcycle_csr();

    print_nl();
    print("========================================"); print_nl();
#if USE_TAP_BLOCKED
    print(" resnet1202: BD4 HWCB tap-blocked path "); print_nl();
    print("  BDMac32 + per-tap channel blocking   "); print_nl();
#else
    print(" resnet1202 Step4: BD4 activations only"); print_nl();
    print("  BDMac32 @ 0x40031000, no int8 path  "); print_nl();
#endif
    print("========================================"); print_nl();
    print_nl();

    /* [1] BDMac32 self-test */
    {
        int pass = bdd_selftest();
        tag("hw");
        if (pass) print("BDMac32 self-test: PASS");
        else      print("BDMac32 self-test: FAIL");
        print_nl();
        if (!pass) { while(1); }
    }

    /* [0] Validate VWB2 blob */
    if (vwb2_verify_header(BLOB_HDR) != WB_OK) {
        tag("init"); print("FATAL: invalid VWB2 blob"); print_nl(); while(1);
    }
    /* [0b] Layout guard: last tensor = "rn1202.layout_flags" sentinel.
     * 0x3F800000 (1.0f) = tap-blocked,  0x00000000 (0.0f) = flat.
     * Hard-fail on mismatch so wrong blob never silently corrupts logits. */
    {
        const vwb2_entry_t *e =
            &vwb2_table(BLOB_HDR)[BLOB_HDR->tensor_count - 1u];
        const uint8_t *fp = vwb2_tensor_data(BLOB_HDR, e);
        uint32_t fb = (uint32_t)fp[0] | ((uint32_t)fp[1] << 8)
                    | ((uint32_t)fp[2] << 16) | ((uint32_t)fp[3] << 24);
        uint32_t expect = USE_TAP_BLOCKED ? 0x3F800000u : 0x00000000u;
        if (e->name_hash != RN1202_LAYOUT_FLAGS_HASH
            || e->dtype   != VWB2_DTYPE_FLOAT32
            || fb         != expect) {
            tag("init"); print("FATAL:layout 0x"); print_hex(fb, 8);
            print_nl(); while(1);
        }
    }
    tag("init");

    /* [0c] Weight spot-check: verify header + start of data */
    weight_spot_check(BLOB_HDR);
    print("blob OK  tc="); print_dec(BLOB_HDR->tensor_count);
    print(" db="); print_dec(BLOB_HDR->data_bytes); print_nl();

    /* ── conv1 stem: int8 3×32×32 → BD4 16×32×32 ─────────────────────── */
    tag("conv1"); print("stem (int8->BD4)..."); print_nl();
    uint32_t t_conv1_start = rdcycle_csr();
    uint32_t w_bytes_conv1 = vwb2_table(BLOB_HDR)[rn1202_tid_conv1() * 2].tensor_bytes;
    {
        extern const int8_t RN1202_INPUT[];
        const uint8_t *w = rn1202_weight_blocks(BLOB_HDR, RN1202_TID_CONV1);
        const float   *b = rn1202_bias_f32(BLOB_HDR, RN1202_TID_CONV1);
#if USE_TAP_BLOCKED
        conv3x3_bd4_tap(RN1202_INPUT, bd_act_A, w, b,
                        3, 16, 32, 32, 1, CONV_SHIFT_STEM, /*relu=*/1);
#else
        conv3x3_bd4(RN1202_INPUT, bd_act_A, accum_scratch, w, b,
                    3, 16, 32, 32, 1, CONV_SHIFT_STEM, /*relu=*/1);
#endif
    }
    uint32_t t_conv1_end = rdcycle_csr();
    tag("conv1_time"); print("cyc="); print_dec(t_conv1_end - t_conv1_start); 
    print(" w_bytes="); print_dec(w_bytes_conv1); print_nl();

#if USE_TAP_BLOCKED
    print_bd4_cksum("conv1", bd_act_A, ACT_STAGE1_SIZE_BD4_HWCB);
#else
    print_bd4_cksum("conv1", bd_act_A, ACT_STAGE1_SIZE_BD4);
#endif

    /* ── Stage 1-3: BD4 ping-pong via run_basic_block_bd4 ─────────────── */
    uint8_t *bd_cur  = bd_act_A;
    uint8_t *bd_next = bd_act_B;

    /* Stage 1: 200 blocks, 16×32×32 */
    tag("stage1"); print("200 blocks"); print_nl();
    {
        uint32_t ts = rdcycle_csr();
        for (int blk = 0; blk < RN1202_N_PER_STAGE; blk++) {
            tag("stage1"); print_dec((uint32_t)blk);
            print("/200"); print_nl();

            BasicBlockConf conf = rn1202_block_conf(1, blk);
            uint32_t t_blk_start = rdcycle_csr();
            uint32_t w_bytes_blk = vwb2_table(BLOB_HDR)[conf.tid_conv_a * 2].tensor_bytes + 
                                   vwb2_table(BLOB_HDR)[conf.tid_conv_b * 2].tensor_bytes;
            if (conf.has_proj) {
                w_bytes_blk += vwb2_table(BLOB_HDR)[conf.tid_proj * 2].tensor_bytes;
            }
#if USE_TAP_BLOCKED
            run_basic_block_bd4_tap(&conf, BLOB_HDR,
                                    bd_cur, bd_skip_bd4, bd_next, bd_skip_bd4);
#else
            run_basic_block_bd4(&conf, BLOB_HDR,
                                bd_cur, bd_skip_bd4, bd_next, bd_skip_bd4,
                                accum_scratch, CONV_SHIFT_STAGE1,
                                act_unpack_i8);
#endif
            uint32_t t_blk_end = rdcycle_csr();
            tag("stage1_blk_time"); print("cyc="); print_dec(t_blk_end - t_blk_start);
            print(" w_bytes="); print_dec(w_bytes_blk); print_nl();

            uint8_t *t = bd_cur; bd_cur = bd_next; bd_next = t;

        }
        uint32_t te = rdcycle_csr();
        tag("stage1"); print("done (");
        print_dec(te - ts); print(" cyc)"); print_nl();
    }
#if USE_TAP_BLOCKED
    print_bd4_cksum("stage1", bd_cur, ACT_STAGE1_SIZE_BD4_HWCB);
#else
    print_bd4_cksum("stage1", bd_cur, ACT_STAGE1_SIZE_BD4);
#endif

    /* Stage 2: 200 blocks, first is 16→32 stride=2, then 32×16×16 */
    tag("stage2"); print("200 blocks"); print_nl();
    {
        uint32_t ts = rdcycle_csr();
        for (int blk = 0; blk < RN1202_N_PER_STAGE; blk++) {
            tag("stage2"); print_dec((uint32_t)blk);
            print("/200"); print_nl();

            BasicBlockConf conf = rn1202_block_conf(2, blk);
#if USE_TAP_BLOCKED
            run_basic_block_bd4_tap(&conf, BLOB_HDR,
                                    bd_cur, bd_skip_bd4, bd_next, bd_skip_bd4);
#else
            run_basic_block_bd4(&conf, BLOB_HDR,
                                bd_cur, bd_skip_bd4, bd_next, bd_skip_bd4,
                                accum_scratch, CONV_SHIFT_STAGE2,
                                act_unpack_i8);
#endif
            uint8_t *t = bd_cur; bd_cur = bd_next; bd_next = t;
        }
        uint32_t te = rdcycle_csr();
        tag("stage2"); print("done (");
        print_dec(te - ts); print(" cyc)"); print_nl();
    }
    print_bd4_cksum("stage2", bd_cur, ACT_STAGE2_SIZE_BD4);

    /* Stage 3: 200 blocks, first is 32→64 stride=2, then 64×8×8 */
    tag("stage3"); print("200 blocks"); print_nl();
    {
        uint32_t ts = rdcycle_csr();
        for (int blk = 0; blk < RN1202_N_PER_STAGE; blk++) {
            tag("stage3"); print_dec((uint32_t)blk);
            print("/200"); print_nl();

            BasicBlockConf conf = rn1202_block_conf(3, blk);
#if USE_TAP_BLOCKED
            run_basic_block_bd4_tap(&conf, BLOB_HDR,
                                    bd_cur, bd_skip_bd4, bd_next, bd_skip_bd4);
#else
            run_basic_block_bd4(&conf, BLOB_HDR,
                                bd_cur, bd_skip_bd4, bd_next, bd_skip_bd4,
                                accum_scratch, CONV_SHIFT_STAGE3,
                                act_unpack_i8);
#endif
            uint8_t *t = bd_cur; bd_cur = bd_next; bd_next = t;
        }
        uint32_t te = rdcycle_csr();
        tag("stage3"); print("done (");
        print_dec(te - ts); print(" cyc)"); print_nl();
    }
    print_bd4_cksum("stage3", bd_cur, ACT_STAGE3_SIZE_BD4);

    /* ── Global average pool: BD4 64×8×8 → int8[64] ──────────────────── */
    int8_t *avgpool_out = (int8_t *)bd_next;  /* reuse bd_next as temp storage */
#if USE_TAP_BLOCKED
    /* HWCB path: iterate spatially, unpack channel-blocks per pixel */
    global_avgpool_hwcb(bd_cur, avgpool_out, 64, 8, 8);
#else
    /* Flat BD4 path: bulk-unpack once, then int8 avgpool */
    {
        uint32_t nb = ((uint32_t)(64 * 8 * 8) + 31u) / 32u;
        bd_act_unpack_tensor(bd_cur, nb, act_unpack_i8, (uint32_t)(64 * 8 * 8));
        global_avgpool(act_unpack_i8, avgpool_out, 64, 8, 8);
    }
#endif
    tag("avgpool"); print("BD4 64x8x8 -> int8[64]"); print_nl();

    /* ── FC: int8[64] → int32 logits[10] ─────────────────────────────── */
    int32_t logits[10];
    {
        const uint8_t *w = rn1202_weight_blocks(BLOB_HDR, RN1202_TID_FC);
        const float   *b = rn1202_bias_f32(BLOB_HDR, RN1202_TID_FC);
        fc_linear(avgpool_out, logits, w, b, 64, 10);
    }

    /* Find top-1 */
    int top1 = 0;
    for (int i = 1; i < 10; i++) {
        if (logits[i] > logits[top1]) top1 = i;
    }

    /* Print logit array */
    print_nl();
    tag("logits");
    for (int i = 0; i < 10; i++) {
        print("["); print_dec((uint32_t)i); print("]=");
        if (logits[i] < 0) { uart_write(UART,'-'); print_dec((uint32_t)(-logits[i])); }
        else                { print_dec((uint32_t)logits[i]); }
        print("  ");
    }
    print_nl();

    uint32_t logits_hash = 0;
    for (int i = 0; i < 10; i++) logits_hash += (uint32_t)logits[i];
    tag("logits"); print("u32sum=0x"); print_hex(logits_hash, 8); print_nl();

    uint32_t t_end  = rdcycle_csr();
    uint32_t total  = t_end - t_start;

    /* ── Summary ─────────────────────────────────────────────────────── */
    print_nl();
    print("========================================"); print_nl();
    print("[1] hw self-test: PASS"); print_nl();
#if USE_TAP_BLOCKED
    print("[4] BD4 tap-blocked (HWCB per-tap channel blocking)"); print_nl();
    print("[5] BDMac32 hardware dot products: ENABLED"); print_nl();
#else
    print("[4] BD4 activations (bulk-unpack fast path)"); print_nl();
#endif
    print("[7] top-1 class: "); print_dec((uint32_t)top1); print_nl();
    print("[8] total rdcycles: "); print_dec(total); print_nl();
    print("[phase3/step4] DONE"); print_nl();
    print("========================================"); print_nl();

    while(1);
}

