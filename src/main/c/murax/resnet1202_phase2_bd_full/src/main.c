/* resnet1202_phase2_bd_full — Milestone 2: BD4 skip-connection compression
 *
 * Extends phase1_int8 by replacing the int8 skip_buf copy with
 * BlockDialect A4 packed format, exercising the full bd_act round-trip
 * on real ResNet residual tensors.
 *
 * What changes from phase1_int8:
 *   - Skip tensor saved as BD4 (~9 KB) instead of int8 (16 KB) for stage1.
 *   - run_basic_block_bd_skip() used instead of run_basic_block().
 *   - bd_skip_buf (uint8_t[SKIP_BUF_SIZE_BD4])  — BD4-packed skip storage.
 *   - skip_buf    (int8_t[SKIP_BUF_SIZE_INT8])   — temp unpack destination
 *                                                   before the residual add.
 *   - Ping/pong activation buffers (act_A, act_B) remain plain int8.
 *
 * SPRAM budget (worst case: stage1, 16×32×32):
 *   act_A     16,384 B  — int8 ping
 *   act_B     16,384 B  — int8 pong
 *   skip_buf  16,384 B  — int8 unpack scratch (only ~in_elems used at once)
 *   bd_skip    9,216 B  — BD4-packed skip (stage1 worst case)
 *   w_decode   1,224 B
 *   stack      2,048 B
 *   ─────────────────────────────────────────────
 *   Total:   ~61,640 B ≈ 60 KB  << 128 KB SPRAM  ✓
 *
 * Verification: after each stage, u32sum hash is printed and should be
 * compared against the Python quantized_reference.py output with BD
 * activation rounding enabled.
 *
 * §5 Milestone 2 of RESNET1202_FPGA_PLAN.md
 */

#include <stdint.h>
#include <stddef.h>
#include "murax.h"
#include "weight_blob.h"
#include "bd_decode_sw.h"
#include "bd_act.h"
#include "spram_layout.h"
#include "resnet1202_layers.h"
#include "resnet1202_conv.h"
#include "model_constants.h"

/* ── MuraxHyperRAM overrides ─────────────────────────────────────────── */
#undef  UART
#undef  GPIO_A
#define UART    ((Uart_Reg*)  (0x40010000))
#define GPIO_A  ((Gpio_Reg*)  (0x40000000))

/* ── Blob header pointer ─────────────────────────────────────────────── */
#define BLOB_HDR  ((const vwb2_header_t *)WEIGHT_BLOB_ADDR)

/* ── Activation buffers (int8 ping/pong + int8 unpack scratch) ───────── */
DEFINE_ACT_BUFFERS();

/* ── BD4-packed skip connection buffer ───────────────────────────────── */
/* Stage1 worst case: 16×32×32 = 16384 elements → 512 blocks × 18 B = 9216 B.
 * Separate from skip_buf (int8) which is used as unpack destination. */
static uint8_t bd_skip_buf[SKIP_BUF_SIZE_BD4] ACT_SECTION;

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

/* ── Per-block progress bar ──────────────────────────────────────────── */
static void print_progress(int stage, int blk, int total) {
    int done   = blk + 1;
    int filled = done * 20 / total;
    uart_write(UART, '\r');
    print("  Stage"); print_dec((uint32_t)stage); print(" [");
    for (int i = 0; i < 20; i++)
        uart_write(UART, i < filled ? '=' : '.');
    print("] ");
    if (done < 100) uart_write(UART, ' ');
    if (done <  10) uart_write(UART, ' ');
    print_dec((uint32_t)done);
    uart_write(UART, '/');
    print_dec((uint32_t)total);
}

/* ── rdcycle ─────────────────────────────────────────────────────────── */
static inline uint32_t rdcycle_csr(void) {
    uint32_t v; __asm__ volatile("csrr %0, mcycle":"=r"(v)); return v;
}

/* ── Stage output hash ───────────────────────────────────────────────── */
static void print_hash(const char *name, const int8_t *buf, int n) {
    uint32_t h = act_hash(buf, n);
    tag("hash"); print(name); print(": 0x"); print_hex(h, 8); print_nl();
}

/* ── Peak BD4 skip memory tracker ────────────────────────────────────── */
static uint32_t s_peak_bd_skip_bytes = 0;
static void update_peak_bd_skip(uint32_t n_elements) {
    uint32_t bytes = bd_act_storage_bytes(n_elements);
    if (bytes > s_peak_bd_skip_bytes) s_peak_bd_skip_bytes = bytes;
}

/* ── Per-stage fixed output shift ────────────────────────────────────── */
/* Match phase1_int8 shifts so hashes are directly comparable.           */
#define CONV_SHIFT_STEM    7
#define CONV_SHIFT_STAGE1  7
#define CONV_SHIFT_STAGE2  7
#define CONV_SHIFT_STAGE3  7

/* ── IRQ stub ────────────────────────────────────────────────────────── */
void irqCallback(void) { while(1); }

/* ── main ────────────────────────────────────────────────────────────── */
void main(void) {
    uint32_t t_start = rdcycle_csr();

    print_nl();
    print("========================================"); print_nl();
    print(" resnet1202_phase2_bd_full  Milestone 2"); print_nl();
    print("  skip tensors stored as BD4 in SPRAM  "); print_nl();
    print("========================================"); print_nl();
    print_nl();

    /* Validate blob */
    if (vwb2_verify_header(BLOB_HDR) != WB_OK) {
        tag("init"); print("FATAL: invalid VWB2 blob"); print_nl(); while(1);
    }
    tag("init");
    print("blob OK  tensor_count="); print_dec(BLOB_HDR->tensor_count);
    print("  data_bytes="); print_dec(BLOB_HDR->data_bytes); print_nl();

    /* Report BD4 skip buffer memory layout */
    tag("mem");
    print("bd_skip_buf="); print_dec(SKIP_BUF_SIZE_BD4);
    print("B  skip_buf(unpack)="); print_dec(SKIP_BUF_SIZE_INT8);
    print("B  act_A+B="); print_dec(2u * ACT_MAX_INT8_SIZE); print("B"); print_nl();

    /* ── conv1 stem: 3×32×32 → 16×32×32 ──────────────────────────────── */
    tag("conv1"); print("stem 3x3 3->16 32x32..."); print_nl();
    {
        extern const int8_t RN1202_INPUT[];  /* from input.h */
        const uint8_t *w = rn1202_weight_blocks(BLOB_HDR, RN1202_TID_CONV1);
        const float   *b = rn1202_bias_f32(BLOB_HDR, RN1202_TID_CONV1);
        /* conv1: stride=1, no BD skip compression on stem output */
        conv3x3(RN1202_INPUT, act_A, w, b,
                3, 16, 32, 32, 1, CONV_SHIFT_STEM, /*relu=*/1);
        tag("conv1"); print("done"); print_nl();
    }
    print_hash("conv1", act_A, 16*32*32);

    /* ── Stage 1: 200 BasicBlocks, 16×32×32, BD4 skip  ─────────────────
     * Each block packs its input (16×32×32 = 16384 elements = 9216 BD bytes)
     * into bd_skip_buf, runs both convs, then unpacks for the residual add. */
    tag("stage1"); print("200 BasicBlocks 16x32x32 (BD4 skip)..."); print_nl();
    {
        int8_t *in  = act_A;   /* output of conv1 */
        int8_t *out = act_B;
        for (int blk = 0; blk < RN1202_N_PER_STAGE; blk++) {
            BasicBlockConf conf = rn1202_block_conf(1, blk);
            update_peak_bd_skip((uint32_t)conf.in_c
                                * conf.in_h * conf.in_w);
            run_basic_block_bd_skip(&conf, BLOB_HDR,
                                    in, out, out,
                                    skip_buf,    /* int8 unpack scratch  */
                                    bd_skip_buf, /* BD4 packed skip store */
                                    CONV_SHIFT_STAGE1);
            int8_t *tmp = in; in = out; out = tmp;
            print_progress(1, blk, RN1202_N_PER_STAGE);
        }
        /* Normalise: result always in act_A for subsequent stages */
        if (in != act_A) {
            int n = 16*32*32;
            for (int i = 0; i < n; i++) act_A[i] = in[i];
        }
    }
    print_nl();  /* end progress bar line */
    tag("stage1"); print("done"); print_nl();
    print_hash("stage1", act_A, 16*32*32);

    /* ── Stage 2: first 16→32 stride=2, rest 32×16×16, BD4 skip ─────────
     * Projection block (blk=0): bd_skip = zero-pad-stride-2 of int8 input.
     * Identity blocks: bd_skip = BD4-packed input. */
    tag("stage2"); print("200 BasicBlocks 16->32/32x16x16 (BD4 skip)..."); print_nl();
    {
        int8_t *in  = act_A;   /* 16×32×32 stage1 output */
        int8_t *out = act_B;
        for (int blk = 0; blk < RN1202_N_PER_STAGE; blk++) {
            BasicBlockConf conf = rn1202_block_conf(2, blk);
            update_peak_bd_skip((uint32_t)conf.in_c
                                * conf.in_h * conf.in_w);
            run_basic_block_bd_skip(&conf, BLOB_HDR,
                                    in, out, out,
                                    skip_buf,
                                    bd_skip_buf,
                                    CONV_SHIFT_STAGE2);
            int8_t *tmp = in; in = out; out = tmp;
            print_progress(2, blk, RN1202_N_PER_STAGE);
        }
        if (in != act_A) {
            int n = 32*16*16;
            for (int i = 0; i < n; i++) act_A[i] = in[i];
        }
    }
    print_nl();  /* end progress bar line */
    tag("stage2"); print("done"); print_nl();
    print_hash("stage2", act_A, 32*16*16);

    /* ── Stage 3: first 32→64 stride=2, rest 64×8×8, BD4 skip ──────────*/
    tag("stage3"); print("200 BasicBlocks 32->64/64x8x8 (BD4 skip)..."); print_nl();
    {
        int8_t *in  = act_A;   /* 32×16×16 stage2 output */
        int8_t *out = act_B;
        for (int blk = 0; blk < RN1202_N_PER_STAGE; blk++) {
            BasicBlockConf conf = rn1202_block_conf(3, blk);
            update_peak_bd_skip((uint32_t)conf.in_c
                                * conf.in_h * conf.in_w);
            run_basic_block_bd_skip(&conf, BLOB_HDR,
                                    in, out, out,
                                    skip_buf,
                                    bd_skip_buf,
                                    CONV_SHIFT_STAGE3);
            int8_t *tmp = in; in = out; out = tmp;
            print_progress(3, blk, RN1202_N_PER_STAGE);
        }
        if (in != act_A) {
            int n = 64*8*8;
            for (int i = 0; i < n; i++) act_A[i] = in[i];
        }
    }
    print_nl();  /* end progress bar line */
    tag("stage3"); print("done"); print_nl();
    print_hash("stage3", act_A, 64*8*8);

    /* ── Global average pool: 64×8×8 → 64 ──────────────────────────────*/
    global_avgpool(act_A, act_B, 64, 8, 8);
    tag("avgpool"); print("64x8x8 -> 64"); print_nl();

    /* ── FC: 64 → 10 logits ─────────────────────────────────────────── */
    int32_t logits[10];
    {
        const uint8_t *w = rn1202_weight_blocks(BLOB_HDR, RN1202_TID_FC);
        const float   *b = rn1202_bias_f32(BLOB_HDR, RN1202_TID_FC);
        fc_linear(act_B, logits, w, b, 64, 10);
    }

    /* Find top-1 */
    int top1 = 0;
    for (int i = 1; i < 10; i++) {
        if (logits[i] > logits[top1]) top1 = i;
    }

    /* ── Print results ───────────────────────────────────────────────── */
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

    uint32_t t_end = rdcycle_csr();

    /* ── Summary ─────────────────────────────────────────────────────── */
    print_nl();
    print("========================================"); print_nl();
    print("[7] top-1 class: "); print_dec((uint32_t)top1); print_nl();
    print("[8] rdcycles: "); print_dec(t_end - t_start); print_nl();
    print("[BD] peak bd4_skip: "); print_dec(s_peak_bd_skip_bytes);
    print("B  (stage1 max="); print_dec(SKIP_BUF_SIZE_BD4);
    print("B vs int8="); print_dec(SKIP_BUF_SIZE_INT8); print("B)"); print_nl();
    print("[phase2] DONE"); print_nl();
    print("========================================"); print_nl();

    while(1);
}
