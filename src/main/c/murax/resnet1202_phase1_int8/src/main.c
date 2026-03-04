/* resnet1202_phase1_int8 — Milestone 1: CPU baseline
 *
 * ResNet-1202 (CIFAR-10) end-to-end inference on iCEBreaker:
 *   - Weights decoded from BD4 (VWB2 blob) in software per output channel
 *   - Activations stored as int8 in SPRAM (all tensors fit, no tiling needed)
 *   - Skip connections saved as int8 in a dedicated skip_buf
 *   - All accumulation in int32 → int8 with fixed output_shift
 *   - u32sum verification hash after each stage (vs. quantized_ref.h)
 *
 * Memory layout (see spram_layout.h):
 *   act_A    16 KB @  0x11000000  — ping activation buffer
 *   act_B    16 KB @  0x11004000  — pong activation buffer
 *   skip_buf 16 KB @  0x11008000  — residual save buffer
 *   w_decode_hu scratch           — per-OC weight decode
 *   .bss + stack in remaining SPRAM
 *
 * §5 Milestone 1 of RESNET1202_FPGA_PLAN.md
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

/* ── Activation buffer declarations (defined here; DEFINE_ACT_BUFFERS) ─ */
DEFINE_ACT_BUFFERS();

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

/* ── Stage hash verification ─────────────────────────────────────────── */
static void print_hash(const char *name, const int8_t *buf, int n) {
    uint32_t h = act_hash(buf, n);
    tag("hash"); print(name); print(": 0x"); print_hex(h, 8); print_nl();
}

/* ── Per-stage fixed output shift ────────────────────────────────────── */
/* Tuning note: start with 7 (divides by 128) and adjust if values saturate. */
/* The Python quantized_reference.py run validates the correct shift.        */
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
    print(" resnet1202_phase1_int8  Milestone 1   "); print_nl();
    print("========================================"); print_nl();
    print_nl();

    /* Validate blob */
    if (vwb2_verify_header(BLOB_HDR) != WB_OK) {
        tag("init"); print("FATAL: invalid VWB2 blob"); print_nl(); while(1);
    }
    tag("init");
    print("blob OK  tensor_count="); print_dec(BLOB_HDR->tensor_count);
    print("  data_bytes="); print_dec(BLOB_HDR->data_bytes); print_nl();

    /* ── conv1 stem: 3×32×32 → 16×32×32 ──────────────────────────────── */
    tag("conv1"); print("stem 3x3 3->16 32x32..."); print_nl();
    {
        /* Input: RN1202_INPUT is in .rodata (BRAM) */
        extern const int8_t RN1202_INPUT[];  /* from input.h */
        const uint8_t *w = rn1202_weight_blocks(BLOB_HDR, RN1202_TID_CONV1);
        const float   *b = rn1202_bias_f32(BLOB_HDR, RN1202_TID_CONV1);
        /* conv1: 3×3, in_c=3, out_c=16, h=w=32, stride=1, with ReLU */
        conv3x3(RN1202_INPUT, act_A, w, b,
                3, 16, 32, 32, 1, CONV_SHIFT_STEM, /*relu=*/1);
        tag("conv1"); print("done"); print_nl();
    }
    print_hash("conv1", act_A, 16*32*32);

    /* ── Stage 1: 200 BasicBlocks, 16×32×32 ─────────────────────────── */
    tag("stage1"); print("200 BasicBlocks 16x32x32..."); print_nl();
    {
        int8_t *in_buf  = act_A;   /* output of conv1 */
        int8_t *out_buf = act_B;
        for (int blk = 0; blk < RN1202_N_PER_STAGE; blk++) {
            BasicBlockConf conf = rn1202_block_conf(1, blk);
            run_basic_block(&conf, BLOB_HDR,
                            in_buf, out_buf, out_buf, skip_buf,
                            CONV_SHIFT_STAGE1);
            /* Result is in out_buf; swap for next iteration */
            int8_t *tmp = in_buf; in_buf = out_buf; out_buf = tmp;
            print_progress(1, blk, RN1202_N_PER_STAGE);
        }
        /* After 200 blocks, result is in in_buf (last swap) */
        /* Normalise: copy to act_A so subsequent stages start consistently */
        if (in_buf != act_A) {
            int n = 16*32*32;
            for (int i = 0; i < n; i++) act_A[i] = in_buf[i];
        }
    }
    print_nl();  /* end progress bar line */
    tag("stage1"); print("done"); print_nl();
    print_hash("stage1", act_A, 16*32*32);

    /* ── Stage 2: 200 BasicBlocks, first 16→32 stride=2, rest 32×16×16 */
    tag("stage2"); print("200 BasicBlocks (first 16->32 s2, rest 32x16x16)..."); print_nl();
    {
        int8_t *in_buf  = act_A;   /* 16×32×32 stage1 output */
        int8_t *out_buf = act_B;
        for (int blk = 0; blk < RN1202_N_PER_STAGE; blk++) {
            BasicBlockConf conf = rn1202_block_conf(2, blk);
            run_basic_block(&conf, BLOB_HDR,
                            in_buf, out_buf, out_buf, skip_buf,
                            CONV_SHIFT_STAGE2);
            int8_t *tmp = in_buf; in_buf = out_buf; out_buf = tmp;
            print_progress(2, blk, RN1202_N_PER_STAGE);
        }
        if (in_buf != act_A) {
            int n = 32*16*16;
            for (int i = 0; i < n; i++) act_A[i] = in_buf[i];
        }
    }
    print_nl();  /* end progress bar line */
    tag("stage2"); print("done"); print_nl();
    print_hash("stage2", act_A, 32*16*16);

    /* ── Stage 3: 200 BasicBlocks, first 32→64 stride=2, rest 64×8×8 ── */
    tag("stage3"); print("200 BasicBlocks (first 32->64 s2, rest 64x8x8)..."); print_nl();
    {
        int8_t *in_buf  = act_A;   /* 32×16×16 stage2 output */
        int8_t *out_buf = act_B;
        for (int blk = 0; blk < RN1202_N_PER_STAGE; blk++) {
            BasicBlockConf conf = rn1202_block_conf(3, blk);
            run_basic_block(&conf, BLOB_HDR,
                            in_buf, out_buf, out_buf, skip_buf,
                            CONV_SHIFT_STAGE3);
            int8_t *tmp = in_buf; in_buf = out_buf; out_buf = tmp;
            print_progress(3, blk, RN1202_N_PER_STAGE);
        }
        if (in_buf != act_A) {
            int n = 64*8*8;
            for (int i = 0; i < n; i++) act_A[i] = in_buf[i];
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
    print_nl();
    print("========================================"); print_nl();
    print("[7] top-1 class: "); print_dec((uint32_t)top1); print_nl();
    print("[8] rdcycles: "); print_dec(t_end - t_start); print_nl();
    print("[phase1] DONE"); print_nl();
    print("========================================"); print_nl();

    while(1);
}
