/* resnet1202_phase0_smoke — Milestone 0: board smoke test
 *
 * Verifies (§5 Milestone 0, RESNET1202_FPGA_PLAN.md):
 *   1. UART + rdcycle CSR work
 *   2. VWB2 blob magic, version, block_size=32
 *   3. Walk tensor table: print count, first 3 name hashes, total data bytes
 *   4. APB BlockDialect hardware decoder self-test
 *
 * Deliverable UART line on success:
 *   [0] VWB2 blob OK: <data_bytes> MB, <tensor_count> tensors
 *   [1] BD decoder self-test: dialect0 exp15 all-max -> 32x+15  PASS
 *   [phase0] PASS
 */

#include <stdint.h>
#include <stddef.h>
#include "murax.h"
#include "weight_blob.h"

/* ── MuraxHyperRAM peripheral overrides ──────────────────────────────── */
#undef  UART
#undef  GPIO_A
#define UART    ((Uart_Reg*)  (0x40010000))
#define GPIO_A  ((Gpio_Reg*)  (0x40000000))

/* ── APB BlockDialect Decoder (§5 Milestone 3 hardware path) ─────────── */
#define BD_DEC_BASE     0x40030000u
#define BD_META         (*(volatile uint32_t*)(BD_DEC_BASE + 0x00))
#define BD_PACKED(i)    (*(volatile uint32_t*)(BD_DEC_BASE + 0x04 + (i)*4))
#define BD_DECODED(i)   (*(volatile uint32_t*)(BD_DEC_BASE + 0x20 + (i)*4))
#define BD_STATUS       (*(volatile uint32_t*)(BD_DEC_BASE + 0x40))
#define BD_SHARED_EXP   (*(volatile uint32_t*)(BD_DEC_BASE + 0x44))

/* ── Weight blob base (matches model_constants.h WEIGHT_BLOB_ADDR) ───── */
#define WEIGHT_BLOB_ADDR  0x20000000u

/* ── Minimal libc stubs ──────────────────────────────────────────────── */
void *memcpy(void *dst, const void *src, unsigned int n) {
    uint8_t *d = (uint8_t *)dst; const uint8_t *s = (const uint8_t *)src;
    while (n--) *d++ = *s++; return dst;
}
void *memset(void *dst, int c, unsigned int n) {
    uint8_t *d = (uint8_t *)dst; while (n--) *d++ = (uint8_t)c; return dst;
}

/* ── UART helpers ────────────────────────────────────────────────────── */
static void print(const char *s)      { while (*s) uart_write(UART, *s++); }
static void print_nl(void)            { uart_write(UART, '\r'); uart_write(UART, '\n'); }
static void print_hex(uint32_t v, int digits) {
    for (int i = (digits-1)*4; i >= 0; i -= 4) {
        int d = (v >> i) & 0xF;
        uart_write(UART, d < 10 ? '0'+d : 'A'+d-10);
    }
}
static void print_dec(uint32_t v) {
    if (v == 0) { uart_write(UART, '0'); return; }
    char buf[12]; int i = 0;
    while (v) { buf[i++] = '0' + (v % 10); v /= 10; }
    while (i > 0) uart_write(UART, buf[--i]);
}
static void tag(const char *t) { print("["); print(t); print("] "); }

/* ── rdcycle CSR ─────────────────────────────────────────────────────── */
static inline uint32_t rdcycle_csr(void) {
    uint32_t v; __asm__ volatile ("csrr %0, mcycle" : "=r"(v)); return v;
}

/* ── Test 1: rdcycle ─────────────────────────────────────────────────── */
static int test_rdcycle(void) {
    uint32_t t0 = rdcycle_csr();
    volatile uint32_t x = 0; for (int i = 0; i < 64; i++) x += i;
    uint32_t t1 = rdcycle_csr(); (void)x;
    tag("rdcycle");
    print("t0=0x"); print_hex(t0,8); print("  t1=0x"); print_hex(t1,8);
    print("  delta="); print_dec(t1-t0); print_nl();
    if (t1 == t0) { tag("rdcycle"); print("ERROR: counter frozen"); print_nl(); return 0; }
    return 1;
}

/* ── Test 2/3: VWB2 blob header ─────────────────────────────────────── */
static int test_blob(void) {
    const vwb2_header_t *h = (const vwb2_header_t *)WEIGHT_BLOB_ADDR;
    uint32_t magic = h->magic;

    tag("blob");
    print("addr=0x"); print_hex(WEIGHT_BLOB_ADDR, 8);
    print("  magic=0x"); print_hex(magic, 8); print_nl();

    if (magic != VWB2_MAGIC) {
        tag("blob"); print("ERROR: expected 0x"); print_hex(VWB2_MAGIC, 8);
        print(" (VWB2)"); print_nl(); return 0;
    }

    wb_err_t err = vwb2_verify_header(h);
    tag("blob");
    print("version="); print_dec(h->version);
    print("  block_size="); print_dec(h->block_size);
    print("  tensor_count="); print_dec(h->tensor_count);
    print("  data_bytes="); print_dec(h->data_bytes);
    print_nl();

    if (err == WB_ERR_VERSION) { tag("blob"); print("ERROR: unsupported version"); print_nl(); return 0; }
    if (err == WB_ERR_BLKSIZE) { tag("blob"); print("ERROR: block_size != 32"); print_nl(); return 0; }

    /* [1] Print first 4 tensor entries (name hash + type + n_elements) */
    const vwb2_entry_t *tbl = vwb2_table(h);
    uint32_t n_show = h->tensor_count < 4 ? h->tensor_count : 4;
    for (uint32_t i = 0; i < n_show; i++) {
        const vwb2_entry_t *e = &tbl[i];
        const char *dt = e->dtype == VWB2_DTYPE_BD4 ? "BD4" :
                         e->dtype == VWB2_DTYPE_FLOAT32 ? "F32" : "???";
        tag("blob");
        print("["); print_dec(i); print("] hash=0x"); print_hex(e->name_hash, 8);
        print("  "); print(dt);
        print("  n_elem="); print_dec(e->n_elements);
        print("  bytes="); print_dec(e->tensor_bytes);
        print_nl();
    }

    /* Peek first BD4 block meta */
    for (uint32_t i = 0; i < h->tensor_count; i++) {
        if (tbl[i].dtype == VWB2_DTYPE_BD4) {
            const vwb2_bd4_hdr_t *bh = vwb2_bd4_header(h, &tbl[i]);
            const uint8_t *blk = vwb2_bd4_blocks(h, &tbl[i]);
            tag("blob");
            print("BD4[0]: n_elem="); print_dec(bh->n_elements);
            print("  n_blocks="); print_dec(bh->n_blocks);
            print("  dialect="); print_dec(VWB2_META_DIALECT(blk[0],blk[1]));
            print("  shared_exp="); print_dec(VWB2_META_EXP(blk[0],blk[1]));
            print_nl();
            break;
        }
    }

    tag("blob");
    print("VWB2 OK: "); print_dec(h->tensor_count); print(" tensors  ");
    print_dec(h->data_bytes / 1024 / 1024); print(" MB data");
    print_nl();
    return 1;
}

/* ── Test 4: APB decoder self-test ──────────────────────────────────── */
/* Block: dialect=0, shared_exp=15, all 32 codes sign=0 idx=7.
 * dialect 0 idx 7 → half_units=15.  Expected: all 32 values = +15.
 * meta BE u16: dialect=0,exp=15 → (0<<12)|(15<<7)=0x0780 → hi=0x07 lo=0x80
 * code=(0<<3)|7=0x07; packed byte=0x77 */
static int test_bd_decoder(void) {
    static const uint8_t blk[18] = {
        0x07, 0x80,
        0x77,0x77,0x77,0x77, 0x77,0x77,0x77,0x77,
        0x77,0x77,0x77,0x77, 0x77,0x77,0x77,0x77
    };

    BD_META = ((uint32_t)blk[0] << 8) | blk[1];
    for (int w = 0; w < 4; w++) {
        const uint8_t *p = &blk[2 + w*4];
        BD_PACKED(w) = (uint32_t)p[0] | ((uint32_t)p[1]<<8)
                     | ((uint32_t)p[2]<<16) | ((uint32_t)p[3]<<24);
    }

    int8_t out[32];
    for (int w = 0; w < 8; w++) {
        uint32_t v = BD_DECODED(w);
        out[w*4+0] = (int8_t)(v & 0xFF);        out[w*4+1] = (int8_t)((v>>8) & 0xFF);
        out[w*4+2] = (int8_t)((v>>16) & 0xFF);  out[w*4+3] = (int8_t)((v>>24) & 0xFF);
    }
    uint8_t sexp = (uint8_t)(BD_SHARED_EXP & 0x1F);

    int ok = (sexp == 15);
    for (int i = 0; i < 32 && ok; i++) ok = (out[i] == 15);

    tag("decoder");
    if (ok) {
        print("self-test PASS (dialect=0 exp=15 all-max → 32x+15)");
    } else {
        print("self-test FAIL  sexp="); print_dec(sexp);
        print("  out[0..3]=[");
        for (int i = 0; i < 4; i++) {
            if (i) uart_write(UART,' ');
            int8_t v = out[i];
            if (v < 0) { uart_write(UART,'-'); v = (int8_t)(-v); }
            print_dec((uint32_t)(uint8_t)v);
        }
        print("]");
    }
    print_nl();
    return ok;
}

/* ── IRQ stub ────────────────────────────────────────────────────────── */
void irqCallback(void) { while (1); }

/* ── main ────────────────────────────────────────────────────────────── */
void main(void) {
    uint32_t t_start = rdcycle_csr();

    print_nl();
    print("========================================"); print_nl();
    print(" resnet1202_phase0_smoke  Milestone 0  "); print_nl();
    print("========================================"); print_nl();
    print_nl();

    /* Test 1: rdcycle */
    if (!test_rdcycle()) {
        print("[phase0] FATAL: rdcycle broken"); print_nl();
        while (1);
    }
    print_nl();

    /* Tests 2+3: blob header */
    if (!test_blob()) {
        print("[phase0] FATAL: blob invalid"); print_nl();
        while (1);
    }
    print_nl();

    /* Deliverable [0] */
    print("[0] VWB2 blob OK: ");
    {
        const vwb2_header_t *h = (const vwb2_header_t *)WEIGHT_BLOB_ADDR;
        print_dec(h->tensor_count); print(" tensors  ");
        print_dec(h->data_bytes); print(" B data");
    }
    print_nl();

    /* Test 4: HW decoder */
    int dec_ok = test_bd_decoder();
    print("[1] BD decoder self-test: ");
    print(dec_ok ? "PASS" : "FAIL");
    print_nl();
    print_nl();

    uint32_t t_end = rdcycle_csr();
    print("----------------------------------------"); print_nl();
    if (dec_ok) {
        print("[phase0] PASS  cycles="); print_dec(t_end - t_start);
    } else {
        print("[phase0] blob OK but decoder FAILED");
    }
    print_nl();
    print("----------------------------------------"); print_nl();

    while (1);
}
