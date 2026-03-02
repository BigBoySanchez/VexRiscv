/* resnet50_phase0_smoke — Milestone 0: board smoke test
 *
 * Supports both blob formats currently in use:
 *   VWB1 (0x56574231) — sequential blob, no tensor table (original format)
 *   VWB2 (0x56574232) — indexed blob with tensor table (new format)
 *
 * Verifies:
 *   1. UART + rdcycle CSR work
 *   2. Blob magic is VWB1 or VWB2; block_size == 32
 *   3a. VWB1: prints payload_size and peeks the first tensor sub-header
 *   3b. VWB2: walks tensor table (first 4 entries) + BD4 sub-header peek
 *   4. BlockDialect decoder APB self-test (known-good block -> expected half-units)
 *
 * Deliverable UART line on success:
 *   [phase0] found ResNet-50 blob.
 */

#include <stdint.h>
#include <stddef.h>
#include "murax.h"
#include "weight_blob.h"
#include "model_constants.h"

/* --- MuraxHyperRAM peripheral overrides ------------------------------------ */
#undef  UART
#undef  GPIO_A
#define UART    ((Uart_Reg*)  (0x40010000))
#define GPIO_A  ((Gpio_Reg*)  (0x40000000))

/* --- Hardware BlockDialect Decoder (APB @ 0x4003_0000) --------------------- */
#define BD_DEC_BASE     0x40030000u
#define BD_META         (*(volatile uint32_t*)(BD_DEC_BASE + 0x00))
#define BD_PACKED(i)    (*(volatile uint32_t*)(BD_DEC_BASE + 0x04 + (i)*4))
#define BD_DECODED(i)   (*(volatile uint32_t*)(BD_DEC_BASE + 0x20 + (i)*4))
#define BD_STATUS       (*(volatile uint32_t*)(BD_DEC_BASE + 0x40))
#define BD_SHARED_EXP   (*(volatile uint32_t*)(BD_DEC_BASE + 0x44))

/* --- VWB1 header layout (16 bytes, all LE u32) ----------------------------- */
/* magic | payload_size | block_size | reserved | <payload> */
#define VWB1_MAGIC       0x56574231u
#define VWB1_BLOCK_SIZE  32u

typedef struct {
    uint32_t magic;
    uint32_t payload_size;
    uint32_t block_size;
    uint32_t reserved;
} vwb1_header_t;

/* First 8 bytes of each VWB1 tensor payload */
typedef struct {
    uint32_t n_elements;
    uint32_t n_blocks;
} vwb1_tensor_hdr_t;

/* --- Minimal libc stubs ---------------------------------------------------- */
void *memcpy(void *dst, const void *src, unsigned int n) {
    uint8_t *d = (uint8_t *)dst;
    const uint8_t *s = (const uint8_t *)src;
    while (n--) *d++ = *s++;
    return dst;
}
void *memset(void *dst, int c, unsigned int n) {
    uint8_t *d = (uint8_t *)dst;
    while (n--) *d++ = (uint8_t)c;
    return dst;
}

/* --- UART helpers ---------------------------------------------------------- */
static void print(const char *s) { while (*s) uart_write(UART, *s++); }
static void print_nl(void) { uart_write(UART, '\r'); uart_write(UART, '\n'); }

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
static void section(const char *tag) { print("["); print(tag); print("] "); }

/* --- rdcycle CSR ----------------------------------------------------------- */
static inline uint32_t rdcycle_csr(void) {
    uint32_t v;
    __asm__ volatile ("csrr %0, mcycle" : "=r"(v));
    return v;
}

/* --- Test 1: rdcycle ------------------------------------------------------- */
static int test_rdcycle(void) {
    uint32_t t0 = rdcycle_csr();
    volatile uint32_t x = 0;
    for (int i = 0; i < 64; i++) x += i;
    uint32_t t1 = rdcycle_csr();
    (void)x;

    section("rdcycle");
    print("t0=0x"); print_hex(t0,8);
    print("  t1=0x"); print_hex(t1,8);
    print("  delta="); print_dec(t1-t0);
    print_nl();

    if (t1 == t0) {
        section("rdcycle"); print("ERROR: counter did not advance!"); print_nl();
        return 0;
    }
    return 1;
}

/* --- Test 2+3: blob header — handles VWB1 and VWB2 ------------------------ */
/* Returns 1 = VWB1 OK, 2 = VWB2 OK, 0 = failure. */
static int test_blob_header(void) {
    const uint32_t *w = (const uint32_t *)WEIGHT_BLOB_ADDR;
    uint32_t magic = w[0];

    section("blob");
    print("addr=0x"); print_hex(WEIGHT_BLOB_ADDR, 8);
    print("  magic=0x"); print_hex(magic, 8);
    print_nl();

    /* ---- VWB1 ---- */
    if (magic == VWB1_MAGIC) {
        const vwb1_header_t *h = (const vwb1_header_t *)WEIGHT_BLOB_ADDR;

        section("blob");
        print("format=VWB1  payload_bytes="); print_dec(h->payload_size);
        print("  block_size="); print_dec(h->block_size);
        print_nl();

        if (h->block_size != VWB1_BLOCK_SIZE) {
            section("blob"); print("ERROR: block_size != 32"); print_nl();
            return 0;
        }

        /* Peek first tensor sub-header (immediately after 16-byte file header) */
        const vwb1_tensor_hdr_t *th =
            (const vwb1_tensor_hdr_t *)(WEIGHT_BLOB_ADDR + sizeof(vwb1_header_t));
        uint32_t n_elem   = th->n_elements;
        uint32_t n_blocks = th->n_blocks;
        uint32_t expected = (n_elem + VWB1_BLOCK_SIZE - 1) / VWB1_BLOCK_SIZE;

        section("blob");
        print("tensor[0]: n_elements="); print_dec(n_elem);
        print("  n_blocks="); print_dec(n_blocks);
        if (n_blocks != expected) {
            print("  (WARNING expected="); print_dec(expected); uart_write(UART,')');
        }
        print_nl();

        /* Peek block[0] meta */
        const uint8_t *blk = (const uint8_t *)(WEIGHT_BLOB_ADDR + sizeof(vwb1_header_t) + 8);
        uint8_t dialect    = (blk[0] >> 4) & 0x0F;
        uint8_t shared_exp = ((blk[0] & 0x0F) << 1) | ((blk[1] >> 7) & 0x01);

        section("blob");
        print("block[0]: dialect="); print_dec(dialect);
        print("  shared_exp="); print_dec(shared_exp);
        print_nl();

        return 1;
    }

    /* ---- VWB2 ---- */
    if (magic == VWB2_MAGIC) {
        const vwb2_header_t *h = (const vwb2_header_t *)WEIGHT_BLOB_ADDR;
        wb_err_t err = vwb2_verify_header(h);

        section("blob");
        print("format=VWB2  version="); print_dec(h->version);
        print("  block_size="); print_dec(h->block_size);
        print("  tensor_count="); print_dec(h->tensor_count);
        print("  data_bytes="); print_dec(h->data_bytes);
        print_nl();

        if (err == WB_ERR_VERSION) {
            section("blob"); print("ERROR: unsupported VWB2 version"); print_nl();
            return 0;
        }
        if (err == WB_ERR_BLKSIZE) {
            section("blob"); print("ERROR: block_size != 32"); print_nl();
            return 0;
        }

        /* Walk first 4 tensor entries */
        const vwb2_entry_t *tbl = vwb2_table(h);
        uint32_t n_show = h->tensor_count < 4 ? h->tensor_count : 4;
        for (uint32_t i = 0; i < n_show; i++) {
            const vwb2_entry_t *e = &tbl[i];
            const char *dt = e->dtype == VWB2_DTYPE_BD4    ? "BD4" :
                             e->dtype == VWB2_DTYPE_FLOAT32 ? "F32" : "???";
            section("blob");
            print("["); print_dec(i); print("] 0x"); print_hex(e->name_hash,8);
            print("  "); print(dt);
            print("  n_elem="); print_dec(e->n_elements);
            print("  bytes="); print_dec(e->tensor_bytes);
            print_nl();
        }

        /* BD4 sub-header peek */
        for (uint32_t i = 0; i < h->tensor_count; i++) {
            if (tbl[i].dtype == VWB2_DTYPE_BD4) {
                const vwb2_bd4_hdr_t *bh = vwb2_bd4_header(h, &tbl[i]);
                const uint8_t *blk = vwb2_bd4_blocks(h, &tbl[i]);
                section("blob");
                print("BD4[0]: n_elements="); print_dec(bh->n_elements);
                print("  n_blocks="); print_dec(bh->n_blocks);
                print("  dialect="); print_dec(VWB2_META_DIALECT(blk[0],blk[1]));
                print("  shared_exp="); print_dec(VWB2_META_EXP(blk[0],blk[1]));
                print_nl();
                break;
            }
        }
        return 2;
    }

    /* Unknown */
    section("blob");
    print("ERROR: unknown magic 0x"); print_hex(magic,8);
    print("  (VWB1=0x56574231  VWB2=0x56574232)");
    print_nl();
    return 0;
}

/* --- Test 4: APB decoder self-test ---------------------------------------- */
/* Synthetic block: dialect=0, exp=15, all 32 codes sign=0 idx=7.
 * dialect 0 idx 7 -> maxHU=15-(0>>1)=15 -> expected 32 x +15.
 * meta BE u16: (0<<12)|(15<<7) = 0x0780 -> hi=0x07, lo=0x80
 * code=(0<<3)|7=0x07; packed byte=(7<<4)|7=0x77
 */
static int test_bd_decoder(void) {
    static const uint8_t blk[18] = {
        0x07, 0x80,
        0x77,0x77,0x77,0x77, 0x77,0x77,0x77,0x77,
        0x77,0x77,0x77,0x77, 0x77,0x77,0x77,0x77
    };
    BD_META = ((uint32_t)blk[0] << 8) | blk[1];
    for (int w = 0; w < 4; w++) {
        const uint8_t *p = &blk[2+w*4];
        BD_PACKED(w) = (uint32_t)p[0]|((uint32_t)p[1]<<8)
                     |((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24);
    }
    int8_t out[32];
    for (int w = 0; w < 8; w++) {
        uint32_t v = BD_DECODED(w);
        out[w*4+0]=(int8_t)(v&0xFF);      out[w*4+1]=(int8_t)((v>>8)&0xFF);
        out[w*4+2]=(int8_t)((v>>16)&0xFF); out[w*4+3]=(int8_t)((v>>24)&0xFF);
    }
    uint8_t sexp = (uint8_t)(BD_SHARED_EXP & 0x1F);

    int ok = (sexp == 15);
    for (int i = 0; i < 32 && ok; i++) ok = (out[i] == 15);

    section("decoder");
    if (ok) {
        print("self-test PASSED (dialect=0,exp=15,all-max -> 32x+15)");
    } else {
        print("self-test FAILED  first8=[");
        for (int i = 0; i < 8; i++) {
            if (i) uart_write(UART,' ');
            int8_t v = out[i];
            if (v < 0) { uart_write(UART,'-'); v=(int8_t)(-v); }
            print_dec((uint32_t)(uint8_t)v);
        }
        print("]  sexp="); print_dec(sexp);
    }
    print_nl();
    return ok;
}

/* --- IRQ stub -------------------------------------------------------------- */
void irqCallback(void) { while (1); }

/* --- main ------------------------------------------------------------------ */
void main(void) {
    uint32_t t_start = rdcycle_csr();

    print_nl();
    print("========================================"); print_nl();
    print(" resnet50_phase0_smoke  Milestone 0    "); print_nl();
    print("========================================"); print_nl();
    print_nl();

    /* 1. rdcycle */
    if (!test_rdcycle()) {
        print("[phase0] FATAL: rdcycle broken"); print_nl();
        while (1);
    }
    print_nl();

    /* 2+3. Blob header (VWB1 or VWB2) */
    int fmt = test_blob_header();
    print_nl();
    if (!fmt) {
        print("[phase0] FATAL: blob header invalid"); print_nl();
        while (1);
    }

    /* *** Milestone 0 deliverable *** */
    print("[phase0] found ResNet-50 blob."); print_nl();
    print_nl();

    /* 4. HW decoder self-test */
    int dec_ok = test_bd_decoder();
    print_nl();

    uint32_t t_end = rdcycle_csr();
    print("----------------------------------------"); print_nl();
    if (dec_ok) {
        print("[phase0] ALL CHECKS PASSED");
    } else {
        print("[phase0] blob OK but decoder self-test FAILED");
    }
    print("  cycles="); print_dec(t_end - t_start); print_nl();
    print("----------------------------------------"); print_nl();

    while (1);
}
