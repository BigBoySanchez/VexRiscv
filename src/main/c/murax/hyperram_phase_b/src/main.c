#include <stdint.h>
#include <stddef.h>
#include "murax.h"

// Minimal memcpy/memset for compiler-generated calls (bare-metal, no libc)
void* memcpy(void* dest, const void* src, unsigned int n) {
    uint8_t* d = (uint8_t*)dest;
    const uint8_t* s = (const uint8_t*)src;
    while (n--) *d++ = *s++;
    return dest;
}

void* memset(void* dest, int c, unsigned int n) {
    uint8_t* d = (uint8_t*)dest;
    while (n--) *d++ = (uint8_t)c;
    return dest;
}

#include "input.h"     // Generated input image (same as Phase A)

// Phase B Memory Map (same as Phase A)
#define RAM_BASE        0x10000000
#define WEIGHTS_BASE    0x20000000
#define UART_BASE       0x40000000

#undef UART
#undef GPIO_A
#define UART      ((Uart_Reg*)(0x40010000))
#define GPIO_A    ((Gpio_Reg*)(0x40000000))

#define BD_MAGIC        0x56574231  // 'VWB1' as stored in blob (bytes: 31 42 57 56 → LE u32 = 0x56574231)
#define BD_BLOCK_SIZE   32          // elements per block
#define BD_BLOCK_BYTES  18          // 2 (metadata) + 16 (packed codes)

// ============================================================================
// Decoder Output Choice — OPTION A: signed half-units
// ============================================================================
// The hardware BlockDialectDecoder outputs BD_DECODED as *signed half-units*,
// i.e. integers in [-15..15] where the real magnitude is 0.5 × |value|.
//
// Paper formula (arXiv 2501.01144v5, §3.2):
//   real_value = sign × half_units × 0.5 × 2^(shared_exp_bits − 15)
//              = signed_half_unit × 2^(shared_exp_bits − 16)
//
// We adopt the "rescale per block, once" strategy (§6.2):
//   1. decode_block_raw() reads hardware → int8 signed half-units
//   2. scale_half_units()  applies the per-block exponent → int8 weights
//   3. conv2d/batchnorm consume the resulting int8 weights directly
//
// This keeps MAC units simple (no per-element rescaling) while staying
// numerically faithful to the BlockDialect paper representation.
//
// To switch to Option B (int8 direct from SW softdecoder, no hardware step),
// remove scale_half_units() calls and replace decode_block_raw() with a SW
// decode that emits pre-scaled int8 values.
// ============================================================================
// Hardware BlockDialect Decoder (MMIO @ 0x40030000)
// ============================================================================
#define BD_DEC_BASE     0x40030000
#define BD_META         (*(volatile uint32_t*)(BD_DEC_BASE + 0x00))
#define BD_PACKED(i)    (*(volatile uint32_t*)(BD_DEC_BASE + 0x04 + (i)*4))
#define BD_DECODED(i)   (*(volatile uint32_t*)(BD_DEC_BASE + 0x20 + (i)*4))
#define BD_STATUS       (*(volatile uint32_t*)(BD_DEC_BASE + 0x40))
#define BD_SHARED_EXP   (*(volatile uint32_t*)(BD_DEC_BASE + 0x44))

// ============================================================================
// Helper Functions
// ============================================================================

void print(const char*str){
    while(*str){
        uart_write(UART,*str);
        str++;
    }
}

void print_hex(uint32_t val, int digits) {
    for (int i = (digits - 1) * 4; i >= 0; i -= 4) {
        int d = (val >> i) & 0xF;
        uart_write(UART, (d < 10 ? '0' + d : 'A' + d - 10));
    }
}

void print_int(int val) {
    if(val < 0){
        uart_write(UART,'-');
        val = -val;
    }
    char buffer[12];
    int i = 0;
    if (val == 0) {
        uart_write(UART, '0');
        return;
    }
    while(val){
        buffer[i++] = val % 10 + '0';
        val /= 10;
    }
    while(i > 0){
        uart_write(UART, buffer[--i]);
    }
}

static inline uint32_t u32_le_from_bytes(const uint8_t *p) {
    return (uint32_t)p[0]
         | ((uint32_t)p[1] << 8)
         | ((uint32_t)p[2] << 16)
         | ((uint32_t)p[3] << 24);
}

// ============================================================================
// BlockDialect Weight Reader
// ============================================================================

// Pointer into the weight blob (after 16-byte file header)
const uint8_t* BD_BASE;
uint32_t bd_offset = 0;
uint32_t bytes_read_total = 0;  // Track total bytes consumed from WeightStore

void reset_weights() {
    bd_offset = 0;
    bytes_read_total = 0;
    BD_BASE = (const uint8_t*)(WEIGHTS_BASE + 16); // Skip file header
}

// Decode a single block (18 bytes) into 32 signed half-units using hardware decoder
static void decode_block_raw(const uint8_t* block_data, int8_t* out, uint8_t* shared_exp_out) {
    // Write metadata (big-endian uint16 → zero-extended to uint32)
    BD_META = ((uint32_t)block_data[0] << 8) | block_data[1];

    // Write 16 bytes of packed codes as 4 × 32-bit words (little-endian)
    BD_PACKED(0) = u32_le_from_bytes(block_data + 2);
    BD_PACKED(1) = u32_le_from_bytes(block_data + 6);
    BD_PACKED(2) = u32_le_from_bytes(block_data + 10);
    BD_PACKED(3) = u32_le_from_bytes(block_data + 14);

    // Read decoded half-units without assuming out is word-aligned
    for (int w = 0; w < 8; w++) {
        uint32_t v = BD_DECODED(w);
        int idx = w * 4;
        out[idx + 0] = (int8_t)(v & 0xFF);
        out[idx + 1] = (int8_t)((v >> 8) & 0xFF);
        out[idx + 2] = (int8_t)((v >> 16) & 0xFF);
        out[idx + 3] = (int8_t)((v >> 24) & 0xFF);
    }

    if (shared_exp_out) {
        *shared_exp_out = (uint8_t)(BD_SHARED_EXP & 0x1F);
    }
}

// scale_half_units: convert one signed half-unit to an int8 weight.
//
// Paper formula: real = signed_hu × 2^(sexp − 16)
//   where sexp = shared_exp_bits (5-bit FP16 exponent bias=15, times-0.5 = -1 → net -16).
// Implemented as an arithmetic shift to avoid floating-point on the CPU.
// Rounding: round-half-away-from-zero (unbiased enough for weights).
// Result clamped to [-127, 127] (not -128, to keep symmetric range).
static int8_t scale_half_units(int8_t hu, uint8_t shared_exp_bits) {
    int32_t v = (int32_t)hu;
    int shift = (int)shared_exp_bits - 16; // value = hu × 2^(sexp − 16)

    if (shift >= 0) {
        v = v << shift;
    } else {
        int rshift = -shift;
        int32_t add = 1 << (rshift - 1);
        if (v < 0) {
            v = -(((-v) + add) >> rshift);
        } else {
            v = (v + add) >> rshift;
        }
    }

    if (v > 127) v = 127;
    if (v < -127) v = -127;
    return (int8_t)v;
}

// Scratch buffer for decoded weights (one get_weights() call at a time).
// Buffer is reused across calls — it need only hold the largest single tensor.
//   Layer 1 conv kernel: out_c × in_c × 3 × 3 = 16 × 3 × 9 = 432 elements
//   → ceil(432/32) = 14 blocks × 32 = 448 elements → fits in 512.
// For future layers with more channels, increase DECODE_BUF_SIZE accordingly
// (e.g., 64×64×3×3 = 36 864 elements → DECODE_BUF_SIZE 36992).
#define DECODE_BUF_SIZE 512
static int8_t decode_buf[DECODE_BUF_SIZE];

// Read and decode N int8 weights from the BlockDialect blob.
// Option A path: reads signed half-units from hardware decoder, then
// calls scale_half_units() to produce int8 — once per block (not per multiply).
const int8_t* get_weights(int count) {
    // Read tensor header: n_elements(4) + n_blocks(4)
    const uint8_t* p = BD_BASE + bd_offset;
    uint32_t n_elements = *(volatile uint32_t*)(p);
    uint32_t n_blocks = *(volatile uint32_t*)(p + 4);
    bd_offset += 8;
    bytes_read_total += 8;

    // Decode blocks
    for (uint32_t b = 0; b < n_blocks; b++) {
        int8_t* out_ptr = decode_buf + b * BD_BLOCK_SIZE;
        // Safety: don't overrun decode buffer
        if ((b + 1) * BD_BLOCK_SIZE > DECODE_BUF_SIZE) break;

        uint8_t shared_exp_bits = 0;
        int8_t raw_block[BD_BLOCK_SIZE];
        decode_block_raw(BD_BASE + bd_offset, raw_block, &shared_exp_bits);
        for (int i = 0; i < BD_BLOCK_SIZE; i++) {
            out_ptr[i] = scale_half_units(raw_block[i], shared_exp_bits);
        }
        bd_offset += BD_BLOCK_BYTES;
        bytes_read_total += BD_BLOCK_BYTES;
    }

    // Align bd_offset to 4 bytes (matching blob alignment)
    while (bd_offset % 4 != 0) {
        bd_offset++;
    }

    return decode_buf;
}

// ============================================================================
// CNN Primitives (same as Phase A)
// ============================================================================

#define MAX_CHANNELS 64
#define IMG_SIZE 32

int8_t buffer_A[32 * 32 * 16]; // 16KB
int8_t buffer_B[32 * 32 * 16]; // 16KB

void conv2d_3x3(const int8_t* input, int8_t* output,
                int in_c, int out_c, int h, int w,
                int stride, int padding) {

    int w_count = out_c * in_c * 3 * 3;
    const int8_t* weights = get_weights(w_count);

    for (int oc = 0; oc < out_c; oc++) {
        for (int y = 0; y < h; y += stride) {
            for (int x = 0; x < w; x += stride) {
                int32_t sum = 0;

                for (int ic = 0; ic < in_c; ic++) {
                    for (int ky = 0; ky < 3; ky++) {
                        for (int kx = 0; kx < 3; kx++) {
                            int iy = y + ky - padding;
                            int ix = x + kx - padding;

                            int8_t val = 0;
                            if (iy >= 0 && iy < h && ix >= 0 && ix < w) {
                                val = input[(ic * h * w) + (iy * w) + ix];
                            }

                            int8_t w_val = weights[((oc * in_c + ic) * 3 + ky) * 3 + kx];
                            sum += val * w_val;
                        }
                    }
                }

                int oy = y / stride;
                int ox = x / stride;
                int ow = w / stride;
                output[(oc * (h/stride) * ow) + (oy * ow) + ox] = (int8_t)(sum >> 7);
            }
        }
    }
}

void batch_norm_relu(int8_t* feature_map, int channels, int h, int w) {
    const int8_t* bn_weight = get_weights(channels);
    // Copy BN weight into local array to avoid aliasing with decode_buf
    int8_t local_w[MAX_CHANNELS];
    for (int i = 0; i < channels; i++) {
        local_w[i] = bn_weight[i];
    }

    const int8_t* bn_bias = get_weights(channels);
    int8_t local_b[MAX_CHANNELS];
    for (int i = 0; i < channels; i++) {
        local_b[i] = bn_bias[i];
    }

    for (int c = 0; c < channels; c++) {
        int8_t w_bn = local_w[c];
        int8_t b_bn = local_b[c];

        for (int i = 0; i < h * w; i++) {
            int idx = c * h * w + i;
            int32_t val = feature_map[idx];
            val = (val * w_bn) >> 6;
            val += b_bn;

            if (val < 0) val = 0;
            if (val > 127) val = 127;
            feature_map[idx] = (int8_t)val;
        }
    }
}

// ============================================================================
// Main
// ============================================================================

void main() {
    print("\r\n[ALIVE] CPU booted OK\r\n");
    print("[Phase B] ResNet-110 Inference (BlockDialect-Lite, HW Decode)\r\n");

    reset_weights();

    // -----------------------------------------------------------------------
    // Self-test 1: hardware decoder → signed half-units
    //   test_block: dialect_id=15, shared_exp_bits=18, packed patterns 0x01..0xEF
    //   Expected half-units: {0,1,2,3,4,5,6,8, 0,-1,-2,-3,-4,-5,-6,-8, …×2}
    //   (dialect 15 table: [0,1,2,3,4,5,6,8] at indices 0..7)
    // -----------------------------------------------------------------------
    {
        const uint8_t test_block[18] = {
            0xF9, 0x00, 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF,
            0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF
        };
        // Signed half-units: magnitude = dialect15[idx], sign from code[3]
        const int8_t expected_hu[32] = {
            0, 1, 2, 3, 4, 5, 6, 8,
            0, -1, -2, -3, -4, -5, -6, -8,
            0, 1, 2, 3, 4, 5, 6, 8,
            0, -1, -2, -3, -4, -5, -6, -8
        };
        int8_t decoded_hu[32];
        uint8_t shared_exp_bits = 0;
        decode_block_raw(test_block, decoded_hu, &shared_exp_bits);

        int ok = (shared_exp_bits == 18);
        for (int i = 0; i < 32 && ok; i++) {
            if (decoded_hu[i] != expected_hu[i]) ok = 0;
        }

        print("[Phase B] Self-test 1 (decoder half-units): ");
        if (ok) {
            print("PASS\r\n");
        } else {
            print("FAIL\r\n");
            print("  shared_exp expected=18 got=");
            print_int(shared_exp_bits); print("\r\n");
            for (int i = 0; i < 32; i++) {
                if (decoded_hu[i] != expected_hu[i]) {
                    print("  elem["); print_int(i); print("] got=");
                    print_int(decoded_hu[i]); print(" want=");
                    print_int(expected_hu[i]); print("\r\n");
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Self-test 2: scale_half_units — Option A full decode→scale chain
    //   Same test_block; shared_exp_bits=18 → shift = 18-16 = 2 → ×4
    //   Expected int8 weights: {0,4,8,12,16,20,24,32, 0,-4,-8,-12,-16,-20,-24,-32, …×2}
    //   Verifies paper formula: scaled = signed_hu × 2^(sexp−16)
    // -----------------------------------------------------------------------
    {
        const uint8_t test_block[18] = {
            0xF9, 0x00, 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF,
            0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF
        };
        // scaled = signed_hu × 4  (shift=2, shared_exp_bits=18)
        const int8_t expected_scaled[32] = {
             0,  4,  8, 12, 16, 20, 24, 32,
             0, -4, -8,-12,-16,-20,-24,-32,
             0,  4,  8, 12, 16, 20, 24, 32,
             0, -4, -8,-12,-16,-20,-24,-32
        };
        int8_t decoded_hu[32];
        int8_t scaled[32];
        uint8_t shared_exp_bits = 0;
        decode_block_raw(test_block, decoded_hu, &shared_exp_bits);
        for (int i = 0; i < 32; i++) {
            scaled[i] = scale_half_units(decoded_hu[i], shared_exp_bits);
        }

        int ok = 1;
        for (int i = 0; i < 32; i++) {
            if (scaled[i] != expected_scaled[i]) { ok = 0; break; }
        }

        print("[Phase B] Self-test 2 (scale_half_units, Option A path): ");
        if (ok) {
            print("PASS\r\n");
        } else {
            print("FAIL\r\n");
            for (int i = 0; i < 32; i++) {
                if (scaled[i] != expected_scaled[i]) {
                    print("  elem["); print_int(i); print("] got=");
                    print_int(scaled[i]); print(" want=");
                    print_int(expected_scaled[i]); print("\r\n");
                }
            }
        }
    }

    // Check Header
    volatile uint32_t* header = (volatile uint32_t*)WEIGHTS_BASE;
    if (header[0] != BD_MAGIC) {
        print("Invalid Magic! Expected VWB1\r\n");
        print("Got: 0x"); print_hex(header[0], 8); print("\r\n");
    }

    uint32_t start_cycles, end_cycles;
    asm volatile("csrr %0, mcycle" : "=r"(start_cycles));

    // Layer 1: Conv2d 3->16 (32x32)
    const int H = 32, W = 32;
    print("Layer 1: Conv2d 3->16 (32x32) [HW BlockDialect decode]...\r\n");

    conv2d_3x3((const int8_t*)INPUT_DATA, buffer_A, 3, 16, H, W, 1, 1);
    batch_norm_relu(buffer_A, 16, H, W);

    asm volatile("csrr %0, mcycle" : "=r"(end_cycles));
    print("Inference Done.\r\n");
    print("Cycles: "); print_int(end_cycles - start_cycles); print("\r\n");

    // Bytes read from WeightStore
    print("Bytes Read: "); print_int(bytes_read_total); print("\r\n");

    // Output hash (same computation as Phase A)
    uint32_t sum = 0;
    for(int i=0; i<H*W*16; i++) sum += buffer_A[i];

    print("Layer1 Hash: 0x"); print_hex(sum, 8); print("\r\n");
    print("SUCCESS: Phase B Run Complete\r\n");
}

void irqCallback() {
    // Disable all interrupts to prevent infinite interrupt loops
    asm volatile("csrw mie, zero");
}
