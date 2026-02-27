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

// ============================================================================
// BlockDialect-Lite Format Constants
// ============================================================================
#define BD_MAGIC        0x56574231  // 'VWB1'
#define BD_BLOCK_SIZE   32          // elements per block
#define BD_BLOCK_BYTES  18          // 2 (metadata) + 16 (packed codes)

// ============================================================================
// Hardware BlockDialect Decoder (MMIO @ 0x40030000)
// ============================================================================
#define BD_DEC_BASE     0x40030000
#define BD_META         (*(volatile uint32_t*)(BD_DEC_BASE + 0x00))
#define BD_PACKED(i)    (*(volatile uint32_t*)(BD_DEC_BASE + 0x04 + (i)*4))
#define BD_DECODED(i)   (*(volatile uint32_t*)(BD_DEC_BASE + 0x20 + (i)*4))
#define BD_STATUS       (*(volatile uint32_t*)(BD_DEC_BASE + 0x40))

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

// Decode a single block (18 bytes) into 32 int8 values using hardware decoder
static void decode_block(const uint8_t* block_data, int8_t* out) {
    // Write metadata (big-endian uint16 → zero-extended to uint32)
    BD_META = ((uint32_t)block_data[0] << 8) | block_data[1];

    // Write 16 bytes of packed codes as 4 × 32-bit words (little-endian)
    const uint32_t* packed_words = (const uint32_t*)(block_data + 2);
    BD_PACKED(0) = packed_words[0];
    BD_PACKED(1) = packed_words[1];
    BD_PACKED(2) = packed_words[2];
    BD_PACKED(3) = packed_words[3];

    // Read 32 decoded bytes as 8 × 32-bit words
    uint32_t* out_words = (uint32_t*)out;
    out_words[0] = BD_DECODED(0);
    out_words[1] = BD_DECODED(1);
    out_words[2] = BD_DECODED(2);
    out_words[3] = BD_DECODED(3);
    out_words[4] = BD_DECODED(4);
    out_words[5] = BD_DECODED(5);
    out_words[6] = BD_DECODED(6);
    out_words[7] = BD_DECODED(7);
}

// Scratch buffer for decoded weights (one tensor at a time)
// Biggest tensor for layer 1: 16 * 3 * 3 * 3 = 432 elements
// Round up to multiple of BLOCK_SIZE = 448
#define DECODE_BUF_SIZE 512
static int8_t decode_buf[DECODE_BUF_SIZE];

// Read and decode N int8 weights from the BlockDialect blob
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

        decode_block(BD_BASE + bd_offset, out_ptr);
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
