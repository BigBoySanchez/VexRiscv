#include <stdint.h>
#include <stddef.h>
#include "murax.h"
#include "input.h"     // Generated input image
#include "expected_full.h" // Generated expected output

// Memory Map
#define RAM_BASE        0x10000000
#define WEIGHTS_BASE    0x20000000
#define UART_BASE       0x40000000

#undef UART
#undef GPIO_A
#define UART      ((Uart_Reg*)(0x40010000))
#define GPIO_A    ((Gpio_Reg*)(0x40000000))

// --- Helper Functions ---
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

void print_hash(const char* name, int8_t* buffer, int size) {
    uint32_t sum = 0;
    for(int i = 0; i < size; i++) {
        sum += (uint32_t)(int32_t)buffer[i];
    }
    print("Hash "); print(name); print(": 0x"); print_hex(sum, 8); print("\r\n");
}

// --- Accessors ---
const uint8_t* WEIGHTS_PTR = (const uint8_t*)(WEIGHTS_BASE + 16); // Skip header
uint32_t w_offset = 0;

void reset_weights() {
    w_offset = 0;
}

const int8_t* get_weights(int count) {
    const int8_t* p = (const int8_t*)(WEIGHTS_PTR + w_offset);
    w_offset += count;
    while (w_offset % 4 != 0) w_offset++;
    return p;
}

// --- CNN Primitives (Int8) ---
// Buffer Strategy (fits in 64KB SPRAM)
// We need at most:
// Stage 1: 16*32*32 = 16KB (Input), 16KB (Output), 16KB (Shortcut)
// Stage 2: 32*16*16 = 8KB
// Stage 3: 64*8*8 = 4KB
int8_t buffer_A[16 * 32 * 32];
int8_t buffer_B[16 * 32 * 32];

void* memcpy(void* dest, const void* src, size_t n) {
    char* d = (char*)dest;
    const char* s = (const char*)src;
    while(n--) *d++ = *s++;
    return dest;
}

void* memset(void* s, int c, size_t n) {
    char* p = (char*)s;
    while(n--) *p++ = c;
    return s;
}

int8_t buffer_temp[16 * 32 * 32]; // Used for residual shortcuts

// Cached Weights
// Instead of caching ALL weights, we cache PER OUTPUT CHANNEL.
int8_t channel_weight_cache[64 * 3 * 3]; // Max 576 bytes per output channel (64 in_c * 3 * 3)

void conv2d_3x3(const int8_t* input, int8_t* output, 
                int in_c, int out_c, int h, int w, 
                int stride, int padding) {
    
    int w_count = out_c * in_c * 3 * 3;
    const int8_t* weights = get_weights(w_count);
    
    int out_h = h / stride;
    int out_w = w / stride;
    
    for (int oc = 0; oc < out_c; oc++) {
        // Cache weights for this output channel to avoid flash read stalls
        for(int i = 0; i < in_c * 9; i++) {
            channel_weight_cache[i] = weights[oc * in_c * 9 + i];
        }
        
        for (int y = 0; y < out_h; y++) {
            for (int x = 0; x < out_w; x++) {
                int32_t sum = 0;
                
                int in_y_base = y * stride - padding;
                int in_x_base = x * stride - padding;
                
                for (int ic = 0; ic < in_c; ic++) {
                    int w_idx_base = ic * 9;
                    int in_idx_base = (ic * h * w);
                    
                    for (int ky = 0; ky < 3; ky++) {
                        int iy = in_y_base + ky;
                        if (iy >= 0 && iy < h) {
                            for (int kx = 0; kx < 3; kx++) {
                                int ix = in_x_base + kx;
                                if (ix >= 0 && ix < w) {
                                    int8_t val = input[in_idx_base + (iy * w) + ix];
                                    int8_t w_val = channel_weight_cache[w_idx_base + ky * 3 + kx];
                                    sum += val * w_val;
                                }
                            }
                        }
                    }
                }
                
                output[(oc * out_h * out_w) + (y * out_w) + x] = (int8_t)(sum >> 7);
            }
        }
    }
}

void batch_norm(int8_t* feature_map, int channels, int h, int w, int apply_relu) {
    const int8_t* bn_weight = get_weights(channels);
    const int8_t* bn_bias   = get_weights(channels);
    
    // Cache BN params
    int8_t w_cache[64];
    int8_t b_cache[64];
    for(int c=0; c<channels; c++){
        w_cache[c] = bn_weight[c];
        b_cache[c] = bn_bias[c];
    }
    
    for (int c = 0; c < channels; c++) {
        int8_t w_bn = w_cache[c];
        int8_t b_bn = b_cache[c];
        
        for (int i = 0; i < h * w; i++) {
            int idx = c * h * w + i;
            int32_t val = feature_map[idx];
            val = (val * w_bn) >> 6;
            val += b_bn;
            
            if (apply_relu) {
                if (val < 0) val = 0;
                if (val > 127) val = 127;
            } else {
                if (val < -128) val = -128;
                if (val > 127) val = 127;
            }
            feature_map[idx] = (int8_t)val;
        }
    }
}

void option_a_downsample(const int8_t* input, int8_t* output, int in_c, int out_c, int h, int w) {
    int out_h = h / 2;
    int out_w = w / 2;
    int pad_c = (out_c - in_c) / 2;
    
    for(int i=0; i<out_c * out_h * out_w; i++) output[i] = 0;
    
    for(int c=0; c<in_c; c++) {
        int out_c_idx = c + pad_c;
        for(int y=0; y<out_h; y++) {
            for(int x=0; x<out_w; x++) {
                output[(out_c_idx * out_h * out_w) + (y * out_w) + x] = input[(c * h * w) + (y * 2 * w) + (x * 2)];
            }
        }
    }
}

void add_relu(int8_t* dst, const int8_t* src, int size) {
    for(int i=0; i<size; i++) {
        int32_t val = (int32_t)dst[i] + (int32_t)src[i];
        if (val < 0) val = 0;
        if (val > 127) val = 127;
        dst[i] = (int8_t)val;
    }
}

void avgpool(const int8_t* input, int8_t* output, int channels) {
    for(int c=0; c<channels; c++) {
        int32_t sum = 0;
        for(int i=0; i<64; i++) {
            sum += input[c * 64 + i];
        }
        output[c] = (int8_t)(sum >> 6);
    }
}

// Basic Block
void basic_block(const char* name, int8_t* in_buf, int8_t* out_buf, 
                 int in_c, int out_c, int h, int w, int stride) {
    
    int out_h = h / stride;
    int out_w = w / stride;
    int out_size = out_c * out_h * out_w;
    
    // conv1 + bn + relu
    conv2d_3x3(in_buf, out_buf, in_c, out_c, h, w, stride, 1);
    batch_norm(out_buf, out_c, out_h, out_w, 1);
    
    // conv2 + bn (no relu yet)
    // Need a temp buffer for inner operation if we want to preserve shortcut.
    if (stride != 1 || in_c != out_c) {
        option_a_downsample(in_buf, buffer_temp, in_c, out_c, h, w);
    } else {
        // copy in_buf to temp so we don't overwrite it
        for(int i=0; i<out_size; i++) buffer_temp[i] = in_buf[i];
    }
    
    // Second conv uses in_buf as temporary scratch to hold conv result
    conv2d_3x3(out_buf, in_buf, out_c, out_c, out_h, out_w, 1, 1);
    batch_norm(in_buf, out_c, out_h, out_w, 0);
    
    // Add shortcut (buffer_temp) and ReLU -> write back to out_buf
    for(int i=0; i<out_size; i++) out_buf[i] = in_buf[i]; // copy result
    add_relu(out_buf, buffer_temp, out_size);
    
    print_hash(name, out_buf, out_size);
}

// Start Main
void main() {
    print("\r\n[ALIVE] CPU booted OK\r\n");
    print("Phase Full: ResNet-20 Inference\r\n");
    
    reset_weights();
    
    // Check Header
    volatile uint32_t* header = (volatile uint32_t*)WEIGHTS_BASE;
    if (header[0] != 0x56574230) {
        print("Invalid Magic!\r\n");
    }
    
    uint32_t start_cycles, end_cycles;
    asm volatile("csrr %0, mcycle" : "=r"(start_cycles));
    
    print("Layer 1: Conv2d 3->16...\r\n");
    conv2d_3x3((const int8_t*)INPUT_DATA, buffer_A, 3, 16, 32, 32, 1, 1);
    batch_norm(buffer_A, 16, 32, 32, 1);
    print_hash("conv1          ", buffer_A, 16*32*32);
    
    // Stage 1
    basic_block("layer1_0       ", buffer_A, buffer_B, 16, 16, 32, 32, 1);
    basic_block("layer1_1       ", buffer_B, buffer_A, 16, 16, 32, 32, 1);
    basic_block("layer1_2       ", buffer_A, buffer_B, 16, 16, 32, 32, 1);
    
    // Stage 2
    basic_block("layer2_0       ", buffer_B, buffer_A, 16, 32, 32, 32, 2);
    basic_block("layer2_1       ", buffer_A, buffer_B, 32, 32, 16, 16, 1);
    basic_block("layer2_2       ", buffer_B, buffer_A, 32, 32, 16, 16, 1);
    
    // Stage 3
    basic_block("layer3_0       ", buffer_A, buffer_B, 32, 64, 16, 16, 2);
    basic_block("layer3_1       ", buffer_B, buffer_A, 64, 64, 8, 8, 1);
    basic_block("layer3_2       ", buffer_A, buffer_B, 64, 64, 8, 8, 1);
    
    // Pool
    avgpool(buffer_B, buffer_A, 64);
    print_hash("pool           ", buffer_A, 64);
    
    // FC Layer
    const int8_t* fc_w = get_weights(10 * 64);
    const int8_t* fc_b = get_weights(10);
    
    int32_t logits[10];
    int best_class = 0;
    int32_t best_score = -9999999;
    
    for(int i=0; i<10; i++) {
        int32_t sum = 0;
        for(int c=0; c<64; c++) {
            sum += (int32_t)buffer_A[c] * (int32_t)fc_w[i * 64 + c];
        }
        sum += (int32_t)fc_b[i];
        logits[i] = sum;
        if (sum > best_score) {
            best_score = sum;
            best_class = i;
        }
    }
    
    asm volatile("csrr %0, mcycle" : "=r"(end_cycles));
    
    print("Final Logits: \r\n");
    for(int i=0; i<10; i++) {
        print_int(logits[i]); print(" ");
    }
    print("\r\n");
    
    print("Expected: \r\n");
    for(int i=0; i<10; i++) {
        print_int(EXPECTED_LOGITS[i]); print(" ");
    }
    print("\r\n");
    
    print("Predicted Class: "); print_int(best_class); print("\r\n");
    print("Cycles: "); print_int(end_cycles - start_cycles); print("\r\n");
    print("SUCCESS: Run Complete\r\n");
}

void irqCallback(){ }
