#include <stdint.h>
#include "murax.h"

// Define the custom instruction macro
// Opcode 0x0B = 0001011 (Custom-0)
// funct3 = 0, funct7 = 0
// Format: .insn r opcode, func3, func7, rd, rs1, rs2
#define custom_mx_plus_b(rd, rs1) \
    asm volatile ( \
        ".insn r 0x0B, 0, 0, %0, %1, x0" \
        : "=r"(rd) \
        : "r"(rs1) \
    )

void print(const char*str){
	while(*str){
		uart_write(UART,*str);
		str++;
	}
}

void print_int(int val) {
    if(val < 0){
        uart_write(UART,'-');
        val = -val;
    }
    if(val / 10){
        print_int(val / 10);
    }
    uart_write(UART, val % 10 + '0');
}

char uart_read(Uart_Reg *reg){
	while(uart_readOccupancy(reg) == 0);
	return reg->DATA;
}

int read_int() {
    int value = 0;
    int sign = 1;
    int first = 1;
    while (1) {
        char c = uart_read(UART);
        uart_write(UART, c);
        if (first && c == '-') {
            sign = -1;
            first = 0;
        } else if (c >= '0' && c <= '9') {
            value = (value * 10) + (c - '0');
            first = 0;
        } else if (c == '\n' || c == '\r') {
            return value * sign;
        }
    }
}

void main() {
    GPIO_A->OUTPUT_ENABLE = 0xFF; // Enable LEDs
    GPIO_A->OUTPUT = 0x03; // Turn on LEDs for debug

    print("MURAX Started. Custom Instruction Demo (y = mx + b)\r\n");

    while(1){
        print("\r\nEnter value for x: ");
        int input = read_int();
        print("\r\nInput: ");
        print_int(input);

        int result;
        custom_mx_plus_b(result, input);

        print("\r\nResult: ");
        print_int(result);
        
        // Visualize on LEDs (bottom 8 bits)
        GPIO_A->OUTPUT = result & 0xFF;
    }
}

void irqCallback(){
}
