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

void print_int(int val){
    char buffer[32];
    // Simple itoa implementation or similar would go here
    // For now just printing hex
    uart_write(UART, '0');
    uart_write(UART, 'x');
    for(int i=28; i>=0; i-=4){
        int hex = (val >> i) & 0xF;
        if(hex < 10) uart_write(UART, hex + '0');
        else uart_write(UART, hex - 10 + 'A');
    }
    uart_write(UART, '\n');
}

char uart_read(Uart_Reg *reg){
	while(uart_readOccupancy(reg) == 0);
	return reg->DATA;
}

int read_int(){
    int value = 0;
    char c;
    while(1){
        c = uart_read(UART);
        uart_write(UART, c); // Echo
        if(c >= '0' && c <= '9'){
            value = value * 10 + (c - '0');
        } else if(c == '\n' || c == '\r'){
            return value;
        }
    }
}

void main() {
    GPIO_A->OUTPUT_ENABLE = 0xFF; // Enable LEDs
    GPIO_A->OUTPUT = 0x03; // Turn on LEDs for debug

    print("MURAX Started. Custom Instruction Demo (y = 5x + 10)\n");

    while(1){
        print("\nEnter value for x: ");
        int input = read_int();
        print("\nInput: ");
        print_int(input);

        int result;
        custom_mx_plus_b(result, input);

        print("Result: ");
        print_int(result);
        
        // Visualize on LEDs (bottom 8 bits)
        GPIO_A->OUTPUT = result & 0xFF;
    }
}

void irqCallback(){
}
