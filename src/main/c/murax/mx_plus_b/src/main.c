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

void main() {
    print("Testing Custom Instruction y = 5x + 10\n");

    int input = 2;
    int result;

    print("Input: ");
    print_int(input);

    // Call custom instruction
    custom_mx_plus_b(result, input);

    print("Result: ");
    print_int(result);

    // Expected: 5*2 + 10 = 20 (0x14)
    if(result == 20) {
        print("PASS\n");
    } else {
        print("FAIL\n");
    }

    while(1);
}

void irqCallback(){
}
