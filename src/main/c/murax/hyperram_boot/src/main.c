#include <stdint.h>
#include "murax.h"

#undef UART
#define UART ((Uart_Reg*)(0x40010000))

static void print(const char* s){
  while(*s) uart_write(UART, *s++);
}

#define FLASH_WIN_BASE 0x20400000u
#define SPRAM_BASE     0x11000000u

typedef void (*entry_fn_t)(void);

void irqCallback() {}

int main(){
  print("BOOT\r\n");

  volatile uint32_t *hdr = (volatile uint32_t*)(FLASH_WIN_BASE);
  uint32_t magic = hdr[0];
  uint32_t len   = hdr[1];
  uint32_t entry = hdr[2];

  if(magic != 0xB00710ADu){
    print("BAD MAGIC\r\n");
    while(1);
  }

  // Copy payload (starts at hdr[3]) into SPRAM_BASE
  volatile uint32_t *src = (volatile uint32_t*)(FLASH_WIN_BASE + 12);
  volatile uint32_t *dst = (volatile uint32_t*)(SPRAM_BASE);

  for(uint32_t i = 0; i < (len + 3)/4; i++){
    dst[i] = src[i];
  }

  print("JUMP\r\n");
  ((entry_fn_t)entry)();

  while(1);
}
