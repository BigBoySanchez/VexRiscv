`timescale 1ns / 1ps

module toplevel(
    input   clk,
    output  TX,
    input   RX,
    output  LED_R,
    output  LED_G,
    input   BTN_N,
    // SPI Flash
    output  FLASH_SCK,
    output  FLASH_SSB,
    output  FLASH_IO0,
    input   FLASH_IO1
  );

  wire io_mainClk;
  wire [31:0] io_gpioA_read;
  wire [31:0] io_gpioA_write;
  wire [31:0] io_gpioA_writeEnable;

  // 12MHz Clock Buffer
  SB_GB mainClkBuffer (
    .USER_SIGNAL_TO_GLOBAL_BUFFER (clk),
    .GLOBAL_BUFFER_OUTPUT ( io_mainClk)
  );

  // Blink LEDs with GPIO
  assign LED_R = io_gpioA_write[0];
  assign LED_G = io_gpioA_write[1];
  
  // Read Button
  assign io_gpioA_read[0] = BTN_N;
  assign io_gpioA_read[31:1] = 31'b0;

  MuraxHyperRAM murax (
    .io_asyncReset(0),
    .io_mainClk (io_mainClk),
    .io_jtag_tck(1'b0),
    .io_jtag_tdi(1'b0),
    .io_jtag_tdo(),
    .io_jtag_tms(1'b0),
    .io_gpioA_read       (io_gpioA_read),
    .io_gpioA_write      (io_gpioA_write),
    .io_gpioA_writeEnable(io_gpioA_writeEnable),
    .io_uart_txd(TX),
    .io_uart_rxd(RX),
    // SPI Flash for weight store
    .io_spiFlash_sclk(FLASH_SCK),
    .io_spiFlash_cs_n(FLASH_SSB),
    .io_spiFlash_mosi(FLASH_IO0),
    .io_spiFlash_miso(FLASH_IO1)
  );		
endmodule
