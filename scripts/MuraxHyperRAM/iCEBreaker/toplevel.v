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
    input   FLASH_IO1,
    output  FLASH_IO2,
    output  FLASH_IO3
  );

  // Keep QSPI chip out of HOLD/WP in single-SPI mode.
  assign FLASH_IO2 = 1'b1; // /WP high
  assign FLASH_IO3 = 1'b1; // /HOLD high

  wire io_mainClk;
  wire [31:0] io_gpioA_read;
  wire [31:0] io_gpioA_write;
  wire [31:0] io_gpioA_writeEnable;

  // -----------------------------
  // 1) PLL: 12 MHz -> 24 MHz
  // -----------------------------
  wire pll_clk;     // raw PLL output
  wire pll_lock;    // PLL is stable when 1

  SB_PLL40_PAD #(
      .FEEDBACK_PATH("SIMPLE"),
      .DIVR(4'b0000),        // / (DIVR+1) = /1
      .DIVF(7'b0111111),     // * (DIVF+1) = *64
      .DIVQ(3'b101),         // / 2^DIVQ    = /32
      .FILTER_RANGE(3'b001)  // 12 * 64 / 32 = 24 MHz
  ) pll_i (
      .PACKAGEPIN(clk),
      .PLLOUTCORE(pll_clk),
      .LOCK(pll_lock),
      .RESETB(1'b1),
      .BYPASS(1'b0)
  );

  // Put PLL clock onto global clock routing
  // 12MHz -> 24MHz Clock Buffer
  SB_GB mainClkBuffer (
    .USER_SIGNAL_TO_GLOBAL_BUFFER (pll_clk),
    .GLOBAL_BUFFER_OUTPUT ( io_mainClk)
  );

  // Blink LEDs with GPIO
  assign LED_R = io_gpioA_write[0];
  assign LED_G = pll_lock;
  
  // Read Button
  assign io_gpioA_read[0] = BTN_N;
  assign io_gpioA_read[31:1] = 31'b0;

  // -----------------------------
  // 2) Reset: hold SoC until PLL locks
  // -----------------------------
  wire io_asyncReset = ~pll_lock;

  MuraxHyperRAM murax (
    .io_asyncReset(io_asyncReset),
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
