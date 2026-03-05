`timescale 1ns / 1ps

module toplevel(
    input  clk,      // 12 MHz from iCEBreaker
    output TX,
    input  RX,
    output LED_R,
    output LED_G,
    input  BTN_N
);

    // -----------------------------
    // 1) PLL: 12 MHz -> 36 MHz
    // -----------------------------
    wire pll_clk;     // raw PLL output
    wire pll_lock;    // PLL is stable when 1

    SB_PLL40_CORE #(
        .FEEDBACK_PATH("SIMPLE"),
        .DIVR(4'b0000),        // / (DIVR+1) = /1
        .DIVF(7'b0101111),     // * (DIVF+1) = *48  (47+1)
        .DIVQ(3'b100),         // / 2^DIVQ    = /16
        .FILTER_RANGE(3'b001)
    ) pll_i (
        .REFERENCECLK(clk),
        .PLLOUTCORE(pll_clk),
        .LOCK(pll_lock),
        .RESETB(1'b1),
        .BYPASS(1'b0)
    );

    // Put PLL clock onto global clock routing
    wire io_mainClk;
    SB_GB mainClkBuffer (
        .USER_SIGNAL_TO_GLOBAL_BUFFER(pll_clk),
        .GLOBAL_BUFFER_OUTPUT(io_mainClk)
    );

    // -----------------------------
    // 2) Reset: hold SoC until PLL locks
    // -----------------------------
    wire io_asyncReset = ~pll_lock;

    // -----------------------------
    // 3) SoC IO (unchanged)
    // -----------------------------
    wire [31:0] io_gpioA_read;
    wire [31:0] io_gpioA_write;
    wire [31:0] io_gpioA_writeEnable;

    assign LED_R = io_gpioA_write[0];
    assign LED_G = pll_lock;

    assign io_gpioA_read[0]    = BTN_N;
    assign io_gpioA_read[31:1] = 31'b0;

    Murax murax (
        .io_asyncReset(io_asyncReset),
        .io_mainClk(io_mainClk),

        .io_jtag_tck(1'b0),
        .io_jtag_tdi(1'b0),
        .io_jtag_tdo(),
        .io_jtag_tms(1'b0),

        .io_gpioA_read(io_gpioA_read),
        .io_gpioA_write(io_gpioA_write),
        .io_gpioA_writeEnable(io_gpioA_writeEnable),

        .io_uart_txd(TX),
        .io_uart_rxd(RX)
    );

endmodule