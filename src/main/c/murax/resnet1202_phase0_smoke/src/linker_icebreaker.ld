/*
 * linker_icebreaker.ld — ResNet-1202 firmware linker script for iCEBreaker
 *
 * Memory map (§0 of RESNET1202_FPGA_PLAN.md):
 *
 *   Address       Region   Size   Contents
 *   ──────────────────────────────────────────────────────────────────
 *   0x10000000    BRAM     12 KB  .text, .data, .rodata, .ctors
 *   0x11000000    SPRAM   128 KB  activation buffers, .bss, stack
 *   0x20000000    WeightStore (flash window, read-only)
 *
 * SPRAM layout (hardcoded addresses used in spram_layout.h):
 *
 *   0x11000000   .act_buffers   ~50 KB  int8 act_A/act_B/skip_buf + scratch
 *   0x1100D000   .bss             ~1 KB  zero-init globals
 *   0x1100D400   .noinit          --     opt. uninit globals
 *   0x1100D800   ._stack        2 KB   stack (grows downward from _stack_start)
 *
 *   Worst-case activation peak (stage1, BD4 skip): ~44.5 KB ≤ 50 KB reserved.
 *   Total SPRAM used: ≈ 53 KB.  Headroom: ~75 KB free above ._stack.
 *
 * All activation arrays MUST be declared with __attribute__((section(".act_buffers")))
 * or ".act_buffers.<name>" so the linker places them at 0x11000000.
 */

OUTPUT_FORMAT("elf32-littleriscv", "elf32-littleriscv", "elf32-littleriscv")
OUTPUT_ARCH(riscv)
ENTRY(crtStart)

MEMORY {
  BRAM      (rx) : ORIGIN = 0x10000000, LENGTH = 12k
  SPRAM     (rw) : ORIGIN = 0x11000000, LENGTH = 128k
}

_stack_size = DEFINED(_stack_size) ? _stack_size : 2048;
_heap_size  = DEFINED(_heap_size)  ? _heap_size  : 0;

SECTIONS {

  /* ── BRAM: firmware code and read-only data ─────────────────────── */

  ._vector ORIGIN(BRAM): {
    *crt.o(.start_jump);
    *crt.o(.text);
  } > BRAM

  .data : {
    *(.rdata)
    *(.rodata .rodata.*)
    *(.gnu.linkonce.r.*)
    *(.data .data.*)
    *(.gnu.linkonce.d.*)
    . = ALIGN(8);
    PROVIDE( __global_pointer$ = . + 0x800 );
    *(.sdata .sdata.*)
    *(.gnu.linkonce.s.*)
    . = ALIGN(8);
    *(.srodata.cst16)
    *(.srodata.cst8)
    *(.srodata.cst4)
    *(.srodata.cst2)
    *(.srodata .srodata.*)
  } > BRAM

  .memory : {
    *(.text);
    end = .;
  } > BRAM

  .ctors : {
    . = ALIGN(4);
    _ctors_start = .;
    KEEP(*(.init_array*))
    KEEP (*(SORT(.ctors.*)))
    KEEP (*(.ctors))
    . = ALIGN(4);
    _ctors_end = .;
  } > BRAM

  /* ── SPRAM: activation buffers first, then .bss and stack ───────── */

  /* §0: Activation buffers are placed at the very start of SPRAM so
   * their addresses match the ACT_*_BASE constants in spram_layout.h.
   * All buffers declared with __attribute__((section(".act_buffers")))
   * land here, before any .bss globals that would otherwise start at
   * ORIGIN(SPRAM) = 0x11000000 and conflict with the hardcoded constants.
   *
   * Reserved block: 52 KB (covers worst-case stage1 + weight scratch).
   * See spram_layout.h for the per-field breakdown.
   */
  .act_buffers (NOLOAD) : {
    . = ALIGN(4);
    PROVIDE (_act_buffers_start = .);
    *(.act_buffers .act_buffers.*)
    . = ALIGN(4);
    PROVIDE (_act_buffers_end = .);
    /* Guard: if activated code overflows this reservation the ASSERT
     * in spram_layout.h will fire at compile time. */
  } > SPRAM

  .bss (NOLOAD) : {
    . = ALIGN(4);
    _bss_start = .;
    *(.sbss*)
    *(.gnu.linkonce.sb.*)
    *(.bss .bss.*)
    *(.gnu.linkonce.b.*)
    *(COMMON)
    . = ALIGN(4);
    _bss_end = .;
  } > SPRAM

  .noinit (NOLOAD) : {
    . = ALIGN(4);
    *(.noinit .noinit.*)
    . = ALIGN(4);
  } > SPRAM

  ._user_heap (NOLOAD) : {
    . = ALIGN(8);
    PROVIDE ( _heap_start = . );
    . = . + _heap_size;
    . = ALIGN(8);
    PROVIDE ( _heap_end = . );
  } > SPRAM

  /* Stack at end of SPRAM, growing downward */
  ._stack (NOLOAD) : {
    . = ALIGN(16);
    PROVIDE (_stack_end   = .);
    . = . + _stack_size;
    . = ALIGN(16);
    PROVIDE (_stack_start = .);
    PROVIDE ( end  = . );
    PROVIDE ( _end = . );
  } > SPRAM
}
