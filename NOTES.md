1. dependencies
    - JDK 17.0.7
    - sbt 1.11.7
    - Verilator 5.006
3. follow custom instruction example
4. run regression test (to ensure prev instructions didn't break; 
ideally on a single config to save time, runs all configs otherwise)

5. write and run tests for custom instruction separately
6. flash code to a real fpga (TODO: need more research on 
does prof actually need it/how to do this)

## Pitfalls
- **EVERYTHING** has to be run in Linux. 
- When using WSL, don't work in a Windows dir.

## "Hello, World" Plan

Here’s a fresh, Murax-specific plan to get **“Hello, World” in C** running on a **VexRiscv core** in simulation, keeping things as simple as we reasonably can.
CRUCIAL NOTE: MuraxWithRamInit ALREADY HAS AN INTERACTIVE PROGRAM ON IT

---

## High-level path

Since you’re using **Murax**:

1. Use the **Murax SoC + its existing Verilator testbench**.
2. Compile a tiny bare-metal C program that:

   * Initializes the UART (if needed).
   * Prints `"Hello, world\n"` using Murax’s UART TX.
3. Convert the ELF into the format the Murax sim expects and run it with the existing Makefile hooks.

So you’re not inventing a new SoC or testbench—just dropping your own firmware into the Murax pipeline.

---

## Step 1 – Confirm Murax build + sim flow

You likely already did something like:

```bash
# Generate Murax Verilog
sbt "runMain vexriscv.demo.Murax"
# or Murax with RAM init:
sbt "runMain vexriscv.demo.MuraxWithRamInit"
```

Then:

```bash
cd src/test/cpp/murax
make clean run
```

That flow usually:

* Verilates Murax.
* Loads some default firmware image (ROM/RAM init).
* Runs the simulation and shows UART output in the console.

Your job: **replace that default firmware with your own hello-world program**.

If `make run` already prints something over UART, that’s proof the pipeline is alive. You’ll reuse this, not rebuild it.

---

## Step 2 – Understand the pieces you’ll need to touch

You’ll have three main artifacts:

1. **C code**: `main.c` with `printf`-style UART writes.
2. **Minimal runtime**:

   * A `crt0.S` or similar startup that sets up `sp` and calls `main`.
   * Optionally, some minimal `libc` stubs (`memcpy`, `memset`, etc.) if the toolchain requires them.
3. **Linker script**: `link.ld` to place `.text`, `.data`, `.bss`, and stack into Murax’s RAM.

Murax already defines:

* A **reset vector** (e.g., 0x80000000 is common).
* A **RAM region** mapped at some base address.
* A **UART peripheral** at a fixed MMIO address.

You’ll line your firmware up with these addresses instead of inventing new ones.

---

## Step 3 – Toolchain choices (simple defaults)

Use a standard bare-metal RISC-V GCC:

* **Compiler**: `riscv32-unknown-elf-gcc`
* **ISA**: Murax is typically RV32IM (no compressed / no FPU), so:

  * `-march=rv32im`
  * `-mabi=ilp32`

Compiler flags (minimal, no OS):

```bash
riscv32-unknown-elf-gcc \
  -march=rv32im -mabi=ilp32 \
  -nostartfiles -nostdlib -ffreestanding \
  -T link.ld \
  crt0.S main.c \
  -o hello.elf
```

If you don’t know your exact Murax config, this `rv32im/ilp32` combo is usually correct and safe.

---

## Step 4 – Minimal C “Hello, world” for Murax UART

Conceptually:

* Murax exposes a UART TX register at a specific address.
* Writing a byte there sends it out; the Verilator TB prints it.

Your program just needs:

```c
#define UART_BASE   0xF0010000u  // example – use Murax’s real value
#define UART_TX     (UART_BASE + 0x0) // TX register offset if needed

static volatile unsigned int * const UART = (unsigned int *)UART_TX;

static void uart_putc(char c) {
    *UART = (unsigned int)c;
}

static void uart_puts(const char *s) {
    while (*s) {
        uart_putc(*s++);
    }
}

int main(void) {
    uart_puts("Hello, world\n");

    // Spin forever so CPU doesn't wander
    while (1) {
        // Optionally, low-power wait
        asm volatile ("wfi");
    }

    return 0;
}
```

You’ll fill in the **actual UART address + offsets** from the Murax code (or its existing firmware examples). But structurally, this is all you need.

---

## Step 5 – Startup code (crt0)

A simple `crt0.S` (in pseudo-ish form):

```asm
    .section .init
    .globl _start
_start:
    la   sp, _stack_top      # set stack pointer

    # (Optional) .data/.bss init if needed
    # For simplest-case where everything is linked directly into RAM,
    # you can skip copying .data from ROM and just zero .bss.

    la   a0, __bss_start
    la   a1, __bss_end
1:
    bge  a0, a1, 2f
    sw   x0, 0(a0)
    addi a0, a0, 4
    j    1b
2:

    call main

hang:
    j hang
```

The labels (`_stack_top`, `__bss_start`, `__bss_end`) come from the linker script in the next step.

For your first iteration, you can keep `.data` simple or even avoid it entirely by not using global initialized data that needs copy-from-ROM.

---

## Step 6 – Linker script aligned with Murax RAM

You need a `link.ld` that matches Murax’s RAM base & size.

Typical simple layout (example numbers):

```ld
ENTRY(_start)

MEMORY
{
  RAM (xrw) : ORIGIN = 0x80000000, LENGTH = 64K
}

SECTIONS
{
  .text : {
    _text_start = .;
    *(.init)
    *(.text*)
    *(.rodata*)
    _text_end = .;
  } > RAM

  .data : ALIGN(4) {
    _data_start = .;
    *(.data*)
    _data_end = .;
  } > RAM

  .bss : ALIGN(4) {
    __bss_start = .;
    *(.bss*)
    *(COMMON)
    __bss_end = .;
  } > RAM

  .stack (NOLOAD) : ALIGN(8) {
    _stack_bottom = .;
    . += 4K;
    _stack_top = .;
  } > RAM
}
```

Adjust:

* `ORIGIN` and `LENGTH` to **exactly match** Murax’s RAM region.
* If Murax uses a ROM + RAM split, you can still simplify by:

  * Either placing everything in RAM from reset, or
  * Using a small `.text.init` in ROM that copies code/data into RAM and jumps there (slightly more complex; skip this if you can).

For “keep it simple”, use the **single RAM region** approach if Murax’s design allows it (MuraxWithRamInit often does—RAM is preinitialized with your code).

---

## Step 7 – Make the binary in the format Murax sim expects

Murax’s Verilator testbench usually:

* Loads a **hex** or **binary** file into RAM, or
* Uses a ROM module with an embedded `.hex`/`.bin`.

Common patterns:

* Use `objcopy` to get a Verilog-hex:

  ```bash
  riscv32-unknown-elf-objcopy -O verilog hello.elf hello.hex
  ```

* Or a raw binary:

  ```bash
  riscv32-unknown-elf-objcopy -O binary hello.elf hello.bin
  ```

Then:

* Update `src/test/cpp/murax/Makefile` or the RAM init file path so the sim loads `hello.hex`/`hello.bin` instead of the default test binary.
* If Murax has a `MuraxWithRamInit` generator, point its init image to your new hex.

Your goal: when you run `make run` in the Murax testbench folder, **your program** is what’s sitting in the on-chip RAM at reset.

---

## Step 8 – Run the sim and watch for “Hello, world”

Once the firmware is wired in:

```bash
cd src/test/cpp/murax
make clean run
```

You should see something like:

```text
Hello, world
```

printed by the Verilator testbench, coming from the UART TX writes of your C program.

If you don’t:

1. Double-check:

   * Reset PC (Murax’s reset address).
   * RAM ORIGIN (linker) == that address.
   * That the ELF actually has `_start` at that address (`objdump -d hello.elf`).
2. Confirm the UART base address & offsets match Murax’s spec.
3. Make sure the Makefile / init script is actually loading **your** image, not an old default.

---

## Step 9 – (Optional) Nice-to-have polish

If you want to go a tiny bit further—but still keep it simple:

* Add a **small delay loop** after each UART write if the UART has a “ready” bit and needs pacing.
* Add a `TRACE=yes` flag to your Verilator run (if supported) to dump a VCD and visually confirm:

  * PC increments through your `_start` and `main`.
  * UART TX strobes in sync with the printed string.

---

If you want, next step I can:

* Draft a **concrete `main.c`, `crt0.S`, and `link.ld`** specifically for “Murax, RAM at 0x80000000, UART at X” (you can plug in the exact UART address), and
* Suggest the minimal edits to the Murax `Makefile`/testbench to swap in `hello.elf`.
