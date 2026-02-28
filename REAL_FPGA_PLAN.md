Below is a concrete “porting plan” to get this **resnet** branch running on a **real iCEBreaker v1.1 (iCE40UP5K-SG48)**, written at the “CS-undergrad-who-can-code-but-is-new-to-FPGA-bring‑up” level.

A key point up front: **this repo already contains an iCEBreaker build flow for Murax** (a simpler VexRiscv SoC) under `scripts/Murax/iCEBreaker/` (Makefile + wrapper Verilog + pin constraints). You’ll use that as your “known-good” baseline, then swap in the **MuraxHyperRAM / resnet** SoC and fix the two big hardware realities on UP5K:

1. **UP5K has limited initialized BRAM**, and
2. **UP5K’s SPRAM is *not* preloaded/initialized at configuration** (so you can’t rely on “RAM init files” for your program if it maps into SPRAM). ([Manuals+][1])

---

## 0) Understand what you’re starting from

### What the repo provides (relevant pieces)

* **iCEBreaker Murax flow**:

  * `scripts/Murax/iCEBreaker/Makefile` uses `yosys` + `nextpnr-ice40 --up5k --package sg48` + `icepack`, then programs with `iceprog`. ([GitHub][2])
  * `scripts/Murax/iCEBreaker/toplevel.v` is a tiny board wrapper: clock buffer, UART pins, LED pins, button to GPIO, instantiates `Murax`. ([GitHub][3])
  * `scripts/Murax/iCEBreaker/io.pcf` sets pins for **CLK=35, RX=6, TX=9, LEDs=11/37, BTN_N=10**. ([GitHub][4])
* **ResNet/weights work**:

  * `src/main/scala/vexriscv/demo/MuraxHyperRAM.scala` defines a SoC with:

    * RAM at `0x1000_0000`
    * Weight store at `0x2000_0000`
    * APB peripherals at `0x4000_0000` (GPIO/UART/timer/BlockDialect, etc.) ([GitHub][5])
  * `src/main/c/murax/hyperram_phase_a/src/main.c` expects:

    * `RAM_BASE = 0x10000000`
    * `WEIGHTS_BASE = 0x20000000`
    * `UART = 0x40010000`
    * and it allocates two ~16KB buffers (tight RAM assumptions). ([GitHub][6])
  * `src/main/c/murax/hyperram_phase_a/src/linker.ld` currently assumes **64KB RAM at 0x10000000**. ([GitHub][7])

### Board facts you must wire correctly (iCEBreaker v1.x)

* 12 MHz clock on pin 35; UART pins RX=6 / TX=9; BTN_N=10; LED pins include 11 and 37 (often labeled active-low in canonical PCFs). ([GitHub][8])
* QSPI flash pins (if you use flash for weights): `FLASH_SCK=15`, `FLASH_SSB=16`, `FLASH_IO0=14`, `FLASH_IO1=17`, `FLASH_IO2=12`, `FLASH_IO3=13`. ([GitHub][8])

---

## 1) Stage 1 bring-up: prove your toolchain + board can run *something*

**Goal:** Get the existing Murax demo in this repo onto the board, blink LEDs, see UART output. Do not touch resnet yet.

1. Install the open-source FPGA toolchain (Yosys/nextpnr/icestorm) and a RISC‑V GCC toolchain.

   * The branch includes a helper script `scripts/setup_toolchain.sh` (it downloads an OSS CAD Suite bundle). ([GitHub][9])
   * You can also use distro packages, but using one known bundle reduces version surprises.

2. Build and program the baseline Murax design:

   ```bash
   cd scripts/Murax/iCEBreaker
   make
   make prog
   ```

   This uses the Makefile flow targeting UltraPlus (`synth_ice40 -device u`, `nextpnr-ice40 --up5k --package sg48`). ([GitHub][2])

3. Open a serial terminal to the iCEBreaker’s FTDI UART at 115200 baud (exact device name depends on OS).

   * If you don’t see output, still check: LEDs toggle? button reads?
   * If LEDs look “inverted,” that’s normal: many iCEBreaker LED pins are active-low in reference PCFs. ([GitHub][8])

**Why this stage matters:** it proves your cable, permissions, toolchain versions, and basic pin constraints before you introduce resnet complexity.

---

## 2) Stage 2: create an iCEBreaker build target for MuraxHyperRAM (the resnet SoC)

Right now the iCEBreaker scripts instantiate **`Murax`**, not **`MuraxHyperRAM`**. You’ll clone the board support folder and swap the SoC.

### 2.1 Copy the working iCEBreaker scaffolding

Create a new directory (recommended so you don’t break the baseline):

```
scripts/MuraxHyperRAM/iCEBreaker/
  Makefile
  toplevel.v
  io.pcf
```

Start by copying from `scripts/Murax/iCEBreaker/`:

* Keep the same approach: wrapper `toplevel.v`, constraints `io.pcf`, and Makefile structure. ([GitHub][2])

### 2.2 Update the generator target in the Makefile

In the existing iCEBreaker Makefile, the generator line is essentially:

* `sbt "runMain vexriscv.demo.MuraxWithMxPlusB"` ([GitHub][2])

You will change that to run a `runMain` that generates **MuraxHyperRAM**. You have two options:

**Option A (cleanest):** create a new Scala `object` like `MuraxHyperRAM_iCEBreaker` in `src/main/scala/vexriscv/demo/` that:

* sets `coreFrequency = 12 MHz` (so UART baud is correct),
* sets RAM/weight sizes for UP5K reality (next section),
* points `onChipRamHexFile` at your compiled firmware hex.

Then your Makefile calls:

```make
sbt "runMain vexriscv.demo.MuraxHyperRAM_iCEBreaker"
```

**Option B (quick hack):** modify an existing `runMain` object in the branch (less recommended because you’ll lose a clean “sim vs board” split).

---

## 3) Stage 3: fix memory for iCE40UP5K (this is the “real hardware” part)

Your current Phase A linker assumes **64KB RAM at 0x10000000**. ([GitHub][7])
On UP5K, you effectively have:

* a small amount of BRAM that *can* be initialized from the bitstream,
* **128 KiB SPRAM** (single-port, 16-bit) that **is not preloaded at configuration**. ([Project F][10])

### 3.1 Why this matters

The current VexRiscv flow initializes RAM from a hex file using `HexTools.initRam(...)` (see `MuraxPipelinedMemoryBusRam`). ([GitHub][11])
That works great if the memory maps to BRAM. It does **not** work if your program ends up in SPRAM (because SPRAM doesn’t come up preloaded). ([Manuals+][1])

### 3.2 A practical memory map for iCEBreaker (recommended)

Use **two memory regions**:

1. **BRAM “boot/code” RAM** (small, initialized)

   * Base: `0x1000_0000` (keep what the SoC expects) ([GitHub][5])
   * Size: keep it small enough to fit in BRAM (think: a few KB to ~15KB total BRAM budget on UP5K)

2. **SPRAM “data/buffers/stack” RAM** (bigger, not initialized)

   * Pick a new base, e.g. `0x1100_0000` (any aligned region not used elsewhere)
   * Size: up to 128 KiB total is available in principle ([Project F][10])

### 3.3 How to implement that in the SoC (SpinalHDL level)

You already have the building blocks:

* `MuraxPipelinedMemoryBusRam(...)` is a memory block behind a PipelinedMemoryBus and can be mapped at an address. ([GitHub][11])
* `MuraxPipelinedMemoryBusDecoder(...)` lets you map multiple slaves into different address windows. ([GitHub][11])

So in your `MuraxHyperRAM_iCEBreaker` generator:

* Keep the existing initialized RAM at `0x1000_0000` but set its size smaller.
* Add a second RAM instance mapped at `0x1100_0000` intended to become SPRAM.

**Important:** don’t rely on hex initialization for the SPRAM region; treat it as `.bss`/runtime-only storage.

### 3.4 Update the linker script to match

Your current linker has only one region (`RAM`) at 0x10000000, 64k. ([GitHub][7])
Make a new linker, e.g. `linker_icebreaker.ld`, with:

* `BRAM` at `0x1000_0000`, LENGTH = small
* `SPRAM` at `0x1100_0000`, LENGTH = large

Then:

* Put `.text` + `.rodata` + `.data` into **BRAM** (so it’s actually present at reset).
* Put `.bss` + heap + stack into **SPRAM**.

This directly addresses the Phase A program’s big buffers (those `buffer_A`/`buffer_B` live in `.bss`). ([GitHub][6])

---

## 4) Stage 4: make weights work on real iCEBreaker (flash-backed weight store)

Your Phase A code assumes weights are readable at `WEIGHTS_BASE = 0x2000_0000` and it just returns pointers into that region. ([GitHub][6])
In simulation, the repo fakes that with an internal “weight store RAM” mapped there. ([GitHub][5])

On real iCEBreaker you can’t afford multi‑MB internal weight RAM, so you need weights in **external QSPI flash** and expose it at that address.

### 4.1 Wire up QSPI flash pins

Update your `io.pcf` to include flash pins (from canonical iCEBreaker constraints): ([GitHub][8])

* `FLASH_SCK 15`
* `FLASH_SSB 16`
* `FLASH_IO0 14`
* `FLASH_IO1 17`
* `FLASH_IO2 12`
* `FLASH_IO3 13`

(Keep CLK/UART/LED/BTN pins the same.)

### 4.2 Add a “weight-store = flash window” in hardware

Conceptually you want:

CPU reads from 0x2000_0000…  →  your RTL translates that into 24-bit/32-bit flash reads  →  returns 32-bit words.

Implementation approaches (in increasing sophistication):

**Approach 1 (simplest, slowest):** a tiny read-only SPI controller + small FIFO

* Only support sequential reads.
* Stall the bus until data arrives.
* Great for “it works” bring-up.

**Approach 2 (recommended):** use an existing memory-mapped SPI/QSPI “XIP-like” block
Murax (the original SoC) has an XIP SPI block in upstream VexRiscv; you can port the same concept into MuraxHyperRAM so weights are just memory-mapped. (This is exactly the kind of peripheral Murax was designed to showcase.)

Either way, the interface you expose to the CPU should match what `main.c` expects: pointer reads from `WEIGHTS_BASE`. ([GitHub][6])

### 4.3 Programming the flash with both bitstream *and* weights

The iCEBreaker flash is also used for FPGA configuration (bitstream at offset 0), so you must not clobber it when adding weights.

`iceprog`’s common manpage documents reading flash (`-r`), reading ID (`-t`), and writing a file to flash, but it doesn’t document “write at offset.” ([Ubuntu Manpages][12])
So the robust workflow is:

1. **Figure out flash size / ID**

   ```bash
   iceprog -t
   ```

   (Then look up the capacity for that ID, or just use `iceprog -r dump.bin` and check the dump file size.) ([Ubuntu Manpages][12])

2. **Build a combined flash image** on your PC:

   * Put `toplevel.bin` at offset 0
   * Put `weights.bin` at some safe offset (e.g., after a generous gap; you choose)

Example (fill in your actual sizes/offsets):

```bash
# Example only — choose FLASH_SIZE and WEIGHT_OFFSET for your board/bitstream
FLASH_SIZE=$((4*1024*1024))      # use the real size you determined
WEIGHT_OFFSET=$((1*1024*1024))   # pick a safe offset after bitstream

dd if=/dev/zero of=flash.img bs=1 count=$FLASH_SIZE
dd if=bin/toplevel.bin of=flash.img conv=notrunc
dd if=weights.bin of=flash.img conv=notrunc bs=1 seek=$WEIGHT_OFFSET

iceprog flash.img
```

3. **Hardcode (or register-configure) that `WEIGHT_OFFSET`** in your weight-store RTL, so:

```
flash_address = WEIGHT_OFFSET + (cpu_address - 0x2000_0000)
```

---

## 5) Stage 5: adapt the Phase A firmware so it fits (and is debuggable)

### 5.1 Keep the memory-map assumptions consistent

Phase A code uses:

* UART at `0x40010000`
* GPIO at `0x40000000`
* weights at `0x20000000` ([GitHub][6])

MuraxHyperRAM maps peripherals under APB at `0x40000000` and weight store at `0x20000000`. ([GitHub][5])
So you want to preserve these addresses to avoid rewriting lots of firmware.

### 5.2 Update the linker + startup

* Use the two-region linker approach (BRAM for `.text/.rodata/.data`, SPRAM for `.bss/stack`).
* Your `crt.S` clears `.bss` by writing zeros across it. That’s good: it means SPRAM-backed `.bss` becomes deterministic even though SPRAM powers up undefined. ([GitHub][13])

### 5.3 Make bring-up observable

Before trying a convolution:

1. Print a boot banner over UART.
2. Toggle LEDs in a loop.
3. Read the first 16 bytes of the weight header and print them (Phase A checks a magic value at `WEIGHTS_BASE`). ([GitHub][6])

Only once that works should you run the conv.

---

## 6) Stage 6: “definition of done” checklist for real hardware

You’ll know you’re genuinely running on the iCEBreaker when:

1. **Bitstream loads on power-up** (no need to reprogram every reset).
2. UART prints:

   * `"[ALIVE] CPU booted OK"` (or your own banner) ([GitHub][6])
3. LED behavior matches GPIO writes (maybe inverted).
4. Weight header read works (magic matches or at least is stable).
5. A minimal layer inference runs (even slowly), and you can print cycle counts (Phase A reads `mcycle`). ([GitHub][6])

---

## 7) Optional upgrades once it works

### 7.1 Faster clock (PLL)

Start at 12 MHz (simplifies UART correctness). Once stable, add an iCE40 PLL to run, say, 48 MHz and adjust any baud divider assumptions accordingly.

### 7.2 Phase B / BlockDialect weight decoding

MuraxHyperRAM includes a BlockDialect decoder region in the APB map. ([GitHub][5])
Once flash-backed weights work for Phase A, you can:

* Store BlockDialect-compressed weights in flash
* Memory-map them
* Use the decoder peripheral to expand blocks into a small decode buffer in SPRAM/BRAM.

### 7.3 Better flash interface

If your “simple SPI read” is too slow:

* Use quad reads
* Add a tiny cache line buffer (e.g., 32 or 64 bytes)
* Prefetch sequential weight blobs

---

## Summary: the shortest realistic path

1. Run the existing `scripts/Murax/iCEBreaker` flow unchanged to verify toolchain + board. ([GitHub][2])
2. Clone that board support folder and switch it to generate/instantiate `MuraxHyperRAM`. ([GitHub][5])
3. Split memory: BRAM for code/init, SPRAM for `.bss/stack` (because SPRAM doesn’t preload). ([Manuals+][1])
4. Replace the fake “weight store RAM” with a flash-backed memory window at `0x2000_0000`, wiring QSPI pins from the canonical iCEBreaker PCF. ([GitHub][8])
5. Program a combined flash image containing both bitstream and weights; use UART/LEDs to debug each step. ([Ubuntu Manpages][12])

If you want, I can also sketch the exact file edits you’d make in:

* the new `MuraxHyperRAM_iCEBreaker` Scala generator (what to map where),
* a two-region `linker_icebreaker.ld`,
* and a minimal “flash weight-store” RTL interface (read-only, bus-stalling, simplest-first).

---

## What's Next: Testing on Hardware (Stage 4 Complete)

Now that the flash-backed weight storage is implemented, you can test the full Phase A inference directly on your iCEBreaker board:

1. **Connect your board**: Ensure your iCEBreaker is plugged in via USB (should show up in `lsusb` as `0403:6010`).
2. **Generate the weights**: Create the binary weights file using the Python script:
   ```bash
   cd ~/VexRiscv/scripts
   python3 gen_resnet_model.py
   ```
3. **Open a Serial Terminal**: In a separate terminal window, start listening to the UART **before** you program the board. This ensures you catch the initial boot message when `iceprog` resets the CPU:
   ```bash
   picocom -b 115200 /dev/ttyUSB1
   ```
4. **Flash the Board**: Build the combined image (bitstream + weights) and program the flash:
   ```bash
   cd ~/VexRiscv/scripts/MuraxHyperRAM/iCEBreaker
   make flash_image WEIGHTS_BIN=../../../scripts/weights.bin
   make prog_flash
   ```
5. **Verify the Output**: Look at your `picocom` terminal. You should see the `[ALIVE] CPU booted OK` banner, followed by the ResNet-20 layer computations, and a successful Layer 1 Hash (`0x00003629`).

Once you've verified the output locally on the board, the hardware baseline is fully operational. We can then begin **Phase B**: implementing and verifying the custom hardware `BlockDialect` decoder!

[1]: https://manuals.plus/m/98d6bbe68afa96fd78cd835264176a627eeacc979febd48f87f8be4530f079f7 "https://manuals.plus/m/98d6bbe68afa96fd78cd835264176a627eeacc979febd48f87f8be4530f079f7"
[2]: https://raw.githubusercontent.com/BigBoySanchez/VexRiscv/resnet/scripts/Murax/iCEBreaker/Makefile "https://raw.githubusercontent.com/BigBoySanchez/VexRiscv/resnet/scripts/Murax/iCEBreaker/Makefile"
[3]: https://raw.githubusercontent.com/BigBoySanchez/VexRiscv/resnet/scripts/Murax/iCEBreaker/toplevel.v "https://raw.githubusercontent.com/BigBoySanchez/VexRiscv/resnet/scripts/Murax/iCEBreaker/toplevel.v"
[4]: https://raw.githubusercontent.com/BigBoySanchez/VexRiscv/resnet/scripts/Murax/iCEBreaker/io.pcf "https://raw.githubusercontent.com/BigBoySanchez/VexRiscv/resnet/scripts/Murax/iCEBreaker/io.pcf"
[5]: https://raw.githubusercontent.com/BigBoySanchez/VexRiscv/resnet/src/main/scala/vexriscv/demo/MuraxHyperRAM.scala "https://raw.githubusercontent.com/BigBoySanchez/VexRiscv/resnet/src/main/scala/vexriscv/demo/MuraxHyperRAM.scala"
[6]: https://raw.githubusercontent.com/BigBoySanchez/VexRiscv/resnet/src/main/c/murax/hyperram_phase_a/src/main.c "https://raw.githubusercontent.com/BigBoySanchez/VexRiscv/resnet/src/main/c/murax/hyperram_phase_a/src/main.c"
[7]: https://raw.githubusercontent.com/BigBoySanchez/VexRiscv/resnet/src/main/c/murax/hyperram_phase_a/src/linker.ld "https://raw.githubusercontent.com/BigBoySanchez/VexRiscv/resnet/src/main/c/murax/hyperram_phase_a/src/linker.ld"
[8]: https://github.com/YosysHQ/icestorm/blob/master/examples/icebreaker/icebreaker.pcf "https://github.com/YosysHQ/icestorm/blob/master/examples/icebreaker/icebreaker.pcf"
[9]: https://raw.githubusercontent.com/BigBoySanchez/VexRiscv/resnet/scripts/setup_toolchain.sh "https://raw.githubusercontent.com/BigBoySanchez/VexRiscv/resnet/scripts/setup_toolchain.sh"
[10]: https://projectf.io/posts/spram-ice40-fpga/ "https://projectf.io/posts/spram-ice40-fpga/"
[11]: https://raw.githubusercontent.com/BigBoySanchez/VexRiscv/resnet/src/main/scala/vexriscv/demo/MuraxUtiles.scala "https://raw.githubusercontent.com/BigBoySanchez/VexRiscv/resnet/src/main/scala/vexriscv/demo/MuraxUtiles.scala"
[12]: https://manpages.ubuntu.com/manpages/questing/en/man1/iceprog.1.html "Ubuntu Manpage:

       iceprog - simple programming tool for FTDI-based Lattice iCE programmers
    "
[13]: https://raw.githubusercontent.com/BigBoySanchez/VexRiscv/resnet/src/main/c/murax/hyperram_phase_a/src/crt.S "https://raw.githubusercontent.com/BigBoySanchez/VexRiscv/resnet/src/main/c/murax/hyperram_phase_a/src/crt.S"
