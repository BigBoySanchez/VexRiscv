---
name: running
description: How to build and run VexRiscv simulations (Verilator) and synthesise/flash FPGA bitstreams for the Murax and MuraxHyperRAM demos. Use this skill when the user asks to simulate, run a test, generate Verilog, build a bitstream, program an FPGA, or debug sim issues.
---

# Building & Running – VexRiscv

## Prerequisites

- JDK 17+ and sbt 1.11.7+ (on PATH)
- Verilator 5.006+ (minimum 3.9)
- OSS CAD Suite for FPGA tools: `export PATH=$HOME/oss-cad-suite/bin:$PATH`
- All work must be done in a Linux environment (native or WSL2 with a Linux filesystem).

---

## 1. Generating Verilog

Run from the repo root (`/home/tim/VexRiscv`):

```bash
# Plain Murax SoC
sbt "runMain vexriscv.demo.Murax"

# Murax with RAM init (UART echo + blinking LEDs)
sbt "runMain vexriscv.demo.MuraxWithRamInit"

# Murax with custom MxPlusB instruction (iCEBreaker target)
sbt "runMain vexriscv.demo.MuraxWithMxPlusB"

# MuraxHyperRAM variant (iCEBreaker target)
sbt "runMain vexriscv.demo.MuraxHyperRAM_iCEBreaker"
```

Output files are written to the repo root (e.g., `Murax.v`, `MuraxHyperRAM.v`).

---

## 2. Running Simulations (Verilator)

### C++ Verilator testbench

```bash
cd src/test/cpp/murax
make clean run
```

This verilates `Murax.v`, loads the default firmware hex, and streams UART output to the console.

### SpinalHDL / Scala sim (MuraxSim)

```bash
sbt "runMain vexriscv.MuraxSim"
```

Opens a GUI window with an LED array and switch controls, JTAG TCP bridge, and live UART output.

### Regression suite

From the repo root:

```bash
# Random configs (default)
cd scripts/regression && make regression_random

# Linux-booting configs
make regression_random_linux

# Bare-metal only
make regression_random_baremetal

# Dhrystone benchmark
cd /home/tim/VexRiscv && sbt "testOnly vexriscv.DhrystoneBench"

# All individual features
sbt "testOnly vexriscv.TestIndividualFeatures"
```

Environment variables to tune regression scale:
- `VEXRISCV_REGRESSION_CONFIG_COUNT` – number of random CPU configs
- `VEXRISCV_REGRESSION_FREERTOS_COUNT` – FreeRTOS test count
- `VEXRISCV_REGRESSION_ZEPHYR_COUNT` – Zephyr test count
- `VEXRISCV_REGRESSION_THREAD_COUNT` – parallel threads

---

## 3. Fixing Sim Issues – Clear the simWorkspace Cache

SpinalHDL caches compiled Verilator models in a `simWorkspace/` directory (created at the repo root). **Stale cache entries are the most common cause of mysterious simulation failures** (wrong waveforms, segfaults, "already defined" errors after changing RTL).

```bash
# From the repo root
rm -rf simWorkspace
```

Then re-run the sim command. This forces a full recompilation of the Verilator model.

---

## 4. Building FPGA Bitstreams

### Murax – iCEBreaker (iCE40 UP5K)

```bash
cd scripts/Murax/iCEBreaker
make          # generates + synthesises → bin/toplevel.bin
make prog     # flashes via iceprog
make clean    # removes Murax.v and bin/
```

The `make` target also triggers `sbt "runMain vexriscv.demo.MuraxWithMxPlusB"` if `Murax.v` is missing.

### MuraxHyperRAM – iCEBreaker

```bash
cd scripts/MuraxHyperRAM/iCEBreaker
make           # generate + synth
make prog      # flash bitstream only
make prog_weights WEIGHTS_BIN=weights.bin   # flash weights at 1 MiB offset
make prog_app   APP_BIN=app.bin             # flash app at 5 MiB offset
make clean
```

### Other boards

| Board | Path |
|---|---|
| iCE40-HX8K breakout | `scripts/Murax/iCE40-hx8k_breakout_board/` |
| iCE40-HX8K breakout (XIP) | `scripts/Murax/iCE40-hx8k_breakout_board_xip/` |
| iCE40HX8K-EVB | `scripts/Murax/iCE40HX8K-EVB/` |
| iCESugar | `scripts/Murax/iCESugar/` |
| Arty A7 | `scripts/Murax/arty_a7/` |

All boards follow the same `make` / `make prog` pattern.

---

## 5. Verifying Serial Output

After flashing, connect at 115200 baud:

```bash
picocom -b 115200 /dev/ttyUSB1
```

(Adjust the device path; `lsusb` and `ls /dev/ttyUSB*` help identify it.)

---

## 6. Toolchain Installation (first time)

```bash
# All-in-one
source tools.sh && install_tools

# Individual components
install_verilator   # builds Verilator 4.216 from source
install_ghdl        # builds GHDL from source
install_iverilog    # builds Icarus Verilog 10.3
install_cocotb      # pip install + apt packages
```

Tools are installed to `~/tools`.  Add `~/tools/bin` to your `PATH`.

---

## 7. A note on RiscV Compilation

The RiscV CC is in `~/tools/xpack-riscv-none-elf-gcc-13.2.0-2/bin`, which is alread in `PATH`.