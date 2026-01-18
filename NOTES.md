## VexRiscv Murax Demo

### Video Demo
[![YouTube Video](http://img.youtube.com/vi/z0yJHjjw2JI/0.jpg)](http://www.youtube.com/watch?v=z0yJHjjw2JI)


### Pitfalls
- **EVERYTHING** has to be run in Linux. 
- When using WSL, work in a Linux directory.

### Steps
1. Install dependencies
    - JDK 17.0.7 (minimum version: JDK 8)
    - sbt 1.11.7
    - Verilator 5.006 (minimum version: 3.9)

2. In the repo root, run:

```bash
# Generate Murax Verilog
sbt "runMain vexriscv.demo.Murax"
# or Murax with echoing over UART + blinking LEDs:
sbt "runMain vexriscv.demo.MuraxWithRamInit"
```

3. To simulate using Verilator, run:

```bash
cd src/test/cpp/murax
make clean run
```

That flow:

* Verilates Murax.
* Loads some default firmware image (ROM/RAM init).
* Runs the simulation and shows UART output in the console.

---

# iCEBreaker VexRiscv Walkthrough

This guide explains how to build and flash the VexRiscv Murax demo on the iCEBreaker FPGA board.

## Prerequisites

Ensure you have the toolchain installed (done via `scripts/setup_toolchain.sh`) and added to your PATH.
```bash
export PATH=$HOME/oss-cad-suite/bin:$PATH
```

## Building the Bitstream

1. Navigate to the iCEBreaker script directory:
   ```bash
   cd scripts/Murax/iCEBreaker
   ```

2. Run the make command:
   ```bash
   make
   ```
   This will generate `bin/toplevel.bin`.

## Flashing the FPGA

Differs per board. Just find a way to flash the bitstream to the FPGA.

### iCEBreaker

1. Connect your iCEBreaker board via USB.
2. Run the programming command:
   ```bash
   make prog
   ```
   *Note: If you are on WSL, ensure the USB device is attached to WSL using `usbipd` (Windows) or similar tools. See the USB Passthrough section below.*

### USB Passthrough (WSL2)

If you are using WSL2, you must forward the USB device from Windows to Linux.

1.  **Install usbipd-win** on Windows (from [GitHub](https://github.com/dorssel/usbipd-win/releases)).
2.  Open **PowerShell as Administrator**.
3.  **List Devices** to find the Bus ID of your iCEBreaker:
    ```powershell
    usbipd list
    ```
4.  **Bind the Device** (only needed once):
    ```powershell
    usbipd bind --busid <BUSID>
    ```
    (Replace `<BUSID>` with the ID from step 3, e.g., `1-1`).
5.  **Attach to WSL**:
    ```powershell
    usbipd attach --wsl --busid <BUSID>
    ```
    Now the device should appear in WSL as `/dev/ttyUSBx`.

## Verifying Output

### LEDs
The "Hello World" firmware basically blinks the LEDs.
- **Red LED**: Corresponds to GPIO 0.
- **Green LED**: Corresponds to GPIO 1.
They should blink in sequence.

### UART (Serial Console)
The firmware also prints "hello world arty a7 v1" (legacy string) to the serial port.

#### PuTTY Setup
To see the serial output on Windows:

1.  **Open Device Manager** on Windows and find the COM port for "iCEBreaker" (or FTDI device). usually `COMx` (e.g., COM3).
2.  **Open PuTTY**.
3.  **Session** Settings:
    *   **Connection type**: Serial
    *   **Serial line**: `COM3` (Replace with your actual port).
    *   **Speed**: `115200`.
4.  Click **Open**.
5.  Press the **User Button** (BTN_N) or Reset button on the iCEBreaker to restart the CPU and see the message again.

> [!TIP]
> If using WSL, you can also use `screen /dev/ttyUSB1 115200` if the device is passed through as `/dev/ttyUSB1`.

---

# Custom Instructions

This guide provides a comprehensive technical breakdown of how to extend the RISC-V ISA within the VexRiscv ecosystem.

---

## 1. Architectural Overview: The Plugin System

VexRiscv is not a static Verilog file; it is a **SpinalHDL hardware generator**. Every processor feature (ALU, CSRs, Caches) is a **Plugin**. When adding a custom instruction, you are essentially "hooking" a new execution unit into the base pipeline.

### The Pipeline Hook
VexRiscv uses a standard 5-stage pipeline (Fetch, Decode, Execute, Memory, Writeback). Your custom logic will primarily affect the **Decode** stage (to identify the instruction) and the **Execute** stage (to compute the result).

---

## 2. Instruction Encoding & The Decoder

To avoid conflicts with the standard RISC-V ISA, we use the `Custom-0` opcode space (`0001011`).

### R-Type Instruction Format
We implement the `y = mx + b` instruction using the **R-Type** (Register-Register) format:
`| funct7 (7b) | rs2 (5b) | rs1 (5b) | funct3 (3b) | rd (5b) | opcode (7b) |`

### The Masked Literal
In SpinalHDL, we define the "identity" of our instruction using a `MaskedLiteral` string. This allows the decoder to ignore the register fields:
`"0000000----------000-----0001011"`
- `0000000`: `funct7` (Distinguishes our instruction from others in Custom-0).
- `----------`: `rs2` and `rs1` (Operand locations).
- `000`: `funct3` (Additional sub-opcode space).
- `-----`: `rd` (Destination register location).
- `0001011`: `opcode` (Custom-0).

---

## 3. Implementing the Plugin (Hardware)

A VexRiscv plugin has two critical methods: `setup` and `build`.

### The `setup` Method: Decoder Configuration
This phase "claims" the bit pattern and tells the processor which control signals to assert when this instruction is decoded.

```scala
// Inside MxPlusBPlugin.scala
decoderService.add(
  key = MaskedLiteral(instructionPattern),
  List(
    IS_MX_PLUS_B             -> True, // Our custom "active" flag
    REGFILE_WRITE_VALID      -> True, // Trigger the Register File write logic
    BYPASSABLE_EXECUTE_STAGE -> True, // Optimization: result ready in Execute
    RS1_USE                  -> True  // Tell the scheduler we need data from RS1
  )
)
```

### The `build` Method: Datapath Implementation
This phase implements the actual logic. We "plug" our logic into the `execute` stage area.

```scala
execute plug new Area {
  // 1. Retrieve the value of RS1 from the pipeline
  val rs1 = execute.input(RS1).asUInt
  
  // 2. Compute y = mx + b (m and b are synthesis-time constants)
  val result = ((rs1 * U(m, 32 bits)).resize(32) + U(b, 32 bits)).resize(32)

  // 3. Drive the result into the REGFILE_WRITE_DATA path
  when(execute.input(IS_MX_PLUS_B)) {
    execute.output(REGFILE_WRITE_DATA) := result.asBits
  }
}
```

---

## 4. Software Interface (C & Assembly)

To use the instruction in C, we wrap the RISC-V `.insn` directive in a macro. This is cleaner than naked `asm` blocks.

### Inline Assembly Breakdown
```c
#define custom_mx_plus_b(rd, rs1) \
    asm volatile ( \
        ".insn r 0x0B, 0, 0, %0, %1, x0" \
        : "=r"(rd) \
        : "r"(rs1) \
    )
```
- `.insn r`: R-Type encoding.
- `0x0B`: Opcode (decimal 11).
- `0, 0`: `funct3` and `funct7`.
- `%0, %1`: Compiler-selected registers for `rd` and `rs1`.
- `x0`: Constant register 0 for the unused `rs2` field.

---

## 5. Deployment Flow: The Hardware-Software Link

One unique aspect of SoC development on iCEBreaker is how programs are loaded:

1.  **C Compiler**: Generates a `.hex` file.
2.  **SpinalHDL**: During Verilog generation, it reads that `.hex` file and generates a Verilog `initial` block that pre-loads the processor's Block RAM (BRAM).
3.  **Synthesis**: Yosys takes that initialized BRAM and maps it to physical FPGA memory.

> [!IMPORTANT]
> Since the software is literally etched into the hardware's initial state, **any change to the C code requires a full hardware rebuild** (Verilog -> Bitstream -> Flash).

---

## 7. Common Pitfalls

1.  **Plugin Instance Reuse**: Always instantiate your configuration and plugins **INSIDE** the `SpinalVerilog` block. Reusing plugin instances across elaboration attempts (which can happen automatically) triggers `AssertionError` in the decoder maps.
2.  **Bit Width Mismatches**: SpinalHDL multiplication (e.g., `rs1 * rs2`) produces a result with the sum of the input widths (e.g., 64 bits for 32x32). You must explicitly `.resize(32)` the result before assigning it to `REGFILE_WRITE_DATA`.
3.  **Bypassing**: Always set `BYPASSABLE_EXECUTE_STAGE -> True` in the decoder if your instruction is single-cycle. This ensures subsequent instructions can use the result immediately without a pipeline stall.

Refer to [PITFALLS.md](PITFALLS.md) for more detailed explanations.
