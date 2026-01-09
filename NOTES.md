## Git Clone -> Simulate Demo Program

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

1. Connect your iCEBreaker board via USB.
2. Run the programming command:
   ```bash
   make prog
   ```
   *Note: If you are on WSL, ensure the USB device is attached to WSL using `usbipd` (Windows) or similar tools. See the USB Passthrough section below.*

## USB Passthrough (WSL2)

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

