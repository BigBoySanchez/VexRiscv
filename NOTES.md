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

### Extra Notes (Not Important for "Hello, World" Demo)
1. dependencies
    - JDK 17.0.7
    - sbt 1.11.7
    - Verilator 5.006
3. follow custom instruction example
4. run regression test (to ensure prev instructions didn't break; 
ideally on a single config to save time, runs all configs otherwise)

5. write and run tests for custom instruction separately
6. flash code to a real fpga (TODO: need more research on 
does prof actually need it/how to do this/hardware requirements)
