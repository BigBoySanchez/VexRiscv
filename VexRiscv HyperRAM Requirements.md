# Starting Design Doc (v0.1)

## iCEBreaker \+ VexRiscv — Two-Phase “Before vs After” Demo (Baseline → BlockDialect)

**Document status:** starter design for implementing the requirements in *Requirements v0.5*  
**Primary goal:** produce a **comparison** demo:

- **Phase A:** VexRiscv-only, **simulation-only** baseline  
- **Phase B:** VexRiscv \+ **BlockDialect-Lite** \+ **custom instruction**, simulation \+ *(stretch)* FPGA run

---

## 1\) Goals and non-goals

### 1.1 Goals

1. Demonstrate a **\>1,000,000 parameter** inference workload on a minimal RISC‑V SoC.  
2. Establish a **baseline ceiling** (Phase A) using only VexRiscv \+ firmware.  
3. Demonstrate **measurable improvements** (Phase B) using BlockDialect-Lite:  
   - fewer flash bytes read per inference and/or  
   - fewer cycles per inference  
4. Keep the architecture modular so flash can be swapped for **HyperRAM later** via a WeightStore abstraction.

### 1.2 Non-goals (for this 9-day sprint)

- Full BlockDialect feature parity (activation-side online quantization, all dialects, etc.).  
- Highest possible performance; correctness \+ comparison-first.  
- Production-grade QSPI controller (simulation-first; FPGA is stretch).

---

## 2\) System overview

### 2.1 High-level blocks

- **CPU:** VexRiscv (RV32IM, minimal config acceptable)  
- **SoC fabric:** simple memory-mapped bus (Wishbone/AXI-lite style depending on existing codebase)  
- **On-chip RAM:** small scratch (for firmware \+ activation tiles \+ output buffers)  
- **UART:** console output for logs/metrics  
- **Weight storage:**  
  - **Phase A:** simulation-only ROM/flash stub (readmem)  
  - **Phase B:** same in sim, plus optional SPI/QSPI read path for FPGA stretch

### 2.2 Two configurations

- **Phase A config:** CPU only \+ RAM \+ UART \+ WeightStore stub  
  *No BlockDialect RTL, no accelerator RTL.*  
- **Phase B config:** CPU \+ RAM \+ UART \+ WeightStore \+ BlockDialect decoder \+ custom instruction accelerator

---

## 3\) Inference workload

### 3.1 Default workload

**ResNet-110 (CIFAR-style)** as the main target (parameter count \>1M).

### 3.2 Fallback workload (schedule safety)

A **\>1M parameter MLP** (e.g., large hidden layer) if convolution datapath risk becomes too high.

### 3.3 Determinism

For both phases:

- fixed test input (embedded array or deterministic generator)  
- fixed weight blob  
- expected output signature (hash or a few reference logits)

---

## 4\) Memory map (proposed)

Exact addresses can be adapted to existing SoC template. This section defines intent.

- **0x0000\_0000 – 0x0000\_FFFF:** Boot ROM (optional; could boot from RAM in sim)  
- **0x1000\_0000 – 0x1000\_FFFF:** On-chip RAM (scratch \+ firmware; size board-dependent)  
- **0x2000\_0000 – 0x20FF\_FFFF:** WeightStore window (read-only view into flash/ROM)  
- **0x4000\_0000 – 0x4000\_0FFF:** UART registers  
- **0x5000\_0000 – 0x5000\_0FFF:** (Phase B) BlockDialect decoder control/status  
- **0x6000\_0000 – 0x6000\_0FFF:** (Phase B) Accelerator / custom instruction CSRs (if any MMIO is needed)

**Design preference:** avoid per-weight MMIO reads; use streaming/FIFO interfaces.

---

## 5\) Phase A design (VexRiscv-only simulation)

### 5.1 WeightStore stub (simulation)

Implement a simple read-only region backing **weights \+ metadata** via:

- `$readmemh` initialized memory (Verilog ROM)  
- or a behavioral “flash” model if already present

**Firmware API:**

- `const uint8_t* weights = (const uint8_t*)WEIGHT_BASE;`  
- `memcpy()` or direct loads

### 5.2 Baseline weight format (Phase A)

**Option A (recommended):** int8 tensors with per-tensor scale (or per-channel if needed).  
**Option B:** int16 if accuracy/debugging needs it.

**Binary layout (baseline blob):**

- Header:  
  - magic `VWB0` (Vex Weights Baseline v0)  
  - version  
  - model ID  
  - tensor count  
  - offsets table  
  - CRC32 (optional)  
- Tensor entries:  
  - name hash / id  
  - shape (up to 4D)  
  - dtype (int8/int16)  
  - scale (Q-format or float32)  
  - raw bytes

### 5.3 Baseline inference (firmware)

- Implement fixed-point ops in C.  
- Use tiling to fit activations in RAM.  
- Print:  
  - bytes read (tracked by pointer deltas, or explicit counter)  
  - cycles (use `rdcycle` if enabled; else instruction count proxy or sim ticks)  
  - output signature

**Phase A principle:** “If it’s not firmware, it doesn’t exist.”

---

## 6\) Phase B design (BlockDialect-Lite \+ custom instruction)

### 6.1 WeightStore abstraction (Phase B-ready)

Define a clean internal interface so flash/HyperRAM can swap later.

**Stream interface (RTL):**

- Request channel:  
  - `req_valid, req_ready`  
  - `req_addr` (byte address)  
  - `req_len` (bytes)  
- Response stream:  
  - `rsp_valid, rsp_ready`  
  - `rsp_data[7:0]` (or wider if convenient)  
  - `rsp_last` (end of burst)  
- Status:  
  - `busy, error`

**In simulation:** a simple ROM-backed responder that honors bursts.  
**On FPGA (stretch):** SPI/QSPI read controller producing the same stream.

### 6.2 BlockDialect-Lite format (practical subset)

Implement the **weight-side** concept:

- Blockwise packed 4-bit codes  
- Per-block metadata selecting a small FP4 “dialect” (format ID)  
- Optional block scale

**Recommended minimal dialect set (example):**

- `D0`: e2m1-like (wider exponent)  
- `D1`: e1m2-like (more mantissa)  
- `D2`: signed int4-like (uniform)  
- `D3`: reserved (future)

**Block size:** 32 or 64 weights per block (pick one and stick to it).

**Per-block metadata (minimum):**

- `dialect_id` (2–3 bits)  
- `scale` (optional, e.g., int8 or small float-ish)

The exact encoding is open, but must be fully specified and tested.

### 6.3 Offline toolchain (host-side)

Create a small tool (Python preferred) that:

1. Loads baseline tensors (e.g., from a known file format or from exported arrays)  
2. Packs them into BlockDialect-Lite:  
   - choose dialect per block  
   - pack 4-bit codes  
   - emit metadata  
3. Writes:  
   - a single flash-friendly blob  
   - a JSON sidecar (for debugging only) containing tensor offsets/shapes

### 6.4 RTL BlockDialect decoder

**Inputs:** WeightStore byte stream  
**Outputs:** decoded weight stream (signed int8/int16) for MAC consumption

Decoder responsibilities:

- parse tensor headers/offsets for the requested tensor region (or rely on firmware to provide offsets)  
- for each block:  
  - read metadata  
  - read packed codes  
  - expand to weight values (int form)  
  - apply scale if present  
- expose a small FIFO (elastic buffer) to smooth bursts

**Verification:**

- build a software reference decoder (same Python tool)  
- randomized tests:  
  - generate random blocks  
  - encode → decode (SW)  
  - decode (RTL) and compare bit-exact output stream  
- minimum: thousands of blocks across dialect IDs

---

## 7\) Custom instruction / accelerator (Phase B)

### 7.1 Why custom instruction here

To demonstrate BlockDialect’s *system-level* value, we want:

- streaming decoded weights consumed efficiently, and  
- measurable cycle reduction vs firmware-only baseline

### 7.2 Candidate accelerator primitive

**Tile MAC primitive**:

- Multiply-accumulate a small activation tile with a corresponding weight tile  
- Emit partial sums to scratch

**Minimal supported op (enough for demo):**

- dot product (1D) or small matrix-vector tile  
- later extensible to conv lowering (im2col) if time allows

### 7.3 Integration approach

Two acceptable options:

**Option A: VexRiscv plugin “CUSTOM” instruction**

- add a custom opcode that:  
  - launches a multi-cycle operation  
  - stalls CPU until completion (or uses busy flag and polling)  
- operands encode pointers/length/flags

**Option B: Memory-mapped accelerator \+ optional custom instruction**

- firmware writes MMIO registers to configure  
- firmware triggers start  
- firmware polls done / reads result  
- still qualifies if the operation is multi-cycle and cleanly integrated

**Requirement must-haves:**

- multi-cycle FSM with `busy/done`  
- handles multiple inputs/outputs (e.g., src ptr, dst ptr, len, status)

### 7.4 Proposed register/operand set (example)

Inputs:

- `A_ptr` (activation tile in RAM)  
- `W_ptr` (weight tensor region / stream source)  
- `Y_ptr` (output/partial sums in RAM)  
- `K` (tile length)  
- `flags` (mode, accumulation, scaling)

Outputs:

- `status` (done/error)  
- optional: `cycles_used` counter for the accelerator itself

---

## 8\) Firmware architecture

### 8.1 Common code (both phases)

- UART logging  
- minimal runtime (no malloc if avoidable)  
- tensor table parsing (offsets/shapes)  
- tiling controller for activations

### 8.2 Phase A firmware path

- read baseline weights directly from WeightStore window  
- pure-C compute

### 8.3 Phase B firmware path

- read tensor table/offsets  
- configure BlockDialect decoder or accelerator to stream decoded weights  
- call custom instruction or MMIO accelerator for tile operations  
- fall back to pure-C for any unsupported ops (allowed, but document it)

---

## 9\) Simulation & build flows

### 9.1 Simulation flow (Phase A)

1. Build SoC RTL \+ VexRiscv  
2. Load firmware into RAM/ROM  
3. Load baseline weight blob into WeightStore stub memory  
4. Run test; capture UART output (signature \+ metrics)

### 9.2 Simulation flow (Phase B)

1. Build SoC RTL with decoder \+ accelerator  
2. Load firmware  
3. Load BlockDialect-Lite blob  
4. Run:  
   - decoder unit tests (can be separate top)  
   - end-to-end inference sim  
5. Capture before/after metrics

---

## 10\) FPGA flow (stretch)

- Replace WeightStore stub with real SPI/QSPI flash reader (if feasible in time)  
- Program iCEBreaker bitstream  
- Place firmware \+ weight blob in flash layout  
- UART shows:  
  - blob magic/version ok  
  - deterministic output signature  
  - metrics

If FPGA run slips:

- ensure the sim demo is polished and comparison is clear.

---

## 11\) Measurement plan (comparison-first)

Log in both phases:

- **bytes read** from WeightStore per inference  
- **cycles** to completion (rdcycle preferred)  
- peak scratch usage (tracked by buffers or instrumentation)  
- output signature (hash / logits)

**Expected outcomes (directional):**

- Phase B should reduce weight bytes by \~2× vs int8 (plus metadata)  
- Phase B should reduce cycles if accelerator is effective

---

## 12\) Verification strategy

### 12.1 Unit tests

- BlockDialect encoder/decoder consistency tests (SW-only)  
- RTL decoder vs SW reference (randomized)  
- Accelerator directed tests (small vectors with known outputs)

### 12.2 End-to-end tests

- fixed input \+ expected output signature for:  
  - Phase A config  
  - Phase B config

---

## 13\) Risks and mitigations

1. **QSPI complexity for FPGA stretch**  
   - Mitigation: keep FPGA run as stretch; prioritize sim completeness.  
2. **Convolution datapath complexity**  
   - Mitigation: allow \>1M MLP fallback; keep comparison narrative intact.  
3. **Bit-exact decode mismatches**  
   - Mitigation: lock down packing/endianness early; build randomized tests day 1 of decoder work.  
4. **Too little on-chip RAM for activations**  
   - Mitigation: aggressive tiling; stream weights; keep activations minimal.

---

## 14\) Open design decisions (choose early)

- Block size (32 vs 64\) and exact metadata fields  
- Output representation of decoder (int8 vs int16)  
- Custom instruction vs MMIO accelerator integration style  
- Workload: ResNet-110 vs \>1M MLP fallback

---

## 15\) Definition of Done for the design

This doc is “good enough to build” when:

- Weight blob formats (baseline \+ BlockDialect-Lite) are fully specified  
- WeightStore streaming interface is nailed down  
- Accelerator interface (instruction operands or MMIO map) is defined  
- Test plan exists for decoder \+ end-to-end runs  
- Metrics are defined and printed consistently in both phases

---

*End of starting design doc.*  
