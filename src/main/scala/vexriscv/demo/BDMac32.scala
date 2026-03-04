package vexriscv.demo

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba3.apb.{Apb3, Apb3SlaveFactory}

/**
 * BDMac32 — APB3 32-element BlockDialect fused multiply-accumulate peripheral.
 *
 * Computes one 32-element BD4 dot product per invocation.  Uses a sequential
 * state machine with two shared DialectFP4DecodeCore instances (one for weights,
 * one for activations), processing one element per clock cycle.  This keeps
 * the design within the iCE40UP5K LC budget (5280 LCs).
 *
 * Latency: write CTRL=1, then poll DONE.  Done goes high after 32 clock cycles
 * (~2.7 µs at 12 MHz) — negligible overhead compared to the APB bus transactions.
 *
 * === Firmware usage pattern ===
 * {{{
 *   // 1. Write weight block (16 bytes codes + 2 bytes meta)
 *   BDMAC32->W_PACKED0 = w[0..3];   BDMAC32->W_PACKED1 = w[4..7];
 *   BDMAC32->W_PACKED2 = w[8..11];  BDMAC32->W_PACKED3 = w[12..15];
 *   BDMAC32->W_META    = w_meta16;  // dialect_id[15:12] | shared_exp[11:7]
 *
 *   // 2. Write activation block
 *   BDMAC32->A_PACKED0 = a[0..3];   BDMAC32->A_PACKED1 = a[4..7];
 *   BDMAC32->A_PACKED2 = a[8..11];  BDMAC32->A_PACKED3 = a[12..15];
 *   BDMAC32->A_META    = a_meta16;
 *
 *   // 3. Trigger and poll
 *   BDMAC32->CTRL = 1;
 *   while (!(BDMAC32->DONE & 1)) {}   // ~32 cycles (~2.7 µs @ 12 MHz)
 *
 *   // 4. Read result
 *   int32_t partial = (int32_t) BDMAC32->PARTIAL_SUM;
 *   uint32_t expsum = BDMAC32->EXP_SUM & 0x3F;
 *
 *   // 5. Firmware scaling (paper §3.4)
 *   int64_t scaled = (int64_t)partial * (1LL << expsum);
 *   scaled >>= 2;   // 0.5 * 0.5 = 1/4 per half-unit multiply
 * }}}
 *
 * === Register map (base = 0x40031000) ===
 * {{{
 *  Offset  Name          Dir  Bits   Description
 *  0x00    W_PACKED0      W   31:0   w_packed[31:0]   — elements 0..7
 *  0x04    W_PACKED1      W   31:0   w_packed[63:32]  — elements 8..15
 *  0x08    W_PACKED2      W   31:0   w_packed[95:64]  — elements 16..23
 *  0x0C    W_PACKED3      W   31:0   w_packed[127:96] — elements 24..31
 *  0x10    W_META         W   15:0   dialect_id[15:12] | shared_exp[11:7]
 *  0x14    A_PACKED0      W   31:0   a_packed[31:0]
 *  0x18    A_PACKED1      W   31:0   a_packed[63:32]
 *  0x1C    A_PACKED2      W   31:0   a_packed[95:64]
 *  0x20    A_PACKED3      W   31:0   a_packed[127:96]
 *  0x24    A_META         W   15:0   dialect_id[15:12] | shared_exp[11:7]
 *  0x28    CTRL           W    0     write 1 to start; clears DONE and begins 32-cycle compute
 *  0x30    PARTIAL_SUM    R   31:0   signed result (sum of 32 products, in half-units²)
 *  0x34    EXP_SUM        R    5:0   w_shared_exp + a_shared_exp (for firmware scaling)
 *  0x38    DONE           R    0     0 while computing, 1 when result is valid
 * }}}
 *
 * === Packed code layout ===
 *   Byte i of packed = (code[2i] << 4) | code[2i+1]
 *   code[j][3]   = sign  (1 = negative)
 *   code[j][2:0] = index into dialect magnitude table (0..7)
 *
 * === Inner computation per element i (sequential) ===
 *   w_mag   = DialectFP4DecodeCore(w_dialect_id, w_code[i][2:0])
 *   a_mag   = DialectFP4DecodeCore(a_dialect_id, a_code[i][2:0])
 *   product_mag  = w_mag * a_mag        (4b × 4b → 8b unsigned, max 225)
 *   product_sign = w_code[i][3] XOR a_code[i][3]
 *   product      = product_sign ? −product_mag : +product_mag  (signed)
 *   accum       += product              (signed 24-bit)
 *   On last element (i=31): latch accum → partialSumReg, assert done
 *
 * Design notes:
 *   The combinational (64-lane parallel) design required ~8060 LCs on the
 *   iCE40UP5K (152% of the 5280 LC capacity).  The sequential design uses
 *   only 2 DialectFP4DecodeCore instances + 1 multiplier + a 24-bit accumulator,
 *   reducing the LC count by ~2000–3000 LCs.
 *
 * This module was introduced in Milestone 5 of the ResNet-1202 FPGA plan.
 * It depends on DialectFP4DecodeCore (Milestone 4).
 */
class BDMac32 extends Component {
  val io = new Bundle {
    val apb = slave(Apb3(
      addressWidth = 8,
      dataWidth    = 32
    ))
  }

  // ── Input registers ───────────────────────────────────────────────────────
  val wPackedRegs = Vec(Reg(Bits(32 bits)) init(0), 4)
  val wMetaReg    = Reg(UInt(16 bits)) init(0)

  val aPackedRegs = Vec(Reg(Bits(32 bits)) init(0), 4)
  val aMetaReg    = Reg(UInt(16 bits)) init(0)

  // ── Metadata extraction ───────────────────────────────────────────────────
  val wDialectId = wMetaReg(15 downto 12)
  val wSharedExp = wMetaReg(11 downto  7)

  val aDialectId = aMetaReg(15 downto 12)
  val aSharedExp = aMetaReg(11 downto  7)

  // ── Pre-wire all 32 4-bit codes from packed registers ────────────────────
  // Pure wiring (no LUTs); synthesises to direct bit connections.
  // The runtime MUX selecting codes(counter) costs ~20 LUTs per port.
  val wCodes = Vec(Bits(4 bits), 32)
  val aCodes = Vec(Bits(4 bits), 32)

  for (elemIdx <- 0 until 32) {
    val byteIdx      = elemIdx / 2
    val regIdx       = byteIdx / 4
    val byteInReg    = byteIdx % 4
    val isHighNibble = (elemIdx % 2) == 0

    def nibble(regs: Vec[Bits]): Bits = {
      val b = regs(regIdx)((byteInReg * 8 + 7) downto (byteInReg * 8))
      if (isHighNibble) b(7 downto 4) else b(3 downto 0)
    }
    wCodes(elemIdx) := nibble(wPackedRegs)
    aCodes(elemIdx) := nibble(aPackedRegs)
  }

  // ── Sequential state machine ─────────────────────────────────────────────
  // States: idle (running=0, done=0), running (running=1, done=0), done (running=0, done=1)
  val counter = Reg(UInt(5 bits)) init(0)
  val running = Reg(Bool())       init(False)
  val doneReg = Reg(Bool())       init(False)
  val accum   = Reg(SInt(24 bits)) init(0)

  // Output result registers — updated atomically when the last element is processed
  val partialSumReg = Reg(SInt(32 bits)) init(0)
  val expSumReg     = Reg(UInt(6 bits))  init(0)

  // ── Shared decode instances — driven by counter-selected code each cycle ──
  val wCode_i = wCodes(counter)
  val aCode_i = aCodes(counter)

  val wSign = wCode_i(3)
  val wIdx  = wCode_i(2 downto 0).asUInt
  val aSign = aCode_i(3)
  val aIdx  = aCode_i(2 downto 0).asUInt

  val wCore = new DialectFP4DecodeCore()
  wCore.io.dialect_id := wDialectId
  wCore.io.idx        := wIdx
  val wMag = wCore.io.mag   // UInt(4 bits)

  val aCore = new DialectFP4DecodeCore()
  aCore.io.dialect_id := aDialectId
  aCore.io.idx        := aIdx
  val aMag = aCore.io.mag   // UInt(4 bits)

  // product_mag: UInt(9 bits), max 225; product: SInt(10 bits), −225..+225
  val productMag  = (wMag * aMag).resize(9)
  val productSign = wSign ^ aSign
  val product     = SInt(10 bits)
  when(productSign) {
    product := -(productMag.asSInt).resize(10)
  } otherwise {
    product := productMag.asSInt.resize(10)
  }

  // ── State transitions and accumulation ───────────────────────────────────
  // start signal: APB write to CTRL (0x28) with bit 0 set
  val busCtrl   = Apb3SlaveFactory(io.apb)
  // Trigger on any write to CTRL (firmware always writes 1; any write suffices)
  val startPulse = busCtrl.isWriting(0x28)

  when(startPulse) {
    // Restart: clear accumulator and state, begin processing element 0
    accum   := 0
    counter := 0
    running := True
    doneReg := False
  } .elsewhen(running) {
    val nextAccum = accum + product.resize(24)
    accum   := nextAccum
    counter := counter + 1
    when(counter === 31) {
      // Last element: latch result and transition to done
      running       := False
      doneReg       := True
      partialSumReg := nextAccum.resize(32)
      expSumReg     := (wSharedExp +^ aSharedExp).resize(6)
    }
  }

  // ── APB3 register map ─────────────────────────────────────────────────────
  for (i <- 0 until 4) busCtrl.write(wPackedRegs(i), 0x00 + i * 4)
  busCtrl.write(wMetaReg, 0x10)
  for (i <- 0 until 4) busCtrl.write(aPackedRegs(i), 0x14 + i * 4)
  busCtrl.write(aMetaReg, 0x24)
  // 0x28 CTRL handled above via startPulse / busCtrl.isWriting

  busCtrl.read(partialSumReg.asBits,           0x30)
  busCtrl.read(expSumReg.asBits.resize(32),    0x34)
  busCtrl.read(doneReg.asBits.resize(32),      0x38)
}

/** Convenience elaboration entry point (standalone Verilog generation). */
object BDMac32 {
  def main(args: Array[String]): Unit = {
    SpinalVerilog(new BDMac32())
  }
}
