package vexriscv.demo

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba3.apb.{Apb3, Apb3SlaveFactory}

/**
 * BlockDialectDecoderAPB — APB3 wrapper for 32-lane BD4 weight decoding.
 *
 * Instantiates one DialectFP4DecodeCore per element lane (32 total), all
 * purely combinational.  The APB register interface is unchanged from the
 * original BlockDialectDecoder, so no firmware changes are required.
 *
 * Register map (from APB base, e.g. 0x40030000):
 *   0x00  META      (W)  dialect_id[15:12] | shared_exp[11:7] | pad[6:0]
 *   0x04  PACKED0   (W)  packed codes bytes 0–3
 *   0x08  PACKED1   (W)  packed codes bytes 4–7
 *   0x0C  PACKED2   (W)  packed codes bytes 8–11
 *   0x10  PACKED3   (W)  packed codes bytes 12–15
 *   0x20–0x3C  DECODED0–7 (R)  32 decoded int8 half-units (4 per word, little-endian)
 *   0x40  STATUS    (R)  bit 0 = 1 (always ready)
 *   0x44  SHARED_EXP (R) shared_exp_bits[4:0] (from META)
 *
 * Architecture:
 *   Each of the 32 element lanes has one DialectFP4DecodeCore instance.
 *   The 4-bit code for lane i is extracted from the packed registers
 *   (2 codes per byte, low nibble = even element, high nibble = odd element,
 *   little-endian byte order within each packed word).
 *
 * This file was produced by Milestone 4 of the ResNet-1202 FPGA plan:
 * splitting the lookup logic (DialectFP4DecodeCore) from the APB bus glue
 * (BlockDialectDecoderAPB) to enable re-use of the core in BDMac32 (M5).
 */
class BlockDialectDecoderAPB extends Component {
  val io = new Bundle {
    val apb = slave(Apb3(
      addressWidth = 8,
      dataWidth = 32
    ))
  }

  // ── Input registers ───────────────────────────────────────────────────────
  val metaReg    = Reg(UInt(16 bits)) init(0)
  val packedRegs = Vec(Reg(Bits(32 bits)) init(0), 4)

  // ── Metadata fields ───────────────────────────────────────────────────────
  val dialectId = metaReg(15 downto 12)
  val sharedExp = metaReg(11 downto  7)

  // ── 32-lane decode ────────────────────────────────────────────────────────
  val decodedBytes = Vec(Bits(8 bits), 32)

  for (elemIdx <- 0 until 32) {
    val byteIdx      = elemIdx / 2
    val regIdx       = byteIdx / 4
    val byteInReg    = byteIdx % 4
    val isHighNibble = (elemIdx % 2) == 0

    val packedByte = packedRegs(regIdx)((byteInReg * 8 + 7) downto (byteInReg * 8))
    val code = (if (isHighNibble) packedByte(7 downto 4) else packedByte(3 downto 0)).asUInt

    val sign  = code(3)
    val index = code(2 downto 0)

    // One DialectFP4DecodeCore per lane — purely combinational
    val decCore = new DialectFP4DecodeCore()
    decCore.io.dialect_id := dialectId
    decCore.io.idx        := index
    val halfUnits = decCore.io.mag

    val signedVal   = SInt(8 bits)
    val halfUnitsS8 = halfUnits.resize(8).asSInt
    when(sign) {
      signedVal := -halfUnitsS8
    } otherwise {
      signedVal := halfUnitsS8
    }

    decodedBytes(elemIdx) := signedVal.asBits
  }

  // ── Pack 4 bytes per 32-bit read word, little-endian ─────────────────────
  val decodedWords = Vec(Bits(32 bits), 8)
  for (w <- 0 until 8) {
    val b0 = decodedBytes(w * 4 + 0)
    val b1 = decodedBytes(w * 4 + 1)
    val b2 = decodedBytes(w * 4 + 2)
    val b3 = decodedBytes(w * 4 + 3)
    decodedWords(w) := b3 ## b2 ## b1 ## b0
  }

  // ── APB3 bus interface ────────────────────────────────────────────────────
  val busCtrl = Apb3SlaveFactory(io.apb)
  busCtrl.write(metaReg, 0x00)
  for (i <- 0 until 4) {
    busCtrl.write(packedRegs(i), 0x04 + i * 4)
  }
  for (i <- 0 until 8) {
    busCtrl.read(decodedWords(i), 0x20 + i * 4)
  }
  busCtrl.read(B(1, 32 bits), 0x40)
  busCtrl.read(sharedExp.asBits.resize(32), 0x44)
}
