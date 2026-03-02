package vexriscv.demo

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba3.apb.{Apb3, Apb3SlaveFactory}

/**
 * BlockDialect-Lite hardware decoder — APB3 peripheral.
 *
 * Register map (from APB base):
 *   0x00  META     (W)   dialect_id[15:12] | shared_exp[11:7] | pad[6:0]
 *   0x04  PACKED0  (W)   packed codes bytes 0–3
 *   0x08  PACKED1  (W)   packed codes bytes 4–7
 *   0x0C  PACKED2  (W)   packed codes bytes 8–11
 *   0x10  PACKED3  (W)   packed codes bytes 12–15
 *   0x20–0x3C  DECODED0–7 (R)   32 decoded int8 half-units (signed, packed 4 per word, little-endian)
 *   0x40  STATUS   (R)   bit 0 = 1 (always ready)
 *   0x44  SHARED_EXP (R) shared_exp_bits[4:0] (from META)
 */
class BlockDialectDecoder extends Component {
  val io = new Bundle {
    val apb = slave(Apb3(
      addressWidth = 8,
      dataWidth = 32
    ))
  }

  // Variant table for idx==6 (dialects 0..14). Dialect 15 is special-cased.
  private val variantIdx6 = Vec(Seq(
    11, 9, 11, 9, 10, 8, 10, 8,
     9, 7,  9, 7,  8, 7,  7
  ).map(v => U(v, 4 bits)))

  private def decodeHalfUnits(dialectId: UInt, index: UInt): UInt = {
    val halfUnits = UInt(4 bits)
    halfUnits := 0

    when(dialectId === U(15)) {
      switch(index) {
        is(0) { halfUnits := U(0) }
        is(1) { halfUnits := U(1) }
        is(2) { halfUnits := U(2) }
        is(3) { halfUnits := U(3) }
        is(4) { halfUnits := U(4) }
        is(5) { halfUnits := U(5) }
        is(6) { halfUnits := U(6) }
        is(7) { halfUnits := U(8) }
      }
    } otherwise {
      val pair = (dialectId >> 1).resized
      val maxHU = (U(15, 4 bits) - pair).resized

      switch(index) {
        is(0) { halfUnits := U(0) }
        is(1) { halfUnits := U(1) }
        is(2) { halfUnits := U(2) }
        is(3) { halfUnits := U(3) }
        is(4) { halfUnits := U(4) }
        is(5) { halfUnits := U(6) }
        is(6) { halfUnits := variantIdx6(dialectId) }
        is(7) { halfUnits := maxHU }
      }
    }

    halfUnits
  }

  // Input registers
  val metaReg    = Reg(UInt(16 bits)) init(0)
  val packedRegs = Vec(Reg(Bits(32 bits)) init(0), 4)

  // Metadata fields
  val dialectId = metaReg(15 downto 12)
  val sharedExp = metaReg(11 downto  7)


  // Decode all 32 elements into bytes
  val decodedBytes = Vec(Bits(8 bits), 32)

  for (elemIdx <- 0 until 32) {
    val byteIdx       = elemIdx / 2
    val regIdx        = byteIdx / 4
    val byteInReg     = byteIdx % 4
    val isHighNibble  = (elemIdx % 2) == 0

    val packedByte = packedRegs(regIdx)((byteInReg * 8 + 7) downto (byteInReg * 8))
    val code = (if (isHighNibble) packedByte(7 downto 4) else packedByte(3 downto 0)).asUInt

    val sign  = code(3)
    val index = code(2 downto 0)

    val halfUnits = decodeHalfUnits(dialectId, index)

    val signedVal = SInt(8 bits)
    val halfUnitsS8 = halfUnits.resize(8).asSInt
    when(sign) {
      signedVal := -halfUnitsS8
    } otherwise {
      signedVal := halfUnitsS8
    }

    decodedBytes(elemIdx) := signedVal.asBits
  }

  // Pack 4 bytes per 32-bit word, little-endian (byte 0 in bits 7..0)
  val decodedWords = Vec(Bits(32 bits), 8)
  for (w <- 0 until 8) {
    val b0 = decodedBytes(w * 4 + 0)
    val b1 = decodedBytes(w * 4 + 1)
    val b2 = decodedBytes(w * 4 + 2)
    val b3 = decodedBytes(w * 4 + 3)
    decodedWords(w) := b3 ## b2 ## b1 ## b0
  }

  // APB3 bus
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
