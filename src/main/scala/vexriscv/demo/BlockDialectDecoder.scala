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
 *   0x20–0x3C  DECODED0–7 (R)   32 decoded int8 values (packed 4 per word, little-endian)
 *   0x40  STATUS   (R)   bit 0 = 1 (always ready)
 */
class BlockDialectDecoder extends Component {
  val io = new Bundle {
    val apb = slave(Apb3(
      addressWidth = 8,
      dataWidth = 32
    ))
  }

  // DialectFP4 LUT as Vec (constant logic, no Mem — avoids readAsync port issues)
  private val lutRom = Vec(Seq(
    0, 1, 2, 3, 4, 4, 4, 4,   // D0
    0, 1, 2, 3, 3, 3, 4, 4,   // D1
    0, 1, 2, 3, 4, 5, 5, 5,   // D2
    0, 1, 2, 3, 3, 4, 5, 5,   // D3
    0, 1, 2, 3, 4, 5, 6, 6,   // D4
    0, 1, 2, 3, 4, 4, 6, 6,   // D5
    0, 1, 2, 3, 4, 5, 6, 7,   // D6
    0, 1, 2, 3, 4, 5, 7, 7,   // D7
    0, 1, 2, 3, 4, 6, 7, 8,   // D8
    0, 1, 2, 3, 4, 6, 8, 8,   // D9
    0, 1, 2, 3, 4, 6, 8, 10,  // D10
    0, 1, 2, 3, 4, 6, 10, 10, // D11
    0, 1, 2, 3, 4, 6, 10, 12, // D12
    0, 1, 2, 3, 4, 6, 12, 12, // D13
    0, 1, 2, 3, 4, 6, 12, 15, // D14
    0, 1, 2, 3, 4, 6, 13, 15  // D15
  ).map(v => U(v, 4 bits)))

  // Input registers
  val metaReg    = Reg(UInt(16 bits)) init(0)
  val packedRegs = Vec(Reg(Bits(32 bits)) init(0), 4)

  // Metadata fields
  val dialectId = metaReg(15 downto 12)
  val sharedExp = metaReg(11 downto  7)

  /** Apply shared exponent as per the python reference:
    *   if exp==0: round(0.5 * magScaled) => (magScaled + 1) >> 1
    *   else:      magScaled * 2^(exp-1)
    * then clamp to 127.
    */
  private def applyExponent(mag4: UInt, exp5: UInt): UInt = {
    val mag16 = mag4.resize(16)
    val tmp   = UInt(16 bits)
    tmp := 0

    // exp5 is 5-bit, but in your int8 weight blob it should only be 0..5.
    // This switch avoids inferring a large barrel shifter.
    switch(exp5) {
      is(0)  { tmp := ((mag16 + 1) >> 1).resized } // round half up
      is(1)  { tmp := mag16 }
      is(2)  { tmp := (mag16 << 1).resized }
      is(3)  { tmp := (mag16 << 2).resized }
      is(4)  { tmp := (mag16 << 3).resized }
      is(5)  { tmp := (mag16 << 4).resized }
      default { tmp := (mag16 << 4).resized }
    }

    val result = UInt(8 bits)
    when(tmp > U(127, 16 bits)) {
      result := U(127, 8 bits)
    } otherwise {
      result := tmp(7 downto 0)
    }
    result
  }

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

    val lutAddr   = (dialectId @@ index)
    val magScaled = lutRom(lutAddr)

    val realMag = applyExponent(magScaled, sharedExp)

    val signedVal = SInt(8 bits)
    when(sign) {
      signedVal := -(realMag.asSInt.resized)
    } otherwise {
      signedVal := realMag.asSInt.resized
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
}
