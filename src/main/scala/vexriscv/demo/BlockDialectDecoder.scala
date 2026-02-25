package vexriscv.demo

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba3.apb.{Apb3, Apb3Config, Apb3SlaveFactory}

/**
 * BlockDialect-Lite hardware decoder — APB3 peripheral.
 *
 * Firmware writes a packed block (2-byte metadata + 16 bytes of 4-bit codes)
 * and reads back 32 decoded int8 values.
 *
 * Register map (from APB base):
 *   0x00  META     (W)   dialect_id[15:12] | shared_exp[11:7] | pad[6:0]
 *   0x04  PACKED0  (W)   packed codes bytes 0–3
 *   0x08  PACKED1  (W)   packed codes bytes 4–7
 *   0x0C  PACKED2  (W)   packed codes bytes 8–11
 *   0x10  PACKED3  (W)   packed codes bytes 12–15
 *   0x20  DECODED0 (R)   decoded bytes 0–3
 *   0x24  DECODED1 (R)   decoded bytes 4–7
 *   ...
 *   0x3C  DECODED7 (R)   decoded bytes 28–31
 *   0x40  STATUS   (R)   bit 0 = 1 (always ready, combinational decode)
 */
class BlockDialectDecoder extends Component {
  val io = new Bundle {
    val apb = slave(Apb3(
      addressWidth = 8,
      dataWidth = 32
    ))
  }

  // DialectFP4 LUT: 16 dialects × 8 magnitudes = 128 entries, 4 bits each
  val lutInit = Seq(
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
  )
  val lutMem = Mem(UInt(4 bits), 128) initBigInt(lutInit.map(BigInt(_)))

  // Input registers (written by firmware)
  val metaReg    = Reg(UInt(16 bits)) init(0)
  val packedRegs = Vec(Reg(Bits(32 bits)) init(0), 4)

  // Extract metadata fields
  val dialectId = metaReg(15 downto 12)  // 4 bits
  val sharedExp = metaReg(11 downto  7)  // 5 bits

  // Helper: apply exponent shift using a switch (avoids dynamic shift)
  def applyExponent(mag4: UInt, exp5: UInt): UInt = {
    val result = UInt(8 bits)
    result := 0
    switch(exp5) {
      is(0)  { result := ((mag4.resize(5 bits) + 1) >> 1).resized }
      is(1)  { result := mag4.resized }
      is(2)  { result := (mag4 << 1).resized }
      is(3)  { result := Mux(mag4 > U(15), U(127, 8 bits), (mag4 << 2).resized) }
      is(4)  { result := Mux(mag4 > U(7),  U(127, 8 bits), (mag4 << 3).resized) }
      is(5)  { result := Mux(mag4 > U(3),  U(127, 8 bits), (mag4 << 4).resized) }
      default { result := 127 }
    }
    result
  }

  // Helper: decode one 4-bit code into 8-bit signed value
  def decodeElement(code: UInt): Bits = {
    val sign  = code(3)
    val index = code(2 downto 0)

    val lutAddr = dialectId @@ index
    val magScaled = lutMem.readAsync(lutAddr)

    val realMag = applyExponent(magScaled, sharedExp)

    val signedVal = SInt(8 bits)
    when(sign) {
      signedVal := -(realMag.asSInt.resized)
    } otherwise {
      signedVal := realMag.asSInt.resized
    }
    signedVal.asBits
  }

  // Build decoded output words (combinational from input registers)
  // Each word is 4 decoded bytes (little-endian)
  val decodedWords = Vec(Bits(32 bits), 8)
  for (wordIdx <- 0 until 8) {
    val bytes = Vec(Bits(8 bits), 4)
    for (bytePos <- 0 until 4) {
      val elemIdx = wordIdx * 4 + bytePos
      val byteIdx = elemIdx / 2
      val regIdx  = byteIdx / 4
      val byteInReg = byteIdx % 4
      val isHighNibble = (elemIdx % 2) == 0

      val packedByte = packedRegs(regIdx)((byteInReg * 8 + 7) downto (byteInReg * 8))
      val code = if (isHighNibble) {
        packedByte(7 downto 4).asUInt
      } else {
        packedByte(3 downto 0).asUInt
      }

      bytes(bytePos) := decodeElement(code)
    }
    decodedWords(wordIdx) := bytes(3) ## bytes(2) ## bytes(1) ## bytes(0)
  }

  // APB3 bus interface
  val busCtrl = Apb3SlaveFactory(io.apb)

  // Write registers
  busCtrl.write(metaReg, 0x00)
  for (i <- 0 until 4) {
    busCtrl.write(packedRegs(i), 0x04 + i * 4)
  }

  // Read decoded output (combinational, no register delay)
  for (i <- 0 until 8) {
    busCtrl.read(decodedWords(i), 0x20 + i * 4)
  }

  // Status register (always ready)
  busCtrl.read(B(1, 32 bits), 0x40)
}
