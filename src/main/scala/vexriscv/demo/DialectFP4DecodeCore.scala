package vexriscv.demo

import spinal.core._

/**
 * DialectFP4DecodeCore — pure combinational half-unit magnitude lookup.
 *
 * Given a BlockDialect dialect_id (4 bits) and a 3-bit code index,
 * returns the unsigned magnitude in half-units (4 bits, range 0–15).
 *
 * The sign bit (code[3]) is NOT handled here; the caller negates the
 * result when sign == 1 (see BlockDialectDecoder for the usage pattern).
 *
 * This is the factored-out core of the lookup originally inlined as
 * `decodeHalfUnits()` inside BlockDialectDecoder.scala.  Extracting it
 * enables:
 *   • Independent unit-testing of the dialect table
 *   • Re-use in a future BDMac32 fused multiply-accumulate peripheral
 *   • Cleaner timing analysis (purely combinational 4→4-bit look-up)
 *
 * Interface:
 *   io.dialect_id  [3:0]  in   — which of the 16 BD dialects to use
 *   io.idx         [2:0]  in   — 3-bit code index (code[2:0])
 *   io.mag         [3:0]  out  — unsigned half-unit magnitude
 *
 * Variant table (idx == 6, dialects 0..14):
 *   Dialect:  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14
 *   Mag:      11   9  11   9  10   8  10   8   9   7   9   7   8   7   7
 *
 * Dialect 15 is the identity dialect: idx maps directly to mag
 * (idx=7 maps to 8, all others map to themselves).
 *
 * All other dialects follow:
 *   idx 0→0, 1→1, 2→2, 3→3, 4→4, 5→6,
 *   idx 6 → variantIdx6(dialect_id),
 *   idx 7 → maxHU = 15 - floor(dialect_id / 2)
 */
class DialectFP4DecodeCore extends Component {
  val io = new Bundle {
    val dialect_id = in UInt(4 bits)
    val idx        = in UInt(3 bits)
    val mag        = out UInt(4 bits)
  }

  // Variant table for idx == 6, dialects 0..14
  private val variantIdx6 = Vec(Seq(
    11, 9, 11, 9, 10, 8, 10, 8,
     9, 7,  9, 7,  8, 7,  7
  ).map(v => U(v, 4 bits)))

  val mag = UInt(4 bits)
  mag := 0

  when(io.dialect_id === U(15)) {
    switch(io.idx) {
      is(0) { mag := U(0) }
      is(1) { mag := U(1) }
      is(2) { mag := U(2) }
      is(3) { mag := U(3) }
      is(4) { mag := U(4) }
      is(5) { mag := U(5) }
      is(6) { mag := U(6) }
      is(7) { mag := U(8) }
    }
  } otherwise {
    val pair  = (io.dialect_id >> 1).resized
    val maxHU = (U(15, 4 bits) - pair).resized

    switch(io.idx) {
      is(0) { mag := U(0) }
      is(1) { mag := U(1) }
      is(2) { mag := U(2) }
      is(3) { mag := U(3) }
      is(4) { mag := U(4) }
      is(5) { mag := U(6) }
      is(6) { mag := variantIdx6(io.dialect_id) }
      is(7) { mag := maxHU }
    }
  }

  io.mag := mag
}
