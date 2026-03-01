package vexriscv.demo

import spinal.core._

/**
 * iCEBreaker (iCE40UP5K) board target for MuraxHyperRAM.
 *
 * Generates Verilog with iCEBreaker-appropriate defaults:
 *   - 12 MHz core frequency (matches board crystal)
 *   - 8 KB on-chip BRAM (initialized from hex)
 *   - 64 KB SPRAM for .bss/stack
 *   - Flash-backed weight store at 0x2000_0000 (1 MB window)
 *   - Weights expected at flash offset 0x100000 (1 MiB)
 *   - BlockDialect decoder disabled to reduce LC usage
 */
object MuraxHyperRAM_iCEBreaker {
  def main(args: Array[String]) {
    SpinalVerilog(MuraxHyperRAM(MuraxHyperRAMConfig.default().copy(
      coreFrequency    = 12 MHz,
      onChipRamSize    = 32 kB,
      onChipRamHexFile = "src/main/c/murax/hyperram_phase_full/build/hello_world.hex",
      spramSize        = 64 kB,
      weightStoreSize  = 2 MB,
      weightStoreHexFile = null,
      flashWeightStore = true,
      flashOffset      = 0x100000,  // 1 MiB — weights start after bitstream
      includeBdDecoder = false
    )))
  }
}
