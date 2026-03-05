package vexriscv.demo

import spinal.core._

/**
 * iCEBreaker (iCE40UP5K) board target for MuraxHyperRAM.
 *
 * Flash layout (16 MB SPI flash):
 *   0x000000  bitstream  (firmware is baked into BRAM via onChipRamHexFile —
 *                         reflash bitstream to update firmware)
 *   0x100000  weights    ~11 MB VWB2 blob  (iceprog -o 1M weights_bd.bin)
 *
 * flashOffset = 0x100000 → CPU 0x20000000 = flash 0x100000 (weights start)
 * BRAM (12 KB) holds the firmware directly; no bootloader needed.
 *
 * Generates Verilog with iCEBreaker-appropriate defaults:
 *   - 12 MHz core frequency (matches board crystal)
 *   - 12 KB on-chip BRAM
 *   - 128 KB SPRAM for activation buffers, .bss, and stack
 *   - BlockDialect decoder @ 0x40030000 (Milestone 3/4)
 *   - BDMac32 32-lane dot-product accelerator @ 0x40031000 (Milestone 5)
 */
object MuraxHyperRAM_iCEBreaker {
  def main(args: Array[String]) {
    SpinalVerilog(MuraxHyperRAM(MuraxHyperRAMConfig.default().copy(
      coreFrequency    = 27 MHz,
      onChipRamSize    = 12 kB,
      onChipRamHexFile = "src/main/c/murax/resnet1202_phase3_hw_decode/build/resnet1202_phase3_hw_decode.hex",
      spramSize        = 128 kB,
      weightStoreSize  = 15 MB,
      weightStoreHexFile = null,
      flashWeightStore = true,
      flashOffset      = 0x100000,  // weights start at 1 MB in flash; CPU 0x20000000 = flash 0x100000
      pipelineMainBus  = true,      // break critical path for higher clock speeds
      includeBdDecoder = false,     // 32-lane parallel decoder: too many LCs alongside BDMac32
      includeBdMac     = true       // Sequential MAC, 2 DialectFP4DecodeCore, fits UP5K @ 0x40031000
    )))
  }
}
