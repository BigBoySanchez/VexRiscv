package vexriscv.demo

import spinal.core._

/**
 * BlockDialectDecoder — backward-compatibility shim.
 *
 * All APB logic and 32-lane DialectFP4DecodeCore instantiation now live in
 * BlockDialectDecoderAPB (Milestone 4 refactor).  This class extends it so
 * that existing call sites (MuraxHyperRAM, CheckDecoder, firmware) require
 * no changes.
 *
 * For new code (e.g. BDMac32, Milestone 5) prefer BlockDialectDecoderAPB
 * or DialectFP4DecodeCore directly.
 */
class BlockDialectDecoder extends BlockDialectDecoderAPB
