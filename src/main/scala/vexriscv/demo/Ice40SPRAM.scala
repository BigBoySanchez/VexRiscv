package vexriscv.demo

import spinal.core._
import spinal.lib._
import spinal.lib.bus.simple._

/**
 * iCE40 UP5K SB_SPRAM256KA BlackBox.
 *
 * Each SPRAM primitive is 256Kbit = 32KB, organized as 16K x 16-bit words.
 * The UP5K has 4 SPRAM blocks total (128KB).
 */
case class SB_SPRAM256KA() extends BlackBox {
  val io = new Bundle {
    val ADDRESS    = in  UInt(14 bits)
    val DATAIN     = in  Bits(16 bits)
    val MASKWREN   = in  Bits(4 bits)  // nibble-level write mask
    val WREN       = in  Bool()
    val CHIPSELECT = in  Bool()
    val CLOCK      = in  Bool()
    val STANDBY    = in  Bool()
    val SLEEP      = in  Bool()
    val POWEROFF   = in  Bool()
    val DATAOUT    = out Bits(16 bits)
  }
  noIoPrefix()
  mapCurrentClockDomain(clock = io.CLOCK)
}

/**
 * 64KB SPRAM behind a PipelinedMemoryBus interface.
 *
 * Uses two SB_SPRAM256KA blocks (32KB each) side-by-side to form
 * a 32-bit-wide, 16K-entry memory = 64KB.
 */
case class Ice40SPRAM_64K(pipelinedMemoryBusConfig: PipelinedMemoryBusConfig) extends Component {
  val io = new Bundle {
    val bus = slave(PipelinedMemoryBus(pipelinedMemoryBusConfig))
  }

  // Two 16-bit SPRAM blocks → one 32-bit word
  val spramLo = SB_SPRAM256KA()
  val spramHi = SB_SPRAM256KA()

  // Address: word-aligned, so shift right by 2 to get 14-bit word index
  val wordAddr = (io.bus.cmd.address >> 2).resize(14 bits)

  // Write mask: PipelinedMemoryBus mask is 4 bits (one per byte).
  // SPRAM MASKWREN is 4 bits per 16-bit block (nibble-level):
  //   bit 0 = data[3:0], bit 1 = data[7:4], bit 2 = data[11:8], bit 3 = data[15:12]
  // Map byte mask to nibble mask:
  //   bus.mask(0) → spramLo bits [7:0]  → MASKWREN(0,1)
  //   bus.mask(1) → spramLo bits [15:8] → MASKWREN(2,3)
  //   bus.mask(2) → spramHi bits [7:0]  → MASKWREN(0,1)
  //   bus.mask(3) → spramHi bits [15:8] → MASKWREN(2,3)
  val maskLo = B(4 bits,
    0 -> io.bus.cmd.mask(0),
    1 -> io.bus.cmd.mask(0),
    2 -> io.bus.cmd.mask(1),
    3 -> io.bus.cmd.mask(1)
  )
  val maskHi = B(4 bits,
    0 -> io.bus.cmd.mask(2),
    1 -> io.bus.cmd.mask(2),
    2 -> io.bus.cmd.mask(3),
    3 -> io.bus.cmd.mask(3)
  )

  // Connect low 16-bit SPRAM
  spramLo.io.ADDRESS    := wordAddr
  spramLo.io.DATAIN     := io.bus.cmd.data(15 downto 0)
  spramLo.io.MASKWREN   := maskLo
  spramLo.io.WREN       := io.bus.cmd.write
  spramLo.io.CHIPSELECT := io.bus.cmd.valid
  spramLo.io.STANDBY    := False
  spramLo.io.SLEEP      := False
  spramLo.io.POWEROFF   := True  // Active-low: True = powered on

  // Connect high 16-bit SPRAM
  spramHi.io.ADDRESS    := wordAddr
  spramHi.io.DATAIN     := io.bus.cmd.data(31 downto 16)
  spramHi.io.MASKWREN   := maskHi
  spramHi.io.WREN       := io.bus.cmd.write
  spramHi.io.CHIPSELECT := io.bus.cmd.valid
  spramHi.io.STANDBY    := False
  spramHi.io.SLEEP      := False
  spramHi.io.POWEROFF   := True

  // Response: 1 cycle latency (synchronous read)
  io.bus.rsp.valid := RegNext(io.bus.cmd.fire && !io.bus.cmd.write) init(False)
  io.bus.rsp.data  := spramHi.io.DATAOUT ## spramLo.io.DATAOUT
  io.bus.cmd.ready := True
}
