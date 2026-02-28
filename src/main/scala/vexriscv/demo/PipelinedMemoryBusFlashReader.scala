package vexriscv.demo

import spinal.core._
import spinal.lib._
import spinal.lib.bus.simple._
import spinal.lib.fsm._

/**
 * SPI flash I/O bundle (directly exposed to top-level pins).
 */
case class SpiFlashIo() extends Bundle with IMasterSlave {
  val sclk = Bool()
  val cs_n = Bool()
  val mosi = Bool()
  val miso = Bool()

  override def asMaster(): Unit = {
    out(sclk, cs_n, mosi)
    in(miso)
  }
}

/**
 * Read-only SPI flash controller behind a PipelinedMemoryBus slave interface.
 *
 * Maps CPU reads at bus addresses into SPI flash reads using the standard
 * READ command (0x03) with a 24-bit address.
 *
 * Flash address = flashOffset + bus_address
 *
 * The bus is stalled during the SPI transaction. Writes are accepted and
 * silently ignored (no stall).
 *
 * SPI Mode 0 (CPOL=0, CPHA=0):
 *   - SCLK idles low
 *   - MOSI changes on falling edge (stable before rising edge)
 *   - MISO sampled on rising edge
 */
case class PipelinedMemoryBusFlashReader(
  pipelinedMemoryBusConfig: PipelinedMemoryBusConfig,
  flashOffset: BigInt = 0x100000,   // 1 MiB default
  spiClkDiv: Int = 0                // sclk = mainClk / (2*(div+1))
) extends Component {

  val io = new Bundle {
    val bus = slave(PipelinedMemoryBus(pipelinedMemoryBusConfig))
    val spiFlash = master(SpiFlashIo())
  }

  // SPI output registers
  val sclkReg = RegInit(False)
  val csReg   = RegInit(True)   // CS active-low, start deasserted
  val mosiReg = RegInit(False)

  io.spiFlash.sclk := sclkReg
  io.spiFlash.cs_n := csReg
  io.spiFlash.mosi := mosiReg

  // Clock divider
  val divCounter = Reg(UInt(log2Up(spiClkDiv + 1) max 1 bits)) init(0)
  val spiTick = divCounter === spiClkDiv

  when(spiTick) {
    divCounter := 0
  } otherwise {
    divCounter := divCounter + 1
  }

  // Transaction state
  val shiftOut     = Reg(Bits(32 bits)) init(0)  // CMD(8) + ADDR(24) shift register
  val shiftIn      = Reg(Bits(32 bits)) init(0)  // 32-bit data read back
  val bitCounter   = Reg(UInt(6 bits)) init(0)   // counts up to 32
  val busy         = RegInit(False)
  val rspValid     = RegInit(False)
  val rspData      = Reg(Bits(32 bits)) init(0)
  val sclkPhase    = RegInit(False)  // False=will do falling/setup, True=will do rising/sample

  // Bus interface
  io.bus.cmd.ready := False
  io.bus.rsp.valid := rspValid
  io.bus.rsp.data  := rspData

  // Clear rspValid after one cycle
  when(rspValid) {
    rspValid := False
  }

  // Accept write commands immediately (silently discard)
  when(io.bus.cmd.valid && io.bus.cmd.write && !busy) {
    io.bus.cmd.ready := True
  }

  // FSM for SPI transactions
  // SPI Mode 0 timing:
  //   1. CS goes low (assert)
  //   2. Set MOSI to first bit (while SCLK is still low)
  //   3. Raise SCLK (rising edge - flash samples MOSI, we sample MISO)
  //   4. Lower SCLK (falling edge - set next MOSI bit)
  //   5. Repeat from 3 for remaining bits
  //   6. CS goes high (deassert)
  val fsm = new StateMachine {
    val IDLE      = new State with EntryPoint
    val SETUP_BIT = new State   // Set MOSI while SCLK is low
    val CLOCK_HI  = new State   // Raise SCLK, sample MISO
    val CLOCK_LO  = new State   // Lower SCLK, advance to next bit
    val READ_SETUP = new State  // Transition to read phase
    val READ_HI   = new State   // Raise SCLK, sample MISO for read
    val READ_LO   = new State   // Lower SCLK for read
    val FINISH    = new State

    IDLE.whenIsActive {
      when(io.bus.cmd.valid && !io.bus.cmd.write) {
        // Latch 24-bit flash address = flashOffset + bus word address
        val flashAddr = (io.bus.cmd.address + U(flashOffset, 32 bits)).resize(24 bits)
        shiftOut := B"8'h03" ## flashAddr.asBits  // READ command + 24-bit address
        bitCounter := 0
        csReg := False     // Assert CS
        busy := True
        sclkReg := False   // Ensure SCLK starts low
        goto(SETUP_BIT)
      }
    }

    // --- Command Phase (send 8-bit opcode + 24-bit address = 32 bits) ---

    SETUP_BIT.whenIsActive {
      // Set MOSI while SCLK is low (data setup before rising edge)
      mosiReg := shiftOut(31 - bitCounter.resized)
      when(spiTick) {
        goto(CLOCK_HI)
      }
    }

    CLOCK_HI.whenIsActive {
      when(spiTick) {
        sclkReg := True    // Rising edge - flash samples MOSI
        goto(CLOCK_LO)
      }
    }

    CLOCK_LO.whenIsActive {
      when(spiTick) {
        sclkReg := False   // Falling edge
        bitCounter := bitCounter + 1
        when(bitCounter === 31) {
          bitCounter := 0
          goto(READ_SETUP)
        } otherwise {
          goto(SETUP_BIT)  // Setup next bit
        }
      }
    }

    // --- Read Phase (receive 32 bits from flash) ---

    READ_SETUP.whenIsActive {
      // MOSI don't-care during read, just wait one tick for setup
      mosiReg := False
      when(spiTick) {
        goto(READ_HI)
      }
    }

    READ_HI.whenIsActive {
      when(spiTick) {
        sclkReg := True    // Rising edge - sample MISO
        shiftIn(31 - bitCounter.resized) := io.spiFlash.miso
        goto(READ_LO)
      }
    }

    READ_LO.whenIsActive {
      when(spiTick) {
        sclkReg := False   // Falling edge - flash drives next bit
        bitCounter := bitCounter + 1
        when(bitCounter === 31) {
          goto(FINISH)
        } otherwise {
          goto(READ_HI)    // Sample next bit
        }
      }
    }

    FINISH.whenIsActive {
      // Deassert CS, provide response
      csReg := True
      mosiReg := False

      // SPI flash returns bytes MSB-first: address A+0 first, then A+1, A+2, A+3
      // shiftIn has: [31:24]=byte@A+0, [23:16]=byte@A+1, [15:8]=byte@A+2, [7:0]=byte@A+3
      // CPU little-endian: data[7:0]=byte@A+0, data[15:8]=byte@A+1, etc.
      // Byte-swap:
      rspData := shiftIn(7 downto 0) ## shiftIn(15 downto 8) ## shiftIn(23 downto 16) ## shiftIn(31 downto 24)
      rspValid := True
      busy := False

      // Accept the bus command
      io.bus.cmd.ready := True

      goto(IDLE)
    }
  }
}
