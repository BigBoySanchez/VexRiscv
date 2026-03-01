package vexriscv.demo

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba3.apb._
import spinal.lib.bus.misc.SizeMapping
import spinal.lib.bus.simple.PipelinedMemoryBus
import spinal.lib.com.jtag.Jtag
import spinal.lib.com.uart._
import spinal.lib.io.{InOutWrapper, TriStateArray}
import spinal.lib.misc.{InterruptCtrl, Prescaler, Timer}
import vexriscv.plugin._
import vexriscv.{VexRiscv, VexRiscvConfig, plugin}
import spinal.lib.com.spi.ddr._
import spinal.lib.bus.simple._
import scala.collection.mutable.ArrayBuffer
import scala.collection.Seq
import spinal.lib.com.jtag.JtagTapInstructionCtrl

case class MuraxHyperRAMConfig(coreFrequency : HertzNumber,
                       onChipRamSize      : BigInt,
                       onChipRamHexFile   : String,
                       spramSize          : BigInt,
                       weightStoreSize    : BigInt,
                       weightStoreHexFile : String,
                       pipelineDBus       : Boolean,
                       pipelineMainBus    : Boolean,
                       pipelineApbBridge  : Boolean,
                       gpioWidth          : Int,
                       uartCtrlConfig     : UartCtrlMemoryMappedConfig,
                       hardwareBreakpointCount : Int,
                       cpuPlugins         : ArrayBuffer[Plugin[VexRiscv]],
                       includeBdDecoder   : Boolean = true,
                       flashWeightStore   : Boolean = false,
                       flashOffset        : BigInt = 0x100000  // 1 MiB default offset in flash
                       ){
  require(pipelineApbBridge || pipelineMainBus, "At least pipelineMainBus or pipelineApbBridge should be enable to avoid wipe transactions")
}

object MuraxHyperRAMConfig{
  // Default Phase A Configuration
  def default : MuraxHyperRAMConfig = default(false)
  def default(bigEndian : Boolean = false) =  MuraxHyperRAMConfig(
    coreFrequency         = 12 MHz,
    onChipRamSize         = 8 kB,
    onChipRamHexFile      = null,
    spramSize             = 0 kB,
    weightStoreSize       = 64 kB,
    weightStoreHexFile    = null,
    pipelineDBus          = true,
    pipelineMainBus       = false,
    pipelineApbBridge     = true,
    gpioWidth = 32,
    hardwareBreakpointCount = 0,
    cpuPlugins = ArrayBuffer(
      new IBusSimplePlugin(
        resetVector = 0x10000000l, // RAM Base
        cmdForkOnSecondStage = true,
        cmdForkPersistence = false,
        prediction = NONE,
        catchAccessFault = false,
        compressedGen = false,
        bigEndian = bigEndian
      ),
      new DBusSimplePlugin(
        catchAddressMisaligned = false,
        catchAccessFault = false,
        earlyInjection = false,
        bigEndian = bigEndian
      ),
      new CsrPlugin(CsrPluginConfig.smallest(mtvecInit = 0x10000020l).copy(mcycleAccess = CsrAccess.READ_ONLY)), // RAM Base + 0x20, mcycle enabled
      new DecoderSimplePlugin(
        catchIllegalInstruction = false
      ),
      new RegFilePlugin(
        regFileReadyKind = plugin.SYNC,
        zeroBoot = false
      ),
      new IntAluPlugin,
      new SrcPlugin(
        separatedAddSub = false,
        executeInsertion = false
      ),
      new LightShifterPlugin,
      new HazardSimplePlugin(
        bypassExecute = false,
        bypassMemory = false,
        bypassWriteBack = false,
        bypassWriteBackBuffer = false,
        pessimisticUseSrc = false,
        pessimisticWriteRegFile = false,
        pessimisticAddressMatch = false
      ),
      new BranchPlugin(
        earlyBranch = false,
        catchAddressMisaligned = false
      ),
      new MulPlugin,
      new DivPlugin
      // No YamlPlugin for now
    ),
    uartCtrlConfig = UartCtrlMemoryMappedConfig(
      uartCtrlConfig = UartCtrlGenerics(
        dataWidthMax      = 8,
        clockDividerWidth = 20,
        preSamplingSize   = 1,
        samplingSize      = 3,
        postSamplingSize  = 1
      ),
      initConfig = UartCtrlInitConfig(
        baudrate = 115200,
        dataLength = 7,  //7 => 8 bits
        parity = UartParityType.NONE,
        stop = UartStopType.ONE
      ),
      busCanWriteClockDividerConfig = false,
      busCanWriteFrameConfig = false,
      txFifoDepth = 16,
      rxFifoDepth = 16
    )
  )
}

case class MuraxHyperRAM(config : MuraxHyperRAMConfig) extends Component{
  import config._
  println("MuraxHyperRAM (Refactored): Elaborating...")

  val io = new Bundle {
    //Clocks / reset
    val asyncReset = in Bool()
    val mainClk = in Bool()

    //Main components IO
    val jtag = slave(Jtag())

    //Peripherals IO
    val gpioA = master(TriStateArray(gpioWidth bits))
    val uart = master(Uart())

    // SPI Flash (for flash-backed weight store)
    val spiFlash = ifGen(flashWeightStore)(master(SpiFlashIo()))
  }

  val resetCtrlClockDomain = ClockDomain(
    clock = io.mainClk,
    config = ClockDomainConfig(
      resetKind = BOOT
    )
  )

  val resetCtrl = new ClockingArea(resetCtrlClockDomain) {
    val mainClkResetUnbuffered  = False

    //Implement an counter to keep the reset axiResetOrder high 64 cycles
    // Also this counter will automatically do a reset when the system boot.
    val systemClkResetCounter = Reg(UInt(6 bits)) init(0)
    when(systemClkResetCounter =/= U(systemClkResetCounter.range -> true)){
      systemClkResetCounter := systemClkResetCounter + 1
      mainClkResetUnbuffered := True
    }
    when(BufferCC(io.asyncReset)){
      systemClkResetCounter := 0
    }

    //Create all reset used later in the design
    val mainClkReset = RegNext(mainClkResetUnbuffered)
    val systemReset  = RegNext(mainClkResetUnbuffered)
  }

  val systemClockDomain = ClockDomain(
    clock = io.mainClk,
    reset = resetCtrl.systemReset,
    frequency = FixedFrequency(coreFrequency)
  )

  val debugClockDomain = ClockDomain(
    clock = io.mainClk,
    reset = resetCtrl.mainClkReset,
    frequency = FixedFrequency(coreFrequency)
  )

  val system = new ClockingArea(systemClockDomain) {
    val pipelinedMemoryBusConfig = PipelinedMemoryBusConfig(
      addressWidth = 32,
      dataWidth = 32
    )

    val bigEndianDBus = config.cpuPlugins.exists(_ match{ case plugin : DBusSimplePlugin => plugin.bigEndian case _ => false})

    //Arbiter of the cpu dBus/iBus to drive the mainBus
    val mainBusArbiter = new MuraxMasterArbiter(pipelinedMemoryBusConfig, bigEndianDBus)

    //Instanciate the CPU
    val cpu = new VexRiscv(
      config = VexRiscvConfig(
        plugins = cpuPlugins += new DebugPlugin(debugClockDomain, hardwareBreakpointCount)
      )
    )

    //Checkout plugins used to instanciate the CPU to connect them to the SoC
    val timerInterrupt = False
    val externalInterrupt = False
    for(plugin <- cpu.plugins) plugin match{
      case plugin : IBusSimplePlugin =>
        mainBusArbiter.io.iBus.cmd <> plugin.iBus.cmd
        mainBusArbiter.io.iBus.rsp <> plugin.iBus.rsp
      case plugin : DBusSimplePlugin => {
        if(!pipelineDBus)
          mainBusArbiter.io.dBus <> plugin.dBus
        else {
          mainBusArbiter.io.dBus.cmd << plugin.dBus.cmd.halfPipe()
          mainBusArbiter.io.dBus.rsp <> plugin.dBus.rsp
        }
      }
      case plugin : CsrPlugin        => {
        plugin.externalInterrupt := externalInterrupt
        plugin.timerInterrupt := timerInterrupt
      }
      case plugin : DebugPlugin         => plugin.debugClockDomain{
        resetCtrl.systemReset setWhen(RegNext(plugin.io.resetOut))
        io.jtag <> plugin.io.bus.fromJtag()
      }
      case _ =>
    }

    //****** MainBus slaves ********
    val mainBusMapping = ArrayBuffer[(PipelinedMemoryBus,SizeMapping)]()

    // RAM @ 0x1000_0000
    val ram = new MuraxPipelinedMemoryBusRam(
      onChipRamSize = onChipRamSize,
      onChipRamHexFile = onChipRamHexFile,
      pipelinedMemoryBusConfig = pipelinedMemoryBusConfig,
      bigEndian = bigEndianDBus,
      baseAddress = 0x10000000l
    )
    mainBusMapping += ram.io.bus -> (0x10000000l, onChipRamSize)

    // SPRAM @ 0x1100_0000 (if enabled)
    if(spramSize > 0){
      require(spramSize == (64 kB) || spramSize == (128 kB), "spramSize must be 64kB or 128kB on iCE40UP5K")
      if(spramSize == (64 kB)) {
        val spram = new Ice40SPRAM_64K(pipelinedMemoryBusConfig)
        mainBusMapping += spram.io.bus -> (0x11000000l, spramSize)
      } else {
        val spram = new Ice40SPRAM_128K(pipelinedMemoryBusConfig)
        mainBusMapping += spram.io.bus -> (0x11000000l, spramSize)
      }
    }

    // WeightStore @ 0x2000_0000
    if(flashWeightStore) {
      // Flash-backed weight store: SPI flash reader
      val flashReader = new PipelinedMemoryBusFlashReader(
        pipelinedMemoryBusConfig = pipelinedMemoryBusConfig,
        flashOffset = flashOffset
      )
      flashReader.io.spiFlash <> io.spiFlash
      mainBusMapping += flashReader.io.bus -> (0x20000000l, weightStoreSize)
    } else if(weightStoreSize > 0) {
      // Internal RAM weight store (simulation / non-flash targets)
      val weightStore = new MuraxPipelinedMemoryBusRam(
        onChipRamSize = weightStoreSize,
        onChipRamHexFile = weightStoreHexFile,
        pipelinedMemoryBusConfig = pipelinedMemoryBusConfig,
        bigEndian = bigEndianDBus,
        baseAddress = 0x20000000l
      )
      mainBusMapping += weightStore.io.bus -> (0x20000000l, weightStoreSize)
    }

    // APB Bridge @ 0x4000_0000
    val apbBridge = new PipelinedMemoryBusToApbBridge(
      apb3Config = Apb3Config(
        addressWidth = 20,
        dataWidth = 32
      ),
      pipelineBridge = pipelineApbBridge,
      pipelinedMemoryBusConfig = pipelinedMemoryBusConfig
    )
    mainBusMapping += apbBridge.io.pipelinedMemoryBus -> (0x40000000l, 1 MB)

    //******** APB peripherals *********
    val apbMapping = ArrayBuffer[(Apb3, SizeMapping)]()

    // GPIO @ 0x00000 (0x40000000)
    val gpioACtrl = Apb3Gpio(gpioWidth = gpioWidth, withReadSync = true)
    io.gpioA <> gpioACtrl.io.gpio
    apbMapping += gpioACtrl.io.apb -> (0x00000, 4 kB)

    // UART @ 0x10000 (0x40010000)
    val uartCtrl = Apb3UartCtrl(uartCtrlConfig)
    uartCtrl.io.uart <> io.uart
    externalInterrupt setWhen(uartCtrl.io.interrupt)
    apbMapping += uartCtrl.io.apb  -> (0x10000, 4 kB)

    val timer = new MuraxApb3Timer()
    timerInterrupt setWhen(timer.io.interrupt)
    apbMapping += timer.io.apb     -> (0x20000, 4 kB)

    // BlockDialect Decoder @ 0x30000 (0x40030000) — Phase B hardware decode
    if(includeBdDecoder) {
      val bdDecoder = new BlockDialectDecoder()
      apbMapping += bdDecoder.io.apb -> (0x30000, 4 kB)
    }
    //******** Memory mappings *********
    val apbDecoder = Apb3Decoder(
      master = apbBridge.io.apb,
      slaves = apbMapping.toSeq
    )

    val mainBusDecoder = new Area {
      val logic = new MuraxPipelinedMemoryBusDecoder(
        master = mainBusArbiter.io.masterBus,
        specification = mainBusMapping.toSeq,
        pipelineMaster = pipelineMainBus
      )
    }
  }
}

object MuraxHyperRAM {
  def main(args: Array[String]) {
    SpinalVerilog(MuraxHyperRAM(MuraxHyperRAMConfig.default()))
  }
}
