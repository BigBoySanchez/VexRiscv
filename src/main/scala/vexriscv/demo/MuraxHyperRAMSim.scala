package vexriscv.demo

import spinal.core._
import spinal.core.sim._
// import spinal.lib.com.uart.sim.{UartDecoder, UartEncoder}

import java.io.{FileWriter, PrintWriter}


object MuraxHyperRAMSim {
  def main(args: Array[String]): Unit = {
    val logFile = new PrintWriter(new FileWriter("sim_output.log"))
    def log(s: String): Unit = { logFile.println(s); logFile.flush(); System.out.println(s); System.out.flush() }
    try {
      // Phase selection via environment variable (default: A)
      val phase = sys.env.getOrElse("SIM_PHASE", "A").toUpperCase
      log(s"Phase: $phase")

      val (firmwareHex, weightsHexFile) = phase match {
        case "B" => (
          "src/main/c/murax/hyperram_phase_b/build/hello_world.hex",
          "scripts/weights_bd_ihex.hex"
        )
        case _ => (
          "src/main/c/murax/hyperram_phase_a/build/hello_world.hex",
          "scripts/weights_ihex.hex"
        )
      }
      log(s"Firmware: $firmwareHex")
      log(s"Weights:  $weightsHexFile")

      val config = SpinalConfig(
        defaultClockDomainFrequency = FixedFrequency(12 MHz)
      )

      val muraxConfig = MuraxHyperRAMConfig.default().copy(
        onChipRamSize      = 64 kB,
        onChipRamHexFile   = firmwareHex,
        weightStoreSize    = 2 MB,  // 2MB covers both Phase A (1.73MB) and Phase B (975KB)
        weightStoreHexFile = weightsHexFile
      )

      SimConfig
        .withConfig(config)
        // .withWave  // Disabled: VCD dumps exhaust WSL memory for long runs
        .compile(MuraxHyperRAM(muraxConfig))
        .doSim { dut =>
          log("Starting sim...")

          dut.io.mainClk #= false
          dut.io.asyncReset #= true
          sleep(100)
          dut.io.asyncReset #= false
          sleep(100)

          // Manual UART decoding
          // UART decoder timing
          // Hardware UART config: 5 samples/bit (pre=1, samp=3, post=1)
          // clockDivider = 12MHz / 115200 / 5 - 1 = ~19
          // Bit period in clocks = (19+1) * 5 = 100 clock cycles
          // Each sim clock cycle = 2 time units (sleep(1) high + sleep(1) low)
          // So bit period = 100 * 2 = 200 time units
          val uartPeriod = 200

          fork {
            var first = true
            while(true) {
              if (dut.io.uart.txd.toBoolean == false && first) {
                 log("Debug: UART Start Bit Detected")
                 first = false
              }
              waitUntil(dut.io.uart.txd.toBoolean == false) // Start bit
              sleep(uartPeriod / 2) // Middle of start bit
              if(dut.io.uart.txd.toBoolean == false) {
                sleep(uartPeriod) // To bit 0
                var buffer = 0
                for(i <- 0 to 7) {
                  if(dut.io.uart.txd.toBoolean) buffer |= (1 << i)
                  sleep(uartPeriod)
                }
                // Write UART char to both stdout and log file
                logFile.print(buffer.toChar); logFile.flush()
                System.out.print(buffer.toChar); System.out.flush()
              }
            }
          }

          // Run simulation
          // 50M cycles for full 32x32 conv2d layer with MulPlugin
          val envMax = sys.env.getOrElse("SIM_MAX_CYCLES", "50000000").toInt
          var cycles = 0
          val maxCycles = envMax
          while(cycles < maxCycles) {
            dut.io.mainClk #= true
            sleep(1)
            dut.io.mainClk #= false
            sleep(1)
            cycles += 1
            if(cycles == 1000) {
              log(s"[info] Reset counter phase complete (1000 cycles)")
            }
            if(cycles % 10000000 == 0) {
              log(s"[${cycles/1000000}M cycles]")
            }
          }
          log(s"\nSimulation finished after $cycles cycles.")
          logFile.close()
        }
    } catch {
      case e: Throwable => 
        log("EXCEPTION DURING SIMULATION:")
        e.printStackTrace(logFile)
        logFile.close()
        throw e
    }
  }
}
