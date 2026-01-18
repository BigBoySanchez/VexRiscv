package vexriscv

import java.awt
import java.awt.event.{ActionEvent, ActionListener, MouseEvent, MouseListener}

import spinal.sim._
import spinal.core._
import spinal.core.sim._
import vexriscv.demo.{Murax, MuraxConfig, MxPlusBPlugin}
import javax.swing._

import spinal.lib.com.jtag.sim.JtagTcp
// UartDecoder/UartEncoder inlined below for better control over flushing and input handling
import vexriscv.test.{JLedArray, JSwitchArray}

import scala.collection.mutable



object MuraxSim {
  def main(args: Array[String]): Unit = {
//    def config = MuraxConfig.default.copy(onChipRamSize = 256 kB)
    def config = {
      val c = MuraxConfig.default(withXip = false).copy(onChipRamSize = 4 kB, onChipRamHexFile = "src/main/c/murax/mx_plus_b/build/hello_world.hex")
      c.cpuPlugins += new MxPlusBPlugin(
        m = 5,
        b = 10,
        instructionPattern = "0000000----------000-----0001011"
      )
      c
    }
    val simSlowDown = false
    SimConfig.allOptimisation.compile(new Murax(config)).doSimUntilVoid{dut =>
      val mainClkPeriod = (1e12/dut.config.coreFrequency.toDouble).toLong
      val jtagClkPeriod = mainClkPeriod*4
      val uartBaudRate = 115200
      val uartBaudPeriod = (1e12/uartBaudRate).toLong

      val clockDomain = ClockDomain(dut.io.mainClk, dut.io.asyncReset)
      clockDomain.forkStimulus(mainClkPeriod)
//      clockDomain.forkSimSpeedPrinter(2)

      val tcpJtag = JtagTcp(
        jtag = dut.io.jtag,
        jtagClkPeriod = jtagClkPeriod
      )

      // Custom UART decoder that flushes output immediately
      val uartTx = fork {
        sleep(1)
        waitUntil(dut.io.uart.txd.toBoolean == true)
        
        while(true) {
          waitUntil(dut.io.uart.txd.toBoolean == false)
          sleep(uartBaudPeriod/2)
          
          if(dut.io.uart.txd.toBoolean != false) {
            println("UART FRAME ERROR")
          } else {
            sleep(uartBaudPeriod)
            
            var buffer = 0
            (0 to 7).foreach { bitId =>
              if (dut.io.uart.txd.toBoolean)
                buffer |= 1 << bitId
              sleep(uartBaudPeriod)
            }
            
            if (dut.io.uart.txd.toBoolean != true) {
              println("UART FRAME ERROR")
            } else if (buffer.toChar != '\r') {
              print(buffer.toChar)
              System.out.flush()  // Flush immediately so output is visible
            }
          }
        }
      }
      
      // Shared queue for UART input (from GUI or stdin)
      val uartInputQueue = new java.util.concurrent.LinkedBlockingQueue[Int]()
      
      // Custom UART encoder that reads from the shared queue
      val uartRx = fork {
        dut.io.uart.rxd #= true
        while(true) {
          // Check both the queue and stdin
          val byteToSend: Option[Int] = {
            val queued = uartInputQueue.poll()
            if (queued != null) {
              Some(queued.intValue())
            } else if (System.in.available() != 0) {
              Some(System.in.read())
            } else {
              None
            }
          }
          
          byteToSend match {
            case Some(buffer) =>
              dut.io.uart.rxd #= false
              sleep(uartBaudPeriod)
              
              (0 to 7).foreach { bitId =>
                dut.io.uart.rxd #= ((buffer >> bitId) & 1) != 0
                sleep(uartBaudPeriod)
              }
              
              dut.io.uart.rxd #= true
              sleep(uartBaudPeriod)
            case None =>
              sleep(uartBaudPeriod * 10)
          }
        }
      }

      if(config.xipConfig != null)dut.io.xip.data(1).read #= 0

      val guiThread = fork{
        val guiToSim = mutable.Queue[Any]()

        var ledsValue = 0l
        var switchValue : () => BigInt = null
        val ledsFrame = new JFrame{
          setLayout(new BoxLayout(getContentPane, BoxLayout.Y_AXIS))

          add(new JLedArray(8){
            override def getValue = ledsValue
          })
          add{
            val switches = new JSwitchArray(8)
            switchValue = switches.getValue
            switches
          }
          
          // UART input panel
          add{
            val inputPanel = new JPanel()
            inputPanel.setLayout(new java.awt.FlowLayout())
            
            val textField = new JTextField(10)
            val sendButton = new JButton("Send")
            
            sendButton.addActionListener(new ActionListener {
              override def actionPerformed(actionEvent: ActionEvent): Unit = {
                val text = textField.getText + "\n"
                text.foreach(c => uartInputQueue.add(c.toInt))
                textField.setText("")
                println(s"[GUI] Sent: ${text.trim}")
              }
            })
            
            // Also send on Enter key
            textField.addActionListener(new ActionListener {
              override def actionPerformed(actionEvent: ActionEvent): Unit = {
                val text = textField.getText + "\n"
                text.foreach(c => uartInputQueue.add(c.toInt))
                textField.setText("")
                println(s"[GUI] Sent: ${text.trim}")
              }
            })
            
            inputPanel.add(new JLabel("UART Input:"))
            inputPanel.add(textField)
            inputPanel.add(sendButton)
            inputPanel
          }

          add(new JButton("Reset"){
            addActionListener(new ActionListener {
              override def actionPerformed(actionEvent: ActionEvent): Unit = {
                println("ASYNC RESET")
                guiToSim.enqueue("asyncReset")
              }
            })
            setAlignmentX(awt.Component.CENTER_ALIGNMENT)
          })
          setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE)
          pack()
          setVisible(true)

        }

        //Fast refresh
//        clockDomain.onSampling{
//          dut.io.gpioA.read #= (dut.io.gpioA.write.toLong & dut.io.gpioA.writeEnable.toLong) | (switchValue() << 8)
//        }

        //Slow refresh
        while(true){
          sleep(mainClkPeriod*50000)

          val dummy = if(guiToSim.nonEmpty){
            val request = guiToSim.dequeue()
            if(request == "asyncReset"){
              dut.io.asyncReset #= true
              sleep(mainClkPeriod*32)
              dut.io.asyncReset #= false
            }
          }

          dut.io.gpioA.read #= (dut.io.gpioA.write.toLong & dut.io.gpioA.writeEnable.toLong) | (switchValue() << 8)
          ledsValue = dut.io.gpioA.write.toLong
          if ((ledsValue & 0xFF) == 0xAA) {
             println("SIMULATION: PASS (LEDs = 0xAA)")
             simSuccess()
          } else if ((ledsValue & 0xFF) == 0x55) {
             println("SIMULATION: FAIL (LEDs = 0x55)")
             simFailure()
          }
          ledsFrame.repaint()
          if(simSlowDown) Thread.sleep(400)
        }
      }
    }
  }
}
