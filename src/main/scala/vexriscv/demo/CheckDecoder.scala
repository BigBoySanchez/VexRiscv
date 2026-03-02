package vexriscv.demo
import spinal.core._
object CheckDecoder {
  def main(args: Array[String]) {
    SpinalVerilog(new BlockDialectDecoder())
  }
}
