package vexriscv.demo

import spinal.core._
import vexriscv.plugin.Plugin
import vexriscv.{Stageable, DecoderService, VexRiscv}

/**
 * A plugin that implements the linear function y = mx + b.
 * m and b are constants defined at synthesis time (Scala parameters).
 *
 * @param m The slope constant.
 * @param b The intercept constant.
 * @param instructionPattern The bit pattern for decoding (e.g., "0000000----------000-----0001011")
 */
class MxPlusBPlugin(m: Int, b: Int, instructionPattern: String) extends Plugin[VexRiscv]{
  //Unique signal to identify this specific plugin's instruction in the pipeline
  object IS_MX_PLUS_B extends Stageable(Bool)

  // Guard to prevent double setup
  var setupDone = false

  override def setup(pipeline: VexRiscv): Unit = {
    if(setupDone) return
    setupDone = true

    import pipeline.config._

    //Retrieve the DecoderService definition
    val decoderService = pipeline.service(classOf[DecoderService])

    //Default value: Not this instruction
    decoderService.addDefault(IS_MX_PLUS_B, False)

    //Register the decoding logic
    decoderService.add(
      key = MaskedLiteral(instructionPattern),
      List(
        IS_MX_PLUS_B             -> True,
        REGFILE_WRITE_VALID      -> True, // Write to RD
        BYPASSABLE_EXECUTE_STAGE -> True, // Result ready in Execute stage
        BYPASSABLE_MEMORY_STAGE  -> True, // Result ready in Memory stage
        RS1_USE                  -> True  // Uses RS1
        // RS2 is not used
      )
    )
  }

  override def build(pipeline: VexRiscv): Unit = {
    import pipeline._
    import pipeline.config._

    //Execute stage logic
    execute plug new Area {
      //Get RS1 value
      val rs1 = execute.input(RS1).asUInt
      
      //Compute y = mx + b
      //We perform 32-bit arithmetic. Overflows wrap around.
      val result = ((rs1 * U(m, 32 bits)).resize(32) + U(b, 32 bits)).resize(32)

      //Provide the result to the writeback path when strictly this instruction is active
      when(execute.input(IS_MX_PLUS_B)) {
        execute.output(REGFILE_WRITE_DATA) := result.asBits
      }
    }
  }
}
