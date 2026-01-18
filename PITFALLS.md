# Common Pitfalls in VexRiscv Custom Instruction Development

This document records technical issues encountered during the development of custom instructions for VexRiscv and their solutions.

## 1. Plugin Instance Reuse & Double Setup AssertionError

### Symptom
When running the generation script (e.g., via `sbt`), the build fails with an `AssertionError` in `DecoderSimplePlugin.scala` before reporting the actual elaboration error.

```text
[error] Exception in thread "main" java.lang.AssertionError: assertion failed
[error] 	at vexriscv.plugin.DecoderSimplePlugin.addDefault(DecoderSimplePlugin.scala:67)
[error] 	at vexriscv.plugin.DBusSimplePlugin.setup(DBusSimplePlugin.scala:348)
```

### Cause
SpinalHDL has a feature where it restarts the elaboration process if it detects certain types of errors (or for other internal reasons).
If your `VexRiscvConfig` (and thus the `cpuPlugins` list) is defined **outside** the `SpinalVerilog` block, the *same* plugin instances are reused for the second elaboration attempt.
Since `setup()` registers default values in a persistent map (in `DecoderSimplePlugin`), running `setup()` a second time on the same instance triggers an assertion that the key already exists.

### Solution
**Always instantiate your configuration and plugins INSIDE the `SpinalVerilog` block.**  This ensures that fresh plugin instances are created for every elaboration attempt.

**Bad Pattern:**
```scala
val config = MyConfig.default // Plugins created here
SpinalVerilog(new VexRiscv(config)) // Reused on restart
```

**Good Pattern:**
```scala
SpinalVerilog {
  val config = MyConfig.default // Fresh plugins created here
  new VexRiscv(config)
}
```

## 2. Bit Width Mismatches (Silent or Late Failure)

### Symptom
The build fails with a `WIDTH MISMATCH` error during the `PhaseNormalizeNodeInputs` phase, often pointing to an assignment in the `build()` method of your plugin.

```text
[error] WIDTH MISMATCH (32 bits <- 64 bits) on (toplevel/system_cpu/??? :  Bits[32 bits]) := (UInt -> Bits of 64 bits)
```

### Cause
SpinalHDL infers widths automatically for operations. For example, multiplying two 32-bit `UInt` values (`rs1 * rs2`) produces a **64-bit** result.
The VexRiscv `REGFILE_WRITE_DATA` signal is 32 bits wide. Attempting to assign a 64-bit result directly to it causes a width mismatch error.

### Solution
Explicitly resize or truncate the result to match the destination width using `.resize(width)`.

**Example:**
```scala
// rs1 is 32 bits. m is 32 bits.
// result is implicitly 64 bits.
// .resize(32) truncates it back to 32 bits.
val result = (rs1 * U(m, 32 bits)).resize(32) + U(b, 32 bits)

execute.output(REGFILE_WRITE_DATA) := result.asBits.resize(32)
```
