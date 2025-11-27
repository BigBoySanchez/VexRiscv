1. dependencies
    - JDK 17.0.7
    - sbt 1.11.7
    - Verilator 5.006
3. follow custom instruction example
4. run regression test (to ensure prev instructions didn't break; 
ideally on a single config to save time, runs all configs otherwise)

5. write and run tests for custom instruction separately
6. flash code to a real fpga (TODO: need more research on 
does prof actually need it/how to do this)

## Pitfalls
- **EVERYTHING** has to be run in Linux. 
- When using WSL, don't work in a Windows dir.