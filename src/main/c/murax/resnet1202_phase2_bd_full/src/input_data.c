/* input_data.c — Pulls the RN1202_INPUT array into the firmware image.
 *
 * input.h defines `const int8_t RN1202_INPUT[3072]` (CIFAR-10 test image).
 * Including it here places the array in .rodata → BRAM.
 * main.c references it via `extern const int8_t RN1202_INPUT[]`.
 */
#include "input.h"
