#ifndef EXPECTED_H
#define EXPECTED_H

#include <stdint.h>

// ============================================================================
// Expected values for Phase B self-tests (Option A: half-units → scale to int8)
//
// Test block: dialect_id=15, shared_exp_bits=18 (shift=2, ×4)
//   metadata bytes: 0xF9 0x00
//   packed bytes:   0x01 0x23 0x45 0x67 0x89 0xAB 0xCD 0xEF (repeated ×2)
//
// Self-test 1 — decoder raw half-units (BD_DECODED output, signed int8):
//   Dialect 15 table (half-units): [0,1,2,3,4,5,6,8]
//   Elements: {0,1,2,3,4,5,6,8, 0,-1,-2,-3,-4,-5,-6,-8} ×2
//
// Self-test 2 — scale_half_units(hu, sexp=18) = hu × 2^(18-16) = hu × 4:
//   Elements: {0,4,8,12,16,20,24,32, 0,-4,-8,-12,-16,-20,-24,-32} ×2
// ============================================================================

// Self-test 1: expected signed half-units from hardware decoder
const int8_t EXPECTED_HALF_UNITS[32] = {
     0,  1,  2,  3,  4,  5,  6,  8,
     0, -1, -2, -3, -4, -5, -6, -8,
     0,  1,  2,  3,  4,  5,  6,  8,
     0, -1, -2, -3, -4, -5, -6, -8,
};

// Self-test 2: expected int8 weights after scale_half_units (Option A path)
const int8_t EXPECTED_SCALED[32] = {
     0,  4,  8, 12, 16, 20, 24, 32,
     0, -4, -8,-12,-16,-20,-24,-32,
     0,  4,  8, 12, 16, 20, 24, 32,
     0, -4, -8,-12,-16,-20,-24,-32,
};

#endif
