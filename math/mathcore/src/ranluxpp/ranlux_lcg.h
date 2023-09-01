// @(#)root/mathcore:$Id$
// Author: Jonas Hahnfeld 05/2021

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef RANLUXPP_RANLUX_LCG_H
#define RANLUXPP_RANLUX_LCG_H

#include "helpers.h"

#include <cstdint>

/// Convert RANLUX numbers to an LCG state
///
/// \param[in] ranlux the RANLUX numbers as 576 bits
/// \param[out] lcg the 576 bits of the LCG state, smaller than m
/// \param[in] c the carry bit of the RANLUX state
///
/// \f$ m = 2^{576} - 2^{240} + 1 \f$
static void to_lcg(const uint64_t *ranlux, unsigned c, uint64_t *lcg)
{
   unsigned carry = 0;
   // Subtract the final 240 bits.
   for (int i = 0; i < 9; i++) {
      uint64_t ranlux_i = ranlux[i];
      uint64_t lcg_i = sub_overflow(ranlux_i, carry, carry);

      uint64_t bits = 0;
      if (i < 4) {
         bits += ranlux[i + 5] >> 16;
         if (i < 3) {
            bits += ranlux[i + 6] << 48;
         }
      }
      lcg_i = sub_carry(lcg_i, bits, carry);
      lcg[i] = lcg_i;
   }

   // Add and propagate the carry bit.
   for (int i = 0; i < 9; i++) {
      lcg[i] = add_overflow(lcg[i], c, c);
   }
}

/// Convert an LCG state to RANLUX numbers
///
/// \param[in] lcg the 576 bits of the LCG state, must be smaller than m
/// \param[out] ranlux the RANLUX numbers as 576 bits
/// \param[out] c_out the carry bit of the RANLUX state
///
/// \f$ m = 2^{576} - 2^{240} + 1 \f$
static void to_ranlux(const uint64_t *lcg, uint64_t *ranlux, unsigned &c_out)
{
   uint64_t r[9] = {0};
   int64_t c = compute_r(lcg, r);

   // ranlux = t1 + t2 + c
   unsigned carry = 0;
   for (int i = 0; i < 9; i++) {
      uint64_t in_i = lcg[i];
      uint64_t tmp_i = add_overflow(in_i, carry, carry);

      uint64_t bits = 0;
      if (i < 4) {
         bits += lcg[i + 5] >> 16;
         if (i < 3) {
            bits += lcg[i + 6] << 48;
         }
      }
      tmp_i = add_carry(tmp_i, bits, carry);
      ranlux[i] = tmp_i;
   }

   // If c = -1, we need to add it to all components.
   int64_t c1 = c >> 1;
   ranlux[0] = add_overflow(ranlux[0], c, carry);
   for (int i = 1; i < 9; i++) {
      uint64_t ranlux_i = ranlux[i];
      ranlux_i = add_overflow(ranlux_i, carry, carry);
      ranlux_i = add_carry(ranlux_i, c1, carry);
   }

   c_out = carry;
}

#endif
