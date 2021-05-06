// @(#)root/mathcore:$Id$
// Author: Jonas Hahnfeld 11/2020

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef RANLUXPP_HELPERS_H
#define RANLUXPP_HELPERS_H

#include <cstdint>

/// Compute `a + b` and set `overflow` accordingly.
static inline uint64_t add_overflow(uint64_t a, uint64_t b, unsigned &overflow)
{
   uint64_t add = a + b;
   overflow = (add < a);
   return add;
}

/// Compute `a + b` and increment `carry` if there was an overflow
static inline uint64_t add_carry(uint64_t a, uint64_t b, unsigned &carry)
{
   unsigned overflow;
   uint64_t add = add_overflow(a, b, overflow);
   // Do NOT branch on overflow to avoid jumping code, just add 0 if there was
   // no overflow.
   carry += overflow;
   return add;
}

/// Compute `a - b` and set `overflow` accordingly
static inline uint64_t sub_overflow(uint64_t a, uint64_t b, unsigned &overflow)
{
   uint64_t sub = a - b;
   overflow = (sub > a);
   return sub;
}

/// Compute `a - b` and increment `carry` if there was an overflow
static inline uint64_t sub_carry(uint64_t a, uint64_t b, unsigned &carry)
{
   unsigned overflow;
   uint64_t sub = sub_overflow(a, b, overflow);
   // Do NOT branch on overflow to avoid jumping code, just add 0 if there was
   // no overflow.
   carry += overflow;
   return sub;
}

/// Update r = r - (t1 + t2) + (t3 + t2) * b ** 10
///
/// This function also yields cbar = floor(r / m) as its return value (int64_t
/// because the value can be -1). With an initial value of r = t0, this can
/// be used for computing the remainder after division by m (see the function
/// mod_m in mulmod.h). The function to_ranlux passes r = 0 and uses only the
/// return value to obtain the decimal expansion after divison by m.
static inline int64_t compute_r(const uint64_t *upper, uint64_t *r)
{
   // Subtract t1 (24 * 24 = 576 bits)
   unsigned carry = 0;
   for (int i = 0; i < 9; i++) {
      uint64_t r_i = r[i];
      r_i = sub_overflow(r_i, carry, carry);

      uint64_t t1_i = upper[i];
      r_i = sub_carry(r_i, t1_i, carry);
      r[i] = r_i;
   }
   int64_t c = -((int64_t)carry);

   // Subtract t2 (only 240 bits, so need to extend)
   carry = 0;
   for (int i = 0; i < 9; i++) {
      uint64_t r_i = r[i];
      r_i = sub_overflow(r_i, carry, carry);

      uint64_t t2_bits = 0;
      if (i < 4) {
         t2_bits += upper[i + 5] >> 16;
         if (i < 3) {
            t2_bits += upper[i + 6] << 48;
         }
      }
      r_i = sub_carry(r_i, t2_bits, carry);
      r[i] = r_i;
   }
   c -= carry;

   // r += (t3 + t2) * 2 ** 240
   carry = 0;
   {
      uint64_t r_3 = r[3];
      // 16 upper bits
      uint64_t t2_bits = (upper[5] >> 16) << 48;
      uint64_t t3_bits = (upper[0] << 48);

      r_3 = add_carry(r_3, t2_bits, carry);
      r_3 = add_carry(r_3, t3_bits, carry);

      r[3] = r_3;
   }
   for (int i = 0; i < 3; i++) {
      uint64_t r_i = r[i + 4];
      r_i = add_overflow(r_i, carry, carry);

      uint64_t t2_bits = (upper[5 + i] >> 32) + (upper[6 + i] << 32);
      uint64_t t3_bits = (upper[i] >> 16) + (upper[1 + i] << 48);

      r_i = add_carry(r_i, t2_bits, carry);
      r_i = add_carry(r_i, t3_bits, carry);

      r[i + 4] = r_i;
   }
   {
      uint64_t r_7 = r[7];
      r_7 = add_overflow(r_7, carry, carry);

      uint64_t t2_bits = (upper[8] >> 32);
      uint64_t t3_bits = (upper[3] >> 16) + (upper[4] << 48);

      r_7 = add_carry(r_7, t2_bits, carry);
      r_7 = add_carry(r_7, t3_bits, carry);

      r[7] = r_7;
   }
   {
      uint64_t r_8 = r[8];
      r_8 = add_overflow(r_8, carry, carry);

      uint64_t t3_bits = (upper[4] >> 16) + (upper[5] << 48);

      r_8 = add_carry(r_8, t3_bits, carry);

      r[8] = r_8;
   }
   c += carry;

   // c = floor(r / 2 ** 576) has been computed along the way via the carry
   // flags. Now if c = 0 and the value currently stored in r is greater or
   // equal to m, we need cbar = 1 and subtract m, otherwise cbar = c. The
   // value currently in r is greater or equal to m, if and only if one of
   // the last 240 bits is set and the upper bits are all set.
   bool greater_m = r[0] | r[1] | r[2] | (r[3] & 0x0000ffffffffffff);
   greater_m &= (r[3] >> 48) == 0xffff;
   for (int i = 4; i < 9; i++) {
      greater_m &= (r[i] == UINT64_MAX);
   }
   return c + (c == 0 && greater_m);
}

#endif
