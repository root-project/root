// @(#)root/mathcore:$Id$
// Author: Jonas Hahnfeld 11/2020

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <cstdint>

/// Multiply two 576 bit numbers, stored as 9 numbers of 64 bits each
///
/// \param[in] in1 first factor as 9 numbers of 64 bits each
/// \param[in] in2 second factor as 9 numbers of 64 bits each
/// \param[out] out result with 18 numbers of 64 bits each
void multiply9x9(const uint64_t *in1, const uint64_t *in2, uint64_t *out)
{
   uint64_t next = 0;
   unsigned nextCarry = 0;

#if defined(__clang__) || defined(__INTEL_COMPILER) || defined(__CUDACC__)
#pragma unroll
#elif defined(__GNUC__) && __GNUC__ >= 8
// This pragma was introduced in GCC version 8.
#pragma GCC unroll 18
#endif
   for (int i = 0; i < 18; i++) {
      uint64_t current = next;
      unsigned carry = nextCarry;

      next = 0;
      nextCarry = 0;

#if defined(__clang__) || defined(__INTEL_COMPILER) || defined(__CUDACC__)
#pragma unroll
#elif defined(__GNUC__) && __GNUC__ >= 8
// This pragma was introduced in GCC version 8.
#pragma GCC unroll 9
#endif
      for (int j = 0; j < 9; j++) {
         int k = i - j;
         if (k < 0 || k >= 9)
            continue;

         uint64_t upper1 = in1[j] >> 32;
         uint64_t lower1 = static_cast<uint32_t>(in1[j]);

         uint64_t upper2 = in2[k] >> 32;
         uint64_t lower2 = static_cast<uint32_t>(in2[k]);

         // Multiply 32-bit parts.
         uint64_t upper = upper1 * upper2;
         uint64_t middle1 = upper1 * lower2;
         uint64_t middle2 = lower1 * upper2;
         uint64_t middle = middle1 + middle2;
         if (middle < middle1) {
            // This can never overflow because the maximum value of upper is
            // (2 ** 32 - 1) ** 2 = 2 ** 64 - 2 * 2 ** 32 + 1. When now adding
            // another 2 ** 32, the result 2 ** 64 - 2 ** 32 + 1 is still smaller
            // than the maximum 2 ** 64 - 1 that can be stored in a uint64_t.
            upper += uint64_t(1) << 32;
         }
         uint64_t lower = lower1 * lower2;

         uint64_t middle_upper = middle >> 32;
         uint64_t middle_lower = middle << 32;

         // Add to current, remember carry.
         current += lower;
         if (current < lower)
            carry++;
         current += middle_lower;
         if (current < middle_lower)
            carry++;

         // Add to next, remember nextCarry.
         next += middle_upper;
         if (next < middle_upper)
            nextCarry++;
         next += upper;
         if (next < upper)
            nextCarry++;
      }

      next += carry;
      if (next < carry)
         nextCarry++;

      out[i] = current;
   }
}

/// Compute a value congruent to mul modulo m less than 2 ** 576
///
/// \param[in] mul product from multiply9x9 with 18 numbers of 64 bits each
/// \param[out] out result with 9 numbers of 64 bits each
///
/// \f$ m = 2^{576} - 2^{240} + 1 \f$
///
/// Note that this function does *not* return the smallest value congruent to
/// the modulus, it only guarantees a value smaller than \f$ 2^{576} \$!
void mod_m(const uint64_t *mul, uint64_t *out)
{
   uint64_t r[9] = {0};

   // r = t0 - t1 (24 * 24 = 576 bits)
   unsigned carry = 0;
   for (int i = 0; i < 9; i++) {
      uint64_t t0_i = mul[i];

      uint64_t r_ic = t0_i - carry;
      if (r_ic > t0_i) {
         carry = 1;
      } else {
         carry = 0;
      }

      uint64_t t1_i = mul[i + 9];
      uint64_t r_i = r_ic - t1_i;
      if (r_i > r_ic)
         carry++;
      r[i] = r_i;
   }
   int64_t c = -((int64_t)carry);

   // r -= t2 (only 240 bits, so need to extend)
   carry = 0;
   for (int i = 0; i < 9; i++) {
      uint64_t r_i = r[i];

      uint64_t r_ic = r_i - carry;
      if (r_ic > r_i) {
         carry = 1;
      } else {
         carry = 0;
      }

      uint64_t t2_bits = 0;
      if (i < 4) {
         t2_bits += mul[i + 14] >> 16;
         if (i < 3) {
            t2_bits += mul[i + 15] << 48;
         }
      }
      r_i = r_ic - t2_bits;
      if (r_i > r_ic)
         carry++;
      r[i] = r_i;
   }
   c -= carry;

   // r += (t3 + t2) * 2 ** 240; copy to output array.
   carry = 0;
   out[0] = r[0];
   out[1] = r[1];
   out[2] = r[2];
   {
      uint64_t r_3 = r[3];
      // 16 upper bits
      uint64_t t2_bits = (mul[14] >> 16) << 48;
      uint64_t t3_bits = (mul[9] << 48);

      r_3 += t2_bits;
      if (r_3 < t2_bits)
         carry++;
      r_3 += t3_bits;
      if (r_3 < t3_bits)
         carry++;

      out[3] = r_3;
   }
   for (int i = 0; i < 3; i++) {
      uint64_t r_i = r[i + 4];
      r_i += carry;
      if (r_i < carry) {
         carry = 1;
      } else {
         carry = 0;
      }

      uint64_t t2_bits = (mul[14 + i] >> 32) + (mul[15 + i] << 32);
      uint64_t t3_bits = (mul[9 + i] >> 16) + (mul[10 + i] << 48);

      r_i += t2_bits;
      if (r_i < t2_bits)
         carry++;
      r_i += t3_bits;
      if (r_i < t3_bits)
         carry++;

      out[i + 4] = r_i;
   }
   {
      uint64_t r_7 = r[7];
      r_7 += carry;
      if (r_7 < carry) {
         carry = 1;
      } else {
         carry = 0;
      }

      uint64_t t2_bits = (mul[17] >> 32);
      uint64_t t3_bits = (mul[12] >> 16) + (mul[13] << 48);

      r_7 += t2_bits;
      if (r_7 < t2_bits)
         carry++;
      r_7 += t3_bits;
      if (r_7 < t3_bits)
         carry++;

      out[7] = r_7;
   }
   {
      uint64_t r_8 = r[8];
      r_8 += carry;
      if (r_8 < carry) {
         carry = 1;
      } else {
         carry = 0;
      }

      uint64_t t3_bits = (mul[13] >> 16) + (mul[14] << 48);

      r_8 += t3_bits;
      if (r_8 < t3_bits)
         carry++;

      out[8] = r_8;
   }
   c += carry;

   // c = floor(r / 2 ** 576) has been computed along the way via the carry
   // flags. Now to update r = r - c * m, it suffices to know c * (-2 ** 240 + 1)
   // because the 2 ** 576 will cancel out. Also note that c may be zero, but
   // the operation is still performed to avoid branching.

   // c * (-2 ** 240 + 1) in 576 bits looks as follows, depending on c:
   //  - if c = 0, the number is zero.
   //  - if c = 1: bits 576 to 240 are set,
   //              bits 239 to 1 are zero, and
   //              the last one is set
   //  - if c = -1, which corresponds to all bits set (signed int64_t):
   //              bits 576 to 240 are zero and the rest is set.
   // Note that all bits except the last are exactly complimentary (unless c = 0)
   // and the last byte is conveniently represented by c already.
   // Now construct the three bit patterns from c, their names correspond to the
   // assembly implementation by Alexei Sibidanov.

   // c = 0 -> t0 = 0; c = 1 -> t0 = 0; c = -1 -> all bits set (sign extension)
   // (The assembly implementation shifts by 63, which gives the same result.)
   int64_t t0 = c >> 1;

   // c = 0 -> t2 = 0; c = 1 -> upper 16 bits set; c = -1 -> lower 48 bits set
   int64_t t2 = t0 - (c << 48);

   // c = 0 -> t1 = 0; c = 1 -> all bits set; c = -1 -> t1 = 0
   // (The assembly implementation shifts by 63, which gives the same result.)
   int64_t t1 = t2 >> 48;

   carry = 0;
   {
      uint64_t r_0 = out[0];

      uint64_t out_0 = r_0 - c;
      if (out_0 > r_0)
         carry++;
      out[0] = out_0;
   }
   for (int i = 1; i < 3; i++) {
      uint64_t r_i = out[i];

      uint64_t r_ic = r_i - carry;
      if (r_ic > r_i) {
         carry = 1;
      } else {
         carry = 0;
      }

      uint64_t out_i = r_ic - t0;
      if (out_i > r_ic)
         carry++;
      out[i] = out_i;
   }
   {
      uint64_t r_3 = out[3];

      uint64_t r_3c = r_3 - carry;
      if (r_3c > r_3) {
         carry = 1;
      } else {
         carry = 0;
      }

      uint64_t out_3 = r_3c - t2;
      if (out_3 > r_3c)
         carry++;
      out[3] = out_3;
   }
   for (int i = 4; i < 9; i++) {
      uint64_t r_i = out[i];

      uint64_t r_ic = r_i - carry;
      if (r_ic > r_i) {
         carry = 1;
      } else {
         carry = 0;
      }

      uint64_t out_i = r_ic - t1;
      if (out_i > r_ic)
         carry++;
      out[i] = out_i;
   }
}

/// Combine multiply9x9 and mod_m with internal temporary storage
///
/// \param[in] in1 first factor with 9 numbers of 64 bits each
/// \param[inout] inout second factor and also the output of the same size
void mulmod(const uint64_t *in1, uint64_t *inout)
{
   uint64_t mul[2 * 9] = {0};
   multiply9x9(in1, inout, mul);
   mod_m(mul, inout);
}

/// Compute base to the n modulo m
///
/// \param[in] base with 9 numbers of 64 bits each
/// \param[out] res output with 9 numbers of 64 bits each
/// \param[in] n exponent
///
/// The arguments base and res may point to the same location.
void powermod(const uint64_t *base, uint64_t *res, uint64_t n)
{
   uint64_t fac[9] = {0};
   fac[0] = base[0];
   res[0] = 1;
   for (int i = 1; i < 9; i++) {
      fac[i] = base[i];
      res[i] = 0;
   }

   uint64_t mul[18] = {0};
   while (n) {
      if (n & 1) {
         multiply9x9(res, fac, mul);
         mod_m(mul, res);
      }
      n >>= 1;
      if (!n)
         break;
      multiply9x9(fac, fac, mul);
      mod_m(mul, fac);
   }
}
