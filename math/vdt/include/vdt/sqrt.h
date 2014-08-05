/*
 * sqrt.h
 * Implementations born on the Quake 3 fast inverse square root
 * function.
 * http://en.wikipedia.org/wiki/Fast_inverse_square_root
 *
 *  Created on: Jun 24, 2012
 *      Author: Danilo Piparo, Thomas Hauth, Vincenzo Innocente
 */

/*
 * VDT is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser Public License for more details.
 *
 * You should have received a copy of the GNU Lesser Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef SQRT_H_
#define SQRT_H_

#include "vdtcore_common.h"

namespace vdt{

//------------------------------------------------------------------------------


/// Sqrt implmentation from Quake3
inline double fast_isqrt_general(double x, const uint32_t ISQRT_ITERATIONS) {

  const double threehalfs = 1.5;
  const double x2 = x * 0.5;
  double y  = x;
  uint64_t i  = details::dp2uint64(y);
  // Evil!
  i  = 0x5fe6eb50c7aa19f9ULL  - ( i >> 1 );
  y  = details::uint642dp(i);
  for (uint32_t j=0;j<ISQRT_ITERATIONS;++j)
      y *= threehalfs - ( x2 * y * y ) ;

  return y;
}

//------------------------------------------------------------------------------

/// Four iterations
inline double fast_isqrt(double x) {return fast_isqrt_general(x,4);}

/// Three iterations
inline double fast_approx_isqrt(double x) {return fast_isqrt_general(x,3);}

//------------------------------------------------------------------------------

/// For comparisons
inline double isqrt (double x) {return 1./std::sqrt(x);}

//------------------------------------------------------------------------------

/// Sqrt implmentation from Quake3
inline float fast_isqrtf_general(float x, const uint32_t ISQRT_ITERATIONS) {

   const float threehalfs = 1.5f;
   const float x2 = x * 0.5f;
   float y  = x;
   uint32_t i  = details::sp2uint32(y);
   i  = 0x5f3759df - ( i >> 1 );
   y  = details::uint322sp(i);
   for (uint32_t j=0;j<ISQRT_ITERATIONS;++j)
      y  *= ( threehalfs - ( x2 * y * y ) );

   return y;
}

//------------------------------------------------------------------------------

/// Two iterations
inline float fast_isqrtf(float x) {return fast_isqrtf_general(x,2);}

/// One (!) iterations
inline float fast_approx_isqrtf(float x) {return fast_isqrtf_general(x,1);}

//------------------------------------------------------------------------------

/// For comparisons
inline float isqrtf (float x) {return 1.f/std::sqrt(x);}

//------------------------------------------------------------------------------

// void isqrtv(const uint32_t size, double const * __restrict__ iarray, double* __restrict__ oarray);
// void fast_isqrtv(const uint32_t size, double const * __restrict__ iarray, double* __restrict__ oarray);
// void fast_approx_isqrtv(const uint32_t size, double const * __restrict__ iarray, double* __restrict__ oarray);
// void isqrtfv(const uint32_t size, float const * __restrict__ iarray, float* __restrict__ oarray);
// void fast_isqrtfv(const uint32_t size, float const * __restrict__ iarray, float* __restrict__ oarray);
// void fast_approx_isqrtfv(const uint32_t size, float const * __restrict__ iarray, float* __restrict__ oarray);

} // end namespace vdt

#endif /* SQRT_H_ */
