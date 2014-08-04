/*
 * inv.h
 * An experiment: implement division with the square fo the approximate
 * inverse square root.
 * In other words one transforms a shift, multiplications and sums into a
 * sqrt.
 *
 *  Created on: Jun 24, 2012
 *      Author: Danilo Piparo, Thomas Hauth, Vincenzo Innocente
 *
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

#ifndef INV_H_
#define INV_H_

#include "vdtcore_common.h"
#include "sqrt.h"
#include <cmath>
#include <limits>

namespace vdt{

//------------------------------------------------------------------------------

/// General implementation of the inversion
inline double fast_inv_general(double x, const uint32_t isqrt_iterations) {
  const uint64_t sign_mask = details::getSignMask(x);
  const double sqrt_one_over_x = fast_isqrt_general(std::fabs(x),
                                                   isqrt_iterations);
  return sqrt_one_over_x*(details::dpORuint64(sqrt_one_over_x , sign_mask ));
}

//------------------------------------------------------------------------------

/// Four iterations inversion
inline double fast_inv(double x) {return fast_inv_general(x,4);}

//------------------------------------------------------------------------------

/// Three iterations
inline double fast_approx_inv(double x) {return fast_inv_general(x,3);}

//------------------------------------------------------------------------------

/// For comparisons
inline double inv (double x) {return 1./x;}

//------------------------------------------------------------------------------
// Single precision



/// General implementation of the inversion
inline float fast_invf_general(float x, const uint32_t isqrt_iterations) {
  const uint32_t sign_mask = details::getSignMask(x);
  const float sqrt_one_over_x = fast_isqrtf_general(std::fabs(x),
                                                   isqrt_iterations);
  return sqrt_one_over_x*(details::spORuint32(sqrt_one_over_x , sign_mask ));
}

//------------------------------------------------------------------------------

/// Two iterations
inline float fast_invf(float x) {return fast_invf_general(x,2);}

//------------------------------------------------------------------------------

/// One iterations
inline float fast_approx_invf(float x) {return fast_invf_general(x,1);}

//------------------------------------------------------------------------------

/// For comparisons
inline float invf (float x) {return 1.f/x;}

//------------------------------------------------------------------------------

// void invv(const uint32_t size, double const * __restrict__ iarray, double* __restrict__ oarray);
// void fast_invv(const uint32_t size, double const * __restrict__ iarray, double* __restrict__ oarray);
// void fast_approx_invv(const uint32_t size, double const * __restrict__ iarray, double* __restrict__ oarray);
// void invfv(const uint32_t size, float const * __restrict__ iarray, float* __restrict__ oarray);
// void fast_invfv(const uint32_t size, float const * __restrict__ iarray, float* __restrict__ oarray);
// void fast_approx_invfv(const uint32_t size, float const * __restrict__ iarray, float* __restrict__ oarray);

} // end namespace vdt

#endif /* INV_H_ */
