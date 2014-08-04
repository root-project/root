/*
 * cos.h
 * The basic idea is to exploit Pade polynomials.
 * A lot of ideas were inspired by the cephes math library (by Stephen L. Moshier
 * moshier@na-net.ornl.gov) as well as actual code.
 * The Cephes library can be found here:  http://www.netlib.org/cephes/
 *
 *  Created on: Jun 23, 2012
 *      Author: Danilo Piparo, Thomas Hauth, Vincenzo Innocente
 */

#ifndef COS_H_
#define COS_H_

#include "sincos.h"

namespace vdt{

// Cos double precision --------------------------------------------------------

/// Double precision cosine: just call sincos.
inline double fast_cos(double x){double s,c;fast_sincos(x,s,c);return c;}

//------------------------------------------------------------------------------

inline float fast_cosf(float x){float s,c;fast_sincosf(x,s,c);return c;}

//------------------------------------------------------------------------------
// void cosv(const uint32_t size, double const * __restrict__ iarray, double* __restrict__ oarray);
// void fast_cosv(const uint32_t size, double const * __restrict__ iarray, double* __restrict__ oarray);
// void cosfv(const uint32_t size, float const * __restrict__ iarray, float* __restrict__ oarray);
// void fast_cosfv(const uint32_t size, float const * __restrict__ iarray, float* __restrict__ oarray);

} //vdt namespace

#endif /* COS_H_ */
