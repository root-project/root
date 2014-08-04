/*
 * vdtcore_common.h
 * Common functions for the vdt routines.
 * The basic idea is to exploit Pade polynomials.
 * A lot of ideas were inspired by the cephes math library (by Stephen L. Moshier
 * moshier@na-net.ornl.gov) as well as actual code for the exp, log, sin, cos,
 * tan, asin, acos and atan functions. The Cephes library can be found here:
 * http://www.netlib.org/cephes/
 *
 *  Created on: Jun 23, 2012
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

#ifndef VDTCOMMON_H_
#define VDTCOMMON_H_

#include "inttypes.h"
#include <cmath>

namespace vdt{

namespace details{

// Constants
const double TWOPI = 2.*M_PI;
const double PI = M_PI;
const double PIO2 = M_PI_2;
const double PIO4 = M_PI_4;
const double ONEOPIO4 = 4./M_PI;

const float TWOPIF = 2.*M_PI;
const float PIF = M_PI;
const float PIO2F = M_PI_2;
const float PIO4F = M_PI_4;
const float ONEOPIO4F = 4./M_PI;

const double MOREBITS = 6.123233995736765886130E-17;


const float MAXNUMF = 3.4028234663852885981170418348451692544e38f;

//------------------------------------------------------------------------------

/// Used to switch between different type of interpretations of the data (64 bits)
union ieee754{
   ieee754 () {};
   ieee754 (double thed) {d=thed;};
   ieee754 (uint64_t thell) {ll=thell;};
   ieee754 (float thef) {f[0]=thef;};
   ieee754 (uint32_t thei) {i[0]=thei;};
  double d;
  float f[2];
  uint32_t i[2];
  uint64_t ll;
  uint16_t s[4];
};

//------------------------------------------------------------------------------

/// Converts an unsigned long long to a double
inline double uint642dp(uint64_t ll) {
  ieee754 tmp;
  tmp.ll=ll;
  return tmp.d;
}

//------------------------------------------------------------------------------

/// Converts a double to an unsigned long long
inline uint64_t dp2uint64(double x) {
  ieee754 tmp;
  tmp.d=x;
  return tmp.ll;
}

//------------------------------------------------------------------------------
/// Makes an AND of a double and a unsigned long long
inline double dpANDuint64(const double x, const uint64_t i ){
  return uint642dp(dp2uint64(x) & i);
}
//------------------------------------------------------------------------------
/// Makes an OR of a double and a unsigned long long
inline double dpORuint64(const double x, const uint64_t i ){
  return uint642dp(dp2uint64(x) | i);
}

/// Makes a XOR of a double and a unsigned long long
inline double dpXORuint64(const double x, const uint64_t i ){
  return uint642dp(dp2uint64(x) ^ i);
}

//------------------------------------------------------------------------------
inline uint64_t getSignMask(const double x){
  const uint64_t mask=0x8000000000000000ULL;
  return dp2uint64(x) & mask;
}

//------------------------------------------------------------------------------
/// Converts an int to a float
inline float uint322sp(int x) {
    ieee754 tmp;
    tmp.i[0]=x;
    return tmp.f[0];
  }

//------------------------------------------------------------------------------
/// Converts a float to an int
inline uint32_t sp2uint32(float x) {
    ieee754 tmp;
    tmp.f[0]=x;
    return tmp.i[0];
  }

//------------------------------------------------------------------------------
/// Makes an AND of a float and a unsigned long
inline float spANDuint32(const float x, const uint32_t i ){
  return uint322sp(sp2uint32(x) & i);
}
//------------------------------------------------------------------------------
/// Makes an OR of a float and a unsigned long
inline float spORuint32(const float x, const uint32_t i ){
  return uint322sp(sp2uint32(x) | i);
}

//------------------------------------------------------------------------------
/// Makes an OR of a float and a unsigned long
inline float spXORuint32(const float x, const uint32_t i ){
  return uint322sp(sp2uint32(x) ^ i);
}
//------------------------------------------------------------------------------
/// Get the sign mask
inline uint32_t getSignMask(const float x){
  const uint32_t mask=0x80000000;
  return sp2uint32(x) & mask;
}

//------------------------------------------------------------------------------
/// Like frexp but vectorising and the exponent is a double.
inline double getMantExponent(const double x, double & fe){

  uint64_t n = dp2uint64(x);

  // Shift to the right up to the beginning of the exponent.
  // Then with a mask, cut off the sign bit
  uint64_t le = (n >> 52);

  // chop the head of the number: an int contains more than 11 bits (32)
  int32_t e = le; // This is important since sums on uint64_t do not vectorise
  fe = e-1023 ;

  // This puts to 11 zeroes the exponent
  n &=0x800FFFFFFFFFFFFFULL;
  // build a mask which is 0.5, i.e. an exponent equal to 1022
  // which means *2, see the above +1.
  const uint64_t p05 = 0x3FE0000000000000ULL; //dp2uint64(0.5);
  n |= p05;

  return uint642dp(n);
}

//------------------------------------------------------------------------------
/// Like frexp but vectorising and the exponent is a float.
inline float getMantExponentf(const float x, float & fe){

    uint32_t n = sp2uint32(x);
    int32_t e = (n >> 23)-127;
    fe = e;

    // fractional part
    const uint32_t p05f = 0x3f000000; // //sp2uint32(0.5);
    n &= 0x807fffff;// ~0x7f800000;
    n |= p05f;

    return uint322sp(n);

}

//------------------------------------------------------------------------------
/// Converts a fp to an int
inline uint32_t fp2uint(float x) {
    return sp2uint32(x);
  }
/// Converts a fp to an int
inline uint64_t fp2uint(double x) {
    return dp2uint64(x);
  }
/// Converts an int to fp
inline float int2fp(uint32_t i) {
    return uint322sp(i);
  }
/// Converts an int to fp
inline double int2fp(uint64_t i) {
    return uint642dp(i);
  }

//------------------------------------------------------------------------------
/**
 * A vectorisable floor implementation, not only triggered by fast-math.
 * These functions do not distinguish between -0.0 and 0.0, so are not IEC6509
 * compliant for argument -0.0
**/
inline double fpfloor(const double x){
  // no problem since exp is defined between -708 and 708. Int is enough for it!
  int32_t ret = int32_t (x);
  ret-=(sp2uint32(x)>>31);
  return ret;

}
//------------------------------------------------------------------------------
/**
 * A vectorisable floor implementation, not only triggered by fast-math.
 * These functions do not distinguish between -0.0 and 0.0, so are not IEC6509
 * compliant for argument -0.0
**/
inline float fpfloor(const float x){
  int32_t ret = int32_t (x);
  ret-=(sp2uint32(x)>>31);
  return ret;

}

//------------------------------------------------------------------------------




}

} // end of namespace vdt

#endif /* VDTCOMMON_H_ */
