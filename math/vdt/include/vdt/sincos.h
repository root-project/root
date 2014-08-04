/*
 * sincos_common.h
 * The basic idea is to exploit Pade polynomials.
 * A lot of ideas were inspired by the cephes math library (by Stephen L. Moshier
 * moshier@na-net.ornl.gov) as well as actual code.
 * The Cephes library can be found here:  http://www.netlib.org/cephes/
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

#include "vdtcore_common.h"
#include <cmath>
#include <limits>

#ifndef SINCOS_COMMON_H_
#define SINCOS_COMMON_H_

namespace vdt{

namespace details{

// double precision constants

const double DP1sc = 7.85398125648498535156E-1;
const double DP2sc = 3.77489470793079817668E-8;
const double DP3sc = 2.69515142907905952645E-15;

const double C1sin = 1.58962301576546568060E-10;
const double C2sin =-2.50507477628578072866E-8;
const double C3sin = 2.75573136213857245213E-6;
const double C4sin =-1.98412698295895385996E-4;
const double C5sin = 8.33333333332211858878E-3;
const double C6sin =-1.66666666666666307295E-1;

const double C1cos =-1.13585365213876817300E-11;
const double C2cos = 2.08757008419747316778E-9;
const double C3cos =-2.75573141792967388112E-7;
const double C4cos = 2.48015872888517045348E-5;
const double C5cos =-1.38888888888730564116E-3;
const double C6cos = 4.16666666666665929218E-2;

const double DP1 = 7.853981554508209228515625E-1;
const double DP2 = 7.94662735614792836714E-9;
const double DP3 = 3.06161699786838294307E-17;

// single precision constants

const float DP1F = 0.78515625;
const float DP2F = 2.4187564849853515625e-4;
const float DP3F = 3.77489497744594108e-8;

const float T24M1 = 16777215.;

//------------------------------------------------------------------------------

inline double get_sin_px(const double x){
   double px=C1sin;
   px *= x;
   px += C2sin;
   px *= x;
   px += C3sin;
   px *= x;
   px += C4sin;
   px *= x;
   px += C5sin;
   px *= x;
   px += C6sin;
   return px;
}

//------------------------------------------------------------------------------

inline double get_cos_px(const double x){
   double px=C1cos;
   px *= x;
   px += C2cos;
   px *= x;
   px += C3cos;
   px *= x;
   px += C4cos;
   px *= x;
   px += C5cos;
   px *= x;
   px += C6cos;
   return px;
}


//------------------------------------------------------------------------------
/// Reduce to 0 to 45
inline double reduce2quadrant(double x, int32_t& quad) {

    x = fabs(x);
    quad = int (ONEOPIO4 * x); // always positive, so (int) == std::floor
    quad = (quad+1) & (~1);
    const double y = double (quad);
    // Extended precision modular arithmetic
    return ((x - y * DP1) - y * DP2) - y * DP3;
  }

//------------------------------------------------------------------------------
/// Sincos only for -45deg < x < 45deg
inline void fast_sincos_m45_45( const double z, double & s, double &c ) {

    double zz = z * z;
    s = z  +  z * zz * get_sin_px(zz);
    c = 1.0 - zz * .5 + zz * zz * get_cos_px(zz);
  }


//------------------------------------------------------------------------------

} // End namespace details

/// Double precision sincos
inline void fast_sincos( const double xx, double & s, double &c ) {
    // I have to use doubles to make it vectorise...

    int j;
    double x = details::reduce2quadrant(xx,j);
    const double signS = (j&4);

    j-=2;

    const double signC = (j&4);
    const double poly = j&2;

    details::fast_sincos_m45_45(x,s,c);

    //swap
    if( poly==0 ) {
      const double tmp = c;
      c=s;
      s=tmp;
    }

    if(signC == 0.)
      c = -c;
    if(signS != 0.)
      s = -s;
    if (xx < 0.)
      s = -s;

  }


// Single precision functions

namespace details {
//------------------------------------------------------------------------------
/// Reduce to 0 to 45
inline float reduce2quadrant(float x, int & quad) {
    /* make argument positive */
    x = fabs(x);

    quad = int (ONEOPIO4F * x); /* integer part of x/PIO4 */

    quad = (quad+1) & (~1);
    const float y = float(quad);
    // quad &=4;
    // Extended precision modular arithmetic
    return ((x - y * DP1F) - y * DP2F) - y * DP3F;
  }


//------------------------------------------------------------------------------



/// Sincos only for -45deg < x < 45deg
inline void fast_sincosf_m45_45( const float x, float & s, float &c ) {

    float z = x * x;

    s = (((-1.9515295891E-4f * z
       + 8.3321608736E-3f) * z
      - 1.6666654611E-1f) * z * x)
      + x;

    c = ((  2.443315711809948E-005f * z
        - 1.388731625493765E-003f) * z
     + 4.166664568298827E-002f) * z * z
      - 0.5f * z + 1.0f;
  }

//------------------------------------------------------------------------------

} // end details namespace

/// Single precision sincos
inline void fast_sincosf( const float xx, float & s, float &c ) {


    int j;
    const float x = details::reduce2quadrant(xx,j);
    int signS = (j&4);

    j-=2;

    const int signC = (j&4);
    const int poly = j&2;

    float ls,lc;
    details::fast_sincosf_m45_45(x,ls,lc);

    //swap
    if( poly==0 ) {
      const float tmp = lc;
      lc=ls; ls=tmp;
    }

    if(signC == 0) lc = -lc;
    if(signS != 0) ls = -ls;
    if (xx<0)  ls = -ls;
    c=lc;
    s=ls;
  }


} // end namespace vdt

#endif
