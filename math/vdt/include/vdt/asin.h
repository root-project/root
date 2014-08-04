/*
 * aasin.h
 * The basic idea is to exploit Pade' polynomials.
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

#ifndef ASIN_H_
#define ASIN_H_

#include "vdtcore_common.h"

namespace vdt{

namespace details{

const double RX1asin = 2.967721961301243206100E-3;
const double RX2asin = -5.634242780008963776856E-1;
const double RX3asin = 6.968710824104713396794E0;
const double RX4asin = -2.556901049652824852289E1;
const double RX5asin = 2.853665548261061424989E1;

const double SX1asin = -2.194779531642920639778E1;
const double SX2asin =  1.470656354026814941758E2;
const double SX3asin = -3.838770957603691357202E2;
const double SX4asin = 3.424398657913078477438E2;

const double PX1asin = 4.253011369004428248960E-3;
const double PX2asin = -6.019598008014123785661E-1;
const double PX3asin = 5.444622390564711410273E0;
const double PX4asin = -1.626247967210700244449E1;
const double PX5asin = 1.956261983317594739197E1;
const double PX6asin = -8.198089802484824371615E0;

const double QX1asin = -1.474091372988853791896E1;
const double QX2asin =  7.049610280856842141659E1;
const double QX3asin = -1.471791292232726029859E2;
const double QX4asin = 1.395105614657485689735E2;
const double QX5asin = -4.918853881490881290097E1;

inline double getRX(const double x){
   double rx = RX1asin;
   rx*= x;
   rx+= RX2asin;
   rx*= x;
   rx+= RX3asin;
   rx*= x;
   rx+= RX4asin;
   rx*= x;
   rx+= RX5asin;
   return rx;
}
inline double getSX(const double x){
   double sx = x;
   sx+= SX1asin;
   sx*= x;
   sx+= SX2asin;
   sx*= x;
   sx+= SX3asin;
   sx*= x;
   sx+= SX4asin;
   return sx;
}

inline double getPX(const double x){
   double px = PX1asin;
   px*= x;
   px+= PX2asin;
   px*= x;
   px+= PX3asin;
   px*= x;
   px+= PX4asin;
   px*= x;
   px+= PX5asin;
   px*= x;
   px+= PX6asin;
   return px;
}

inline double getQX(const double x){
   double qx = x;
   qx+= QX1asin;
   qx*= x;
   qx+= QX2asin;
   qx*= x;
   qx+= QX3asin;
   qx*= x;
   qx+= QX4asin;
   qx*= x;
   qx+= QX5asin;
   return qx;
   }
}

}

namespace vdt{

// asin double precision --------------------------------------------------------
/// Double Precision asin
inline double fast_asin(double x){

   const uint64_t sign_mask = details::getSignMask(x);
   x = std::fabs(x);
   const double a = x;


   double zz = 1.0 - a;
   double px = details::getRX(zz);
   double qx = details::getSX(zz);

   const double p = zz * px/qx;

   zz = std::sqrt(zz+zz);
   double z = details::PIO4 - zz;
   zz = zz * p - details::MOREBITS;
   z -= zz;
   z += details::PIO4;

   if( a < 0.625 ){
      zz = a * a;
      px = details::getPX(zz);
      qx = details::getQX(zz);
      z = zz*px/qx;
      z = a * z + a;
   }


   // Linear approx, not sooo needed but seable. Price is cheap though
   double res = a < 1e-8? a : z ;
        // Restore Sign
   return details::dpORuint64(res,sign_mask);

}

//------------------------------------------------------------------------------
/// Single Precision asin
inline float fast_asinf(float x){


    uint32_t flag=0;

    const uint32_t sign_mask = details::getSignMask(x);
    const float a = std::fabs(x);

    float z;
    if( a > 0.5f )
        {
        z = 0.5f * (1.0f - a);
        x = sqrtf( z );
        flag = 1;
        }
    else
        {
        x = a;
        z = x * x;
        }

    z = (((( 4.2163199048E-2f * z
            + 2.4181311049E-2f) * z
            + 4.5470025998E-2f) * z
            + 7.4953002686E-2f) * z
            + 1.6666752422E-1f) * z * x
            + x;

//     if( flag != 0 )
//       {
//       z = z + z;
//       z = PIO2F - z;
//       }

    // No branch with the two coefficients

    float tmp = z + z;
    tmp = details::PIO2F - tmp;

    // Linear approx, not sooo needed but seable. Price is cheap though
    float res = a < 1e-4f? a : tmp * flag + (1-flag) * z ;

    // Restore Sign
    return details::spORuint32(res,sign_mask);

    return( z );
}

//------------------------------------------------------------------------------
// The cos is in this file as well

inline double fast_acos( double x ){return details::PIO2  - fast_asin(x);}

//------------------------------------------------------------------------------

inline float fast_acosf( float x ){return details::PIO2F  - fast_asinf(x);}

//------------------------------------------------------------------------------

// // Vector signatures
//
// void asinv(const uint32_t size, double const * __restrict__ iarray, double* __restrict__ oarray);
// void fast_asinv(const uint32_t size, double const * __restrict__ iarray, double* __restrict__ oarray);
// void asinfv(const uint32_t size, float const * __restrict__ iarray, float* __restrict__ oarray);
// void fast_asinfv(const uint32_t size, float const * __restrict__ iarray, float* __restrict__ oarray);
//
// void acosv(const uint32_t size, double const * __restrict__ iarray, double* __restrict__ oarray);
// void fast_acosv(const uint32_t size, double const * __restrict__ iarray, double* __restrict__ oarray);
// void acosfv(const uint32_t size, float const * __restrict__ iarray, float* __restrict__ oarray);
// void fast_acosfv(const uint32_t size, float const * __restrict__ iarray, float* __restrict__ oarray);

} //vdt namespace

#endif /* ASIN_H_ */
