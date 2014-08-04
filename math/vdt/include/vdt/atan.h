/*
 * atan.h
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

#ifndef ATAN_H_
#define ATAN_H_

#include "vdtcore_common.h"

namespace vdt{

namespace details{
const double T3PO8 = 2.41421356237309504880;
const double MOREBITSO2 = MOREBITS * 0.5;

inline double get_atan_px(const double x2){

   const double PX1atan = -8.750608600031904122785E-1;
   const double PX2atan = -1.615753718733365076637E1;
   const double PX3atan = -7.500855792314704667340E1;
   const double PX4atan = -1.228866684490136173410E2;
   const double PX5atan = -6.485021904942025371773E1;

   double px = PX1atan;
   px *= x2;
   px += PX2atan;
   px *= x2;
   px += PX3atan;
   px *= x2;
   px += PX4atan;
   px *= x2;
   px += PX5atan;

   return px;
}


inline double get_atan_qx(const double x2){
   const double QX1atan = 2.485846490142306297962E1;
   const double QX2atan = 1.650270098316988542046E2;
   const double QX3atan = 4.328810604912902668951E2;
   const double QX4atan = 4.853903996359136964868E2;
   const double QX5atan = 1.945506571482613964425E2;

   double qx=x2;
   qx += QX1atan;
   qx *=x2;
   qx += QX2atan;
   qx *=x2;
   qx += QX3atan;
   qx *=x2;
   qx += QX4atan;
   qx *=x2;
   qx += QX5atan;

   return qx;
}

}



/// Fast Atan implementation double precision
inline double fast_atan(double x){

   /* make argument positive and save the sign */
   const uint64_t sign_mask = details::getSignMask(x);
   x=std::fabs(x);

   /* range reduction */
   const double originalx=x;

   double y = details::PIO4;
   double factor = details::MOREBITSO2;
   x = (x-1.0) / (x+1.0);

   if( originalx > details::T3PO8 ) {
      y = details::PIO2;
      factor = details::MOREBITS;
      x = -1.0 / originalx ;
   }
   if ( originalx <= 0.66 ) {
      y = 0.;
      factor = 0.;
      x = originalx;
   }

   const double x2 = x * x;

   const double px = details::get_atan_px(x2);
   const double qx = details::get_atan_qx(x2);

   //double res = y +x * x2 * px / qx + x +factor;

   const double poq=px / qx;

   double res = x * x2 * poq + x;
   res+=y;

   res+=factor;

   return details::dpORuint64(res,sign_mask);
}

//------------------------------------------------------------------------------
/// Fast Atan implementation single precision
inline float fast_atanf( float xx ) {

   const uint32_t sign_mask = details::getSignMask(xx);

   float x= std::fabs(xx);
   const float x0=x;
   float y=0.0f;

   /* range reduction */
   if( x0 > 0.4142135623730950f ){ // * tan pi/8
      x = (x0-1.0f)/(x0+1.0f);
      y = details::PIO4F;
   }
   if( x0 > 2.414213562373095f ){  // tan 3pi/8
      x = -( 1.0f/x0 );
      y = details::PIO2F;
   }


   const float x2 = x * x;
   y +=
         ((( 8.05374449538e-2f * x2
               - 1.38776856032E-1f) * x2
               + 1.99777106478E-1f) * x2
               - 3.33329491539E-1f) * x2 * x
               + x;

   return details::spORuint32(y,sign_mask);
}

//------------------------------------------------------------------------------
// // Vector signatures
//
// void atanv(const uint32_t size, double const * __restrict__ iarray, double* __restrict__ oarray);
// void fast_atanv(const uint32_t size, double const * __restrict__ iarray, double* __restrict__ oarray);
// void atanfv(const uint32_t size, float const * __restrict__ iarray, float* __restrict__ oarray);
// void fast_atanfv(const uint32_t size, float const * __restrict__ iarray, float* __restrict__ oarray);

}// end of vdt

#endif // end of atan
