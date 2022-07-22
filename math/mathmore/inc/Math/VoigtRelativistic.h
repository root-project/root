// @(#)root/mathmore:$Id$
// Author: J. Lindon Wed Jun 15 02:35:26 2022

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2022  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 * This library is free software; you can redistribute it and/or      *
 * modify it under the terms of the GNU General Public License        *
 * as published by the Free Software Foundation; either version 2     *
 * of the License, or (at your option) any later version.             *
 *                                                                    *
 * This library is distributed in the hope that it will be useful,    *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of     *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU   *
 * General Public License for more details.                           *
 *                                                                    *
 * You should have received a copy of the GNU General Public License  *
 * along with this library (see file COPYING); if not, write          *
 * to the Free Software Foundation, Inc., 59 Temple Place, Suite      *
 * 330, Boston, MA 02111-1307 USA, or contact the author.             *
 *                                                                    *
 **********************************************************************/

// Header file for class VoigtRelativistic

#ifndef ROOT_Math_VoigtRelativistic
#define ROOT_Math_VoigtRelativistic

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
// VoigtRelativistic                                                        //
//                                                                          //
// Calculates the relativistic voigt function                               //
// (convolution of a relativistic breit wigner and a Gaussian)              //
// Implemented as equation 9 of https://doi.org/10.1016/j.jmaa.2018.03.065  //
// With normalization changed to match TMath::Voigt (non relativistic voigt)//
//                                                                          //
//////////////////////////////////////////////////////////////////////////////


namespace ROOT {
  namespace Math {

    class VoigtRelativistic
    {
    public:
      // The relativistic voigt function
      static double evaluate(double x, double median, double sigma, double lg, double integrationRange=26.615717509251260);//t=26.615717509251260 gives exp(-t^2)=minimum value stored by double.
      static double dumpingFunction(double median, double sigma, double lg, double integrationRange=26.615717509251260);//t=26.615717509251260 gives exp(-t^2)=minimum value stored by double.
      
      // Include an empty virtual desctructor to eliminate compiler warnings
      virtual ~VoigtRelativistic() {}

    protected:

    };

  } // namespace Math
} // namespace ROOT


#endif
