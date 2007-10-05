// @(#)root/mathmore:$Id$
// Author: Magdalena Slawinska 08/2007

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2004 ROOT Foundation,  CERN/PH-SFT                   *
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


#ifndef ROOT_Math_MCParameters
#define ROOT_Math_MCParameters


namespace ROOT {
namespace Math {



/**
   structures collecting parameters 
   for VEGAS multidimensional integration

   @ingroup MCIntegration
*/
struct VegasParameters{
  double sigma;
  double chisq;
  double alpha;
  size_t iterations;
  
  VegasParameters():
    alpha( 1.5),
    iterations(5)
  {} 
 

//int stage;
  //int mode;
  //int verbose;

};


/**
   structures collecting parameters 
   for MISER multidimensional integration

   @ingroup MCIntegration
*/
struct MiserParameters{
  double estimate_frac;
  size_t min_calls;
  size_t min_calls_per_bisection;
  double alpha;

  MiserParameters(unsigned int dim):
    estimate_frac(0.1),
    min_calls(16*dim),
    min_calls_per_bisection(32*min_calls) ,
    alpha(2.)
  {}
};

struct PlainParameters{

};

} // namespace Math
} // namespace ROOT

#endif 
