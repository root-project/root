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

#include <cstring>   // for size_t

namespace ROOT {
namespace Math {


class IOptions;

/**
   structures collecting parameters
   for VEGAS multidimensional integration
   FOr implementation of default parameters see file
   mathmore/src/GSLMCIntegrationWorkspace.h

   @ingroup MCIntegration
*/
struct VegasParameters{
   double alpha;
   size_t iterations;
   int stage;
   int mode;
   int verbose;

   // constructor of default parameters
   VegasParameters() { SetDefaultValues(); }

   // construct from GenAlgoOptions
   // parameter not specified are ignored
   VegasParameters(const ROOT::Math::IOptions & opt);

   void SetDefaultValues();

   VegasParameters & operator=(const ROOT::Math::IOptions & opt);

   /// convert to options (return object is managed by the user)
   ROOT::Math::IOptions * operator() () const;
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
   double dither;

   // constructor of default parameters
   // needs dimension since min_calls = 16 * dim
   MiserParameters(size_t dim=10) { SetDefaultValues(dim); }

   void SetDefaultValues(size_t dim=10);

   // construct from GenAlgoOptions
   // parameter not specified are ignored
   MiserParameters(const ROOT::Math::IOptions & opt, size_t dim = 10);

   MiserParameters & operator=(const ROOT::Math::IOptions & opt);

   /// convert to options (return object is managed by the user)
   ROOT::Math::IOptions * operator() () const;

};

struct PlainParameters{
};

} // namespace Math
} // namespace ROOT

#endif
