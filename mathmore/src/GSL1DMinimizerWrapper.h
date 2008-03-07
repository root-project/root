// @(#)root/mathmore:$Id$
// Authors: L. Moneta, A. Zsenei   08/2005
 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2004 moneta,  CERN/PH-SFT                            *
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

// Header file for class GSL1DMinimizerWrapper
// 
// Created by: moneta  at Wed Dec  1 17:25:44 2004
// 
// Last update: Wed Dec  1 17:25:44 2004
// 
#ifndef ROOT_Math_GSL1DMinimizerWrapper
#define ROOT_Math_GSL1DMinimizerWrapper

#include "gsl/gsl_min.h"


namespace ROOT { 

namespace Math { 

/**
   wrapper class for gsl_min_fminimizer structure
   @ingroup Min1D
*/
class GSL1DMinimizerWrapper {

public: 
   GSL1DMinimizerWrapper( const gsl_min_fminimizer_type * T) 
   {
      fMinimizer = gsl_min_fminimizer_alloc(T); 
   }
   virtual ~GSL1DMinimizerWrapper() { 
      gsl_min_fminimizer_free(fMinimizer);
   }

private:
// usually copying is non trivial, so we make this unaccessible
   GSL1DMinimizerWrapper(const GSL1DMinimizerWrapper &); 
   GSL1DMinimizerWrapper & operator = (const GSL1DMinimizerWrapper &); 

public: 

   gsl_min_fminimizer * Get() const { 
      return fMinimizer; 
   }


private: 

   gsl_min_fminimizer * fMinimizer; 

}; 

} // end namespace Math
} // end namespace ROOT

#endif /* ROOT_Math_GSL1DMinimizerWrapper */
