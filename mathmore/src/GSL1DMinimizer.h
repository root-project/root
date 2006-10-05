// @(#)root/mathmore:$Name:  $:$Id: Integrator.h,v 1.2 2006/06/16 10:34:08 moneta Exp $
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

// Header file for class GSL1DMinimizer
// 
// Created by: moneta  at Wed Dec  1 17:25:44 2004
// 
// Last update: Wed Dec  1 17:25:44 2004
// 
#ifndef ROOT_Math_GSL1DMinimizer
#define ROOT_Math_GSL1DMinimizer

#include "gsl/gsl_min.h"

/**
   wrapper class for gsl_min_fminimizer structure
   @ingroup Min1D
*/

namespace ROOT { 

namespace Math { 

class GSL1DMinimizer {

public: 
   GSL1DMinimizer( const gsl_min_fminimizer_type * T) 
   {
      fMinimizer = gsl_min_fminimizer_alloc(T); 
   }
   virtual ~GSL1DMinimizer() { 
      gsl_min_fminimizer_free(fMinimizer);
   }

private:
// usually copying is non trivial, so we make this unaccessible
   GSL1DMinimizer(const GSL1DMinimizer &); 
   GSL1DMinimizer & operator = (const GSL1DMinimizer &); 

public: 

   gsl_min_fminimizer * Get() const { 
      return fMinimizer; 
   }


private: 

   gsl_min_fminimizer * fMinimizer; 

}; 

} // end namespace Math
} // end namespace ROOT

#endif /* ROOT_Math_GSL1DMinimizer */
