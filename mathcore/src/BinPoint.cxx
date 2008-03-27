// @(#)root/mathcore:$Id$
// Author: L. Moneta Wed Aug 30 11:10:03 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Implementation file for class BinPoint

#include "Fit/BinPoint.h"
#include "Fit/DataRange.h"

#include <cassert> 

namespace ROOT { 

   namespace Fit { 


bool BinPoint::IsInRange(const DataRange & range) const 
{
   // check if given point is inside the given range
  
   unsigned int ndim = NDim(); 
   // need to check that datarange size is same as point size 
   if (range.NDim() == 0) return true; // (range is empty is equivalent to (-inf, + inf) 
   // in case not zero dimension must be equal to the coordinates
   assert( ndim == range.NDim() );  
   for (unsigned int i = 0; i < ndim; ++i) { 
      if ( ! range.IsInside( fCoords[i] ) ) return false; 
   }
   return true; 
}

   } // end namespace Fit

} // end namespace ROOT

