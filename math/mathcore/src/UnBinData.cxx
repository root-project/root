// @(#)root/mathcore:$Id$
// Author: L. Moneta Wed Aug 30 11:10:03 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Implementation file for class UnBinData

#include "Fit/UnBinData.h"
#include "Math/Error.h"

#include <cassert>
#include <cmath>

namespace ROOT {

   namespace Fit {

/*
void UnBinData::Initialize(unsigned int maxpoints, unsigned int dim, bool isWeighted ) {
   //   preallocate a data set given size and dimension
   unsigned int pointSize = (isWeighted) ? dim+1 : dim;
   if ( (dim != fDim || pointSize != fPointSize) && fDataVector) {
//       MATH_INFO_MSGVAL("BinData::Initialize"," Reset amd re-initialize with a new fit point size of ",
//                        dim);
      delete fDataVector;
      fDataVector = 0;
   }
   fDim = dim;
   fPointSize = pointSize;
   unsigned int n = fPointSize*maxpoints;
   if ( n > MaxSize() ) {
      MATH_ERROR_MSGVAL("UnBinData::Initialize","Invalid data size", n );
      return;
   }
   if (fDataVector)
      (fDataVector->Data()).resize( fDataVector->Size() + n );
   else
      fDataVector = new DataVector( n);
}

void UnBinData::Resize(unsigned int npoints) {
   // resize vector to new points
   if (fDim == 0) return;
   if ( npoints > MaxSize() ) {
      MATH_ERROR_MSGVAL("BinData::Resize"," Invalid data size  ", npoints );
      return;
   }
   if (fDataVector != 0)  {
      int nextraPoints = npoints -  fDataVector->Size()/fPointSize;
      if  (nextraPoints < 0) {
         // delete extra points
         (fDataVector->Data()).resize( npoints * fPointSize);
      }
      else if (nextraPoints > 0) {
         // add extra points
         Initialize(nextraPoints, fDim, IsWeighted()  );
      }
      else // nextraPoints == 0
         return;
   }
   else // no DataVector create it
      fDataVector = new DataVector( npoints*fPointSize);
}
*/


   } // end namespace Fit

} // end namespace ROOT

