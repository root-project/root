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

UnBinData::UnBinData(unsigned int maxpoints, unsigned int dim ) : 
   FitData(),
   fDim(dim),
   fNPoints(0),
   fDataVector(0), 
   fDataWrapper(0)
{ 
   // constructor with default option and range
   unsigned int n = dim*maxpoints; 
   if ( n > MaxSize() ) 
      MATH_ERROR_MSGVAL("UnBinData","Invalid data size n - no allocation done", n )
   else if (n > 0) 
      fDataVector = new DataVector(n);
} 

UnBinData::UnBinData (const DataRange & range,  unsigned int maxpoints , unsigned int dim ) : 
   FitData(range), 
   fDim(dim),
   fNPoints(0), 
   fDataVector(0), 
   fDataWrapper(0)
{
   // constructor from option and default range
   unsigned int n = dim*maxpoints; 
   if ( n > MaxSize() ) 
      MATH_ERROR_MSGVAL("UnBinData","Invalid data size n - no allocation done", n )
   else if (n > 0) 
      fDataVector = new DataVector(n);
} 

UnBinData::UnBinData (const DataOptions & opt, const DataRange & range,  unsigned int maxpoints, unsigned int dim ) : 
   FitData( opt, range), 
   fDim(dim),
   fNPoints(0),
   fDataVector(0), 
   fDataWrapper(0)
{
   // constructor from options and range
   unsigned int n = dim*maxpoints; 
   if ( n > MaxSize() ) 
      MATH_ERROR_MSGVAL("UnBinData","Invalid data size n - no allocation done", n )
   else if (n > 0) 
      fDataVector = new DataVector(n);
} 
      
UnBinData::UnBinData(unsigned int n, const double * dataX ) : 
   FitData( ), 
   fDim(1), 
   fNPoints(n),
   fDataVector(0)
{ 
   // constructor for 1D external data
   fDataWrapper = new DataWrapper(dataX);
} 
      
UnBinData::UnBinData(unsigned int n, const double * dataX, const double * dataY ) : 
   FitData( ), 
   fDim(2), 
   fNPoints(n),
   fDataVector(0),
   fDataWrapper(0)
{ 
   //    constructor for 2D external data
   fDataWrapper = new DataWrapper(dataX, dataY, 0, 0, 0, 0);
} 

UnBinData::UnBinData(unsigned int n, const double * dataX, const double * dataY, const double * dataZ ) : 
   FitData( ), 
   fDim(3), 
   fNPoints(n),
   fDataVector(0)
{ 
   //   constructor for 3D external data
   fDataWrapper = new DataWrapper(dataX, dataY, dataZ, 0, 0, 0, 0, 0);
} 

UnBinData::UnBinData(unsigned int n, const double * dataX, const DataRange & range ) : 
   FitData(range), 
   fDim(1), 
   fNPoints(0),
   fDataVector(0),
   fDataWrapper(0)
{ 
   // constructor for 1D array data using a range to select the data
   if ( n > MaxSize() ) 
      MATH_ERROR_MSGVAL("UnBinData","Invalid data size n - no allocation done", n )
   else if (n > 0) {
      fDataVector = new DataVector(n);

      for (unsigned int i = 0; i < n; ++i)  
         if ( range.IsInside(dataX[i]) ) Add(dataX[i] ); 
      
      if (fNPoints < n) (fDataVector->Data()).resize(fNPoints); 
   } 
}
      
UnBinData::UnBinData(unsigned int n, const double * dataX, const double * dataY, const DataRange & range ) : 
   FitData(range), 
   fDim(2), 
   fNPoints(0),
   fDataVector(0),
   fDataWrapper(0)
{ 
   // constructor for 2D array data using a range to select the data
   if ( n > MaxSize() ) 
      MATH_ERROR_MSGVAL("UnBinData","Invalid data size n - no allocation done", n )
   else if (n > 0) {
      fDataVector = new DataVector(2*n);

      for (unsigned int i = 0; i < n; ++i)  
         if ( range.IsInside(dataX[i],0) && 
              range.IsInside(dataY[i],1) )
            Add(dataX[i], dataY[i] ); 
      
      if (fNPoints < n) (fDataVector->Data()).resize(2*fNPoints); 
   } 
} 

UnBinData::UnBinData(unsigned int n, const double * dataX, const double * dataY, const double * dataZ, 
                     const DataRange & range ) : 
   FitData(range ), 
   fDim(3), 
   fNPoints(0),
   fDataVector(0),
   fDataWrapper(0)
{ 
   // constructor for 3D array data using a range to select the data
   if ( n > MaxSize() ) 
      MATH_ERROR_MSGVAL("UnBinData","Invalid data size n - no allocation done", n )
   else if (n > 0) {
      fDataVector = new DataVector(3*n);

      for (unsigned int i = 0; i < n; ++i)  
         if ( range.IsInside(dataX[i],0) && 
              range.IsInside(dataY[i],1) &&
              range.IsInside(dataZ[i],2) )
            Add(dataX[i], dataY[i], dataZ[i] ); 
      
      if (fNPoints < n) (fDataVector->Data()).resize(3*fNPoints); 
   } 
} 

void UnBinData::Initialize(unsigned int maxpoints, unsigned int dim ) { 
   //   preallocate a data set given size and dimension
   if ( dim != fDim && fDataVector) { 
//       MATH_INFO_MSGVAL("BinData::Initialize"," Reset amd re-initialize with a new fit point size of ",
//                        dim);
      delete fDataVector; 
      fDataVector = 0; 
   }
   fDim = dim;
   unsigned int n = fDim*maxpoints; 
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
      int nextraPoints = npoints -  fDataVector->Size()/fDim; 
      if  (nextraPoints < 0) {
         // delete extra points
         (fDataVector->Data()).resize( npoints * fDim);
      }
      else if (nextraPoints > 0) { 
         // add extra points 
         Initialize(nextraPoints, fDim ); 
      }
      else // nextraPoints == 0
         return; 
   }
   else // no DataVector create
      fDataVector = new DataVector( npoints*fDim);      
}



   } // end namespace Fit

} // end namespace ROOT

