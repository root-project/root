// @(#)root/mathcore:$Id$
// Author: L. Moneta Wed Aug 30 11:10:03 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Implementation file for class BinData

#include "Fit/BinData.h"
#include "Math/Error.h"

#include <cassert> 
#include <cmath>

namespace ROOT { 

   namespace Fit { 

   /**
    */

BinData::BinData(unsigned int maxpoints , unsigned int dim , ErrorType err ) : 
//      constructor from dimension of point  and max number of points (to pre-allocate vector)
//      Give a zero value and then use Initialize later one if the size is not known
   FitData(),
   fDim(dim),
   fPointSize(GetPointSize(err,dim) ),
   fNPoints(0),
   fRefVolume(1.0),
   fDataVector(0),
   fDataWrapper(0)
{ 
   unsigned int n = fPointSize*maxpoints; 
   if ( n > MaxSize() ) 
      MATH_ERROR_MSGVAL("BinData","Invalid data size n - no allocation done", n )
   else if (n > 0) 
      fDataVector = new DataVector(n);
} 

BinData::BinData (const DataOptions & opt, unsigned int maxpoints, unsigned int dim, ErrorType err ) : 
//      constructor from option and default range
      // DataVector( opt, (dim+2)*maxpoints ), 
   FitData(opt),
   fDim(dim),
   fPointSize(GetPointSize(err,dim) ),
   fNPoints(0),
   fRefVolume(1.0),
   fDataVector(0),
   fDataWrapper(0)
{ 
   unsigned int n = fPointSize*maxpoints; 
   if ( n > MaxSize() ) 
      MATH_ERROR_MSGVAL("BinData","Invalid data size n - no allocation done", n )
   else if (n > 0) 
      fDataVector = new DataVector(n);
} 
      
   /**
    */
BinData::BinData (const DataOptions & opt, const DataRange & range, unsigned int maxpoints , unsigned int dim , ErrorType err  ) : 
//      constructor from options and range
//      default is 1D and value errors

      //DataVector( opt, range, (dim+2)*maxpoints ), 
   FitData(opt,range),
   fDim(dim),
   fPointSize(GetPointSize(err,dim) ),
   fNPoints(0),
   fRefVolume(1.0),
   fDataVector(0),
   fDataWrapper(0)
{ 
   unsigned int n = fPointSize*maxpoints; 
   if ( n > MaxSize() ) 
      MATH_ERROR_MSGVAL("BinData","Invalid data size n - no allocation done", n )
   else if (n > 0) 
      fDataVector = new DataVector(n);
} 
      
/** constructurs using external data */
   
   /**
    */
BinData::BinData(unsigned int n, const double * dataX, const double * val, const double * ex , const double * eval ) : 
//      constructor from external data for 1D with errors on  coordinate and value
   fDim(1), 
   fPointSize(2),
   fNPoints(n),
   fRefVolume(1.0),
   fDataVector(0)
{ 
   if (eval != 0) { 
      fPointSize++;
      if (ex != 0) fPointSize++;
   }
   fDataWrapper  = new DataWrapper(dataX, val, eval, ex);
} 

   
   /**

    */
BinData::BinData(unsigned int n, const double * dataX, const double * dataY, const double * val, const double * ex , const double * ey, const double * eval  ) : 
//      constructor from external data for 2D with errors on  coordinate and value      
   fDim(2), 
   fPointSize(3),
   fNPoints(n),
   fRefVolume(1.0),
   fDataVector(0)
{ 
   if (eval != 0) { 
      fPointSize++;
      if (ex != 0 && ey != 0 ) fPointSize += 2;
   }
   fDataWrapper  = new DataWrapper(dataX, dataY, val, eval, ex, ey);
} 

   /**
    */
BinData::BinData(unsigned int n, const double * dataX, const double * dataY, const double * dataZ, const double * val, const double * ex , const double * ey , const double * ez , const double * eval   ) : 
//      constructor from external data for 3D with errors on  coordinate and value
   fDim(3), 
   fPointSize(4),
   fNPoints(n),
   fRefVolume(1.0),
   fDataVector(0)
{ 
   if (eval != 0) { 
      fPointSize++;
      if (ex != 0 && ey != 0 && ez != 0) fPointSize += 3;
   }
   fDataWrapper  = new DataWrapper(dataX, dataY, dataZ, val, eval, ex, ey, ez);
} 


   /// copy constructor 
BinData::BinData(const BinData & rhs) : 
   FitData(rhs.Opt(), rhs.Range()), 
   fDim(rhs.fDim), 
   fPointSize(rhs.fPointSize), 
   fNPoints(rhs.fNPoints), 
   fRefVolume(1.0),
   fDataVector(0),
   fDataWrapper(0), 
   fBinEdge(rhs.fBinEdge)
{
   // copy constructor (copy data vector or just the pointer)
   if (rhs.fDataVector != 0) fDataVector = new DataVector(*rhs.fDataVector);
   else if (rhs.fDataWrapper != 0) fDataWrapper = new DataWrapper(*rhs.fDataWrapper);
}



BinData & BinData::operator= (const BinData & rhs) { 
   // assignment operator
   
   // copy  options but cannot copy  range since cannot be modified afterwards
   DataOptions & opt = Opt();
   opt = rhs.Opt();
   //t.b.c
   //DataRange & range = Range(); 
   //range = rhs.Range();
   //
   // assignment operator  
   if (&rhs == this) return *this; 
   fDim = rhs.fDim;  
   fPointSize = rhs.fPointSize;  
   fNPoints = rhs.fNPoints;  
   fBinEdge = rhs.fBinEdge;
   fRefVolume = rhs.fRefVolume;
   // delete previous pointers 
   if (fDataVector) delete fDataVector; 
   if (fDataWrapper) delete fDataWrapper; 
   if (rhs.fDataVector != 0)  
      fDataVector = new DataVector(*rhs.fDataVector);
   else 
      fDataVector = 0; 
   if (rhs.fDataWrapper != 0) 
      fDataWrapper = new DataWrapper(*rhs.fDataWrapper);
   else 
      fDataWrapper = 0; 

   return *this; 
} 


      
BinData::~BinData() {
   // destructor 
   if (fDataVector) delete fDataVector; 
   if (fDataWrapper) delete fDataWrapper; 
}

void BinData::Initialize(unsigned int maxpoints, unsigned int dim , ErrorType err  ) { 
//       preallocate a data set given size and dimension
//       need to be initialized with the  right dimension before
   if (fDataWrapper) delete fDataWrapper;
   fDataWrapper = 0; 
   unsigned int pointSize = GetPointSize(err,dim);  
   if ( pointSize != fPointSize && fDataVector) { 
//       MATH_INFO_MSGVAL("BinData::Initialize"," Reset amd re-initialize with a new fit point size of ",
//                        pointSize);
      delete fDataVector; 
      fDataVector = 0; 
   }
   fPointSize = pointSize; 
   fDim = dim;
   unsigned int n = fPointSize*maxpoints; 
   if ( n > MaxSize() ) { 
      MATH_ERROR_MSGVAL("BinData::Initialize"," Invalid data size  ", n );
      return; 
   }
   if (fDataVector) { 
      // resize vector by adding the extra points on top of the previously existing ones 
      (fDataVector->Data()).resize( fDataVector->Size() + n);
   }
   else {
      fDataVector = new DataVector(n);
   }
   // reserve space for bin width in case of integral options
   if (Opt().fIntegral) fBinEdge.reserve( maxpoints * fDim);
}

void BinData::Resize(unsigned int npoints) { 
   // resize vector to new points 
   if (fPointSize == 0) return; 
   if ( npoints > MaxSize() ) { 
      MATH_ERROR_MSGVAL("BinData::Resize"," Invalid data size  ", npoints );
      return; 
   }
   int nextraPoints = npoints - DataSize()/ fPointSize;  
   if (nextraPoints == 0) return; 
   else if (nextraPoints < 0) {
      // delete extra points
      if (!fDataVector) return; 
      (fDataVector->Data()).resize( npoints * fPointSize);
   } 
   else 
      Initialize(nextraPoints, fDim, GetErrorType() ); 
}
   /**
   */
void BinData::Add(double x, double y ) { 
//       add one dim data with only coordinate and values
   int index = fNPoints*PointSize();
   assert (fDataVector != 0);
   assert (PointSize() == 2 ); 
   assert (index + PointSize() <= DataSize() ); 
   
   double * itr = &((fDataVector->Data())[ index ]);
   *itr++ = x; 
   *itr++ = y; 
   
   fNPoints++;
}
   
   /**
   */
void BinData::Add(double x, double y, double ey) { 
//       add one dim data with no error in x
//       in this case store the inverse of the error in y
   int index = fNPoints*PointSize(); 

   assert( fDim == 1);
   assert (fDataVector != 0);
   assert (PointSize() == 3 ); 
   assert (index + PointSize() <= DataSize() ); 
   
   double * itr = &((fDataVector->Data())[ index ]);
   *itr++ = x; 
   *itr++ = y; 
   *itr++ =  (ey!= 0) ? 1.0/ey : 0; 
   
   fNPoints++;
}

   /**
   */
void BinData::Add(double x, double y, double ex, double ey) { 
//      add one dim data with  error in x
//      in this case store the y error and not the inverse 
   int index = fNPoints*PointSize(); 
   assert (fDataVector != 0);
   assert( fDim == 1);
   assert (PointSize() == 4 ); 
   assert (index + PointSize() <= DataSize() ); 

   double * itr = &((fDataVector->Data())[ index ]);
   *itr++ = x; 
   *itr++ = y; 
   *itr++ = ex; 
   *itr++ = ey; 
   
   fNPoints++;
}

   /**
   */
void BinData::Add(double x, double y, double ex, double eyl , double eyh) { 
//      add one dim data with  error in x and asymmetric errors in y
//      in this case store the y errors and not the inverse 
   int index = fNPoints*PointSize(); 
   assert (fDataVector != 0);
   assert( fDim == 1);
   assert (PointSize() == 5 ); 
   assert (index + PointSize() <= DataSize() ); 
   
   double * itr = &((fDataVector->Data())[ index ]);
   *itr++ = x; 
   *itr++ = y; 
   *itr++ = ex; 
   *itr++ = eyl; 
   *itr++ = eyh; 
   
   fNPoints++;
}


   /**
   */
void BinData::Add(const double *x, double val) { 
//      add multi dim data with only value (no errors)
   int index = fNPoints*PointSize(); 
   assert (fDataVector != 0);
   assert (PointSize() == fDim + 1 ); 
   
   if (index + PointSize() > DataSize()) 
      MATH_ERROR_MSGVAL("BinData::Add","add a point beyond the data size", DataSize() );

   assert (index + PointSize() <= DataSize() ); 
   
   double * itr = &((fDataVector->Data())[ index ]);
   
   for (unsigned int i = 0; i < fDim; ++i) 
      *itr++ = x[i]; 
   *itr++ = val; 
   
   fNPoints++;
}

   /**
   */
void BinData::Add(const double *x, double val, double  eval) { 
//      add multi dim data with only error in value 
   int index = fNPoints*PointSize(); 
   assert (fDataVector != 0);
   assert (PointSize() == fDim + 2 ); 
   
   if (index + PointSize() > DataSize()) 
      MATH_ERROR_MSGVAL("BinData::Add","add a point beyond the data size", DataSize() );

   assert (index + PointSize() <= DataSize() ); 
   
   double * itr = &((fDataVector->Data())[ index ]);
   
   for (unsigned int i = 0; i < fDim; ++i) 
      *itr++ = x[i]; 
   *itr++ = val; 
   *itr++ =  (eval!= 0) ? 1.0/eval : 0; 
   
   fNPoints++;
}


   /**
   */
void BinData::Add(const double *x, double val, const double * ex, double  eval) { 
   //      add multi dim data with error in coordinates and value 
   int index = fNPoints*PointSize(); 
   assert (fDataVector != 0);
   assert (PointSize() == 2*fDim + 2 ); 
   
   if (index + PointSize() > DataSize()) 
      MATH_ERROR_MSGVAL("BinData::Add","add a point beyond the data size", DataSize() );

   assert (index + PointSize() <= DataSize() ); 
   
   double * itr = &((fDataVector->Data())[ index ]);
   
   for (unsigned int i = 0; i < fDim; ++i) 
      *itr++ = x[i]; 
   *itr++ = val; 
   for (unsigned int i = 0; i < fDim; ++i) 
      *itr++ = ex[i]; 
   *itr++ = eval; 
   
   fNPoints++;
}

   /**
   */
void BinData::Add(const double *x, double val, const double * ex, double  elval, double  ehval) { 
   //      add multi dim data with error in coordinates and asymmetric error in value
   int index = fNPoints*PointSize(); 
   assert (fDataVector != 0);
   assert (PointSize() == 2*fDim + 3 ); 
   
   if (index + PointSize() > DataSize()) 
      MATH_ERROR_MSGVAL("BinData::Add","add a point beyond the data size", DataSize() );

   assert (index + PointSize() <= DataSize() ); 
   
   double * itr = &((fDataVector->Data())[ index ]);
   
   for (unsigned int i = 0; i < fDim; ++i) 
      *itr++ = x[i]; 
   *itr++ = val; 
   for (unsigned int i = 0; i < fDim; ++i) 
      *itr++ = ex[i]; 
   *itr++ = elval; 
   *itr++ = ehval; 
   
   fNPoints++;
}

void BinData::AddBinUpEdge(const double *xup ) { 
//      add multi dim bin upper edge data (coord2)

   fBinEdge.insert( fBinEdge.end(), xup, xup + fDim);
   
   // check that is consistent with number of points added in the data
   assert( fNPoints * fDim == fBinEdge.size() );

   // compute the bin volume 
   const double * xlow = Coords(fNPoints-1);

   double binVolume = 1;
   for (unsigned int j = 0; j < fDim; ++j) {
      binVolume *= (xup[j]-xlow[j]);
   }
      
   // store the minimum bin volume found as  reference for future normalizations
   if (fNPoints == 1) {
      fRefVolume = binVolume;
      return;
   }

   if (binVolume < fRefVolume) 
      fRefVolume = binVolume;
   
}


BinData & BinData::LogTransform() { 
   // apply log transform on the bin data values

   if (fNPoints == 0) return *this; 

   if (fDataVector) {       

      ErrorType type = GetErrorType(); 

      std::vector<double> & data = fDataVector->Data(); 

      typedef std::vector<double>::iterator DataItr; 
      unsigned int ip = 0;
      DataItr itr = data.begin();      

      if (type == kNoError ) { 
         fPointSize = fDim + 2;
      }

      while (ip <  fNPoints ) {     
         assert( itr != data.end() );
         DataItr valitr = itr + fDim; 
         double val = *(valitr); 
         if (val <= 0) { 
            MATH_ERROR_MSG("BinData::TransformLog","Some points have negative values - cannot apply a log transformation");
            // return an empty data-sets
            Resize(0);
            return *this; 
         }
         *(valitr) = std::log(val);
         // change also errors to 1/val * err
         if (type == kNoError ) { 
            // insert new error value 
            DataItr errpos = data.insert(valitr+1,val); 
            // need to get new iterators for right position
            itr = errpos - fDim -1;
            //std::cout << " itr " << *(itr) << " itr +1 " << *(itr+1) << std::endl;
         }
         else if (type == kValueError) { 
             // new weight = val * old weight
            *(valitr+1) *= val; 
         } 
         else {
            // other case (error in value is stored) : new error = old_error/value 
            for (unsigned int j = fDim + 1; j < fPointSize; ++j)  
               *(itr+j) /= val;  
         }
         itr += fPointSize; 
         ip++;
      }
      // in case of Noerror since we added the errors we have changes the type 
   return *this; 
   }
   // case of data wrapper - we copy the data and build a datavector
   if (fDataWrapper == 0) return *this; 

   // asym errors are not supported for data wrapper 
   ErrorType type = kValueError; 
   std::vector<double> errx; 
   if (fDataWrapper->CoordErrors(0) != 0 ) { 
      type = kCoordError; 
      errx.resize(fDim);  // allocate vector to store errors 
   }

   BinData tmpData(fNPoints, fDim, type); 
   for (unsigned int i = 0; i < fNPoints; ++i ) { 
      double val = fDataWrapper->Value(i);
      if (val <= 0) { 
         MATH_ERROR_MSG("BinData::TransformLog","Some points have negative values - cannot apply a log transformation");
         // return an empty data-sets
         Resize(0);
         return *this; 
      } 
      double err = fDataWrapper->Error(i); 
      if (err <= 0) err = 1;
      if (type == kValueError ) 
         tmpData.Add(fDataWrapper->Coords(i), std::log(val), err/val);
      else if (type == kCoordError) { 
         const double * exold = fDataWrapper->CoordErrors(i);
         assert(exold != 0);
         for (unsigned int j = 0; j < fDim; ++j) { 
            std::cout << " j " << j << " val " << val << " " << errx.size() <<  std::endl;
            errx[j] = exold[j]/val; 
         }
         tmpData.Add(fDataWrapper->Coords(i), std::log(val), &errx.front(),  err/val); 
      }                                            
   }
   delete fDataWrapper;
   fDataWrapper = 0; // no needed anymore
   *this = tmpData; 
   return *this; 
}

   } // end namespace Fit

} // end namespace ROOT

