// @(#)root/mathcore:$Id$
// Author: L. Moneta Wed Aug 30 11:10:03 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class BinPoint

#ifndef ROOT_Fit_BinPoint
#define ROOT_Fit_BinPoint




namespace ROOT { 

   namespace Fit { 


      class DataRange; 

/** 
    Obsolete class, no more in use.
    class describing the point with bins ( x coordinates, y and error on y ) 
     but not error in X . For the Error in x one should use onother class

              
*/ 
class BinPoint {

public: 

   
   //typedef  std::vector<double> CoordData; 


   /** 
      Constructor
   */ 
   explicit BinPoint (unsigned int n = 1) : 
      fDim(n),
      fCoords(0 ), 
      fCoordErr( 0),
      fValue(0), 
      fError(1),
      fInvError(1)
   {}

//    /**
//       constructor from a vector of coordinates, y value and y error
//     */
//    BinPoint (const std::vector<double> & x, double y, double ey = 1) : 
//       fCoords(x), 
//       fValue(y), 
//       fInvError( ey!= 0 ? 1.0/ey : 0 )
//    { }
   
//    template <class Iterator> 
//    BinPoint (const Iterator begin, const Iterator end, double y, double ey = 1) : 
//       fCoords(begin,end), 
//       fValue(y), 
//       fInvError( ey!= 0. ? 1.0/ey : 1. )
//    { }

   void Set(const double * x, double value, double invErr) { 
      fCoords = x; 
      fValue = value; 
      fInvError = invErr;
   }

   void Set(const double * x, double value, const double * ex, double err) { 
      fCoords = x; 
      fValue = value;
      fCoordErr = ex; 
      fError = err;
   }


   /** 
      Destructor (no operations)
   */ 
   ~BinPoint ()  {}  

   // use default copy constructor and assignment


   // accessors 

   /**
      return pointer to coordinates 
    */
   //const double *  Coords() const { return &fCoords.front(); }

    /**
      return vector of coordinates 
    */
   const double * Coords() const { return fCoords; }

   /**
      return the value (bin height in case of an histogram)
    */
   double Value() const { return fValue; }

   /**
      return the error on the value 
    */
   double Error() const { 
      //return fInvError != 0 ? 1.0/fInvError : 0; 
      return fError;
   } 

   /**
      return the inverse of error on the value 
    */
   double InvError() const { return fInvError; }

   /** 
     get the dimension (dimension of the cooordinates)
    */
   unsigned int NDim() const { return  fDim; }

   /**
      check if a Point is inside the given range 
    */ 
   bool IsInRange( const DataRange & range) const; 

private: 

   unsigned int fDim;
   //double fCoords[N];
   const double * fCoords; 
   const double * fCoordErr; 
   
   double fValue; 
   // better to store the inverse of the error (is more efficient)
   double fError; 
   double fInvError; 


}; 

   } // end namespace Fit

} // end namespace ROOT

// #ifndef ROOT_Fit_DataRange
// #include "Fit/DataRange.h"
// #endif
// #include <cassert> 

// namespace ROOT { 

//    namespace Fit { 

// template<unsigned int N> 
// bool BinPoint<N>::IsInRange(const DataRange & range) const 
// {
//    // check if given point is inside the given range
  
//    // need to check that datarange size is same as point size 
//    if (range.NDim() == 0) return true; // (range is empty is equivalent to (-inf, + inf) 
//    // in case not zero dimension must be equal to the coordinates
//    assert( kSize == range.NDim() );  
//    for (unsigned int i = 0; i < kSize; ++i) { 
//       if ( ! range.IsInside( fCoords[i] ) ) return false; 
//    }
//    return true; 
// }

//    } // end namespace Fit

// } // end namespace ROOT



#endif /* ROOT_Fit_BinPoint */
