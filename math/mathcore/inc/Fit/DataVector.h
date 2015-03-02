// @(#)root/mathcore:$Id$
// Author: L. Moneta Wed Aug 30 11:15:23 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class DataVector

#ifndef ROOT_Fit_DataVector
#define ROOT_Fit_DataVector

/**
@defgroup FitData Fit Data Classes

Classes for describing the input data for fitting

@ingroup Fit
*/


// #ifndef ROOT_Fit_DataVectorfwd
// #include "Fit/DataVectorfwd.h"
// #endif

#ifndef ROOT_Fit_DataOptions
#include "Fit/DataOptions.h"
#endif


#ifndef ROOT_Fit_DataRange
#include "Fit/DataRange.h"
#endif


#include <vector>
#include <cassert>
#include <iostream>


namespace ROOT {

   namespace Fit {


      //class used for making shared_ptr of data classes managed by the user (i.e. when we don;t want to delete the contained object) 
      template <class T> 
      struct DummyDeleter
      {
         // a deleter not deleting the contained object
         // used to avoid shared_ptr deleting the contained objects if managed externally
         void operator()(T* /* p */) {
            //printf("ROOT::Fit::DummyDeleter called - do not delete object %x \n", p);
         }
      };


/**
   Base class for all the fit data types

   @ingroup FitData
 */

class FitData {

public:

   /// construct with default option and data range
   FitData() {}

   /// dummy virtual destructor
   virtual ~FitData() {}

   /// construct passing options and default data range
   FitData(const DataOptions & opt) :
      fOptions(opt)
   {}


   /// construct passing range and default options
   FitData(const DataRange & range) :
      fRange(range)
   {}

   /// construct passing options and data range
   FitData (const DataOptions & opt, const DataRange & range) :
      fOptions(opt),
      fRange(range)
   {}

   /**
      access to options
    */
   const DataOptions & Opt() const { return fOptions; }
   DataOptions & Opt() { return fOptions; }

   /**
      access to range
    */
   const DataRange & Range() const { return fRange; }

   // range cannot be modified afterwards
   // since fit method functions use all data

   /**
       define a max size to avoid allocating too large arrays
   */
   static unsigned int MaxSize()  {
      return (unsigned int) (-1) / sizeof (double);
   }


private:

      DataOptions fOptions;
      DataRange   fRange;

};



/**
   class holding the fit data points. It is template on the type of point,
   which can be for example a binned or unbinned point.
   It is basicaly a wrapper on an std::vector

   @ingroup FitData

*/

class DataVector {

public:


   typedef std::vector<double>      FData;

   /**
      default constructor for a vector of N -data
   */
   explicit DataVector (size_t n ) :
      fData(std::vector<double>(n))

   {
      //if (n!=0) fData.reserve(n);
   }


   /**
      Destructor (no operations)
   */
   ~DataVector ()  {}

   // use default copy constructor and assignment operator


   /**
      const access to underlying vector
    */
   const FData & Data() const { return fData; }

   /**
      non-const access to underlying vector (in case of insertion/deletion) and iterator
    */
   FData & Data()  { return fData; }

#ifndef __CINT__
   /**
      const iterator access
   */
   typedef FData::const_iterator const_iterator;
   typedef FData::iterator iterator;

   const_iterator begin() const { return fData.begin(); }
   const_iterator end() const { return fData.begin()+fData.size(); }

   /**
      non-const iterator access
   */
   iterator begin() { return fData.begin(); }
   iterator end()   { return fData.end(); }

#endif
   /**
      access to the point
    */
   const double & operator[] (unsigned int i)  const { return fData[i]; }
   double & operator[] (unsigned int i)   { return fData[i]; }


   /**
      full size of data vector (npoints * point size)
    */
   size_t Size() const { return fData.size(); }


private:

      FData fData;
};


//       // usefule typedef's of DataVector
//       class BinPoint;

//       // declaration for various type of data vectors
//       typedef DataVector<ROOT::Fit::BinPoint>                    BinData;
//       typedef DataVector<ROOT::Fit::BinPoint>::const_iterator    BinDataIterator;

/**
   class maintaining a pointer to external data
   Using this class avoids copying the data when performing a fit
   NOTE: this class is not thread-safe and should not be used in parallel fits


   @ingroup FitData
 */

class DataWrapper {

public:

   /**
      specialized constructor for 1D data without errors and values
    */
   explicit DataWrapper(const double * dataX ) :
      fDim(1),
      fValues(0),
      fErrors(0),
      fCoords(std::vector<const double * >(1) ),
      fX(std::vector<double>(1) )
   {
      fCoords[0] = dataX;
   }


   /**
      constructor for 1D data (if errors are not present a null pointer should be passed)
    */
   DataWrapper(const double * dataX, const double * val, const double * eval , const double * ex ) :
      fDim(1),
      fValues(val),
      fErrors(eval),
      fCoords(std::vector<const double * >(1) ),
      fErrCoords(std::vector<const double * >(1) ),
      fX(std::vector<double>(1) ),
      fErr(std::vector<double>(1) )
   {
      fCoords[0] = dataX;
      fErrCoords[0] = ex;
   }

   /**
      constructor for 2D data (if errors are not present a null pointer should be passed)
    */
   DataWrapper(const double * dataX, const double * dataY, const double * val, const double * eval, const double * ex , const double * ey  ) :
      fDim(2),
      fValues(val),
      fErrors(eval),
      fCoords(std::vector<const double * >(2) ),
      fErrCoords(std::vector<const double * >(2) ),
      fX(std::vector<double>(2) ),
      fErr(std::vector<double>(2) )
   {
      fCoords[0] = dataX;
      fCoords[1] = dataY;
      fErrCoords[0] = ex;
      fErrCoords[1] = ey;
   }

   /**
      constructor for 3D data (if errors are not present a null pointer should be passed)
    */
   DataWrapper(const double * dataX, const double * dataY, const double * dataZ, const double * val, const double * eval, const double * ex , const double * ey, const double * ez  ) :
      fDim(3),
      fValues(val),
      fErrors(eval),
      fCoords(std::vector<const double * >(3) ),
      fErrCoords(std::vector<const double * >(3) ),
      fX(std::vector<double>(3) ),
      fErr(std::vector<double>(3) )
   {
      fCoords[0] = dataX;
      fCoords[1] = dataY;
      fCoords[2] = dataZ;
      fErrCoords[0] = ex;
      fErrCoords[1] = ey;
      fErrCoords[2] = ez;
   }

   /**
      constructor for multi-dim data without  errors
    */
   template<class Iterator>
   DataWrapper(unsigned int dim,  Iterator  coordItr ) :
      fDim(dim),
      fValues(0),
      fErrors(0),
      fCoords(std::vector<const double * >(coordItr, coordItr+dim) ),
      fX(std::vector<double>(dim) )
   { }


   /**
      constructor for multi-dim data with errors and values (if errors are not present a null pointer should be passed)
    */
   template<class Iterator>
   DataWrapper(size_t dim, Iterator coordItr, const double * val, const double * eval, Iterator errItr ) :
      // use size_t for dim to avoid allocating huge vector on 64 bits when dim=-1
      fDim(dim),
      fValues(val),
      fErrors(eval),
      fCoords(std::vector<const double * >(coordItr, coordItr+dim) ),
      fErrCoords(std::vector<const double * >(errItr, errItr+dim) ),
      fX(std::vector<double>(dim) ),
      fErr(std::vector<double>(dim) )
   { }

   // destructor
   ~DataWrapper() {
      //printf("Delete Data wrapper\n");
      // no operations
   }

   // use default copy constructor and assignment operator
   // copy the pointer of the data not the data


   const double * Coords(unsigned int ipoint) const {
      for (unsigned int i = 0; i < fDim; ++i) {
         const double * x = fCoords[i];
         assert (x != 0);
         fX[i] = x[ipoint];
      }
      return &fX.front();
   }

   double Coord(unsigned int ipoint, unsigned int icoord) const {
         const double * x = fCoords[icoord];
         assert (x != 0);
         return  x[ipoint];
   }


   const double * CoordErrors(unsigned int ipoint) const {
      for (unsigned int i = 0; i < fDim; ++i) {
         const double * err = fErrCoords[i];
         if (err == 0) return 0;
         fErr[i] = err[ipoint];
      }
      return &fErr.front();
   }

   double CoordError(unsigned int ipoint, unsigned int icoord) const {
         const double * err = fErrCoords[icoord];
         return  (err != 0) ? err[ipoint] : 0;
   }


   double Value(unsigned int ipoint) const {
      return fValues[ipoint];
   }

   double Error(unsigned int ipoint) const {
      return (fErrors) ?  fErrors[ipoint]  : 0. ;
   }



private:


   unsigned int fDim;
   const double * fValues;
   const double * fErrors;
   std::vector<const double *> fCoords;
   std::vector<const double *> fErrCoords;
   // cached vector to return x[] and errors on x
   mutable std::vector<double> fX;
   mutable std::vector<double> fErr;

};



   } // end namespace Fit

} // end namespace ROOT



#endif /* ROOT_Fit_DataVector */
