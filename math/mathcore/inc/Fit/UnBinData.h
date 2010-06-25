// @(#)root/mathcore:$Id$
// Author: L. Moneta Wed Aug 30 11:15:23 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class UnBinData

#ifndef ROOT_Fit_UnBinData
#define ROOT_Fit_UnBinData

#ifndef ROOT_Fit_DataVector
#include "Fit/DataVector.h"
#endif

#ifndef ROOT_Math_Error
#include "Math/Error.h"
#endif



namespace ROOT { 

   namespace Fit { 


//___________________________________________________________________________________
/** 
   Class describing the unbinned data sets (just x coordinates values) of any dimensions

              There is the option to construct UnBindata copying the data in (using the DataVector class) 
              or using pointer to external data (DataWrapper) class. 
              In general is found to be more efficient to copy the data. 
              In case of really large data sets for limiting memory consumption then the other option can be used
              Specialized constructor exists for using external data up to 3 dimensions. 

              When the data are copying in the number of points can be set later (or re-set) using Initialize and 
              the data are inserted one by one using the Add method. 
              It is mandatory to set the size before using the Add method.  

             @ingroup  FitData  
*/ 
class UnBinData : public FitData { 

public : 

   /**
      constructor from dimension of point  and max number of points (to pre-allocate vector)
    */

   explicit UnBinData(unsigned int maxpoints = 0, unsigned int dim = 1 );


   /**
      constructor from range and default option
    */
   explicit UnBinData (const DataRange & range,  unsigned int maxpoints = 0, unsigned int dim = 1);

   /**
      constructor from options and range
    */
   UnBinData (const DataOptions & opt, const DataRange & range,  unsigned int maxpoints = 0, unsigned int dim = 1 );

   /**
      constructor for 1D external data (data are not copied inside)
    */
   UnBinData(unsigned int n, const double * dataX );

   /**
      constructor for 2D external data (data are not copied inside)
    */
   UnBinData(unsigned int n, const double * dataX, const double * dataY );

   /**
      constructor for 3D external data (data are not copied inside)
    */
   UnBinData(unsigned int n, const double * dataX, const double * dataY, const double * dataZ );

   /**
      constructor for multi-dim external data (data are not copied inside)
      Uses as argument an iterator of a list (or vector) containing the const double * of the data
      An example could be the std::vector<const double *>::begin
    */
   template<class Iterator> 
   UnBinData(unsigned int n, unsigned int dim, Iterator dataItr ) : 
      FitData( ), 
      fDim(dim), 
      fNPoints(n),
      fDataVector(0)
   { 
      fDataWrapper = new DataWrapper(dim, dataItr);
   } 

   /**
      constructor for 1D data and a range (data are copied inside according to the given range)
    */
   UnBinData(unsigned int maxpoints, const double * dataX, const DataRange & range);

   /**
      constructor for 2D data and a range (data are copied inside according to the given range)
    */
   UnBinData(unsigned int maxpoints, const double * dataX, const double * dataY, const DataRange & range);

   /**
      constructor for 3D data and a range (data are copied inside according to the given range)
    */
   UnBinData(unsigned int maxpoints, const double * dataX, const double * dataY, const double * dataZ, const DataRange & range);

   /**
      constructor for multi-dim external data and a range (data are copied inside according to the range)
      Uses as argument an iterator of a list (or vector) containing the const double * of the data
      An example could be the std::vector<const double *>::begin
    */
   template<class Iterator> 
   UnBinData(unsigned int maxpoints, unsigned int dim, Iterator dataItr, const DataRange & range ) : 
      FitData( ), 
      fDim(dim), 
      fNPoints(0),
      fDataVector(0),
      fDataWrapper(0)
   { 
      unsigned int n = dim*maxpoints; 
      if ( n > MaxSize() ) {
         MATH_ERROR_MSGVAL("UnBinData","Invalid data size n - no allocation done", n );
      }
      else if (n > 0) { 
         fDataVector = new DataVector(n);

         // use data wrapper to get the data
         ROOT::Fit::DataWrapper wdata(dim, dataItr); 
         for (unsigned int i = 0; i < maxpoints; ++i) { 
            bool isInside = true;
            for (unsigned int icoord = 0; icoord < dim; ++icoord)  
               isInside &= range.IsInside( wdata.Coords(i)[icoord], icoord ); 
            if ( isInside ) Add(wdata.Coords(i)); 
         }
         if (fNPoints < maxpoints) (fDataVector->Data()).resize(fDim*fNPoints);
      } 
   }


private: 
   /// copy constructor (private) 
   UnBinData(const UnBinData &) : FitData() {}
   /// assignment operator  (private) 
   UnBinData & operator= (const UnBinData &) { return *this; } 

public: 

#ifdef LATER
   /**
      Create from a compatible UnBinData set
    */
   
   UnBinData (const UnBinData & data , const DataOptions & opt, const DataRange & range) : 
      DataVector(opt,range, data.DataSize() ), 
      fDim(data.fDim),
      fNPoints(data.fNPoints) 
   {
//       for (Iterator itr = begin; itr != end; ++itr) 
//          if (itr->IsInRange(range) )
//             Add(*itr); 
   } 
#endif

   /**
      destructor, delete pointer to internal data or external data wrapper
    */
   virtual ~UnBinData() {
      if (fDataVector) delete fDataVector; 
      if (fDataWrapper) delete fDataWrapper; 
   }

   /**
      preallocate a data set given size and dimension
      if a vector already exists with correct dimension (point size) extend the existing one
      to a total size of maxpoints (equivalent to a Resize)
    */
   void Initialize(unsigned int maxpoints, unsigned int dim = 1);

   
   /**
      return fit point size (for unbin data is equivalent to coordinate dimension)
    */
   unsigned int PointSize() const { 
      return fDim; 
   }

   /**
      return size of internal data vector (is 0 for external data) 
    */
   unsigned int DataSize() const { 
      return (fDataVector) ? fDataVector->Size() : 0;
   }
   
   /**
      add one dim coordinate data
   */
   void Add(double x) { 
      int index = fNPoints*PointSize(); 
      assert(fDataVector != 0);
      assert(PointSize() == 1);
      assert (index + PointSize() <= DataSize() ); 

      (fDataVector->Data())[ index ] = x;

      fNPoints++;
   }

   /**
      add 2-dim coordinate data
   */
   void Add(double x, double y) { 
      int index = fNPoints*PointSize(); 
      assert(fDataVector != 0);
      assert(PointSize() == 2);
      assert (index + PointSize() <= DataSize() ); 

      (fDataVector->Data())[ index ] = x;
      (fDataVector->Data())[ index+1 ] = y;

      fNPoints++;
   }

   /**
      add 3-dim coordinate data
   */
   void Add(double x, double y, double z) { 
      int index = fNPoints*PointSize(); 
      assert(fDataVector != 0);
      assert(PointSize() == 3);
      assert (index + PointSize() <= DataSize() ); 

      (fDataVector->Data())[ index ] = x;
      (fDataVector->Data())[ index+1 ] = y;
      (fDataVector->Data())[ index+2 ] = z;

      fNPoints++;
   }

   /**
      add multi-dim coordinate data
   */
   void Add(const double *x) { 
      int index = fNPoints*PointSize(); 

      assert(fDataVector != 0);
      assert (index + PointSize() <= DataSize() ); 

      double * itr = &( (fDataVector->Data()) [ index ]);

      for (unsigned int i = 0; i < fDim; ++i) 
         *itr++ = x[i]; 

      fNPoints++;
   }

   /**
      return pointer to coordinate data
    */
   const double * Coords(unsigned int ipoint) const { 
      if (fDataVector) 
         return &( (fDataVector->Data()) [ ipoint*PointSize() ] );
      else 
         return fDataWrapper->Coords(ipoint); 
   }


   /**
      resize the vector to the given npoints 
    */
   void Resize (unsigned int npoints); 


   /**
      return number of contained points 
    */ 
   unsigned int NPoints() const { return fNPoints; } 

   /**
      return number of contained points 
    */ 
   unsigned int Size() const { return fNPoints; }

   /**
      return coordinate data dimension
    */ 
   unsigned int NDim() const { return fDim; } 

protected: 

   void SetNPoints(unsigned int n) { fNPoints = n; }

private: 

   unsigned int fDim;         // coordinate data dimension
   unsigned int fNPoints;     // numer of fit points
   
   DataVector * fDataVector;     // pointer to internal data vector (null for external data)
   DataWrapper * fDataWrapper;   // pointer to structure wrapping external data (null when data are copied in)

}; 

  
   } // end namespace Fit

} // end namespace ROOT



#endif /* ROOT_Fit_UnBinData */
