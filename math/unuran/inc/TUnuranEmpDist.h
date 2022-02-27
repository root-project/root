// @(#)root/unuran:$Id$
// Authors: L. Moneta, J. Leydold Wed Feb 28 2007

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class TUnuranEmpDist


#ifndef ROOT_Math_TUnuranEmpDist
#define ROOT_Math_TUnuranEmpDist


#include "TUnuranBaseDist.h"

#include <vector>

class TH1;


/**
   \class TUnuranEmpDist
   \ingroup Unuran

   TUnuranEmpDist class for describing empirical distributions. It is used by TUnuran
   to generate double random number according to this distribution via TUnuran::Sample() or
   TUnuran::Sample(double *) in case of multi-dimensional empirical distributions.

   An empirical distribution can be one or multi-dimension constructed from a set of unbinned data,
   (the class can be constructed from an iterator to a vector of data) or by using an histogram
   (with a pointer to the TH1 class). If the histogram contains a buffer with the original data they are used by
   default to estimate the empirical distribution, otherwise the bins information is used. In this binned case
   only one dimension is now supported.

   In the case of unbinned data the density distribution is estimated by UNURAN using kernel smoothing and
   then random numbers are generated. In the case of bin data (which can only be one dimension)
   the probability density is estimated directly from the histograms and the random numbers are generated according
   to the histogram (like in TH1::GetRandom). This method requires some initialization time but it is faster
   in generating the random numbers than TH1::GetRandom and it becomes convenient to use when generating
   a large amount of data.

*/


class TUnuranEmpDist : public TUnuranBaseDist {

public:


   /**
      Constructor from a TH1 objects.
      If the histogram has a buffer by default the unbinned data are used
   */
   TUnuranEmpDist (const TH1 * h1 = 0, bool useBuffer = true );

   /**
      Constructor from a set of data using an iterator to specify begin/end of the data
      In the case of multi-dimension the data are assumed to be passed in this order
      x0,y0,...x1,y1,..x2,y2,...
   */
   template<class Iterator>
   TUnuranEmpDist (Iterator begin, Iterator end, unsigned int dim = 1) :
      fData(std::vector<double>(begin,end) ),
      fDim(dim),
      fMin(0), fMax(0),
      fBinned(0)  {}

   /**
      Constructor from a set of 1D data
   */
   TUnuranEmpDist (unsigned int n, double * x);

   /**
      Constructor from a set of 2D data
   */
   TUnuranEmpDist (unsigned int n, double * x, double * y);

   /**
      Constructor from a set of 3D data
   */
   TUnuranEmpDist (unsigned int n, double * x, double * y, double * z);


   /**
      Destructor (no operations)
   */
   ~TUnuranEmpDist () override {}


   /**
      Copy constructor
   */
   TUnuranEmpDist(const TUnuranEmpDist &);


   /**
      Assignment operator
   */
   TUnuranEmpDist & operator = (const TUnuranEmpDist & rhs);

   /**
      Clone (required by base class)
    */
   TUnuranEmpDist * Clone() const override { return new TUnuranEmpDist(*this); }


   /**
      Return reference to data vector (unbinned or binned data)
    */
   const std::vector<double> & Data() const { return fData; }

   /**
      Flag to control if data are binned
    */
   bool IsBinned() const { return fBinned; }

   /**
      Min value of binned data
      (return 0 for unbinned data)
    */
   double LowerBin() const { return fMin; }

   /**
      upper value of binned data
      (return 0 for unbinned data)
    */
   double UpperBin() const { return fMax; }

   /**
      Number of data dimensions
    */
   unsigned int NDim() const { return fDim; }


private:

   std::vector<double>  fData;       ///< pointer to the data vector (used for generation from un-binned data)
   unsigned int fDim;                ///< data dimensionality
   double fMin;                      ///< min values (used in the binned case)
   double fMax;                      ///< max values (used in the binned case)
   bool   fBinned;                   ///< flag for binned/unbinned data

   ClassDefOverride(TUnuranEmpDist,1)        //Wrapper class for empirical distribution


};



#endif /* ROOT_Math_TUnuranEmpDist */
