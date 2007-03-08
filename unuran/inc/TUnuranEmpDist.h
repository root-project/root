// @(#)root/unuran:$Name:  $:$Id: TUnuranEmpDist.h,v 1.3 2007/02/05 10:24:44 moneta Exp $
// Authors: L. Moneta, J. Leydold Wed Feb 28 2007

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class TUnuranEmpDist

//////////////////////////////////////////////////////////////////////
// 
//   TUnuranEmpDistr class 
//   wrapper class for one dimensional empirical distribution
// 
///////////////////////////////////////////////////////////////////////

#ifndef ROOT_Math_TUnuranEmpDist
#define ROOT_Math_TUnuranEmpDist


#ifndef ROOT_Math_TUnuranBaseDist
#include "TUnuranBaseDist.h"
#endif

#include <vector>

class TH1;

//////////////////////////////////////////////////////////////////////
/** 
   TUnuranEmpDist class 
   wrapper class for empiral  distributions obtained for example from an histogram 
   or a vector of data
*/ 
///////////////////////////////////////////////////////////////////////

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
   TUnuranEmpDist (Iterator begin, Iterator end, unsigned int dim ) : 
      fData(std::vector<double>(begin,end) ), 
      fDim(dim) {}

   /** 
      Destructor (no operations)
   */ 
   virtual ~TUnuranEmpDist () {}


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
   TUnuranEmpDist * Clone() const { return new TUnuranEmpDist(*this); } 


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

   std::vector<double>  fData;       //pointer to the data vector (used for generation from un-binned data)
   unsigned int fDim;                 //data dimensionality
   double fMin;                       // min values (used in the binned case)
   double fMax;                       // max values (used in the binned case)
   bool   fBinned;                    // flag for binned/unbinned data 

   //ClassDef(TUnuranEmpDist,1)         //Wrapper class for empirical distribution (One of multi-dim.)


}; 



#endif /* ROOT_Math_TUnuranEmpDist */
