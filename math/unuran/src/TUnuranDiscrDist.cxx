// @(#)root/unuran:$Id$
// Authors: L. Moneta, J. Leydold Wed Feb 28 2007

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Implementation file for class TUnuranDiscrDist

#include "TUnuranDiscrDist.h"

#include "Math/IFunction.h"
#include "TF1.h"
#include "Math/WrappedTF1.h"


#include <cassert>


TUnuranDiscrDist::TUnuranDiscrDist (const ROOT::Math::IGenFunction & func, bool copyFunc) :
   fPmf(&func),
   fCdf(0),
   fXmin(1),
   fXmax(-1),
   fMode(0),
   fSum(0),
   fHasDomain(0),
   fHasMode(0),
   fHasSum(0),
   fOwnFunc(copyFunc)
{
   //Constructor from a generic function object
   if (fOwnFunc) {
      fPmf = fPmf->Clone();
      //if (fCdf) fCdf->Clone();
   }
}


TUnuranDiscrDist::TUnuranDiscrDist (TF1 * func) :
   fPmf( (func) ? new ROOT::Math::WrappedTF1 ( *func) : 0 ),
   fCdf(0),
   fXmin(1),
   fXmax(-1),
   fMode(0),
   fSum(0),
   fHasDomain(0),
   fHasMode(0),
   fHasSum(0),
   fOwnFunc(true)
{
   //Constructor from a TF1 objects
}


TUnuranDiscrDist::TUnuranDiscrDist(const TUnuranDiscrDist & rhs) :
   TUnuranBaseDist(),
   fPmf(0),
   fCdf(0)
{
   // Implementation of copy ctor using aassignment operator
   operator=(rhs);
}

TUnuranDiscrDist & TUnuranDiscrDist::operator = (const TUnuranDiscrDist &rhs)
{
   // Implementation of assignment operator (copy only the function pointer not the function itself)
   if (this == &rhs) return *this;  // time saving self-test
   fPVec = rhs.fPVec;
   fPVecSum = rhs.fPVecSum;
   fXmin = rhs.fXmin;
   fXmax = rhs.fXmax;
   fMode = rhs.fMode;
   fSum  = rhs.fSum;
   fHasDomain = rhs.fHasDomain;
   fHasMode   = rhs.fHasMode;
   fHasSum    = rhs.fHasSum;
   fOwnFunc   = rhs.fOwnFunc;
   if (!fOwnFunc) {
      fPmf   = rhs.fPmf;
      fCdf   = rhs.fCdf;
   }
   else {
      if (fPmf) delete fPmf;
      if (fCdf) delete fCdf;
      fPmf  = (rhs.fPmf)  ? rhs.fPmf->Clone()  : 0;
      fCdf  = (rhs.fCdf)  ? rhs.fCdf->Clone()  : 0;
   }

   return *this;
}

TUnuranDiscrDist::~TUnuranDiscrDist() {
   // destructor implementation
   if (fOwnFunc) {
      if (fPmf) delete fPmf;
      if (fCdf) delete fCdf;
   }
}

void TUnuranDiscrDist::SetCdf(const ROOT::Math::IGenFunction & cdf) {
   //  set cdf distribution using a generic function interface
   fCdf = (fOwnFunc) ? cdf.Clone() : &cdf;
}

void TUnuranDiscrDist::SetCdf(TF1 *  cdf) {
   // set cumulative distribution function from a TF1
   if (!fOwnFunc && fPmf) {
      // need to manage also the pmf
      fPmf = fPmf->Clone();
   }
   else
      if (fCdf) delete fCdf;

   fCdf = (cdf) ? new ROOT::Math::WrappedTF1 ( *cdf) : 0;
   fOwnFunc = true;
}

double TUnuranDiscrDist::Pmf ( int x) const {
   // evaluate the distribution
   if (!fPmf) {
      if (x < static_cast<int>(fPVec.size()) || x >= static_cast<int>(fPVec.size()) ) return 0;
      return fPVec[x];
   }
   return (*fPmf)(double(x));
}

double TUnuranDiscrDist::Cdf ( int x) const {
   // evaluate the cumulative distribution
   // otherwise evaluate from the sum of the probabilities
   if (fHasDomain && x < fXmin) return 0;

   if (fCdf) {
      return (*fCdf)(double(x));
   }

   //estimation from sum of probability
   int vsize = fPVecSum.size();
   if ( x < vsize )
      return fPVecSum[x];

   // calculate the sum
   int x0 = ( fHasDomain) ? fXmin : 0;
   int i0 = vsize;     // starting index
   int iN = x - x0 + 1;  // maximum index
   fPVecSum.resize(iN);
   double sum = ( i0 > 0 ) ? fPVecSum.back() : 0;
   for (int i = i0; i < iN; ++i) {
      sum += Pmf(i + x0);
      fPVecSum[i] =  sum;
   }

   return fPVecSum.back();

}





