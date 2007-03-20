// @(#)root/unuran:$Name:  $:$Id: TUnuranDiscrDist.cxx,v 1.2 2007/02/05 10:24:44 moneta Exp $
// Authors: L. Moneta, J. Leydold Wed Feb 28 2007 

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Implementation file for class TUnuranDiscrDist

#include "TUnuranDiscrDist.h"

#include "TF1.h"

#include <cassert>


TUnuranDiscrDist::TUnuranDiscrDist (const TF1 * func) : 
   fPmf(func), 
   fCdf(0), 
   fXmin(1), 
   fXmax(-1), 
   fMode(0), 
   fSum(0),
   fHasDomain(0),
   fHasMode(0),
   fHasSum(0)
{
   //Constructor from a TF1 objects
} 


TUnuranDiscrDist::TUnuranDiscrDist(const TUnuranDiscrDist & rhs) :
   TUnuranBaseDist()
{
   // Implementation of copy ctor using aassignment operator
   operator=(rhs);
}

TUnuranDiscrDist & TUnuranDiscrDist::operator = (const TUnuranDiscrDist &rhs) 
{
   // Implementation of assignment operator (copy only the funciton pointer not the function itself)
   if (this == &rhs) return *this;  // time saving self-test
   fPVec = rhs.fPVec;
   fPmf  = rhs.fPmf;
   fCdf  = rhs.fCdf;
   fXmin = rhs.fXmin;
   fXmax = rhs.fXmax;
   fMode = rhs.fMode;
   fSum  = rhs.fSum;
   fHasDomain = rhs.fHasDomain;
   fHasMode   = rhs.fHasMode;
   fHasSum    = rhs.fHasSum;
   return *this;
}



double TUnuranDiscrDist::Pmf ( int x) const {  
   // evaluate the distribution 
   if (!fPmf) { 
      if (x < static_cast<int>(fPVec.size()) || x >= static_cast<int>(fPVec.size()) ) return 0; 
      return fPVec[x]; 
   }
   return fPmf->Eval(double(x)); 
}

double TUnuranDiscrDist::Cdf ( int x) const {  
   // evaluate the cumulative distribution 
   // otherwise evaluate from the sum of the probabilities 
   assert(fCdf != 0); 
   return fCdf->Eval(double(x)); 
//naive numerical estimation is too slow  
//    double cdf = 0; 
//    int i0 = ( fHasDomain) ? fXmin : 0; 
//    for (int i = i0; i <= x; ++i) 
//       cdf += Pmf(i); 
//    return cdf; 
}





