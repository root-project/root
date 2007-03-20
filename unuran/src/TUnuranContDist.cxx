// @(#)root/unuran:$Name:  $:$Id: TUnuranContDist.cxx,v 1.2 2007/02/05 10:24:44 moneta Exp $
// Authors: L. Moneta, J. Leydold Wed Feb 28 2007

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Implementation file for class TUnuranContDist

#include "TUnuranContDist.h"

#include "TF1.h"
#include <cassert>

ClassImp(TUnuranContDist)


TUnuranContDist::TUnuranContDist (const TF1 * pdf, const TF1 * deriv, bool isLogPdf  ) : 
   fPdf(pdf),
   fDPdf(deriv),
   fCdf(0), 
   fXmin(1.), 
   fXmax(-1.), 
   fMode(0), 
   fArea(0),
   fIsLogPdf(isLogPdf),
   fHasDomain(0),
   fHasMode(0),
   fHasArea(0)
{
   // Constructor from a TF1 objects
} 


TUnuranContDist::TUnuranContDist(const TUnuranContDist & rhs) : 
   TUnuranBaseDist()
{
   // Implementation of copy constructor (copy just the pointer ) 
   operator=(rhs);
}

TUnuranContDist & TUnuranContDist::operator = (const TUnuranContDist &rhs) 
{
   // Implementation of assignment operator.
   if (this == &rhs) return *this;  // time saving self-test
    fPdf   = rhs.fPdf;
   fDPdf  = rhs.fDPdf;
   fCdf   = rhs.fCdf; 
   fXmin  = rhs.fXmin;  
   fXmax  = rhs.fXmax;  
   fMode  = rhs.fMode; 
   fArea  = rhs.fArea;
   fIsLogPdf  = rhs.fIsLogPdf;
   fHasDomain = rhs.fHasDomain;
   fHasMode   = rhs.fHasMode;
   fHasArea   = rhs.fHasArea;
   return *this;
}

double TUnuranContDist::Pdf ( double x) const { 
   // evaluate the pdf of the destribution    
   assert(fPdf != 0);
   return fPdf->Eval(x); 
}

double TUnuranContDist::DPdf( double x) const { 
   // evaluate the derivative of the pdf
   // if derivative function is not given is evaluated numerically
   if (fDPdf != 0) return fDPdf->Eval(x); 
   // do numerical derivation using algorithm in TF1
   assert(fPdf != 0);
   return fPdf->Derivative(x); 
}

double TUnuranContDist::Cdf(double x) const {   
   // evaluate the integral (cdf)  on the domain
   assert (fCdf != 0); 
   return fCdf->Eval(x);
   // t.b.t if the cdf function is not provided evaluate numerically 
   // (need methods for undefined integration in mathmore), cannot use TF1::Integral
   // assert(fPdf != 0);
   // TF1 * f =  const_cast<TF1*>(fPdf);
   //  return f->Integral(fXmin, x); 
}

