// @(#)root/unuran:$Name:  $:$Id: TUnuranDistr.cxx,v 1.1 2006/11/15 17:40:36 brun Exp $
// Author: L. Moneta Wed Sep 27 11:53:27 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Implementation file for class TUnuranDistr

#include "TUnuranDistr.h"

#include "TF1.h"



TUnuranDistr::TUnuranDistr (const TF1 * func, const TF1 * cdf , const TF1 * deriv  ) : 
   fFunc(func), 
   fCdf(cdf),
   fDeriv(deriv), 
   fXmin(1.), fXmax(-1.), // if min > max range is undef.
   fHasDomain(0)
{
   // Constructor from a TF1 objects
   assert(func != 0); 
   // use by default the range specified in TF1
   func->GetRange(fXmin, fXmax);
} 


TUnuranDistr::TUnuranDistr(const TUnuranDistr & rhs) : 
   fFunc(rhs.fFunc),
   fCdf(rhs.fCdf),
   fDeriv(rhs.fDeriv),
   fXmin(rhs.fXmin), fXmax(rhs.fXmax), 
   fHasDomain(rhs.fHasDomain)
{
   // Implementation of copy constructor (copy just the pointer ) 
}

TUnuranDistr & TUnuranDistr::operator = (const TUnuranDistr &rhs) 
{
   // Implementation of assignment operator.
   if (this == &rhs) return *this;  // time saving self-test
   fFunc = rhs.fFunc;
   fCdf= rhs.fCdf;
   fDeriv = rhs.fDeriv;
   fXmin= rhs.fXmin; 
   fXmax = rhs.fXmax;
   fHasDomain = rhs.fHasDomain;
   return *this;
}

double TUnuranDistr::operator() ( double x) const { 
   // evaluate the destribution 
      return fFunc->Eval(x); 
}

double TUnuranDistr::Derivative( double x) const { 
   // evaluate the derivative of the function
      if (fDeriv != 0) return fDeriv->Eval(x); 
      // do numerical derivation
      return fFunc->Derivative(x); 
}

double TUnuranDistr::Cdf(double x) const {   
   // evaluate the integral (cdf)  on the domain
   if (fCdf != 0) return fCdf->Eval(x);
   TF1 * f =  const_cast<TF1*>(fFunc);
   return f->Integral(fXmin, x); 
}

double TUnuranDistr::Mode() const { 
   // get the mode   (x location of function maximum)  
   return fFunc->GetMaximumX(fXmin, fXmax); 
}
