/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 * @(#)root/roofit:$Id$
 * Authors:                                                                  *
 *   GR, Gerhard Raven,   UC San Diego, Gerhard.Raven@slac.stanford.edu
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

//////////////////////////////////////////////////////////////////////////////
//
// BEGIN_HTML
// Chebychev polynomial p.d.f. of the first kind
// END_HTML
//

#include <cmath>
#include <iostream>

#include "RooFit.h"

#include "Riostream.h"

#include "RooChebychev.h"
#include "RooAbsReal.h"
#include "RooRealVar.h"
#include "RooArgList.h"
#include "RooNameReg.h"

#include "TError.h"

#if defined(__my_func__)
#undef __my_func__
#endif
#if defined(WIN32)
#define __my_func__ __FUNCTION__
#else
#define __my_func__ __func__
#endif

ClassImp(RooChebychev)
;

//_____________________________________________________________________________
RooChebychev::RooChebychev() : _refRangeName(0)
{
}


//_____________________________________________________________________________
RooChebychev::RooChebychev(const char* name, const char* title, 
                           RooAbsReal& x, const RooArgList& coefList): 
  RooAbsPdf(name, title),
  _x("x", "Dependent", this, x),
  _coefList("coefficients","List of coefficients",this),
  _refRangeName(0)
{
  // Constructor
  TIterator* coefIter = coefList.createIterator() ;
  RooAbsArg* coef ;
  while((coef = (RooAbsArg*)coefIter->Next())) {
    if (!dynamic_cast<RooAbsReal*>(coef)) {
	std::cerr << "RooChebychev::ctor(" << GetName() <<
	    ") ERROR: coefficient " << coef->GetName() <<
	    " is not of type RooAbsReal" << std::endl ;
      R__ASSERT(0) ;
    }
    _coefList.add(*coef) ;
  }

  delete coefIter ;
}



//_____________________________________________________________________________
RooChebychev::RooChebychev(const RooChebychev& other, const char* name) :
  RooAbsPdf(other, name), 
  _x("x", this, other._x), 
  _coefList("coefList",this,other._coefList),
  _refRangeName(other._refRangeName)
{
}

//inline static double p0(double ,double a) { return a; }
inline static double p1(double t,double a,double b) { return a*t+b; }
inline static double p2(double t,double a,double b,double c) { return p1(t,p1(t,a,b),c); }
inline static double p3(double t,double a,double b,double c,double d) { return p2(t,p1(t,a,b),c,d); }
//inline static double p4(double t,double a,double b,double c,double d,double e) { return p3(t,p1(t,a,b),c,d,e); }


//_____________________________________________________________________________
void RooChebychev::selectNormalizationRange(const char* rangeName, Bool_t force) 
{
  if (rangeName && (force || !_refRangeName)) {
    _refRangeName = (TNamed*) RooNameReg::instance().constPtr(rangeName) ;
  }
  if (!rangeName) {
    _refRangeName = 0 ;
  }
}


//_____________________________________________________________________________
Double_t RooChebychev::evaluate() const 
{
  
  Double_t xmin = _x.min(_refRangeName?_refRangeName->GetName():0) ; Double_t xmax = _x.max(_refRangeName?_refRangeName->GetName():0);
  Double_t x(-1+2*(_x-xmin)/(xmax-xmin));
  Double_t x2(x*x);
  Double_t sum(0) ;
  switch (_coefList.getSize()) {
  case  7: sum+=((RooAbsReal&)_coefList[6]).getVal()*x*p3(x2,64,-112,56,-7);
  case  6: sum+=((RooAbsReal&)_coefList[5]).getVal()*p3(x2,32,-48,18,-1);
  case  5: sum+=((RooAbsReal&)_coefList[4]).getVal()*x*p2(x2,16,-20,5);
  case  4: sum+=((RooAbsReal&)_coefList[3]).getVal()*p2(x2,8,-8,1);
  case  3: sum+=((RooAbsReal&)_coefList[2]).getVal()*x*p1(x2,4,-3);
  case  2: sum+=((RooAbsReal&)_coefList[1]).getVal()*p1(x2,2,-1);
  case  1: sum+=((RooAbsReal&)_coefList[0]).getVal()*x;
  case  0: sum+=1; break;
  default: std::cerr << "In " << __my_func__ << " (" << __FILE__ << ", line " <<
	       __LINE__ << "): Higher order Chebychev polynomials currently "
	       "unimplemented." << std::endl;
	   R__ASSERT(false);
  }
  return sum;
}


//_____________________________________________________________________________
Int_t RooChebychev::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* /* rangeName */) const 
{
  if (matchArgs(allVars, analVars, _x)) return 1;
  return 0;
}


//_____________________________________________________________________________
Double_t RooChebychev::analyticalIntegral(Int_t code, const char* rangeName) const 
{

  R__ASSERT(1 == code);

  // the full range of the function is mapped to the normalised [-1, 1] range

  const Double_t xminfull(_x.min(_refRangeName?_refRangeName->GetName():0)) ;
  const Double_t xmaxfull(_x.max(_refRangeName?_refRangeName->GetName():0)) ;

  const Double_t fullRange = xmaxfull - xminfull;

  // define limits of the integration range on a normalised scale
  Double_t minScaled = -1., maxScaled = +1.;

  minScaled = -1. + 2. * (_x.min(rangeName) - xminfull) / fullRange;
  maxScaled = +1. - 2. * (xmaxfull - _x.max(rangeName)) / fullRange;

  // return half of the range since the normalised range runs from -1 to 1
  // which has a range of two
  double val =  0.5 * fullRange * (evalAnaInt(maxScaled) - evalAnaInt(minScaled));
  //std::cout << " integral = " << val << std::endl;
  return val;
}

Double_t RooChebychev::evalAnaInt(const Double_t x) const
{
  const Double_t x2 = x * x;
  Double_t sum = 0.;
  switch (_coefList.getSize()) {
    case  7: sum+=((RooAbsReal&)_coefList[6]).getVal()*x2*p3(x2,8.,-112./6.,14.,-7./2.);
    case  6: sum+=((RooAbsReal&)_coefList[5]).getVal()*x*p3(x2,32./7.,-48./5.,6.,-1.);
    case  5: sum+=((RooAbsReal&)_coefList[4]).getVal()*x2*p2(x2,16./6.,-5.,2.5);
    case  4: sum+=((RooAbsReal&)_coefList[3]).getVal()*x*p2(x2,8./5.,-8./3.,1.);
    case  3: sum+=((RooAbsReal&)_coefList[2]).getVal()*x2*p1(x2,1.,-3./2.);
    case  2: sum+=((RooAbsReal&)_coefList[1]).getVal()*x*p1(x2,2./3.,-1.);
    case  1: sum+=((RooAbsReal&)_coefList[0]).getVal()*x2*.5;
    case  0: sum+=x; break;
	     
    default: std::cerr << "In " << __my_func__ << " (" << __FILE__ << ", line " <<
	     __LINE__ << "): Higher order Chebychev polynomials currently "
		 "unimplemented." << std::endl;
	     R__ASSERT(false);
  }
  return sum;
}

#undef __my_func__
