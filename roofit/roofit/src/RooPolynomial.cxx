/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 * @(#)root/roofit:$Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
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
// RooPolynomial implements a polynomial p.d.f of the form
// <pre>
// f(x) = sum_i a_i * x^i
//</pre>
// By default coefficient a_0 is chosen to be 1, as polynomial
// probability density functions have one degree of freedome
// less than polynomial functions due to the normalization condition
// END_HTML
//

#include <cmath>
#include <cassert>

#include "RooPolynomial.h"
#include "RooAbsReal.h"
#include "RooArgList.h"
#include "RooMsgService.h"

#include "TError.h"

using namespace std;

ClassImp(RooPolynomial)
;


//_____________________________________________________________________________
RooPolynomial::RooPolynomial()
{
  // coverity[UNINIT_CTOR]
}


//_____________________________________________________________________________
RooPolynomial::RooPolynomial(const char* name, const char* title, 
			     RooAbsReal& x, const RooArgList& coefList, Int_t lowestOrder) :
  RooAbsPdf(name, title),
  _x("x", "Dependent", this, x),
  _coefList("coefList","List of coefficients",this),
  _lowestOrder(lowestOrder) 
{
  // Constructor

  // Check lowest order
  if (_lowestOrder<0) {
    coutE(InputArguments) << "RooPolynomial::ctor(" << GetName() 
			  << ") WARNING: lowestOrder must be >=0, setting value to 0" << endl ;
    _lowestOrder=0 ;
  }

  RooFIter coefIter = coefList.fwdIterator() ;
  RooAbsArg* coef ;
  while((coef = (RooAbsArg*)coefIter.next())) {
    if (!dynamic_cast<RooAbsReal*>(coef)) {
      coutE(InputArguments) << "RooPolynomial::ctor(" << GetName() << ") ERROR: coefficient " << coef->GetName() 
			    << " is not of type RooAbsReal" << endl ;
      R__ASSERT(0) ;
    }
    _coefList.add(*coef) ;
  }
}



//_____________________________________________________________________________
RooPolynomial::RooPolynomial(const char* name, const char* title,
                           RooAbsReal& x) :
  RooAbsPdf(name, title),
  _x("x", "Dependent", this, x),
  _coefList("coefList","List of coefficients",this),
  _lowestOrder(1)
{ }                                                                                                                                 

//_____________________________________________________________________________
RooPolynomial::RooPolynomial(const RooPolynomial& other, const char* name) :
  RooAbsPdf(other, name), 
  _x("x", this, other._x), 
  _coefList("coefList",this,other._coefList),
  _lowestOrder(other._lowestOrder) 
{
  // Copy constructor
}




//_____________________________________________________________________________
RooPolynomial::~RooPolynomial()
{
  // Destructor
}




//_____________________________________________________________________________
Double_t RooPolynomial::evaluate() const 
{
  // Calculate and return value of polynomial

  const unsigned sz = _coefList.getSize();
  const int lowestOrder = _lowestOrder;
  if (!sz) return lowestOrder ? 1. : 0.;
  _wksp.clear();
  _wksp.reserve(sz);
  {
    const RooArgSet* nset = _coefList.nset();
    RooFIter it = _coefList.fwdIterator();
    RooAbsReal* c;
    while ((c = (RooAbsReal*) it.next())) _wksp.push_back(c->getVal(nset));
  }
  const Double_t x = _x;
  Double_t retVal = _wksp[sz - 1];
  for (unsigned i = sz - 1; i--; ) retVal = _wksp[i] + x * retVal;
  return retVal * std::pow(x, lowestOrder) + (lowestOrder ? 1.0 : 0.0);
}



//_____________________________________________________________________________
Int_t RooPolynomial::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* /*rangeName*/) const 
{
  if (matchArgs(allVars, analVars, _x)) return 1;
  return 0;
}



//_____________________________________________________________________________
Double_t RooPolynomial::analyticalIntegral(Int_t code, const char* rangeName) const 
{
  R__ASSERT(code==1) ;

  const Double_t xmin = _x.min(rangeName), xmax = _x.max(rangeName);
  const int lowestOrder = _lowestOrder;
  const unsigned sz = _coefList.getSize();
  if (!sz) return xmax - xmin;
  _wksp.clear();
  _wksp.reserve(sz);
  {
    const RooArgSet* nset = _coefList.nset();
    RooFIter it = _coefList.fwdIterator();
    unsigned i = 1 + lowestOrder;
    RooAbsReal* c;
    while ((c = (RooAbsReal*) it.next())) {
      _wksp.push_back(c->getVal(nset) / Double_t(i));
      ++i;
    }
  }
  Double_t min = _wksp[sz - 1], max = _wksp[sz - 1];
  for (unsigned i = sz - 1; i--; )
    min = _wksp[i] + xmin * min, max = _wksp[i] + xmax * max;
  return max * std::pow(xmax, 1 + lowestOrder) - min * std::pow(xmin, 1 + lowestOrder) +
      (lowestOrder ? (xmax - xmin) : 0.);
}
