/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
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
// Class RooPolyVar is a RooAbsReal implementing a polynomial in terms
// of a list of RooAbsReal coefficients
// <pre>
// f(x) = sum_i a_i * x
// </pre>
// Class RooPolyvar implements analytical integrals of all polynomials
// it can define.
// END_HTML
//

#include <cmath>

#include "TError.h"
#include "RooPolyVar.h"
#include "RooArgList.h"
#include "RooMsgService.h"

using namespace std;

ClassImp(RooPolyVar)
;


//_____________________________________________________________________________
RooPolyVar::RooPolyVar() : _lowestOrder(0)
{
  // Default constructor
}


//_____________________________________________________________________________
RooPolyVar::RooPolyVar(const char* name, const char* title, 
			     RooAbsReal& x, const RooArgList& coefList, Int_t lowestOrder) :
  RooAbsReal(name, title),
  _x("x", "Dependent", this, x),
  _coefList("coefList","List of coefficients",this),
  _lowestOrder(lowestOrder) 
{
  // Construct polynomial in x with coefficients in coefList. If
  // lowestOrder is not zero, then the first element in coefList is
  // interpreted as as the 'lowestOrder' coefficients and all
  // subsequent coeffient elements are shifted by a similar amount.

  // Check lowest order
  if (_lowestOrder<0) {
    coutE(InputArguments) << "RooPolyVar::ctor(" << GetName() 
			  << ") WARNING: lowestOrder must be >=0, setting value to 0" << endl ;
    _lowestOrder=0 ;
  }

  RooFIter coefIter = coefList.fwdIterator() ;
  RooAbsArg* coef ;
  while((coef = (RooAbsArg*)coefIter.next())) {
    if (!dynamic_cast<RooAbsReal*>(coef)) {
      coutE(InputArguments) << "RooPolyVar::ctor(" << GetName() << ") ERROR: coefficient " << coef->GetName() 
			    << " is not of type RooAbsReal" << endl ;
      R__ASSERT(0) ;
    }
    _coefList.add(*coef) ;
  }
}



//_____________________________________________________________________________
RooPolyVar::RooPolyVar(const char* name, const char* title,
                           RooAbsReal& x) :
  RooAbsReal(name, title),
  _x("x", "Dependent", this, x),
  _coefList("coefList","List of coefficients",this),
  _lowestOrder(1)
{
  // Constructor of flat polynomial function
}                                                                                                                                 



//_____________________________________________________________________________
RooPolyVar::RooPolyVar(const RooPolyVar& other, const char* name) :
  RooAbsReal(other, name), 
  _x("x", this, other._x), 
  _coefList("coefList",this,other._coefList),
  _lowestOrder(other._lowestOrder) 
{
  // Copy constructor
}




//_____________________________________________________________________________
RooPolyVar::~RooPolyVar() 
{
  // Destructor
}




//_____________________________________________________________________________
Double_t RooPolyVar::evaluate() const 
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
  return retVal * std::pow(x, lowestOrder);
}



//_____________________________________________________________________________
Int_t RooPolyVar::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* /*rangeName*/) const 
{
  // Advertise that we can internally integrate over x

  if (matchArgs(allVars, analVars, _x)) return 1;
  return 0;
}



//_____________________________________________________________________________
Double_t RooPolyVar::analyticalIntegral(Int_t code, const char* rangeName) const 
{
  // Calculate and return analytical integral over x
  (void)code;
  assert(code==1) ;

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
  return max * std::pow(xmax, 1 + lowestOrder) - min * std::pow(xmin, 1 + lowestOrder);
}
