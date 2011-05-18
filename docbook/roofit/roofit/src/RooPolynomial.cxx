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

#include "RooFit.h"

#include "Riostream.h"
#include "Riostream.h"
#include "TMath.h"

#include "RooPolynomial.h"
#include "RooAbsReal.h"
#include "RooRealVar.h"
#include "RooArgList.h"

ClassImp(RooPolynomial)
;


//_____________________________________________________________________________
RooPolynomial::RooPolynomial()
{
  // coverity[UNINIT_CTOR]
  _coefIter = _coefList.createIterator() ;
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
  _coefIter = _coefList.createIterator() ;

  // Check lowest order
  if (_lowestOrder<0) {
    cout << "RooPolynomial::ctor(" << GetName() 
	 << ") WARNING: lowestOrder must be >=0, setting value to 0" << endl ;
    _lowestOrder=0 ;
  }

  TIterator* coefIter = coefList.createIterator() ;
  RooAbsArg* coef ;
  while((coef = (RooAbsArg*)coefIter->Next())) {
    if (!dynamic_cast<RooAbsReal*>(coef)) {
      cout << "RooPolynomial::ctor(" << GetName() << ") ERROR: coefficient " << coef->GetName() 
	   << " is not of type RooAbsReal" << endl ;
      assert(0) ;
    }
    _coefList.add(*coef) ;
  }
  delete coefIter ;
}



//_____________________________________________________________________________
RooPolynomial::RooPolynomial(const char* name, const char* title,
                           RooAbsReal& x) :
  RooAbsPdf(name, title),
  _x("x", "Dependent", this, x),
  _coefList("coefList","List of coefficients",this),
  _lowestOrder(1)
{
  _coefIter = _coefList.createIterator() ;
}                                                                                                                                 



//_____________________________________________________________________________
RooPolynomial::RooPolynomial(const RooPolynomial& other, const char* name) :
  RooAbsPdf(other, name), 
  _x("x", this, other._x), 
  _coefList("coefList",this,other._coefList),
  _lowestOrder(other._lowestOrder) 
{
  // Copy constructor
  _coefIter = _coefList.createIterator() ;
}




//_____________________________________________________________________________
RooPolynomial::~RooPolynomial()
{
  // Destructor
  delete _coefIter ;
}




//_____________________________________________________________________________
Double_t RooPolynomial::evaluate() const 
{
  Int_t order(_lowestOrder) ;
  Double_t sum(order<1 ? 0 : 1) ;

  _coefIter->Reset() ;

  RooAbsReal* coef ;
  const RooArgSet* nset = _coefList.nset() ;
  while((coef=(RooAbsReal*)_coefIter->Next())) {
    sum += coef->getVal(nset)*TMath::Power(_x,order++) ;
  }

//   if (sum<=0) {
    //cout << "RooPolynomial sum = " << sum << endl ;  
//   }
  return sum;
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
  assert(code==1) ;

  Int_t order(_lowestOrder) ;
  
  Double_t sum(order>0 ? _x.max(rangeName)-_x.min(rangeName) : 0) ;
  //cout << "RooPolynomial::aI(" << GetName() << ") range = " << _x.min(rangeName) << " - " << _x.max(rangeName) << endl ;
  
  const RooArgSet* nset = _coefList.nset() ;
  _coefIter->Reset() ;
  RooAbsReal* coef ;

  // Primitive = sum(k) coef_k * 1/(k+1) x^(k+1)
  while((coef=(RooAbsReal*)_coefIter->Next())) {
    sum += coef->getVal(nset)*(TMath::Power(_x.max(rangeName),order+1)-TMath::Power(_x.min(rangeName),order+1))/(order+1) ; 
    order++ ;
  }

  return sum;  
  
}
