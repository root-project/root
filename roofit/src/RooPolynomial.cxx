/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitTools
 *    File: $Id: RooPolynomial.cc,v 1.5 2001/02/13 20:20:56 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   05-Jan-2000 DK Created initial version from RooPolynomialProb
 *   02-Jul-2000 DK Add analytic optimizations for generating toy MC
 *                  at low orders
 *
 * Copyright (C) 1999 Stanford University
 *****************************************************************************/

#include <iostream.h>
#include <math.h>

#include "RooFitModels/RooPolynomial.hh"
#include "RooFitCore/RooAbsReal.hh"
#include "RooFitCore/RooRealVar.hh"
#include "RooFitCore/RooArgList.hh"

ClassImp(RooPolynomial)
;

RooPolynomial::RooPolynomial()
{
  _coefIter = _coefList.createIterator() ;
}

RooPolynomial::RooPolynomial(const char* name, const char* title, 
			     RooAbsReal& x, const RooArgList& coefList, Int_t lowestOrder) :
  RooAbsPdf(name, title),
  _x("x", "Dependent", this, x),
  _coefList("coefList","List of coefficients",this),
  _lowestOrder(lowestOrder) 
{
  // Constructor
  _coefIter = coefList.createIterator() ;

  // Check lowest order
  if (_lowestOrder<1) {
    cout << "RooPolynomial::ctor(" << GetName() 
	 << ") WARNING: lowestOrder must be >=1, setting value to 1" << endl ;
    _lowestOrder=1 ;
  }

  TIterator* coefIter = coefList.createIterator() ;
  RooAbsArg* coef ;
  while(coef = (RooAbsArg*)coefIter->Next()) {
    if (!dynamic_cast<RooAbsReal*>(coef)) {
      cout << "RooPolynomial::ctor(" << GetName() << ") ERROR: coefficient " << coef->GetName() 
	   << " is not of type RooAbsReal" << endl ;
      assert(0) ;
    }
    _coefList.add(*coef) ;
  }
  delete coefIter ;
}



RooPolynomial::RooPolynomial(const RooPolynomial& other, const char* name) :
  RooAbsPdf(other, name), 
  _x("x", this, other._x), 
  _coefList("coefList",this,other._coefList),
  _lowestOrder(other._lowestOrder) 
{
  // Copy constructor
  _coefIter = _coefList.createIterator() ;
}




Double_t RooPolynomial::evaluate() const 
{
  Double_t sum(1) ;
  Int_t order(_lowestOrder) ;
  _coefIter->Reset() ;

  RooAbsReal* coef ;
  const RooArgSet* nset = _coefList.nset() ;
  while(coef=(RooAbsReal*)_coefIter->Next()) {
    sum += coef->getVal(nset)*pow(_x,order++) ;
  }

  return sum;
}


Int_t RooPolynomial::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars) const 
{
  if (matchArgs(allVars, analVars, _x)) return 1;
  return 0;
}



Double_t RooPolynomial::analyticalIntegral(Int_t code) const 
{
  assert(code==1) ;

  Double_t sum(_x.max()-_x.min()) ;

  const RooArgSet* nset = _coefList.nset() ;
  Int_t order(_lowestOrder) ;
  _coefIter->Reset() ;
  RooAbsReal* coef ;

  // Primitive = sum(k) coef_k * 1/(k+1) x^(k+1)
  while(coef=(RooAbsReal*)_coefIter->Next()) {
    sum += coef->getVal(nset)*(pow(_x.max(),order+1)-pow(_x.min(),order+1))/(order+1) ; 
    order++ ;
  }

  return sum;  
  
}
