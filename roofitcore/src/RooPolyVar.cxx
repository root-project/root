/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *
 * Copyright (C) 1999 Stanford University
 *****************************************************************************/

// -- CLASS DESCRIPTION [PDF] --

#include <iostream.h>
#include <math.h>

#include "RooFitCore/RooPolyVar.hh"
#include "RooFitCore/RooAbsReal.hh"
#include "RooFitCore/RooRealVar.hh"
#include "RooFitCore/RooArgList.hh"

ClassImp(RooPolyVar)
;

RooPolyVar::RooPolyVar()
{
  _coefIter = _coefList.createIterator() ;
}

RooPolyVar::RooPolyVar(const char* name, const char* title, 
			     RooAbsReal& x, const RooArgList& coefList, Int_t lowestOrder) :
  RooAbsReal(name, title),
  _x("x", "Dependent", this, x),
  _coefList("coefList","List of coefficients",this),
  _lowestOrder(lowestOrder) 
{
  // Constructor
  _coefIter = _coefList.createIterator() ;

  // Check lowest order
  if (_lowestOrder<0) {
    cout << "RooPolyVar::ctor(" << GetName() 
	 << ") WARNING: lowestOrder must be >=1, setting value to 1" << endl ;
    _lowestOrder=0 ;
  }

  TIterator* coefIter = coefList.createIterator() ;
  RooAbsArg* coef ;
  while(coef = (RooAbsArg*)coefIter->Next()) {
    if (!dynamic_cast<RooAbsReal*>(coef)) {
      cout << "RooPolyVar::ctor(" << GetName() << ") ERROR: coefficient " << coef->GetName() 
	   << " is not of type RooAbsReal" << endl ;
      assert(0) ;
    }
    _coefList.add(*coef) ;
  }
  delete coefIter ;
}



RooPolyVar::RooPolyVar(const char* name, const char* title,
                           RooAbsReal& x) :
  RooAbsReal(name, title),
  _x("x", "Dependent", this, x),
  _coefList("coefList","List of coefficients",this),
  _lowestOrder(1)
{
  _coefIter = _coefList.createIterator() ;
}                                                                                                                                 



RooPolyVar::RooPolyVar(const RooPolyVar& other, const char* name) :
  RooAbsReal(other, name), 
  _x("x", this, other._x), 
  _coefList("coefList",this,other._coefList),
  _lowestOrder(other._lowestOrder) 
{
  // Copy constructor
  _coefIter = _coefList.createIterator() ;
}




Double_t RooPolyVar::evaluate() const 
{
  Double_t sum(0) ;
  Int_t order(_lowestOrder) ;
  _coefIter->Reset() ;

  RooAbsReal* coef ;
  const RooArgSet* nset = _coefList.nset() ;
  while(coef=(RooAbsReal*)_coefIter->Next()) {
    sum += coef->getVal(nset)*pow(_x,order++) ;
  }

  return sum;
}


Int_t RooPolyVar::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars) const 
{
  if (matchArgs(allVars, analVars, _x)) return 1;
  return 0;
}



Double_t RooPolyVar::analyticalIntegral(Int_t code) const 
{
  assert(code==1) ;

  Double_t sum(0) ;

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
