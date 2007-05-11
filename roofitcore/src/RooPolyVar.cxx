/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooPolyVar.cc,v 1.11 2005/06/20 15:44:56 wverkerke Exp $
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

// -- CLASS DESCRIPTION [PDF] --

#include "RooFit.h"

#include "Riostream.h"
#include "Riostream.h"
#include <math.h>
#include "TMath.h"

#include "RooPolyVar.h"
#include "RooAbsReal.h"
#include "RooRealVar.h"
#include "RooArgList.h"
#include "TMath.h"

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
  while((coef = (RooAbsArg*)coefIter->Next())) {
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
  while((coef=(RooAbsReal*)_coefIter->Next())) {
    sum += coef->getVal(nset)*TMath::Power(_x,order++) ;
  }

  return sum;
}


Int_t RooPolyVar::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* /*rangeName*/) const 
{
  if (matchArgs(allVars, analVars, _x)) return 1;
  return 0;
}



Double_t RooPolyVar::analyticalIntegral(Int_t code, const char* rangeName) const 
{
  assert(code==1) ;

  Double_t sum(0) ;

  const RooArgSet* nset = _coefList.nset() ;
  Int_t order(_lowestOrder) ;
  _coefIter->Reset() ;
  RooAbsReal* coef ;

  // Primitive = sum(k) coef_k * 1/(k+1) x^(k+1)
  while((coef=(RooAbsReal*)_coefIter->Next())) {
    sum += coef->getVal(nset)*(TMath::Power(_x.max(rangeName),order+1)-TMath::Power(_x.min(rangeName),order+1))/(order+1) ; 
    order++ ;
  }

  return sum;  
  
}
