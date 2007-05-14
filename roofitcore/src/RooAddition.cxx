/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Name:  $:$Id: RooAddition.cxx,v 1.7 2007/05/11 09:11:58 verkerke Exp $
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

// -- CLASS DESCRIPTION [REAL] --
//
// RooAddition calculates the sum of a set of RooAbsReal terms, or
// when constructed with two sets, it sums the product of the terms
// in the two sets. This class does not (yet) do any smart handling of integrals, 
// i.e. all integrals of the product are handled numerically


#include "RooFit.h"

#include "Riostream.h"
#include "Riostream.h"
#include <math.h>

#include "RooAddition.h"
#include "RooAbsReal.h"
#include "RooErrorHandler.h"
#include "RooArgSet.h"
#include "RooNLLVar.h"
#include "RooChi2Var.h"

ClassImp(RooAddition)
;

RooAddition::RooAddition()
{
  _setIter1 = _set1.createIterator() ;
  _setIter2 = _set2.createIterator() ;
}


RooAddition::RooAddition(const char* name, const char* title, const RooArgSet& sumSet, Bool_t takeOwnership) :
  RooAbsReal(name, title),
  _set1("set1","First set of components",this),
  _set2("set2","Second set of components",this)
{
  // Constructor
  _setIter1 = _set1.createIterator() ;
  _setIter2 = 0 ;

  TIterator* inputIter = sumSet.createIterator() ;
  RooAbsArg* comp ;
  while((comp = (RooAbsArg*)inputIter->Next())) {
    if (!dynamic_cast<RooAbsReal*>(comp)) {
      cout << "RooAddition::ctor(" << GetName() << ") ERROR: component " << comp->GetName() 
	   << " is not of type RooAbsReal" << endl ;
      RooErrorHandler::softAbort() ;
    }
    _set1.add(*comp) ;
    if (takeOwnership) {
      _ownedList.addOwned(*comp) ;
    }
  }

  delete inputIter ;
}



RooAddition::RooAddition(const char* name, const char* title, const RooArgList& sumSet1, const RooArgList& sumSet2, Bool_t takeOwnership) :
  RooAbsReal(name, title),
  _set1("set1","First set of components",this),
  _set2("set2","Second set of components",this)
{
  // Constructor
  _setIter1 = _set1.createIterator() ;
  _setIter2 = _set2.createIterator() ;

  if (sumSet1.getSize() != sumSet2.getSize()) {
    cout << "RooAddition::ctor(" << GetName() << ") ERROR: input lists should be of equal length" << endl ;
    RooErrorHandler::softAbort() ;    
  }

  TIterator* inputIter1 = sumSet1.createIterator() ;
  RooAbsArg* comp ;
  while((comp = (RooAbsArg*)inputIter1->Next())) {
    if (!dynamic_cast<RooAbsReal*>(comp)) {
      cout << "RooAddition::ctor(" << GetName() << ") ERROR: component " << comp->GetName() 
	   << " in first list is not of type RooAbsReal" << endl ;
      RooErrorHandler::softAbort() ;
    }
    _set1.add(*comp) ;
    if (takeOwnership) {
      _ownedList.addOwned(*comp) ;
    }
  }
  delete inputIter1 ;


  TIterator* inputIter2 = sumSet2.createIterator() ;
  while((comp = (RooAbsArg*)inputIter2->Next())) {
    if (!dynamic_cast<RooAbsReal*>(comp)) {
      cout << "RooAddition::ctor(" << GetName() << ") ERROR: component " << comp->GetName() 
	   << " in first list is not of type RooAbsReal" << endl ;
      RooErrorHandler::softAbort() ;
    }
    _set2.add(*comp) ;
    if (takeOwnership) {
      _ownedList.addOwned(*comp) ;
    }
  }
  delete inputIter2 ;
}



RooAddition::RooAddition(const RooAddition& other, const char* name) :
  RooAbsReal(other, name), 
  _set1("set1",this,other._set1),
  _set2("set2",this,other._set2)
{
  // Copy constructor
  // Copy constructor
  _setIter1 = _set1.createIterator() ;
  if (other._setIter2) {
    _setIter2 = _set2.createIterator() ;
  } else {
    _setIter2 = 0 ;
  }
  
  // Member _ownedList is intentionally not copy-constructed -- ownership is not transferred
}


RooAddition::~RooAddition() 
{
}



Double_t RooAddition::evaluate() const 
{
  Double_t sum(0);
  RooAbsReal* comp ;
  const RooArgSet* nset = _set1.nset() ;

  _setIter1->Reset() ;

  if (!_setIter2) {

    while((comp=(RooAbsReal*)_setIter1->Next())) {
      sum += comp->getVal(nset) ;
    }

  } else {

    RooAbsReal* comp2 ;
    _setIter2->Reset() ;
    while((comp=(RooAbsReal*)_setIter1->Next())) {
      comp2 = (RooAbsReal*)_setIter2->Next() ;
      sum += comp->getVal(nset)*comp2->getVal(nset) ;
    }


  }
  
  return sum ;
}


Double_t RooAddition::defaultErrorLevel() const 
{
  // See if we contain a RooNLLVar or RooChi2Var object

  RooAbsReal* nllArg(0) ;
  RooAbsReal* chi2Arg(0) ;

  RooAbsArg* arg ;

  _setIter1->Reset() ;
  while((arg=(RooAbsArg*)_setIter1->Next())) {
    if (dynamic_cast<RooNLLVar*>(arg)) {
      nllArg = (RooAbsReal*)arg ;
    }
    if (dynamic_cast<RooChi2Var*>(arg)) {
      chi2Arg = (RooAbsReal*)arg ;
    }
  }

  if (_setIter2) {
    _setIter2->Reset() ;
    while((arg=(RooAbsArg*)_setIter2->Next())) {
      if (dynamic_cast<RooNLLVar*>(arg)) {
	nllArg = (RooAbsReal*)arg ;
      }
      if (dynamic_cast<RooChi2Var*>(arg)) {
	chi2Arg = (RooAbsReal*)arg ;
      }
    }
  }

  if (nllArg && !chi2Arg) {
    cout << "RooAddition::defaultErrorLevel(" << GetName() 
	 << ") Summation contains a RooNLLVar, using its error level" << endl ;
    return nllArg->defaultErrorLevel() ;
  } else if (chi2Arg && !nllArg) {
    cout << "RooAddition::defaultErrorLevel(" << GetName() 
	 << ") Summation contains a RooChi2Var, using its error level" << endl ;
    return chi2Arg->defaultErrorLevel() ;
  } else if (!nllArg && !chi2Arg) {
    cout << "RooAddition::defaultErrorLevel(" << GetName() << ") WARNING: "
	 << "Summation contains neither RooNLLVar nor RooChi2Var server, using default level of 1.0" << endl ;
  } else {
    cout << "RooAddition::defaultErrorLevel(" << GetName() << ") WARNING: "
	 << "Summation contains BOTH RooNLLVar and RooChi2Var server, using default level of 1.0" << endl ;
  }

  return 1.0 ;
}

