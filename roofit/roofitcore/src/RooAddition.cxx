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
// RooAddition calculates the sum of a set of RooAbsReal terms, or
// when constructed with two sets, it sums the product of the terms
// in the two sets. This class does not (yet) do any smart handling of integrals, 
// i.e. all integrals of the product are handled numerically
// END_HTML
//


#include "RooFit.h"

#include "Riostream.h"
#include "Riostream.h"
#include <math.h>

#include "RooAddition.h"
#include "RooAbsReal.h"
#include "RooAbsPdf.h"
#include "RooErrorHandler.h"
#include "RooArgSet.h"
#include "RooNLLVar.h"
#include "RooChi2Var.h"
#include "RooMsgService.h"

ClassImp(RooAddition)
;


//_____________________________________________________________________________
RooAddition::RooAddition()
{
  _setIter1 = _set1.createIterator() ;
  _setIter2 = _set2.createIterator() ;
}



//_____________________________________________________________________________
RooAddition::RooAddition(const char* name, const char* title, const RooArgSet& sumSet, Bool_t takeOwnership) :
  RooAbsReal(name, title),
  _set1("set1","First set of components",this),
  _set2("set2","Second set of components",this)
{
  // Constructor with a single set of RooAbsReals. The value of the function will be
  // the sum of the values in sumSet. If takeOwnership is true the RooAddition object
  // will take ownership of the arguments in sumSet

  _setIter1 = _set1.createIterator() ;
  _setIter2 = 0 ;

  TIterator* inputIter = sumSet.createIterator() ;
  RooAbsArg* comp ;
  while((comp = (RooAbsArg*)inputIter->Next())) {
    if (!dynamic_cast<RooAbsReal*>(comp)) {
      coutE(InputArguments) << "RooAddition::ctor(" << GetName() << ") ERROR: component " << comp->GetName() 
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



//_____________________________________________________________________________
RooAddition::RooAddition(const char* name, const char* title, const RooArgList& sumSet1, const RooArgList& sumSet2, Bool_t takeOwnership) :
  RooAbsReal(name, title),
  _set1("set1","First set of components",this),
  _set2("set2","Second set of components",this)
{
  // Constructor with two set of RooAbsReals. The value of the function will be
  //
  //  A = sum_i sumSet1(i)*sumSet2(i) 
  //
  // If takeOwnership is true the RooAddition object will take ownership of the arguments in sumSet

  _setIter1 = _set1.createIterator() ;
  _setIter2 = _set2.createIterator() ;

  if (sumSet1.getSize() != sumSet2.getSize()) {
    coutE(InputArguments) << "RooAddition::ctor(" << GetName() << ") ERROR: input lists should be of equal length" << endl ;
    RooErrorHandler::softAbort() ;    
  }

  TIterator* inputIter1 = sumSet1.createIterator() ;
  RooAbsArg* comp ;
  while((comp = (RooAbsArg*)inputIter1->Next())) {
    if (!dynamic_cast<RooAbsReal*>(comp)) {
      coutE(InputArguments) << "RooAddition::ctor(" << GetName() << ") ERROR: component " << comp->GetName() 
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
      coutE(InputArguments) << "RooAddition::ctor(" << GetName() << ") ERROR: component " << comp->GetName() 
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



//_____________________________________________________________________________
RooAddition::RooAddition(const RooAddition& other, const char* name) :
  RooAbsReal(other, name), 
  _set1("set1",this,other._set1),
  _set2("set2",this,other._set2)
{
  // Copy constructor

  _setIter1 = _set1.createIterator() ;
  if (other._setIter2) {
    _setIter2 = _set2.createIterator() ;
  } else {
    _setIter2 = 0 ;
  }
  
  // Member _ownedList is intentionally not copy-constructed -- ownership is not transferred
}



//_____________________________________________________________________________
RooAddition::~RooAddition() 
{
  // Destructor

  if (_setIter1) delete _setIter1 ;
  if (_setIter2) delete _setIter2 ;
}




//_____________________________________________________________________________
Double_t RooAddition::evaluate() const 
{
  // Calculate and return current value of self

  Double_t sum(0);
  RooAbsReal* comp ;
  const RooArgSet* nset = _set1.nset() ;

  _setIter1->Reset() ;

  if (_setIter2 && _set2.getSize()==0) {
    delete _setIter2 ;
    _setIter2=0 ;
  }

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



//_____________________________________________________________________________
Double_t RooAddition::defaultErrorLevel() const 
{
  // Return the default error level for MINUIT error analysis
  // If the addition contains one or more RooNLLVars and 
  // no RooChi2Vars, return the defaultErrorLevel() of
  // RooNLLVar. If the addition contains one ore more RooChi2Vars
  // and no RooNLLVars, return the defaultErrorLevel() of
  // RooChi2Var. If the addition contains neither or both
  // issue a warning message and return a value of 1

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
    coutI(Fitting) << "RooAddition::defaultErrorLevel(" << GetName() 
		   << ") Summation contains a RooNLLVar, using its error level" << endl ;
    return nllArg->defaultErrorLevel() ;
  } else if (chi2Arg && !nllArg) {
    coutI(Fitting) << "RooAddition::defaultErrorLevel(" << GetName() 
		   << ") Summation contains a RooChi2Var, using its error level" << endl ;
    return chi2Arg->defaultErrorLevel() ;
  } else if (!nllArg && !chi2Arg) {
    coutI(Fitting) << "RooAddition::defaultErrorLevel(" << GetName() << ") WARNING: "
		   << "Summation contains neither RooNLLVar nor RooChi2Var server, using default level of 1.0" << endl ;
  } else {
    coutI(Fitting) << "RooAddition::defaultErrorLevel(" << GetName() << ") WARNING: "
		   << "Summation contains BOTH RooNLLVar and RooChi2Var server, using default level of 1.0" << endl ;
  }

  return 1.0 ;
}

