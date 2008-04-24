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

// -- CLASS DESCRIPTION [REAL] --
//
// RooConstraintSum calculates the sum of a set of RooAbsReal terms, or
// when constructed with two sets, it sums the product of the terms
// in the two sets. This class does not (yet) do any smart handling of integrals, 
// i.e. all integrals of the product are handled numerically


#include "RooFit.h"

#include "Riostream.h"
#include "Riostream.h"
#include <math.h>

#include "RooConstraintSum.h"
#include "RooAbsReal.h"
#include "RooAbsPdf.h"
#include "RooErrorHandler.h"
#include "RooArgSet.h"
#include "RooNLLVar.h"
#include "RooChi2Var.h"
#include "RooMsgService.h"

ClassImp(RooConstraintSum)
;

RooConstraintSum::RooConstraintSum()
{
  _setIter1 = _set1.createIterator() ;
}


RooConstraintSum::RooConstraintSum(const char* name, const char* title, const RooArgSet& constraintSet) :
  RooAbsReal(name, title),
  _set1("set1","First set of components",this)
{
  // Constructor
  _setIter1 = _set1.createIterator() ;

  TIterator* inputIter = constraintSet.createIterator() ;
  RooAbsArg* comp ;
  while((comp = (RooAbsArg*)inputIter->Next())) {
    if (!dynamic_cast<RooAbsPdf*>(comp)) {
      coutE(InputArguments) << "RooConstraintSum::ctor(" << GetName() << ") ERROR: component " << comp->GetName() 
			    << " is not of type RooAbsPdf" << endl ;
      RooErrorHandler::softAbort() ;
    }
    _set1.add(*comp) ;
  }

  delete inputIter ;
}





RooConstraintSum::RooConstraintSum(const RooConstraintSum& other, const char* name) :
  RooAbsReal(other, name), 
  _set1("set1",this,other._set1)
{
  // Copy constructor
  _setIter1 = _set1.createIterator() ;  
}


RooConstraintSum::~RooConstraintSum() 
{
  if (_setIter1) delete _setIter1 ;
}



Double_t RooConstraintSum::evaluate() const 
{
  Double_t sum(0);
  RooAbsReal* comp ;
  const RooArgSet* nset = _set1.nset() ;

  _setIter1->Reset() ;

  while((comp=(RooAbsReal*)_setIter1->Next())) {
    sum -= ((RooAbsPdf*)comp)->getLogVal(nset) ;
  }
  
  return sum ;
}

