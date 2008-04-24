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
// RooRecursiveFraction calculates the sum of a set of RooAbsReal terms, or
// when constructed with two sets, it sums the product of the terms
// in the two sets. This class does not (yet) do any smart handling of integrals, 
// i.e. all integrals of the product are handled numerically


#include "RooFit.h"

#include "Riostream.h"
#include "Riostream.h"
#include <math.h>

#include "RooRecursiveFraction.h"
#include "RooAbsReal.h"
#include "RooAbsPdf.h"
#include "RooErrorHandler.h"
#include "RooArgSet.h"
#include "RooNLLVar.h"
#include "RooChi2Var.h"
#include "RooMsgService.h"

ClassImp(RooRecursiveFraction)
;

RooRecursiveFraction::RooRecursiveFraction()
{
  _listIter = _list.createIterator() ;
}


RooRecursiveFraction::RooRecursiveFraction(const char* name, const char* title, const RooArgList& fracList) :
  RooAbsReal(name, title),
  _list("list","First set of components",this)
{
  // Constructor
  _listIter = _list.createIterator() ;

  TIterator* inputIter = fracList.createIterator() ;
  RooAbsArg* comp ;
  while((comp = (RooAbsArg*)inputIter->Next())) {
    if (!dynamic_cast<RooAbsReal*>(comp)) {
      coutE(InputArguments) << "RooRecursiveFraction::ctor(" << GetName() << ") ERROR: component " << comp->GetName() 
			    << " is not of type RooAbsReal" << endl ;
      RooErrorHandler::softAbort() ;
    }
    _list.add(*comp) ;
  }
  delete inputIter ;
}



RooRecursiveFraction::RooRecursiveFraction(const RooRecursiveFraction& other, const char* name) :
  RooAbsReal(other, name), 
  _list("list",this,other._list)
{
  // Copy constructor
  _listIter = _list.createIterator() ;
}


RooRecursiveFraction::~RooRecursiveFraction() 
{
  if (_listIter) delete _listIter ;
}



Double_t RooRecursiveFraction::evaluate() const 
{
  RooAbsReal* comp ;
  const RooArgSet* nset = _list.nset() ;

  _listIter->Reset() ;
  comp=(RooAbsReal*)_listIter->Next() ;
  Double_t prod = comp->getVal(nset) ;

  while((comp=(RooAbsReal*)_listIter->Next())) {
    prod *= (1-comp->getVal(nset)) ;
  }
    
  return prod ;
}

