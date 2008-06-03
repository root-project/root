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
// RooFracRemainder calculates the sum of a set of RooAbsReal terms, or
// when constructed with two sets, it sums the product of the terms
// in the two sets. This class does not (yet) do any smart handling of integrals, 
// i.e. all integrals of the product are handled numerically


#include "RooFit.h"

#include "Riostream.h"
#include "Riostream.h"
#include <math.h>

#include "RooFracRemainder.h"
#include "RooAbsReal.h"
#include "RooAbsPdf.h"
#include "RooErrorHandler.h"
#include "RooArgSet.h"
#include "RooMsgService.h"

ClassImp(RooFracRemainder)
;

RooFracRemainder::RooFracRemainder()
{
  _setIter1 = _set1.createIterator() ;
}


RooFracRemainder::RooFracRemainder(const char* name, const char* title, const RooArgSet& sumSet) :
  RooAbsReal(name, title),
  _set1("set1","First set of components",this)
{
  // Constructor
  _setIter1 = _set1.createIterator() ;

  TIterator* inputIter = sumSet.createIterator() ;
  RooAbsArg* comp ;
  while((comp = (RooAbsArg*)inputIter->Next())) {
    if (!dynamic_cast<RooAbsReal*>(comp)) {
      coutE(InputArguments) << "RooFracRemainder::ctor(" << GetName() << ") ERROR: component " << comp->GetName() 
			    << " is not of type RooAbsReal" << endl ;
      RooErrorHandler::softAbort() ;
    }
    _set1.add(*comp) ;
  }

  delete inputIter ;
}




RooFracRemainder::RooFracRemainder(const RooFracRemainder& other, const char* name) :
  RooAbsReal(other, name), 
  _set1("set1",this,other._set1)
{
  // Copy constructor
  _setIter1 = _set1.createIterator() ;
  
  // Member _ownedList is intentionally not copy-constructed -- ownership is not transferred
}


RooFracRemainder::~RooFracRemainder() 
{
  if (_setIter1) delete _setIter1 ;
}



Double_t RooFracRemainder::evaluate() const 
{
  Double_t sum(1);
  RooAbsReal* comp ;
  const RooArgSet* nset = _set1.nset() ;

  _setIter1->Reset() ;

  while((comp=(RooAbsReal*)_setIter1->Next())) {
    sum -= comp->getVal(nset) ;
  }
    
  return sum ;
}

