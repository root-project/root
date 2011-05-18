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
// RooConstraintSum calculates the sum of the -(log) likelihoods of
// a set of RooAbsPfs that represent constraint functions. This class
// is used to calculate the composite -log(L) of constraints to be
// added the regular -log(L) in RooAbsPdf::fitTo() with Constrain(..)
// arguments
// END_HTML
//


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


//_____________________________________________________________________________
RooConstraintSum::RooConstraintSum()
{
  // Default constructor
  _setIter1 = _set1.createIterator() ;
}



//_____________________________________________________________________________
RooConstraintSum::RooConstraintSum(const char* name, const char* title, const RooArgSet& constraintSet, const RooArgSet& paramSet) :
  RooAbsReal(name, title),
  _set1("set1","First set of components",this),
  _paramSet("paramSet","Set of parameters",this)
{
  // Constructor with set of constraint p.d.f.s. All elements in constraintSet must inherit from RooAbsPdf

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

  _paramSet.add(paramSet) ;

  delete inputIter ;
}





//_____________________________________________________________________________
RooConstraintSum::RooConstraintSum(const RooConstraintSum& other, const char* name) :
  RooAbsReal(other, name), 
  _set1("set1",this,other._set1),
  _paramSet("paramSet",this,other._paramSet)
{
  // Copy constructor

  _setIter1 = _set1.createIterator() ;  
}



//_____________________________________________________________________________
RooConstraintSum::~RooConstraintSum() 
{
  // Destructor

  if (_setIter1) delete _setIter1 ;
}



//_____________________________________________________________________________
Double_t RooConstraintSum::evaluate() const 
{
  // Return sum of -log of constraint p.d.f.s

  Double_t sum(0);
  RooAbsReal* comp ;
  _setIter1->Reset() ;

  while((comp=(RooAbsReal*)_setIter1->Next())) {
    sum -= ((RooAbsPdf*)comp)->getLogVal(&_paramSet) ;
  }
  
  return sum ;
}

