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

/**
\file RooRecursiveFraction.cxx
\class RooRecursiveFraction
\ingroup Roofitcore

Class RooRecursiveFraction is a RooAbsReal implementation that
calculates the plain fraction of sum of RooAddPdf components
from a set of recursive fractions: for a given set of input fractions
a_i it returns a_0 * Prod_i (1 - a_i). 
**/


#include "RooFit.h"

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

using namespace std;

ClassImp(RooRecursiveFraction);



////////////////////////////////////////////////////////////////////////////////
/// Default constructor

RooRecursiveFraction::RooRecursiveFraction()
{
  _listIter = _list.createIterator() ;
}



////////////////////////////////////////////////////////////////////////////////
/// Constructor of plain RooAddPdf fraction from list of recursive fractions

RooRecursiveFraction::RooRecursiveFraction(const char* name, const char* title, const RooArgList& fracList) :
  RooAbsReal(name, title),
  _list("list","First set of components",this)
{
  _listIter = _list.createIterator() ;

  for (Int_t ifrac=fracList.getSize()-1 ; ifrac>=0 ; ifrac--) {
    RooAbsArg* comp = fracList.at(ifrac) ;
    if (!dynamic_cast<RooAbsReal*>(comp)) {
      coutE(InputArguments) << "RooRecursiveFraction::ctor(" << GetName() << ") ERROR: component " << comp->GetName() 
			    << " is not of type RooAbsReal" << endl ;
      RooErrorHandler::softAbort() ;
    }
    _list.add(*comp) ;    
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooRecursiveFraction::RooRecursiveFraction(const RooRecursiveFraction& other, const char* name) :
  RooAbsReal(other, name), 
  _list("list",this,other._list)
{
  _listIter = _list.createIterator() ;
}



////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooRecursiveFraction::~RooRecursiveFraction() 
{
  if (_listIter) delete _listIter ;
}



////////////////////////////////////////////////////////////////////////////////
/// Calculate and return value of 1 - prod_i (1 - f_i )

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

