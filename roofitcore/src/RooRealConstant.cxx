/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooRealConstant.cc,v 1.8 2002/09/05 04:33:52 verkerke Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2004, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

// -- CLASS DESCRIPTION [REAL] --
// RooRealConstant provides static functions to create and keep track
// of RooRealVar constants. Instead of creating such constants by
// hand (e.g. RooRealVar one("one","one",1)), simply use
//
//  RooRealConstant::value(1.0)
//
// whenever a reference to RooRealVar with constant value 1.0 is needed.
// RooRealConstant keeps an internal database of previously created
// RooRealVar objects and will recycle them as appropriate.

#include "RooFitCore/RooRealConstant.hh"
#include "RooFitCore/RooConstVar.hh"
#include "RooFitCore/RooArgList.hh"

ClassImp(RooRealConstant)
;


RooArgList* RooRealConstant::_constDB = 0;
TIterator* RooRealConstant::_constDBIter = 0;

RooConstVar& RooConst(Double_t val) { return RooRealConstant::value(val) ; }

RooConstVar& RooRealConstant::value(Double_t value) 
{
  // Lookup existing constant
  init() ;
  RooConstVar* var ;
  while(var=(RooConstVar*)_constDBIter->Next()) {
    if (var->getVal()==value) return *var ;
  }

  // Create new constant
  char label[128] ;
  sprintf(label,"%8.6f",value) ;
  var = new RooConstVar(label,label,value) ;
  var->setAttribute("RooRealConstant_Factory_Object",kTRUE) ;
  _constDB->add(*var) ;

  return *var ;
}



void RooRealConstant::init() 
{
  if (!_constDB) {
    _constDB = new RooArgList("RooRealVar Constants Database") ;
    _constDBIter = _constDB->createIterator() ;
  } else {
    _constDBIter->Reset() ;
  }
}
