/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooRealConstant.cc,v 1.4 2002/04/03 23:37:26 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   16-Aug-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
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

const RooAbsReal& RooConst(Double_t val) { return RooRealConstant::value(val) ; }
const RooAbsReal& RooRealConstant::value(Double_t value) 
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
