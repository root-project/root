/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooRealConstant.cc,v 1.1 2001/10/03 16:16:31 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   16-Aug-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [REAL] --

#include "RooFitCore/RooRealConstant.hh"
#include "RooFitCore/RooRealVar.hh"
#include "RooFitCore/RooArgList.hh"

ClassImp(RooRealConstant)
;


RooArgList* RooRealConstant::_constDB(0) ;
TIterator* RooRealConstant::_constDBIter(0) ;


const RooRealVar& RooRealConstant::value(Double_t value) 
{
  // Lookup existing constant
  init() ;
  RooRealVar* var ;
  while(var=(RooRealVar*)_constDBIter->Next()) {
    if (var->getVal()==value) return *var ;
  }

  // Create new constant
  char label[128] ;
  sprintf(label,"%8.6f",value) ;
  var = new RooRealVar(label,label,value) ;
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
