/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

#include "RooFitCore/RooRealFormula.hh"

ClassImp(RooRealFormula)


RooRealFormula::RooRealFormula(const char *name, const char *title, RooArgSet& dependents) : 
  RooAbsReal(name,title), _formula(name,title,dependents)
{  
  _formula.actualDependents().print() ;

  TIterator* depIter = _formula.actualDependents().MakeIterator() ;
  RooAbsArg* server(0) ;
  while (server=(RooAbsArg*)depIter->Next()) {
    addServer(*server) ;
  }
  setValueDirty(kTRUE) ;
  setShapeDirty(kTRUE) ;
}


RooRealFormula::RooRealFormula(const RooRealFormula& other) : 
  RooAbsReal(other), _formula(other._formula)
{
}


RooRealFormula::~RooRealFormula() 
{
}


Double_t RooRealFormula::evaluate()
{
  // Evaluate embedded formula
  return _formula.eval() ;
}


Bool_t RooRealFormula::isValid() 
{
  return isValid(getVal()) ;
}


Bool_t RooRealFormula::isValid(Double_t value) {
  return kTRUE ;
}


Bool_t RooRealFormula::redirectServersHook(RooArgSet& newServerList, Bool_t mustReplaceAll)
{
  // Propagate server change to formula engine
  return _formula.changeDependents(newServerList,mustReplaceAll) ;
}


void RooRealFormula::printToStream(ostream& os, Option_t* opt= 0)
{
  // Print current value and definition of formula
  os << "RooRealFormula: " << GetName() << " = " << GetTitle() << " = " << getVal();
  if(!_unit.IsNull()) os << ' ' << _unit;
  printAttribList(os) ;
  os << endl ;
} 



