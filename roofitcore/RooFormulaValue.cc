/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooFormulaValue.cc,v 1.3 2001/03/16 07:59:12 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

#include "RooFitCore/RooFormulaValue.hh"

ClassImp(RooFormulaValue)


RooFormulaValue::RooFormulaValue(const char *name, const char *title, RooArgSet& dependents) : 
  RooAbsValue(name,title), _formula(name,title,dependents)
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


RooFormulaValue::RooFormulaValue(const RooFormulaValue& other) : 
  RooAbsValue(other), _formula(other._formula)
{
}


RooFormulaValue::~RooFormulaValue() 
{
}


Double_t RooFormulaValue::evaluate()
{
  // Evaluate embedded formula
  return _formula.eval() ;
}


Bool_t RooFormulaValue::isValid() 
{
  return isValid(getVal()) ;
}


Bool_t RooFormulaValue::isValid(Double_t value) {
  return kTRUE ;
}


Bool_t RooFormulaValue::redirectServersHook(RooArgSet& newServerList, Bool_t mustReplaceAll)
{
  // Propagate server change to formula engine
  return _formula.changeDependents(newServerList,mustReplaceAll) ;
}


void RooFormulaValue::printToStream(ostream& os, Option_t* opt= 0)
{
  // Print current value and definition of formula
  os << "RooFormulaValue: " << GetName() << " = " << GetTitle() << " = " << getVal();
  if(!_unit.IsNull()) os << ' ' << _unit;
  printAttribList(os) ;
  os << endl ;
} 



