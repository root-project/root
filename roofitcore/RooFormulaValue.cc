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

#include "RooFitCore/RooFormulaValue.hh"

ClassImp(RooFormulaValue)


RooFormulaValue::RooFormulaValue(const char *name, const char *title, RooArgSet& dependents) : 
  RooDerivedValue(name,title), _formula(name,title,dependents)
{  
  TIterator* depIter = _formula.actualDependents().MakeIterator() ;
  RooAbsArg* server(0) ;
  while (server=(RooAbsArg*)depIter->Next()) {
    addServer(*server) ;
  }
  setDirty(kTRUE) ;
}

RooFormulaValue::RooFormulaValue(const RooFormulaValue& other) : 
  RooDerivedValue(other), _formula(other._formula)
{
}

RooFormulaValue::~RooFormulaValue() 
{
}

Double_t RooFormulaValue::Evaluate()
{
  // Evaluate embedded formula
  return _formula.Eval() ;
}


Bool_t RooFormulaValue::redirectServers(RooArgSet& newServerList, Bool_t mustReplaceAll)
{
  // Change current servers to new servers with the same name in the given list
  RooDerivedValue::redirectServers(newServerList,mustReplaceAll) ;
  Bool_t ret =  _formula.changeDependents(newServerList,mustReplaceAll) ;
  setDirty(kTRUE) ;
  return ret ;
}

void RooFormulaValue::PrintToStream(ostream& os, Option_t* opt= 0)
{
  // Print current value and definition of formula
  os << "RooFormulaValue: " << GetName() << " = " << GetTitle() << " = " << GetVar();
  if(!_unit.IsNull()) os << ' ' << _unit;
  printAttribList(os) ;
  os << endl ;
} 



