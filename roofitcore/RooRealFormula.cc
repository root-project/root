/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooRealFormula.cc,v 1.5 2001/04/05 01:49:10 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

#include "RooFitCore/RooRealFormula.hh"
#include "RooFitCore/RooStreamParser.hh"

ClassImp(RooRealFormula)


RooRealFormula::RooRealFormula(const char *name, const char *title, RooArgSet& dependents) : 
  RooDerivedReal(name,title), _formula(name,title,dependents)
{  
  TIterator* depIter = _formula.actualDependents().MakeIterator() ;
  RooAbsArg* server(0) ;
  while (server=(RooAbsArg*)depIter->Next()) {
    addServer(*server,kTRUE,kFALSE) ;
  }
}


RooRealFormula::RooRealFormula(const char* name, const RooRealFormula& other) : 
  RooDerivedReal(name, other), _formula(other._formula)
{
}


RooRealFormula::RooRealFormula(const RooRealFormula& other) : 
  RooDerivedReal(other), _formula(other._formula)
{
}


RooRealFormula::~RooRealFormula() 
{
}


RooRealFormula& RooRealFormula::operator=(const RooRealFormula& other)
{
  RooAbsReal::operator=(other) ;
  _formula = other._formula ;
}


RooAbsArg& RooRealFormula::operator=(const RooAbsArg& aother) 
{
  return operator=((const RooRealFormula&)aother) ;  
}


Double_t RooRealFormula::evaluate() const
{
  // Evaluate embedded formula
  return _formula.eval() ;
}


Bool_t RooRealFormula::isValid() const
{
  return isValid(getVal()) ;
}


Bool_t RooRealFormula::setFormula(const char* formula) 
{
  if (_formula.reCompile(formula)) return kTRUE ;
  
  SetTitle(formula) ;
  setValueDirty(kTRUE) ;
  return kFALSE ;
}



Bool_t RooRealFormula::isValid(Double_t value) const {
  return kTRUE ;
}


Bool_t RooRealFormula::redirectServersHook(RooArgSet& newServerList, Bool_t mustReplaceAll)
{
  // Propagate server change to formula engine
  return _formula.changeDependents(newServerList,mustReplaceAll) ;
}


void RooRealFormula::printToStream(ostream& os, PrintOption opt) const
{
  switch(opt) {
  case OneLine:
  case Standard:
    // Print current value and definition of formula
    os << "RooRealFormula: " << GetName() << " = " << GetTitle() << " = " << getVal();
    if(!_unit.IsNull()) os << ' ' << _unit;
    printAttribList(os) ;
    os << endl ;
    break ;

  case Verbose:
    RooAbsArg::printToStream(os,opt) ;
    break ;

  case Shape:
    cout << "RooRealFormula: " << GetName() << " Shape printing not implemented yet" << endl ;
    break ;
  }
} 


Bool_t RooRealFormula::readFromStream(istream& is, Bool_t compact, Bool_t verbose)
{
  if (compact) {
    cout << "RooRealFormula::readFromStream(" << GetName() << "): can't read in compact mode" << endl ;
    return kTRUE ;
  } else {
    RooStreamParser parser(is) ;
    return setFormula(parser.readLine()) ;
  }
}


void RooRealFormula::writeToStream(ostream& os, Bool_t compact) const
{
  if (compact) {
    cout << "RooRealFormula::writeToStream(" << GetName() << "): can't write in compact mode" << endl ;
  } else {
    os << GetTitle() ;
  }
}



