/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooRealFormula.cc,v 1.10 2001/05/10 18:58:48 verkerke Exp $
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


RooRealFormula::RooRealFormula(const char *name, const char *title, const RooArgSet& dependents) : 
  RooAbsReal(name,title), _formula(name,title,dependents)
{  
  TIterator* depIter = _formula.actualDependents().MakeIterator() ;
  RooAbsArg* server(0) ;
  while (server=(RooAbsArg*)depIter->Next()) {
    addServer(*server,kTRUE,kFALSE) ;
  }
}


RooRealFormula::RooRealFormula(const RooRealFormula& other, const char* name) : 
  RooAbsReal(other, name), _formula(other._formula)
{
}


RooRealFormula::~RooRealFormula() 
{
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



Bool_t RooRealFormula::redirectServersHook(const RooArgSet& newServerList, Bool_t mustReplaceAll)
{
  // Propagate server change to formula engine
  return _formula.changeDependents(newServerList,mustReplaceAll) ;
}



Bool_t RooRealFormula::checkDependents(const RooDataSet* set) const 
{
  // We can handle any dependent configuration since RooRealFormula 
  // does an explicit normalization of the top-level PDF over the leafNode servers
  return kFALSE ;
}


void RooRealFormula::printToStream(ostream& os, PrintOption opt, TString indent) const
{
  RooAbsReal::printToStream(os,opt,indent);
  if(opt >= Verbose) {
    indent.Append("  ");
    os << indent;
    _formula.printToStream(os,opt,indent);
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
    cout << getVal() << endl ;
  } else {
    os << GetTitle() ;
  }
}



