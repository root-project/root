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

#include "RooFitCore/RooGenericPdf.hh"
#include "RooFitCore/RooStreamParser.hh"

ClassImp(RooGenericPdf)


RooGenericPdf::RooGenericPdf(const char *name, const char *title, const RooArgSet& dependents) : 
  RooAbsPdf(name,title), _formula(name,title,dependents)
{  
  TIterator* depIter = _formula.actualDependents().MakeIterator() ;
  RooAbsArg* server(0) ;
  while (server=(RooAbsArg*)depIter->Next()) {
    addServer(*server,kTRUE,kFALSE) ;
  }
}


RooGenericPdf::RooGenericPdf(const RooGenericPdf& other, const char* name) : 
  RooAbsPdf(other, name), _formula(other._formula)
{
}


RooGenericPdf::~RooGenericPdf() 
{
}


Double_t RooGenericPdf::evaluate() const
{
  // Evaluate embedded formula
  return _formula.eval() ;
}


Bool_t RooGenericPdf::isValid() const
{
  return isValid(getVal()) ;
}


Bool_t RooGenericPdf::setFormula(const char* formula) 
{
  if (_formula.reCompile(formula)) return kTRUE ;
  
  SetTitle(formula) ;
  setValueDirty(kTRUE) ;
  return kFALSE ;
}


Bool_t RooGenericPdf::isValid(Double_t value) const {
  return kTRUE ;
}



Bool_t RooGenericPdf::redirectServersHook(const RooArgSet& newServerList, Bool_t mustReplaceAll)
{
  // Propagate server change to formula engine
  return _formula.changeDependents(newServerList,mustReplaceAll) ;
}



Bool_t RooGenericPdf::checkDependents(const RooDataSet* set) const 
{
  // We can handle any dependent configuration since RooGenericPdf 
  // does an explicit normalization of the top-level PDF over the leafNode servers
  return kFALSE ;
}


void RooGenericPdf::printToStream(ostream& os, PrintOption opt, TString indent) const
{
  RooAbsPdf::printToStream(os,opt,indent);
  if(opt >= Verbose) {
    indent.Append("  ");
    os << indent;
    _formula.printToStream(os,opt,indent);
  }
}



Bool_t RooGenericPdf::readFromStream(istream& is, Bool_t compact, Bool_t verbose)
{
  if (compact) {
    cout << "RooGenericPdf::readFromStream(" << GetName() << "): can't read in compact mode" << endl ;
    return kTRUE ;
  } else {
    RooStreamParser parser(is) ;
    return setFormula(parser.readLine()) ;
  }
}

void RooGenericPdf::writeToStream(ostream& os, Bool_t compact) const
{
  if (compact) {
    cout << getVal() << endl ;
  } else {
    os << GetTitle() ;
  }
}



