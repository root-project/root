/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooGenericPdf.cc,v 1.1 2001/05/11 23:37:41 verkerke Exp $
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
  // Constructor with formula expression and list of variables
  TIterator* depIter = _formula.actualDependents().MakeIterator() ;
  RooAbsArg* server(0) ;
  while (server=(RooAbsArg*)depIter->Next()) {
    addServer(*server,kTRUE,kFALSE) ;
  }
}


RooGenericPdf::RooGenericPdf(const RooGenericPdf& other, const char* name) : 
  RooAbsPdf(other, name), _formula(other._formula)
{
  // Copy constructor
}


RooGenericPdf::~RooGenericPdf() 
{
  // Destructor
}


Double_t RooGenericPdf::evaluate() const
{
  // Calculate current value of this object
  return _formula.eval() ;
}


Bool_t RooGenericPdf::isValid() const
{
  // Check if current value is valid
  return isValid(getVal()) ;
}


Bool_t RooGenericPdf::setFormula(const char* formula) 
{
  // Change formula expression to given expression
  if (_formula.reCompile(formula)) return kTRUE ;
  
  SetTitle(formula) ;
  setValueDirty(kTRUE) ;
  return kFALSE ;
}


Bool_t RooGenericPdf::isValid(Double_t value) const {
  // Check if given value is valid
  return kTRUE ;
}



Bool_t RooGenericPdf::redirectServersHook(const RooArgSet& newServerList, Bool_t mustReplaceAll)
{
  // Propagate server change to embedded formula objecy
  return _formula.changeDependents(newServerList,mustReplaceAll) ;
}



Bool_t RooGenericPdf::checkDependents(const RooDataSet* set) const 
{
  // Check if this PDF is valid for dependent configuration given by specified data set

  // We can handle any dependent configuration since RooGenericPdf 
  // does an explicit normalization of the top-level PDF over the leafNode servers
  return kFALSE ;
}


void RooGenericPdf::printToStream(ostream& os, PrintOption opt, TString indent) const
{
  // Print info about this object to the specified stream. 
  RooAbsPdf::printToStream(os,opt,indent);
  if(opt >= Verbose) {
    indent.Append("  ");
    os << indent;
    _formula.printToStream(os,opt,indent);
  }
}



Bool_t RooGenericPdf::readFromStream(istream& is, Bool_t compact, Bool_t verbose)
{
  // Read object contents from given stream
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
  // Write object contents to given stream
  if (compact) {
    cout << getVal() << endl ;
  } else {
    os << GetTitle() ;
  }
}



