/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooGenericPdf.cc,v 1.12 2001/10/22 07:12:13 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [PDF] --
// RooGenericPdf is a concrete implementation of a probability density function,
// which takes a RooArgList of servers and a C++ expression string defining how
// its value should be calculated from the given list of servers.
// A fully numerical integration is automatically performed to normalize the given
// expression. RooGenericPdf uses a RooFormula object to perform the expression evaluation
//
// The string expression can be any valid TFormula expression referring to the
// listed servers either by name or by their ordinal list position:
//
//   RooGenericPdf("gen","x*y",RooArgList(x,y))  or
//   RooGenericPdf("gen","@0*@1",RooArgList(x,y)) 
//
// The latter form, while slightly less readable, is more versatile because it
// doesn't hardcode any of the variable names it expects

#include "RooFitCore/RooGenericPdf.hh"
#include "RooFitCore/RooStreamParser.hh"
#include "RooFitCore/RooArgList.hh"

ClassImp(RooGenericPdf)


RooGenericPdf::RooGenericPdf(const char *name, const char *title, const RooArgList& dependents) : 
  RooAbsPdf(name,title), _formula(name,title,dependents)
{  
  // Constructor with title used as formula expression
  TIterator* depIter = _formula.actualDependents().createIterator() ;
  RooAbsArg* server(0) ;
  while (server=(RooAbsArg*)depIter->Next()) {
    addServer(*server,kTRUE,kFALSE) ;
  }
}


RooGenericPdf::RooGenericPdf(const char *name, const char *title, 
			     const char* formula, const RooArgList& dependents) : 
  RooAbsPdf(name,title), _formula(name,formula,dependents)
{  
  // Constructor with separate title and formula expression
  TIterator* depIter = _formula.actualDependents().createIterator() ;
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
  return _formula.eval(_lastNormSet) ;
}


Bool_t RooGenericPdf::setFormula(const char* formula) 
{
  // Change formula expression to given expression
  if (_formula.reCompile(formula)) return kTRUE ;
  
  SetTitle(formula) ;
  setValueDirty() ;
  return kFALSE ;
}


Bool_t RooGenericPdf::isValidReal(Double_t value, Bool_t printError) const {
  // Check if given value is valid
  return kTRUE ;
}



Bool_t RooGenericPdf::redirectServersHook(const RooAbsCollection& newServerList, Bool_t mustReplaceAll, Bool_t nameChange)
{
  // Propagate server changes to embedded formula objecy
  return _formula.changeDependents(newServerList,mustReplaceAll,nameChange) ;
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



