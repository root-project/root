/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooFormulaVar.cc,v 1.12 2001/09/04 01:37:42 david Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION --
// RooFormulaVar is a concrete implementation of a derived real-valued object,
// which takes a RooArgSet of servers and a string expression defining how
// its value should be calculated from the given list of servers.
// 
// RooFormulaVar uses a RooFormula object to perform the expression evaluation


#include "RooFitCore/RooFormulaVar.hh"
#include "RooFitCore/RooStreamParser.hh"

ClassImp(RooFormulaVar)


RooFormulaVar::RooFormulaVar(const char *name, const char *title, const char* formula, const RooArgSet& dependents) : 
  RooAbsReal(name,title), _formula(name,formula,dependents)
{  
  Bool_t anyServers(kFALSE) ;
  // Constructor with formula expression and list of input variables
  TIterator* depIter = _formula.actualDependents().createIterator() ;
  RooAbsArg* server(0) ;
  while (server=(RooAbsArg*)depIter->Next()) {
    addServer(*server,kTRUE,kFALSE) ;
    anyServers=kTRUE ;
  }
  delete depIter ;

  if (!anyServers) _value = traceEval(0) ;
}


RooFormulaVar::RooFormulaVar(const char *name, const char *title, const RooArgSet& dependents) : 
  RooAbsReal(name,title), _formula(name,title,dependents)
{  
  Bool_t anyServers(kFALSE) ;
  // Constructor with formula expression and list of input variables
  TIterator* depIter = _formula.actualDependents().createIterator() ;
  RooAbsArg* server(0) ;
  while (server=(RooAbsArg*)depIter->Next()) {
    addServer(*server,kTRUE,kFALSE) ;
    anyServers=kTRUE ;
  }
  delete depIter ;

  if (!anyServers) _value = traceEval(0) ;
}


RooFormulaVar::RooFormulaVar(const RooFormulaVar& other, const char* name) : 
  RooAbsReal(other, name), _formula(other._formula)
{
  // Copy constructor
}


RooFormulaVar::~RooFormulaVar() 
{
  // Destructor
}


Double_t RooFormulaVar::evaluate() const
{
  // Calculate current value of object
  return _formula.eval(0) ;
}



Bool_t RooFormulaVar::setFormula(const char* formula) 
{
  // Change formula expression
  if (_formula.reCompile(formula)) return kTRUE ;
  
  SetTitle(formula) ;
  setValueDirty() ;
  return kFALSE ;
}


Bool_t RooFormulaVar::isValidReal(Double_t value, Bool_t printError) const {
  // Check if given value is valid
  return kTRUE ;
}



Bool_t RooFormulaVar::redirectServersHook(const RooAbsCollection& newServerList, Bool_t mustReplaceAll)
{
  // Propagate server change information to embedded RooFormula object

  return _formula.changeDependents(newServerList,mustReplaceAll) ;
}



Bool_t RooFormulaVar::checkDependents(const RooArgSet* nset) const 
{
  // Check if dependent configuration of given data set is OK

  // We can handle any dependent configuration since RooFormulaVar 
  // does an explicit normalization of the top-level PDF over the leafNode servers
  return kFALSE ;
}


void RooFormulaVar::printToStream(ostream& os, PrintOption opt, TString indent) const
{
  // Print info about this object to the specified stream.   
  RooAbsReal::printToStream(os,opt,indent);
  if(opt >= Verbose) {
    indent.Append("  ");
    os << indent;
    _formula.printToStream(os,opt,indent);
  }
}



Bool_t RooFormulaVar::readFromStream(istream& is, Bool_t compact, Bool_t verbose)
{
  // Read object contents from given stream
  if (compact) {
    cout << "RooFormulaVar::readFromStream(" << GetName() << "): can't read in compact mode" << endl ;
    return kTRUE ;
  } else {
    RooStreamParser parser(is) ;
    return setFormula(parser.readLine()) ;
  }
}

void RooFormulaVar::writeToStream(ostream& os, Bool_t compact) const
{
  // Write object contents to given stream
  if (compact) {
    cout << getVal() << endl ;
  } else {
    os << GetTitle() ;
  }
}

Int_t RooFormulaVar::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars) const 
{
  // Determine which part (if any) of given integral can be performed analytically.
  // If any analytical integration is possible, return integration scenario code

  analVars.add(allVars);
  return 1 ;
}


Double_t RooFormulaVar::analyticalIntegral(Int_t code) const 
{
  // Return analytical integral defined by given scenario code

  return 1 ;
}
