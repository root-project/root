/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooFormulaVar.cc,v 1.22 2002/04/03 23:37:25 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [REAL] --
// RooRealVar is a generic implementation of a real valued object
// which takes a RooArgList of servers and a C++ expression string defining how
// its value should be calculated from the given list of servers.
// RooRealVar uses a RooFormula object to perform the expression evaluation.
//
// If RooAbsPdf objects are supplied to RooRealVar as servers, their
// raw (unnormalized) values will be evaluated. Use RooGenericPdf, which
// constructs generic PDF functions, to access their properly normalized
// values.
//
// The string expression can be any valid TFormula expression referring to the
// listed servers either by name or by their ordinal list position:
//
//   RooRealVar("gen","x*y",RooArgList(x,y))  or
//   RooRealVar("gen","@0*@1",RooArgList(x,y)) 
//
// The latter form, while slightly less readable, is more versatile because it
// doesn't hardcode any of the variable names it expects


#include "RooFitCore/RooFormulaVar.hh"
#include "RooFitCore/RooStreamParser.hh"

ClassImp(RooFormulaVar)


RooFormulaVar::RooFormulaVar(const char *name, const char *title, const char* formula, const RooArgList& dependents) : 
  RooAbsReal(name,title), _formExpr(formula), _formula(0),
  _actualVars("actualVars","Variables used by formula expression",this)
{  
  // Constructor with formula expression and list of input variables
  RooFormula tmpFormula(name,formula,dependents) ;
  _actualVars.add(tmpFormula.actualDependents()) ;

  if (_actualVars.getSize()==0) _value = traceEval(0) ;
}


RooFormulaVar::RooFormulaVar(const char *name, const char *title, const RooArgList& dependents) : 
  RooAbsReal(name,title), _formExpr(title), _formula(0),
  _actualVars("actualVars","Variables used by formula expression",this)
{  
  // Constructor with formula expression, title and list of input variables
  RooFormula tmpFormula(name,title,dependents) ;
  _actualVars.add(tmpFormula.actualDependents()) ;

  if (_actualVars.getSize()==0) _value = traceEval(0) ;
}


RooFormulaVar::RooFormulaVar(const RooFormulaVar& other, const char* name) : 
  RooAbsReal(other, name), _formExpr(other._formExpr), _formula(0),
  _actualVars("actualVars",this,other._actualVars) 
{
  // Copy constructor
}


RooFormulaVar::~RooFormulaVar() 
{
  // Destructor
  if (_formula) delete _formula ;
}


RooFormula& RooFormulaVar::formula() const
{
  if (!_formula) {
    _formula = new RooFormula(GetName(),_formExpr,_actualVars) ;    
  }
  return *_formula ;
}


Double_t RooFormulaVar::evaluate() const
{
  // Calculate current value of object
  return formula().eval(0) ;
}



Bool_t RooFormulaVar::isValidReal(Double_t value, Bool_t printError) const {
  // Check if given value is valid
  return kTRUE ;
}



Bool_t RooFormulaVar::redirectServersHook(const RooAbsCollection& newServerList, Bool_t mustReplaceAll, Bool_t nameChange)
{
  // Propagate server change information to embedded RooFormula object
  return _formula ? _formula->changeDependents(newServerList,mustReplaceAll,nameChange) : kFALSE ;
}



void RooFormulaVar::printToStream(ostream& os, PrintOption opt, TString indent) const
{
  // Print info about this object to the specified stream.   
  RooAbsReal::printToStream(os,opt,indent);
  if(opt >= Verbose) {
    indent.Append("  ");
    os << indent;
    formula().printToStream(os,opt,indent);
  }
}


Bool_t RooFormulaVar::readFromStream(istream& is, Bool_t compact, Bool_t verbose)
{
  // Read object contents from given stream
  cout << "RooFormulaVar::readFromStream(" << GetName() << "): can't read" << endl ;
  return kTRUE ;
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

