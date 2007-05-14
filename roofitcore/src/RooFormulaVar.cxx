/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Name:  $:$Id: RooFormulaVar.cxx,v 1.37 2007/05/11 09:11:58 verkerke Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
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


#include "RooFit.h"

#include "RooFormulaVar.h"
#include "RooFormulaVar.h"
#include "RooStreamParser.h"
#include "RooNLLVar.h"
#include "RooChi2Var.h"

ClassImp(RooFormulaVar)


RooFormulaVar::RooFormulaVar(const char *name, const char *title, const char* formula, const RooArgList& dependents) : 
  RooAbsReal(name,title), 
  _actualVars("actualVars","Variables used by formula expression",this),
  _formula(0), _formExpr(formula)
{  
  // Constructor with formula expression and list of input variables
//   RooFormula tmpFormula(name,formula,dependents) ;
  _actualVars.add(dependents) ; //tmpFormula.actualDependents()) ;

  if (_actualVars.getSize()==0) _value = traceEval(0) ;
}


RooFormulaVar::RooFormulaVar(const char *name, const char *title, const RooArgList& dependents) : 
  RooAbsReal(name,title),
  _actualVars("actualVars","Variables used by formula expression",this),
  _formula(0), _formExpr(title)
{  
  // Constructor with formula expression, title and list of input variables
//   RooFormula tmpFormula(name,title,dependents) ;
  _actualVars.add(dependents) ; //tmpFormula.actualDependents()) ;

  if (_actualVars.getSize()==0) _value = traceEval(0) ;
}


RooFormulaVar::RooFormulaVar(const RooFormulaVar& other, const char* name) : 
  RooAbsReal(other, name), 
  _actualVars("actualVars",this,other._actualVars),
  _formula(0), _formExpr(other._formExpr)
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



Bool_t RooFormulaVar::isValidReal(Double_t /*value*/, Bool_t /*printError*/) const {
  // Check if given value is valid
  return kTRUE ;
}



Bool_t RooFormulaVar::redirectServersHook(const RooAbsCollection& newServerList, Bool_t mustReplaceAll, Bool_t nameChange, Bool_t /*isRecursive*/)
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


Bool_t RooFormulaVar::readFromStream(istream& /*is*/, Bool_t /*compact*/, Bool_t /*verbose*/)
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

Double_t RooFormulaVar::defaultErrorLevel() const 
{
  // See if we contain a RooNLLVar or RooChi2Var object

  RooAbsReal* nllArg(0) ;
  RooAbsReal* chi2Arg(0) ;

  TIterator* iter = _actualVars.createIterator() ;
  RooAbsArg* arg ;
  while((arg=(RooAbsArg*)iter->Next())) {
    if (dynamic_cast<RooNLLVar*>(arg)) {
      nllArg = (RooAbsReal*)arg ;
    }
    if (dynamic_cast<RooChi2Var*>(arg)) {
      chi2Arg = (RooAbsReal*)arg ;
    }
  }
  delete iter ;

  if (nllArg && !chi2Arg) {
    cout << "RooFormulaVar::defaultErrorLevel(" << GetName() 
	 << ") Formula contains a RooNLLVar, using its error level" << endl ;
    return nllArg->defaultErrorLevel() ;
  } else if (chi2Arg && !nllArg) {
    cout << "RooFormulaVar::defaultErrorLevel(" << GetName() 
	 << ") Formula contains a RooChi2Var, using its error level" << endl ;
    return chi2Arg->defaultErrorLevel() ;
  } else if (!nllArg && !chi2Arg) {
    cout << "RooFormulaVar::defaultErrorLevel(" << GetName() << ") WARNING: "
	 << "Formula contains neither RooNLLVar nor RooChi2Var server, using default level of 1.0" << endl ;
  } else {
    cout << "RooFormulaVar::defaultErrorLevel(" << GetName() << ") WARNING: "
	 << "Formula contains BOTH RooNLLVar and RooChi2Var server, using default level of 1.0" << endl ;
  }

  return 1.0 ;
}




