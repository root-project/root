/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
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

//////////////////////////////////////////////////////////////////////////////
//
// RooFormulaVar is a generic implementation of a real valued object
// which takes a RooArgList of servers and a C++ expression string defining how
// its value should be calculated from the given list of servers.
// RooFormulaVar uses a RooFormula object to perform the expression evaluation.
//
// If RooAbsPdf objects are supplied to RooFormulaVar as servers, their
// raw (unnormalized) values will be evaluated. Use RooGenericPdf, which
// constructs generic PDF functions, to access their properly normalized
// values.
//
// The string expression can be any valid TFormula expression referring to the
// listed servers either by name or by their ordinal list position:
//
//   RooFormulaVar("gen","x*y",RooArgList(x,y))  or
//   RooFormulaVar("gen","@0*@1",RooArgList(x,y)) 
//
// The latter form, while slightly less readable, is more versatile because it
// doesn't hardcode any of the variable names it expects
//


#include "RooFit.h"
#include "Riostream.h"

#include "RooFormulaVar.h"
#include "RooFormulaVar.h"
#include "RooStreamParser.h"
#include "RooNLLVar.h"
#include "RooChi2Var.h"
#include "RooMsgService.h"
#include "RooTrace.h"


using namespace std;

ClassImp(RooFormulaVar)



//_____________________________________________________________________________
RooFormulaVar::RooFormulaVar(const char *name, const char *title, const char* inFormula, const RooArgList& dependents) : 
  RooAbsReal(name,title), 
  _actualVars("actualVars","Variables used by formula expression",this),
  _formula(0), _formExpr(inFormula)
{  
  // Constructor with formula expression and list of input variables

  _actualVars.add(dependents) ; 

  if (_actualVars.getSize()==0) _value = traceEval(0) ;
}



//_____________________________________________________________________________
RooFormulaVar::RooFormulaVar(const char *name, const char *title, const RooArgList& dependents) : 
  RooAbsReal(name,title),
  _actualVars("actualVars","Variables used by formula expression",this),
  _formula(0), _formExpr(title)
{  
  // Constructor with formula expression, title and list of input variables

  _actualVars.add(dependents) ; 

  if (_actualVars.getSize()==0) _value = traceEval(0) ;
}



//_____________________________________________________________________________
RooFormulaVar::RooFormulaVar(const RooFormulaVar& other, const char* name) : 
  RooAbsReal(other, name), 
  _actualVars("actualVars",this,other._actualVars),
  _formula(0), _formExpr(other._formExpr)
{
  // Copy constructor
}



//_____________________________________________________________________________
RooFormulaVar::~RooFormulaVar() 
{
  // Destructor

  if (_formula) delete _formula ;
}



//_____________________________________________________________________________
RooFormula& RooFormulaVar::formula() const
{
  // Return reference to internal RooFormula object

  if (!_formula) {
    _formula = new RooFormula(GetName(),_formExpr,_actualVars) ;    
  }
  return *_formula ;
}



//_____________________________________________________________________________
Double_t RooFormulaVar::evaluate() const
{
  // Calculate current value of object from internal formula
  return formula().eval(_lastNSet) ;
}



//_____________________________________________________________________________
Bool_t RooFormulaVar::isValidReal(Double_t /*value*/, Bool_t /*printError*/) const 
{
  // Check if given value is valid
  return kTRUE ;
}



//_____________________________________________________________________________
Bool_t RooFormulaVar::redirectServersHook(const RooAbsCollection& newServerList, Bool_t mustReplaceAll, Bool_t nameChange, Bool_t /*isRecursive*/)
{
  // Propagate server change information to embedded RooFormula object
  return formula().changeDependents(newServerList,mustReplaceAll,nameChange) ;
}



//_____________________________________________________________________________
void RooFormulaVar::printMultiline(ostream& os, Int_t contents, Bool_t verbose, TString indent) const
{
  // Print info about this object to the specified stream.   

  RooAbsReal::printMultiline(os,contents,verbose,indent);
  if(verbose) {
    indent.Append("  ");
    os << indent;
    formula().printMultiline(os,contents,verbose,indent);
  }
}



//_____________________________________________________________________________
void RooFormulaVar::printMetaArgs(ostream& os) const 
{
  // Add formula expression as meta argument in printing interface
  os << "formula=\"" << _formExpr << "\" " ;
}




//_____________________________________________________________________________
Bool_t RooFormulaVar::readFromStream(istream& /*is*/, Bool_t /*compact*/, Bool_t /*verbose*/)
{
  // Read object contents from given stream

  coutE(InputArguments) << "RooFormulaVar::readFromStream(" << GetName() << "): can't read" << endl ;
  return kTRUE ;
}



//_____________________________________________________________________________
void RooFormulaVar::writeToStream(ostream& os, Bool_t compact) const
{
  // Write object contents to given stream

  if (compact) {
    cout << getVal() << endl ;
  } else {
    os << GetTitle() ;
  }
}



//_____________________________________________________________________________
std::list<Double_t>* RooFormulaVar::binBoundaries(RooAbsRealLValue& obs, Double_t xlo, Double_t xhi) const
{
  // Forward the plot sampling hint from the p.d.f. that defines the observable obs  
  RooFIter iter = _actualVars.fwdIterator() ;
  RooAbsReal* func ;
  while((func=(RooAbsReal*)iter.next())) {
    list<Double_t>* binb = func->binBoundaries(obs,xlo,xhi) ;      
    if (binb) {
      return binb ;
    }
  }
  
  return 0 ;  
}



//_____________________________________________________________________________
std::list<Double_t>* RooFormulaVar::plotSamplingHint(RooAbsRealLValue& obs, Double_t xlo, Double_t xhi) const
{
  // Forward the plot sampling hint from the p.d.f. that defines the observable obs  
  RooFIter iter = _actualVars.fwdIterator() ;
  RooAbsReal* func ;
  while((func=(RooAbsReal*)iter.next())) {
    list<Double_t>* hint = func->plotSamplingHint(obs,xlo,xhi) ;      
    if (hint) {
      return hint ;
    }
  }
  
  return 0 ;
}



//_____________________________________________________________________________
Double_t RooFormulaVar::defaultErrorLevel() const 
{
  // Return the default error level for MINUIT error analysis
  // If the formula contains one or more RooNLLVars and 
  // no RooChi2Vars, return the defaultErrorLevel() of
  // RooNLLVar. If the addition contains one ore more RooChi2Vars
  // and no RooNLLVars, return the defaultErrorLevel() of
  // RooChi2Var. If the addition contains neither or both
  // issue a warning message and return a value of 1

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
    coutI(Minimization) << "RooFormulaVar::defaultErrorLevel(" << GetName() 
			<< ") Formula contains a RooNLLVar, using its error level" << endl ;
    return nllArg->defaultErrorLevel() ;
  } else if (chi2Arg && !nllArg) {
    coutI(Minimization) << "RooFormulaVar::defaultErrorLevel(" << GetName() 
	 << ") Formula contains a RooChi2Var, using its error level" << endl ;
    return chi2Arg->defaultErrorLevel() ;
  } else if (!nllArg && !chi2Arg) {
    coutI(Minimization) << "RooFormulaVar::defaultErrorLevel(" << GetName() << ") WARNING: "
		      << "Formula contains neither RooNLLVar nor RooChi2Var server, using default level of 1.0" << endl ;
  } else {
    coutI(Minimization) << "RooFormulaVar::defaultErrorLevel(" << GetName() << ") WARNING: "
			<< "Formula contains BOTH RooNLLVar and RooChi2Var server, using default level of 1.0" << endl ;
  }

  return 1.0 ;
}




