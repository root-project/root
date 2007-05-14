/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Name:  $:$Id: RooGenericPdf.cxx,v 1.28 2007/05/11 09:11:58 verkerke Exp $
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

#include "RooFit.h"

#include "RooGenericPdf.h"
#include "RooGenericPdf.h"
#include "RooStreamParser.h"
#include "RooArgList.h"

ClassImp(RooGenericPdf)


RooGenericPdf::RooGenericPdf(const char *name, const char *title, const RooArgList& dependents) : 
  RooAbsPdf(name,title), 
  _actualVars("actualVars","Variables used by PDF expression",this),
  _formula(name,title,dependents)
{  
  // Constructor with formula expression and list of input variables
  _actualVars.add(dependents) ; 

  if (_actualVars.getSize()==0) _value = traceEval(0) ;
}


RooGenericPdf::RooGenericPdf(const char *name, const char *title, 
			     const char* formula, const RooArgList& dependents) : 
  RooAbsPdf(name,title), 
  _actualVars("actualVars","Variables used by PDF expression",this),
  _formula(name,formula,dependents)
{  
  _actualVars.add(dependents) ; 

  if (_actualVars.getSize()==0) _value = traceEval(0) ;
}


RooGenericPdf::RooGenericPdf(const RooGenericPdf& other, const char* name) : 
  RooAbsPdf(other, name), 
  _actualVars("actualVars",this,other._actualVars),
  _formula(other._formula)
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
  return _formula.eval(_normMgr.lastNormSet()) ;
}


Bool_t RooGenericPdf::setFormula(const char* formula) 
{
  // Change formula expression to given expression
  if (_formula.reCompile(formula)) return kTRUE ;
  
  SetTitle(formula) ;
  setValueDirty() ;
  return kFALSE ;
}


Bool_t RooGenericPdf::isValidReal(Double_t /*value*/, Bool_t /*printError*/) const {
  // Check if given value is valid
  return kTRUE ;
}



Bool_t RooGenericPdf::redirectServersHook(const RooAbsCollection& newServerList, Bool_t mustReplaceAll, Bool_t nameChange, Bool_t /*isRecursive*/)
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



Bool_t RooGenericPdf::readFromStream(istream& is, Bool_t compact, Bool_t /*verbose*/)
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



