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
// BEGIN_HTML
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
// END_HTML
//

#include "RooFit.h"
#include "Riostream.h"

#include "RooGenericPdf.h"
#include "RooGenericPdf.h"
#include "RooStreamParser.h"
#include "RooMsgService.h"
#include "RooArgList.h"



ClassImp(RooGenericPdf)



//_____________________________________________________________________________
RooGenericPdf::RooGenericPdf(const char *name, const char *title, const RooArgList& dependents) : 
  RooAbsPdf(name,title), 
  _actualVars("actualVars","Variables used by PDF expression",this),
  _formula(0),
  _formExpr(title)
{  
  // Constructor with formula expression and list of input variables
  _actualVars.add(dependents) ; 

  if (_actualVars.getSize()==0) _value = traceEval(0) ;
}



//_____________________________________________________________________________
RooGenericPdf::RooGenericPdf(const char *name, const char *title, 
			     const char* inFormula, const RooArgList& dependents) : 
  RooAbsPdf(name,title), 
  _actualVars("actualVars","Variables used by PDF expression",this),
  _formula(0),
  _formExpr(inFormula)
{  
  // Constructor with a name, title, formula expression and a list of variables

  _actualVars.add(dependents) ; 

  if (_actualVars.getSize()==0) _value = traceEval(0) ;
}



//_____________________________________________________________________________
RooGenericPdf::RooGenericPdf(const RooGenericPdf& other, const char* name) : 
  RooAbsPdf(other, name), 
  _actualVars("actualVars",this,other._actualVars),
  _formula(0),
  _formExpr(other._formExpr)
{
  // Copy constructor
}



//_____________________________________________________________________________
RooGenericPdf::~RooGenericPdf() 
{
  // Destructor
  if (_formula) delete _formula ;
}



//_____________________________________________________________________________
RooFormula& RooGenericPdf::formula() const
{
  if (!_formula) {
    _formula = new RooFormula(GetName(),_formExpr.Data(),_actualVars) ;
  } 
  return *_formula ;
}



//_____________________________________________________________________________
Double_t RooGenericPdf::evaluate() const
{
  // Calculate current value of this object
  
  return formula().eval(_normSet) ;
}



//_____________________________________________________________________________
Bool_t RooGenericPdf::setFormula(const char* inFormula) 
{
  // Change formula expression to given expression

  if (formula().reCompile(inFormula)) return kTRUE ;

  _formExpr = inFormula ;
  setValueDirty() ;
  return kFALSE ;
}



//_____________________________________________________________________________
Bool_t RooGenericPdf::isValidReal(Double_t /*value*/, Bool_t /*printError*/) const 
{
  // Check if given value is valid
  return kTRUE ;
}



//_____________________________________________________________________________
Bool_t RooGenericPdf::redirectServersHook(const RooAbsCollection& newServerList, Bool_t mustReplaceAll, Bool_t nameChange, Bool_t /*isRecursive*/)
{
  // Propagate server changes to embedded formula object

  if (_formula) {
     return _formula->changeDependents(newServerList,mustReplaceAll,nameChange) ;
  } else {
    return kTRUE ;
  }
}



//_____________________________________________________________________________
void RooGenericPdf::printMultiline(ostream& os, Int_t content, Bool_t verbose, TString indent) const
{
  // Print info about this object to the specified stream. 

  RooAbsPdf::printMultiline(os,content,verbose,indent);
  if (verbose) {
    os << " --- RooGenericPdf --- " << endl ;
    indent.Append("  ");
    os << indent ;
    formula().printMultiline(os,content,verbose,indent);
  }
}



//_____________________________________________________________________________
void RooGenericPdf::printMetaArgs(ostream& os) const 
{
  // Add formula expression as meta argument in printing interface
  os << "formula=\"" << _formExpr << "\" " ;
}



//_____________________________________________________________________________
Bool_t RooGenericPdf::readFromStream(istream& is, Bool_t compact, Bool_t /*verbose*/)
{
  // Read object contents from given stream

  if (compact) {
    coutE(InputArguments) << "RooGenericPdf::readFromStream(" << GetName() << "): can't read in compact mode" << endl ;
    return kTRUE ;
  } else {
    RooStreamParser parser(is) ;
    return setFormula(parser.readLine()) ;
  }
}


//_____________________________________________________________________________
void RooGenericPdf::writeToStream(ostream& os, Bool_t compact) const
{
  // Write object contents to given stream

  if (compact) {
    os << getVal() << endl ;
  } else {
    os << GetTitle() ;
  }
}



