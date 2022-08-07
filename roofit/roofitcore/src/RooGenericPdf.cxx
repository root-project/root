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

/**
\file RooGenericPdf.cxx
\class RooGenericPdf
\ingroup Roofitcore

RooGenericPdf is a concrete implementation of a probability density function,
which takes a RooArgList of servers and a C++ expression string defining how
its value should be calculated from the given list of servers.
A fully numerical integration is automatically performed to normalize the given
expression. RooGenericPdf uses a RooFormula object to perform the expression evaluation.

The string expression can be any valid TFormula expression referring to the
listed servers either by name or by their ordinal list position. These three are
equivalent:
```
  RooFormulaVar("gen", "x*y", RooArgList(x,y))       // reference by name
  RooFormulaVar("gen", "@0*@1", RooArgList(x,y))     // reference by ordinal with @
  RooFormulaVar("gen", "x[0]*x[1]", RooArgList(x,y)) // TFormula-builtin reference by ordinal
```
Note that `x[i]` is an expression reserved for TFormula. All variable references
are automatically converted to the TFormula-native format. If a variable with
the name `x` is given, the RooFormula interprets `x[i]` as a list position,
but `x` without brackets as the name of a RooFit object.

The last two versions, while slightly less readable, are more versatile because
the names of the arguments are not hard coded.
**/

#include "RooGenericPdf.h"
#include "Riostream.h"
#include "RooStreamParser.h"
#include "RooMsgService.h"
#include "RooArgList.h"
#include "RunContext.h"

using namespace std;

ClassImp(RooGenericPdf);



////////////////////////////////////////////////////////////////////////////////
/// Constructor with formula expression and list of input variables

RooGenericPdf::RooGenericPdf(const char *name, const char *title, const RooArgList& dependents) :
  RooAbsPdf(name,title),
  _actualVars("actualVars","Variables used by PDF expression",this),
  _formExpr(title)
{
  _actualVars.add(dependents) ;
  formula();

  if (_actualVars.empty()) _value = traceEval(0) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Constructor with a name, title, formula expression and a list of variables

RooGenericPdf::RooGenericPdf(const char *name, const char *title,
              const char* inFormula, const RooArgList& dependents) :
  RooAbsPdf(name,title),
  _actualVars("actualVars","Variables used by PDF expression",this),
  _formExpr(inFormula)
{
  _actualVars.add(dependents) ;
  formula();

  if (_actualVars.empty()) _value = traceEval(0) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooGenericPdf::RooGenericPdf(const RooGenericPdf& other, const char* name) :
  RooAbsPdf(other, name),
  _actualVars("actualVars",this,other._actualVars),
  _formExpr(other._formExpr)
{
  formula();
}


////////////////////////////////////////////////////////////////////////////////

RooFormula& RooGenericPdf::formula() const
{
  if (!_formula) {
    const_cast<std::unique_ptr<RooFormula>&>(_formula).reset(
        new RooFormula(GetName(),_formExpr.Data(),_actualVars));
    const_cast<TString&>(_formExpr) = _formula->formulaString().c_str();
  }
  return *_formula ;
}



////////////////////////////////////////////////////////////////////////////////
/// Calculate current value of this object

double RooGenericPdf::evaluate() const
{
  return formula().eval(_actualVars.nset()) ;
}


////////////////////////////////////////////////////////////////////////////////
/// Evaluate this formula for values found in inputData.
RooSpan<double> RooGenericPdf::evaluateSpan(RooBatchCompute::RunContext& inputData, const RooArgSet* normSet) const {
  if (normSet != nullptr && normSet != _normSet)
    throw std::logic_error("Got conflicting normSets");

  auto results = formula().evaluateSpan(this, inputData, _normSet);
  inputData.spans[this] = results;

  return results;
}

////////////////////////////////////////////////////////////////////////////////
void RooGenericPdf::computeBatch(cudaStream_t* stream, double* output, size_t nEvents, RooFit::Detail::DataMap const& dataMap) const
{
  formula().computeBatch(stream, output, nEvents, dataMap);
}


////////////////////////////////////////////////////////////////////////////////
/// Change formula expression to given expression

bool RooGenericPdf::setFormula(const char* inFormula)
{
  if (formula().reCompile(inFormula)) return true ;

  _formExpr = inFormula ;
  setValueDirty() ;
  return false ;
}



////////////////////////////////////////////////////////////////////////////////
/// Check if given value is valid

bool RooGenericPdf::isValidReal(double /*value*/, bool /*printError*/) const
{
  return true ;
}



////////////////////////////////////////////////////////////////////////////////
/// Propagate server changes to embedded formula object

bool RooGenericPdf::redirectServersHook(const RooAbsCollection& newServerList, bool mustReplaceAll, bool nameChange, bool isRecursive)
{
  bool error = _formula ? _formula->changeDependents(newServerList,mustReplaceAll,nameChange) : true;
  return error || RooAbsPdf::redirectServersHook(newServerList, mustReplaceAll, nameChange, isRecursive);
}



////////////////////////////////////////////////////////////////////////////////
/// Print info about this object to the specified stream.

void RooGenericPdf::printMultiline(ostream& os, Int_t content, bool verbose, TString indent) const
{
  RooAbsPdf::printMultiline(os,content,verbose,indent);
  if (verbose) {
    os << " --- RooGenericPdf --- " << endl ;
    indent.Append("  ");
    os << indent ;
    formula().printMultiline(os,content,verbose,indent);
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Add formula expression as meta argument in printing interface

void RooGenericPdf::printMetaArgs(ostream& os) const
{
  os << "formula=\"" << _formExpr << "\" " ;
}



////////////////////////////////////////////////////////////////////////////////
/// Read object contents from given stream

bool RooGenericPdf::readFromStream(istream& is, bool compact, bool /*verbose*/)
{
  if (compact) {
    coutE(InputArguments) << "RooGenericPdf::readFromStream(" << GetName() << "): can't read in compact mode" << endl ;
    return true ;
  } else {
    RooStreamParser parser(is) ;
    return setFormula(parser.readLine()) ;
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Write object contents to given stream

void RooGenericPdf::writeToStream(ostream& os, bool compact) const
{
  if (compact) {
    os << getVal() << endl ;
  } else {
    os << GetTitle() ;
  }
}



