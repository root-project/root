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

Implementation of a probability density function
that takes a RooArgList of servers and a C++ expression string defining how
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
#include "RooFormula.h"

using std::istream, std::ostream, std::endl;

ClassImp(RooGenericPdf);

RooGenericPdf::RooGenericPdf() {}

RooGenericPdf::~RooGenericPdf()
{
   if(_formula) delete _formula;
}


////////////////////////////////////////////////////////////////////////////////
/// Constructor with formula expression and list of input variables

RooGenericPdf::RooGenericPdf(const char *name, const char *title, const RooArgList& dependents) :
  RooAbsPdf(name,title),
  _actualVars("actualVars","Variables used by PDF expression",this),
  _formExpr(title)
{
  if (dependents.empty()) {
    _value = traceEval(nullptr);
  } else {
    _formula = new RooFormula(GetName(), _formExpr, dependents);
    _formExpr = _formula->formulaString().c_str();
    _actualVars.add(_formula->actualDependents());
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Constructor with a name, title, formula expression and a list of variables

RooGenericPdf::RooGenericPdf(const char *name, const char *title,
              const char* inFormula, const RooArgList& dependents) :
  RooAbsPdf(name,title),
  _actualVars("actualVars","Variables used by PDF expression",this),
  _formExpr(inFormula)
{
  if (dependents.empty()) {
    _value = traceEval(nullptr);
  } else {
    _formula = new RooFormula(GetName(), _formExpr, dependents);
    _formExpr = _formula->formulaString().c_str();
    _actualVars.add(_formula->actualDependents());
  }
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
    _formula = new RooFormula(GetName(),_formExpr.Data(),_actualVars);
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
void RooGenericPdf::doEval(RooFit::EvalContext & ctx) const
{
  formula().doEval(ctx);
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


void RooGenericPdf::dumpFormula() { formula().dump() ; }


////////////////////////////////////////////////////////////////////////////////
/// Read object contents from given stream

bool RooGenericPdf::readFromStream(istream& /*is*/, bool /*compact*/, bool /*verbose*/)
{
  coutE(InputArguments) << "RooGenericPdf::readFromStream(" << GetName() << "): can't read" << std::endl;
  return true;
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

void RooGenericPdf::translate(RooFit::Detail::CodeSquashContext &ctx) const
{
   // If the number of elements to sum is less than 3, just build a sum expression.
   // Otherwise build a loop to sum over the values.
   unsigned int eleSize = _actualVars.size();
   std::string className = GetName();
   std::string varName = "elements" + className;
   std::string sumName = "sum" + className;
   std::string code;
   std::string decl = "double " + varName + "[" + std::to_string(eleSize) + "]{";
   int idx = 0;
   for (RooAbsArg *it : _actualVars) {
      decl += ctx.getResult(*it) + ",";
      ctx.addResult(it, varName + "[" + std::to_string(idx) + "]");
      idx++;
   }
   decl.back() = '}';
   code += decl + ";\n";

   ctx.addResult(this, (_formula->getTFormula()->GetUniqueFuncName() + "(" + varName + ")").Data());
   ctx.addToCodeBody(this, code);
}

