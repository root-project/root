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
#include "RooAbsRealLValue.h"
#include "RooAbsBinning.h"
#include "RooCurve.h"
#include "RooHelpers.h"

using std::istream, std::ostream, std::endl;


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
   for (auto const &item : other._binnings) {
      _binnings[item.first] = std::unique_ptr<RooAbsBinning>{item.second->clone()};
   }
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
/// Declare that this pdf is piecewise constant (flat) within the bins of the
/// given `binning` of the observable `obs`, which must be one of the formula
/// variables. The method can be called several times to set a binning for more
/// than one observable. Use a RooUniformBinning to describe many uniform bins
/// compactly.
///
/// Once set, integrals over `obs` use the fast bin integrator (which sums the
/// central value of each bin times the bin width) instead of the generic
/// numeric integrator, and plotting samples the step shape exactly.
///
/// If `checkFlatness` is true (the default), the function is sampled at several
/// points inside each bin to verify that it is indeed flat; if it is not, an
/// error is issued and the binning is not stored.

void RooGenericPdf::setBinBoundaries(RooAbsRealLValue &obs, const RooAbsBinning &binning, bool checkFlatness)
{
   // Match the observable to a formula variable by name, so that a same-named
   // stand-in for the actual server is accepted too.
   const int idx = _actualVars.index(obs.GetName());
   if (idx < 0) {
      coutE(InputArguments) << "RooGenericPdf::setBinBoundaries(" << GetName() << ") the observable " << obs.GetName()
                            << " is not one of the formula variables of this pdf, nothing done." << std::endl;
      return;
   }

   if (checkFlatness) {
      // Vary the actual formula variable (the server), which may be a different
      // object than `obs` if `obs` is just a same-named stand-in.
      auto *serverObs = dynamic_cast<RooAbsRealLValue *>(_actualVars.at(idx));
      RooAbsRealLValue &flatObs = serverObs ? *serverObs : obs;
      std::span<const double> boundaries{binning.array(), static_cast<std::size_t>(binning.numBoundaries())};
      if (!RooHelpers::isFunctionFlatInBins(*this, flatObs, boundaries)) {
         coutE(InputArguments) << "RooGenericPdf::setBinBoundaries(" << GetName() << ") the expression \"" << _formExpr
                               << "\" is not flat within the given bins of " << obs.GetName()
                               << ". The binning is not set. Pass checkFlatness=false to override this check."
                               << std::endl;
         return;
      }
   }

   // Key the binning by the observable's index in _actualVars (not its name), so
   // that it survives a renaming of the variable or a server redirection.
   _binnings[idx] = std::unique_ptr<RooAbsBinning>{binning.clone()};
}

////////////////////////////////////////////////////////////////////////////////
/// Return true if a binning was set with setBinBoundaries() for every
/// observable in the integration set `obs`.

bool RooGenericPdf::isBinnedDistribution(const RooArgSet &obs) const
{
   if (obs.empty() || _binnings.empty()) {
      return false;
   }
   for (RooAbsArg *o : obs) {
      if (_binnings.find(_actualVars.index(o->GetName())) == _binnings.end()) {
         return false;
      }
   }
   return true;
}

////////////////////////////////////////////////////////////////////////////////
/// Return the boundaries of the binning set with setBinBoundaries() that fall
/// within [xlo, xhi], or a null pointer if no binning was set for this observable.

std::list<double> *RooGenericPdf::binBoundaries(RooAbsRealLValue &obs, double xlo, double xhi) const
{
   auto found = _binnings.find(_actualVars.index(obs.GetName()));
   if (found == _binnings.end()) {
      return nullptr;
   }
   const RooAbsBinning &binning = *found->second;
   return RooHelpers::binBoundariesInRange({binning.array(), static_cast<std::size_t>(binning.numBoundaries())}, xlo,
                                           xhi);
}

////////////////////////////////////////////////////////////////////////////////
/// Return sampling hints that draw the piecewise-flat shape exactly (a pair of
/// points just left and right of every bin boundary), or a null pointer if no
/// binning was set for this observable.

std::list<double> *RooGenericPdf::plotSamplingHint(RooAbsRealLValue &obs, double xlo, double xhi) const
{
   auto found = _binnings.find(_actualVars.index(obs.GetName()));
   if (found == _binnings.end()) {
      return nullptr;
   }
   const RooAbsBinning &binning = *found->second;
   return RooCurve::plotSamplingHintForBinBoundaries(
      {binning.array(), static_cast<std::size_t>(binning.numBoundaries())}, xlo, xhi);
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
  formula().doEval(_actualVars, ctx);
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
    os << " --- RooGenericPdf --- " << std::endl ;
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


void RooGenericPdf::dumpFormula() { formula().printMultiline(std::cout, 0) ; }


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
    os << getVal() << std::endl ;
  } else {
    os << GetTitle() ;
  }
}

std::string RooGenericPdf::getUniqueFuncName() const
{
   return formula().getTFormula()->GetUniqueFuncName().Data();
}
