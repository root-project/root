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
expression. The expression syntax is the same as for RooFormulaVar; see its
class documentation.
**/

#include "RooGenericPdf.h"
#include "Riostream.h"
#include "RooStreamParser.h"
#include "RooMsgService.h"
#include "RooArgList.h"
#include "RooFormulaUtils.h"
#include "RooAbsRealLValue.h"
#include "RooAbsBinning.h"
#include "RooCurve.h"
#include "RooFitImplHelpers.h"

#include "TFormula.h"

using std::istream, std::ostream, std::endl;


RooGenericPdf::RooGenericPdf() {}

RooGenericPdf::~RooGenericPdf() = default;

////////////////////////////////////////////////////////////////////////////////
/// Constructor with formula expression and list of input variables

RooGenericPdf::RooGenericPdf(const char *name, const char *title, const RooArgList &dependents)
   : RooGenericPdf(name, title, title, dependents)
{
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
     auto compiled = RooFormulaUtils::compileFormula(GetName(), _formExpr.Data(), dependents);
     _formExpr = compiled.formula.c_str();
     _actualVars.add(compiled.actualVars);
     _evaluator = std::move(compiled.evaluator);
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
   if (other._evaluator) {
      _evaluator = other._evaluator->clone();
      // Like when the TFormula was still copied directly, the copied TFormula is
      // renamed after the possibly-different name of this object.
      if (TFormula *tFormula = _evaluator->getTFormula()) {
         tFormula->SetName(GetName());
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Return reference to the formula evaluation engine.
/// If it doesn't exist, create it on the fly. Throws if the formula is invalid.

RooFormulaEvaluator &RooGenericPdf::evaluator() const
{
   if (!_evaluator) {
      // After being read from file, the evaluation engine might not exist, yet.
      // Old files may also store the formula expression with name or ordinal
      // references, so it is normalized to the `x[i]` dialect here, where `i`
      // refers to the position in _actualVars.
      std::string processed = RooFormulaUtils::processFormula(_formExpr.Data(), _actualVars, GetName());
      _evaluator = RooFormulaUtils::makeEvaluator(GetName(), processed, _formExpr.Data(), _actualVars);
      const_cast<TString &>(_formExpr) = processed.c_str();
   }
   return *_evaluator;
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

void RooGenericPdf::setBinning(const RooAbsRealLValue &obs, const RooAbsBinning &binning, bool checkFlatness)
{
   // Match the observable to a formula variable by name, so that a same-named
   // stand-in for the actual server is accepted too.
   const int idx = _actualVars.index(obs.GetName());
   if (idx < 0) {
      coutE(InputArguments) << "RooGenericPdf::setBinning(" << GetName() << ") the observable " << obs.GetName()
                            << " is not one of the formula variables of this pdf, nothing done." << std::endl;
      return;
   }

   if (checkFlatness) {
      // Sample the function by varying the actual formula variable (the server),
      // which may be a different object than `obs` if `obs` is just a same-named
      // stand-in: the function's value depends on the server, not on `obs`.
      if (auto *serverObs = dynamic_cast<RooAbsRealLValue *>(_actualVars.at(idx))) {
         std::span<const double> boundaries{binning.array(), static_cast<std::size_t>(binning.numBoundaries())};
         if (!RooHelpers::isFunctionFlatInBins(*this, *serverObs, boundaries)) {
            coutE(InputArguments) << "RooGenericPdf::setBinning(" << GetName() << ") the expression \"" << _formExpr
                                  << "\" is not flat within the given bins of " << obs.GetName()
                                  << ". The binning is not set. Pass checkFlatness=false to override this check."
                                  << std::endl;
            return;
         }
      }
   }

   // Key the binning by the observable's index in _actualVars (not its name), so
   // that it survives a renaming of the variable or a server redirection.
   _binnings[idx] = std::unique_ptr<RooAbsBinning>{binning.clone()};
}

////////////////////////////////////////////////////////////////////////////////
/// Return the binning previously declared with setBinning() for observable
/// `obs`, or nullptr if no binning was declared. The observable is matched to a
/// formula variable by name, consistently with setBinning().

const RooAbsBinning *RooGenericPdf::getBinning(const RooAbsRealLValue &obs) const
{
   auto found = _binnings.find(_actualVars.index(obs.GetName()));
   return found != _binnings.end() ? found->second.get() : nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Remove a binning previously declared with setBinning() for observable `obs`,
/// reverting to the generic numeric integrator for it. Returns true if a binning
/// was removed, false if none was set for `obs`.

bool RooGenericPdf::removeBinning(const RooAbsRealLValue &obs)
{
   return _binnings.erase(_actualVars.index(obs.GetName())) > 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Return true if a binning was set with setBinning() for every
/// observable in the integration set `obs`.

bool RooGenericPdf::isBinnedDistribution(const RooArgSet &obs) const
{
   if (obs.empty() || _binnings.empty()) {
      return false;
   }
   for (RooAbsArg *o : obs) {
      const int idx = _actualVars.index(o->GetName());
      // Observables that are not formula variables of this pdf are ones we do
      // not depend on: the function is constant (hence trivially binned) in
      // them, so they must be ignored here. This matches the convention that
      // composite functions like RooProduct rely on, where each component's
      // isBinnedDistribution() is queried with the full observable set.
      if (idx < 0) {
         continue;
      }
      if (_binnings.find(idx) == _binnings.end()) {
         return false;
      }
   }
   return true;
}

////////////////////////////////////////////////////////////////////////////////
/// Return the boundaries of the binning set with setBinning() that fall
/// within [xlo, xhi], or a null pointer if no binning was set for this observable.

std::list<double> *RooGenericPdf::binBoundaries(RooAbsRealLValue &obs, double xlo, double xhi) const
{
   auto found = _binnings.find(_actualVars.index(obs.GetName()));
   if (found == _binnings.end()) {
      return nullptr;
   }
   const RooAbsBinning &binning = *found->second;
   auto hint = new std::list<double>;
   for (int i = 0; i < binning.numBoundaries(); ++i) {
      const double boundary = binning.array()[i];
      if (boundary >= xlo && boundary <= xhi) {
         hint->push_back(boundary);
      }
   }
   return hint;
}

////////////////////////////////////////////////////////////////////////////////
/// Return sampling hints that draw the piecewise-flat shape exactly (a pair of
/// points just left and right of every bin boundary), or a null pointer if no
/// binning was set for this observable.

std::list<double> *RooGenericPdf::plotSamplingHint(RooAbsRealLValue &obs, double xlo, double xhi) const
{
   const RooAbsBinning *binning = getBinning(obs);
   if (!binning) {
      return nullptr;
   }
   return RooCurve::plotSamplingHintForBinBoundaries(
      {binning->array(), static_cast<std::size_t>(binning->numBoundaries())}, xlo, xhi);
}

////////////////////////////////////////////////////////////////////////////////
/// Calculate current value of this object

double RooGenericPdf::evaluate() const
{
   return RooFormulaUtils::evalFormula(evaluator(), _actualVars, _actualVars.nset());
}


////////////////////////////////////////////////////////////////////////////////
void RooGenericPdf::doEval(RooFit::EvalContext & ctx) const
{
   RooFormulaUtils::doEvalFormula(evaluator(), _actualVars, ctx);
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
    RooFormulaUtils::printFormula(os, indent, _formExpr.Data(), _actualVars);
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Add formula expression as meta argument in printing interface

void RooGenericPdf::printMetaArgs(ostream& os) const
{
  os << "formula=\"" << _formExpr << "\" " ;
}

void RooGenericPdf::dumpFormula()
{
   RooFormulaUtils::printFormula(std::cout, "", _formExpr.Data(), _actualVars);
}

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
   return evaluator().getTFormula()->GetUniqueFuncName().Data();
}
