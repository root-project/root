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
/// \class RooFormulaVar
///
/// A RooFormulaVar is a generic implementation of a real-valued object,
/// which takes a RooArgList of servers and a C++ expression string defining how
/// its value should be calculated from the given list of servers.
///
/// If RooAbsPdf objects are supplied to RooFormulaVar as servers, their
/// raw (unnormalized) values will be evaluated. Use RooGenericPdf, which
/// constructs generic PDF functions, to access their properly normalized
/// values.
///
/// The string expression can be any valid TFormula expression referring to the
/// listed servers either by name or by their ordinal list position. These three are
/// equivalent:
/// ```
///   RooFormulaVar("gen", "x*y", RooArgList(x,y))       // reference by name
///   RooFormulaVar("gen", "@0*@1", RooArgList(x,y))     // reference by ordinal with @
///   RooFormulaVar("gen", "x[0]*x[1]", RooArgList(x,y)) // TFormula-builtin reference by ordinal
/// ```
/// Note that `x[i]` is an expression reserved for TFormula. All variable references
/// are automatically converted to the TFormula-native format. If a variable with
/// the name `x` is given, `x[i]` is interpreted as a list position,
/// but `x` without brackets as the name of a RooFit object.
///
/// The last two versions, while slightly less readable, are more versatile because
/// the names of the arguments are not hard coded.
///

#include "Riostream.h"

#include "RooFormulaVar.h"
#include "RooStreamParser.h"
#include "RooMsgService.h"
#include "RooFormulaUtils.h"
#include "RooAbsRealLValue.h"

#ifdef ROOFIT_LEGACY_EVAL_BACKEND
#include "RooNLLVar.h"
#include "RooChi2Var.h"
#endif

using std::ostream, std::istream, std::list;


RooFormulaVar::RooFormulaVar() {}

RooFormulaVar::~RooFormulaVar() = default;

////////////////////////////////////////////////////////////////////////////////
/// Constructor with formula expression and list of input variables.
/// \param[in] name Name of the formula.
/// \param[in] title Title of the formula.
/// \param[in] inFormula Expression to be evaluated.
/// \param[in] dependents Variables that should be passed to the formula.
/// \param[in] checkVariables Unused parameter.
RooFormulaVar::RooFormulaVar(const char *name, const char *title, const char* inFormula, const RooArgList& dependents,
    bool /*checkVariables*/) :
  RooAbsReal(name,title),
  _actualVars("actualVars","Variables used by formula expression",this),
  _formExpr(inFormula)
{
  if (dependents.empty()) {
    _value = traceEval(nullptr);
  } else {
     RooFormulaUtils::initFormula(_evaluator, _formExpr, _actualVars, dependents, GetName());
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Constructor with formula expression, title and list of input variables.
/// \param[in] name Name of the formula.
/// \param[in] title Formula expression. Will also be used as the title.
/// \param[in] dependents Variables that should be passed to the formula.
/// \param[in] checkVariables Check that all variables from `dependents` are used in the expression.
RooFormulaVar::RooFormulaVar(const char *name, const char *title, const RooArgList &dependents, bool checkVariables)
   : RooFormulaVar(name, title, title, dependents, checkVariables)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooFormulaVar::RooFormulaVar(const RooFormulaVar& other, const char* name) :
  RooAbsReal(other, name),
  _actualVars("actualVars",this,other._actualVars),
  _formExpr(other._formExpr)
{
   _binnings = RooFormulaUtils::cloneBinnings(other._binnings);
   if (other._evaluator) {
      _evaluator = RooFormulaUtils::cloneEvaluator(*other._evaluator, GetName());
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Return reference to the formula evaluation engine.
/// If it doesn't exist, create it on the fly. Throws if the formula is invalid.
RooFormulaEvaluator &RooFormulaVar::evaluator() const
{
   return RooFormulaUtils::ensureEvaluator(_evaluator, const_cast<TString &>(_formExpr), _actualVars, GetName());
}

bool RooFormulaVar::ok() const
{
   evaluator();
   return true;
}

void RooFormulaVar::dumpFormula()
{
   RooFormulaUtils::printFormula(std::cout, "", _formExpr.Data(), _actualVars);
}

////////////////////////////////////////////////////////////////////////////////
/// Calculate current value of object from internal formula

double RooFormulaVar::evaluate() const
{
   return RooFormulaUtils::evalFormula(evaluator(), _actualVars, _actualVars.nset());
}


void RooFormulaVar::doEval(RooFit::EvalContext &ctx) const
{
   RooFormulaUtils::doEvalFormula(evaluator(), _actualVars, ctx);
}

////////////////////////////////////////////////////////////////////////////////
/// Print info about this object to the specified stream.

void RooFormulaVar::printMultiline(ostream& os, Int_t contents, bool verbose, TString indent) const
{
  RooAbsReal::printMultiline(os,contents,verbose,indent);
  if(verbose) {
    indent.Append("  ");
    os << indent;
    RooFormulaUtils::printFormula(os, indent, _formExpr.Data(), _actualVars);
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Add formula expression as meta argument in printing interface

void RooFormulaVar::printMetaArgs(ostream& os) const
{
  os << "formula=\"" << _formExpr << "\" " ;
}




////////////////////////////////////////////////////////////////////////////////
/// Read object contents from given stream

bool RooFormulaVar::readFromStream(istream& /*is*/, bool /*compact*/, bool /*verbose*/)
{
  coutE(InputArguments) << "RooFormulaVar::readFromStream(" << GetName() << "): can't read" << std::endl ;
  return true ;
}



////////////////////////////////////////////////////////////////////////////////
/// Write object contents to given stream

void RooFormulaVar::writeToStream(ostream& os, bool compact) const
{
  if (compact) {
    std::cout << getVal() << std::endl ;
  } else {
    os << GetTitle() ;
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Declare that this function is piecewise constant (flat) within the bins of
/// the given `binning` of the observable `obs`, which must be one of the formula
/// variables. The method can be called several times to set a binning for more
/// than one observable. See RooGenericPdf::setBinning() for details.

void RooFormulaVar::setBinning(const RooAbsRealLValue &obs, const RooAbsBinning &binning, bool checkFlatness)
{
   RooFormulaUtils::setBinning(_binnings, *this, _actualVars, _formExpr.Data(), obs, binning, checkFlatness);
}

////////////////////////////////////////////////////////////////////////////////
/// Return the binning previously declared with setBinning() for observable
/// `obs`, or nullptr if no binning was declared. This reports only binnings
/// owned by this formula, not binning hints forwarded by its servers.

const RooAbsBinning *RooFormulaVar::getBinning(const RooAbsRealLValue &obs) const
{
   return RooFormulaUtils::getBinning(_binnings, _actualVars, obs);
}

////////////////////////////////////////////////////////////////////////////////
/// Remove a binning previously declared with setBinning() for observable `obs`,
/// reverting to the generic numeric integrator for it. Returns true if a binning
/// was removed, false if none was set for `obs`.

bool RooFormulaVar::removeBinning(const RooAbsRealLValue &obs)
{
   return _binnings.erase(_actualVars.index(obs.GetName())) > 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Return true if a binning was set with setBinning() for every
/// observable in the integration set `obs`.

bool RooFormulaVar::isBinnedDistribution(const RooArgSet &obs) const
{
   return RooFormulaUtils::isBinnedDistribution(_binnings, _actualVars, obs);
}

////////////////////////////////////////////////////////////////////////////////
/// Return the boundaries of the binning set with setBinning() that fall
/// within [xlo, xhi]. If no binning was set for this observable, forward the bin
/// boundaries from the server that defines the observable obs.

std::list<double>* RooFormulaVar::binBoundaries(RooAbsRealLValue& obs, double xlo, double xhi) const
{
   if (auto *hint = RooFormulaUtils::binBoundaries(_binnings, _actualVars, obs, xlo, xhi)) {
      return hint;
   }

  for (const auto par : _actualVars) {
    auto func = static_cast<const RooAbsReal*>(par);
    list<double>* binb = nullptr;

    if (func && (binb = func->binBoundaries(obs,xlo,xhi)) ) {
      return binb;
    }
  }

  return nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Return sampling hints that draw the piecewise-flat shape exactly if a binning
/// was set for this observable. Otherwise, forward the plot sampling hint from
/// the server that defines the observable obs.

std::list<double>* RooFormulaVar::plotSamplingHint(RooAbsRealLValue& obs, double xlo, double xhi) const
{
   if (auto *hint = RooFormulaUtils::plotSamplingHint(_binnings, _actualVars, obs, xlo, xhi)) {
      return hint;
   }

  for (const auto par : _actualVars) {
    auto func = dynamic_cast<const RooAbsReal*>(par);
    list<double>* hint = nullptr;

    if (func && (hint = func->plotSamplingHint(obs,xlo,xhi)) ) {
      return hint;
    }
  }

  return nullptr;
}



////////////////////////////////////////////////////////////////////////////////
/// Return the default error level for MINUIT error analysis
/// If the formula contains one or more RooNLLVars and
/// no RooChi2Vars, return the defaultErrorLevel() of
/// RooNLLVar. If the addition contains one ore more RooChi2Vars
/// and no RooNLLVars, return the defaultErrorLevel() of
/// RooChi2Var. If the addition contains neither or both
/// issue a warning message and return a value of 1

double RooFormulaVar::defaultErrorLevel() const
{
  RooAbsReal* nllArg(nullptr) ;
  RooAbsReal* chi2Arg(nullptr) ;

#ifdef ROOFIT_LEGACY_EVAL_BACKEND
  for (const auto arg : _actualVars) {
    if (dynamic_cast<RooNLLVar*>(arg)) {
      nllArg = static_cast<RooAbsReal*>(arg) ;
    }
    if (dynamic_cast<RooChi2Var*>(arg)) {
      chi2Arg = static_cast<RooAbsReal*>(arg) ;
    }
  }
#endif

  if (nllArg && !chi2Arg) {
    coutI(Minimization) << "RooFormulaVar::defaultErrorLevel(" << GetName()
         << ") Formula contains a RooNLLVar, using its error level" << std::endl ;
    return nllArg->defaultErrorLevel() ;
  } else if (chi2Arg && !nllArg) {
    coutI(Minimization) << "RooFormulaVar::defaultErrorLevel(" << GetName()
    << ") Formula contains a RooChi2Var, using its error level" << std::endl ;
    return chi2Arg->defaultErrorLevel() ;
  } else if (!nllArg && !chi2Arg) {
    coutI(Minimization) << "RooFormulaVar::defaultErrorLevel(" << GetName() << ") WARNING: "
            << "Formula contains neither RooNLLVar nor RooChi2Var server, using default level of 1.0" << std::endl ;
  } else {
    coutI(Minimization) << "RooFormulaVar::defaultErrorLevel(" << GetName() << ") WARNING: "
         << "Formula contains BOTH RooNLLVar and RooChi2Var server, using default level of 1.0" << std::endl ;
  }

  return 1.0 ;
}

////////////////////////////////////////////////////////////////////////////////
/// Name of the cling-JIT-compiled function evaluating this formula, which the
/// codegen fallback path calls by name in generated code. Empty when the
/// formula is evaluated by the JIT-free expression backend (codegen then
/// inlines the expression via emitFormulaCpp() instead).
std::string RooFormulaVar::getUniqueFuncName() const
{
   return evaluator().uniqueFuncName();
}

////////////////////////////////////////////////////////////////////////////////
/// If the formula expression can be emitted as inline C++ (i.e. it is
/// evaluated by the JIT-free expression backend), return the emitted
/// expression, with `varName(i)` supplying the generated name of
/// `dependents()[i]`. Return an empty string otherwise; codegen then falls
/// back to calling the cling-JIT-compiled TFormula function by name (see
/// getUniqueFuncName()).
std::string RooFormulaVar::emitFormulaCpp(std::function<std::string(unsigned int)> const &varName) const
{
   return evaluator().emitCpp(varName);
}

////////////////////////////////////////////////////////////////////////////////
/// Whether the formula expression is evaluated by RooFit's built-in JIT-free
/// (AST) formula backend, which is the default for supported expressions.
/// Returns false if it is evaluated by the TFormula (cling JIT) fallback
/// backend instead, either because the built-in parser does not support the
/// expression or because the TFormula backend was forced with the
/// ROOFIT_FORMULA_BACKEND environment variable. Exactly the formulas on the
/// JIT-free backend can be emitted as inline C++ (emitFormulaCpp()); the
/// others report the name of their JIT-compiled function (getUniqueFuncName()).
bool RooFormulaVar::formulaUsesAstBackend() const
{
   return evaluator().canEmitCpp();
}

std::unique_ptr<RooAbsArg>
RooFormulaVar::compileForNormSet(RooArgSet const &normSet, RooFit::Detail::CompileContext &ctx) const
{
   // Some users exploit unnormalized RooAbsPdfs as inputs for RooFormulaVars,
   // relying on what the pdf returns from RooAbsPdf::evaluate(). This is in
   // principle not allowed because every pdf needs to be evaluated with a
   // normalization set, but it's so common in user code that we need to
   // support it. To make this work, we need to make sure that the no
   // normalization over non-dependents is happening at this point, reducing
   // the normalization set to the subset of actual dependents.
   // See also the "PdfAsFunctionInFormulaVar" test in testRooAbsPdf.
   RooArgSet depList;
   getObservables(&normSet, depList);
   auto newArg = std::unique_ptr<RooAbsArg>{static_cast<RooAbsArg *>(Clone())};
   ctx.markAsCompiled(*newArg);
   ctx.compileServers(*newArg, depList);
   return newArg;
}
