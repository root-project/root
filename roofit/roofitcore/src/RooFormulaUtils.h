/// \cond ROOFIT_INTERNAL

/*
 * Project: RooFit
 *
 * Copyright (c) 2026, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef ROO_FORMULA_UTILS
#define ROO_FORMULA_UTILS

#include "RooArgList.h"
#include "RooFit/EvalContext.h"

#include "RooFormulaEvaluator.h"

#include <list>
#include <map>
#include <memory>
#include <string>

class RooAbsBinning;
class RooAbsReal;
class RooAbsRealLValue;

/// Free functions to translate and evaluate user-defined expressions of
/// RooAbsArgs, as used by RooFormulaVar and RooGenericPdf and for cut
/// expressions on datasets. See the RooFormulaVar class documentation for the
/// supported expression dialect (name, `@i` and `x[i]` variable references,
/// and `cat::state` category state references). processFormula() normalizes
/// an expression to the `x[i]`-only dialect that the RooFormulaEvaluator
/// backends understand; the other functions operate on the normalized form.
namespace RooFormulaUtils {

std::string processFormula(std::string formula, RooArgList const &varList, std::string const &callerName);

std::string reconstructFormula(std::string internalRepr, RooArgList const &args,
                               const char *fixedReplacement = nullptr);

std::unique_ptr<RooFormulaEvaluator> makeEvaluator(std::string const &name, std::string const &processedFormula,
                                                   std::string const &origFormula, RooArgList const &varList);

std::unique_ptr<RooFormulaEvaluator>
makeFormulaEvaluator(std::string const &name, std::string const &expression, RooArgList const &varList);

void initFormula(std::unique_ptr<RooFormulaEvaluator> &evaluator, TString &formExpr, RooAbsCollection &actualVars,
                 RooArgList const &dependents, const char *name);

RooFormulaEvaluator &ensureEvaluator(std::unique_ptr<RooFormulaEvaluator> &evaluator, TString &formExpr,
                                     RooArgList const &actualVars, const char *name);

std::unique_ptr<RooFormulaEvaluator> cloneEvaluator(RooFormulaEvaluator const &other, const char *newName);

double evalFormula(RooFormulaEvaluator const &evaluator, RooAbsCollection const &vars, RooArgSet const *nset = nullptr);

void doEvalFormula(RooFormulaEvaluator const &evaluator, RooArgList const &actualVars, RooFit::EvalContext &ctx);

void printFormula(std::ostream &os, TString indent, std::string const &formula, RooArgList const &actualVars);

/// Map of user-defined binnings for a piecewise-flat RooFormulaVar or
/// RooGenericPdf, keyed by the observable's index in the formula variables.
/// The functions below implement the binning interface of these two classes.
using BinningMap = std::map<int, std::unique_ptr<RooAbsBinning>>;

BinningMap cloneBinnings(BinningMap const &binnings);

void setBinning(BinningMap &binnings, RooAbsReal const &caller, RooArgList const &actualVars, const char *formExpr,
                RooAbsRealLValue const &obs, RooAbsBinning const &binning, bool checkFlatness);

const RooAbsBinning *getBinning(BinningMap const &binnings, RooArgList const &actualVars, RooAbsRealLValue const &obs);

bool isBinnedDistribution(BinningMap const &binnings, RooArgList const &actualVars, RooArgSet const &obs);

std::list<double> *binBoundaries(BinningMap const &binnings, RooArgList const &actualVars, RooAbsRealLValue const &obs,
                                 double xlo, double xhi);

std::list<double> *plotSamplingHint(BinningMap const &binnings, RooArgList const &actualVars,
                                    RooAbsRealLValue const &obs, double xlo, double xhi);

} // namespace RooFormulaUtils

namespace RooFormulaInternal {

/// Testing hook: discard the cached ROOFIT_FORMULA_BACKEND setting so that it
/// is read again from the environment on the next evaluator creation.
void resetFormulaBackendForTesting();

} // namespace RooFormulaInternal

#endif

/// \endcond
