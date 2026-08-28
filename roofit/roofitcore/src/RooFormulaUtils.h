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

#include <memory>
#include <string>
#include <vector>

/// Free functions to translate and evaluate user-defined expressions of
/// RooAbsArgs, as used by RooFormulaVar and RooGenericPdf and for cut
/// expressions on datasets. See the RooFormulaVar class documentation for the
/// supported expression dialect (name, `@i` and `x[i]` variable references,
/// and `cat::state` category state references). processFormula() normalizes
/// an expression to the `x[i]`-only dialect that the RooFormulaEvaluator
/// backends understand; the other functions operate on the normalized form.
namespace RooFormulaUtils {

/// The result of compileFormula(): the normalized expression together with the
/// variables it actually uses and the evaluation engine for it.
struct CompiledFormula {
   /// Processed expression in the `x[i]` dialect, where `i` refers to the
   /// position in `actualVars` (variables that turned out to be unused in the
   /// expression are pruned and the indices remapped accordingly).
   std::string formula;
   /// The subset of input variables that the expression actually uses.
   RooArgList actualVars;
   /// The evaluation engine, with `x[i]` referring to `actualVars[i]`.
   std::unique_ptr<RooFormulaEvaluator> evaluator;
};

std::string processFormula(std::string formula, RooArgList const &varList, std::string const &callerName);

std::vector<bool> usedVariables(std::string const &processedFormula, std::size_t nVars);

std::string reindexFormula(std::string const &processedFormula, std::vector<bool> const &varIsUsed);

std::string reconstructFormula(std::string internalRepr, RooArgList const &args,
                               const char *fixedReplacement = nullptr);

std::unique_ptr<RooFormulaEvaluator> makeEvaluator(std::string const &name, std::string const &processedFormula,
                                                   std::string const &origFormula, RooArgList const &varList);

std::unique_ptr<RooFormulaEvaluator>
makeFormulaEvaluator(std::string const &name, std::string const &expression, RooArgList const &varList);

CompiledFormula compileFormula(std::string const &name, std::string const &expression, RooArgList const &varList);

double evalFormula(RooFormulaEvaluator const &evaluator, RooAbsCollection const &vars, RooArgSet const *nset = nullptr);

void doEvalFormula(RooFormulaEvaluator const &evaluator, RooArgList const &actualVars, RooFit::EvalContext &ctx);

void printFormula(std::ostream &os, TString indent, std::string const &formula, RooArgList const &actualVars);

} // namespace RooFormulaUtils

#endif

/// \endcond
