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

#ifndef ROO_FORMULA_PARSER
#define ROO_FORMULA_PARSER

#include "RooExprEvaluator.h"

#include <memory>
#include <string>

namespace RooFormulaParser {

/// Try to compile the given processed formula string (all variables in the
/// `x[i]` dialect produced by RooFormula::processFormula()) into a program for
/// the JIT-free RooExprEvaluator.
///
/// Returns nullptr on *any* construct the JIT-free evaluator does not support
/// (unknown identifier, unknown function, unsupported operator, ...); the
/// caller then falls back to the TFormula backend. If `error` is non-null, it
/// is filled with the reason on failure.
///
/// Successfully compiled programs are cached in a process-wide registry keyed
/// on the formula string, so identical formulas share one immutable
/// instruction vector (mirroring what TFormula's gClingFunctions cache does
/// for JIT-compiled code). Compilation takes a mutex; evaluation of the
/// returned program is lock-free.
std::shared_ptr<const RooExprEvaluator::Program>
compile(std::string const &processedFormula, unsigned int nVars, std::string *error = nullptr);

} // namespace RooFormulaParser

#endif

/// \endcond
