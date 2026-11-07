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

#ifndef ROO_FORMULA_EVALUATOR
#define ROO_FORMULA_EVALUATOR

#include <memory>

class TFormula;

/// Abstract interface for evaluating a processed formula expression, i.e. one
/// normalized by RooFormulaUtils::processFormula() to the `x[i]`-only dialect,
/// so an implementation doesn't need any name resolution.
class RooFormulaEvaluator {
public:
   virtual ~RooFormulaEvaluator() = default;

   /// Evaluate the formula, with `vars[i]` providing the value of `x[i]`.
   virtual double eval(const double *vars) const = 0;

   /// Return a deep copy of this evaluator.
   virtual std::unique_ptr<RooFormulaEvaluator> clone() const = 0;

   /// Return the underlying TFormula. Only the TFormula-backed evaluator
   /// returns a non-nullptr. This accessor only exists to support the
   /// getUniqueFuncName() functions used by the codegen backend and will be
   /// removed together with them.
   virtual TFormula *getTFormula() const { return nullptr; }
};

#endif

/// \endcond
