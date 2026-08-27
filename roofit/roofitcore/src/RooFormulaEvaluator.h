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
#include <string>

class TFormula;

/// Abstract interface for evaluating a processed RooFormula expression, i.e.
/// one normalized by RooFormula::processFormula() to the `x[i]`-only dialect,
/// so an implementation doesn't need any name resolution.
class RooFormulaEvaluator {
public:
   virtual ~RooFormulaEvaluator() = default;

   /// Evaluate the formula, with `vars[i]` providing the value of `x[i]`.
   virtual double eval(const double *vars) const = 0;

   /// Return a deep copy of this evaluator.
   virtual std::unique_ptr<RooFormulaEvaluator> clone() const = 0;

   /// Report whether the variable `x[i]` appears in the formula.
   virtual bool usesVariable(unsigned int i) const = 0;

   /// Return the processed formula string (in the normalized `x[i]` dialect).
   virtual std::string processedFormula() const = 0;

   /// Return the underlying TFormula. Only the TFormula-backed evaluator
   /// returns a non-nullptr. This accessor only exists to support
   /// RooFormula::getTFormula() and will be removed together with it.
   virtual TFormula *getTFormula() const { return nullptr; }
};

#endif

/// \endcond
