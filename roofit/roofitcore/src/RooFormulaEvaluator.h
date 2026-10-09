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

#include <functional>
#include <memory>
#include <string>

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

   /// Whether emitCpp() can emit this expression as C++ source. Only the
   /// JIT-free expression backend can; the TFormula backend cannot.
   virtual bool canEmitCpp() const { return false; }

   /// Emit this expression as C++ source with explicit parenthesization, for
   /// RooFit code generation and automatic differentiation. `varName(i)`
   /// supplies the emitted name for `x[i]`. Returns an empty string if this
   /// evaluator cannot emit C++ (see canEmitCpp()); the codegen caller then
   /// uses the TFormula fallback path via uniqueFuncName().
   virtual std::string emitCpp(std::function<std::string(unsigned int)> const & /*varName*/) const { return {}; }

   /// Name of the cling-JIT-compiled function that evaluates this formula.
   /// Only meaningfully implemented by the TFormula backend, where it serves
   /// the codegen fallback path for formulas that cannot emitCpp(): the
   /// generated code calls that function by name. Empty otherwise.
   virtual std::string uniqueFuncName() const { return {}; }

   /// Propagate a rename of the owning object to any named objects held
   /// by the evaluator (the TFormula backend renames its TFormula).
   virtual void setName(const char * /*name*/) {}
};

#endif

/// \endcond
