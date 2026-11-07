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

#ifndef ROO_T_FORMULA_EVALUATOR
#define ROO_T_FORMULA_EVALUATOR

#include "RooFormulaEvaluator.h"

#include <memory>
#include <string>

class RooArgList;
class TFormula;

/// RooFormulaEvaluator implementation backed by a TFormula, i.e. evaluating
/// via a function compiled with the cling JIT.
class RooTFormulaEvaluator : public RooFormulaEvaluator {
public:
   RooTFormulaEvaluator(const char *name, std::string const &processedFormula, std::string const &origFormula,
                        RooArgList const &varList);
   RooTFormulaEvaluator(RooTFormulaEvaluator const &other);

   RooTFormulaEvaluator &operator=(RooTFormulaEvaluator const &other) = delete;

   ~RooTFormulaEvaluator() override;

   double eval(const double *vars) const override;
   std::unique_ptr<RooFormulaEvaluator> clone() const override;

   TFormula *getTFormula() const override { return _tFormula.get(); }

private:
   std::unique_ptr<TFormula> _tFormula; ///< The formula used to compute values
};

#endif

/// \endcond
