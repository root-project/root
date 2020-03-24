/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   PB, Patrick Bos,     NL eScience Center, p.bos@esciencecenter.nl        *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

#ifndef ROOFIT_TESTSTATISTICS_GRADMINIMIZER_H
#define ROOFIT_TESTSTATISTICS_GRADMINIMIZER_H

#include <vector>
#include <RooMinimizer.h>

#include "Fit/FitResult.h"
#include "Minuit2/MnStrategy.h"
#include "TMatrixDSym.h"
#include "RooGradientFunction.h"

#include "RooGradMinimizer.h"
#include "TestStatistics/LikelihoodGradientWrapper.h"

namespace RooFit {
namespace TestStatistics {

class GradMinimizerFcn : public RooGradientFunction {
public:
   GradMinimizerFcn(RooAbsReal *funct, RooMinimizerGenericPtr context, bool verbose = false);
   GradMinimizerFcn(const GradMinimizerFcn &other);

   ROOT::Math::IMultiGradFunction* Clone() const override;

   void BackProp(const ROOT::Fit::FitResult &results);
   void ApplyCovarianceMatrix(TMatrixDSym &V);

   ROOT::Minuit2::MnStrategy get_strategy() const;
   double get_error_def() const;
   void set_strategy(int istrat);

   Bool_t Synchronize(std::vector<ROOT::Fit::ParameterSettings> &parameter_settings,
                      Bool_t optConst = kTRUE, Bool_t verbose = kFALSE);

   void Gradient(const double *x, double *grad) const override;
   void G2ndDerivative(const double *x, double *g2) const override;
   void GStepSize(const double *x, double *gstep) const override;

private:
   std::vector<ROOT::Fit::ParameterSettings>& parameter_settings() const override;

//   LikelihoodGradientWrapper gradient;

   RooMinimizerGenericPtr _context;
};

using GradMinimizer = RooMinimizerTemplate<GradMinimizerFcn, RooFit::MinimizerType::Minuit2, std::size_t>;

} // namespace TestStatistics
} // namespace RooFit

#endif //ROOFIT_TESTSTATISTICS_GRADMINIMIZER_H
