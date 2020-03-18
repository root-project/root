/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   AL, Alfio Lazzaro,   INFN Milan,        alfio.lazzaro@mi.infn.it        *
 *   PB, Patrick Bos,     NL eScience Center, p.bos@esciencecenter.nl        *
 *   VC, Vince Croft,     DIANA / NYU,        vincent.croft@cern.ch          *
 *                                                                           *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

#ifndef __ROOFIT_NOGradMinimizer

#ifndef ROOT_ROOFIT_TESTSTATISTICS_GRAD_MINIMIZER_FCN
#define ROOT_ROOFIT_TESTSTATISTICS_GRAD_MINIMIZER_FCN

#include "Fit/FitResult.h"
#include "Minuit2/MnStrategy.h"
#include "TMatrixDSym.h"
#include "RooGradientFunction.h"

#include <TestStatistics/GradMinimizer.h>

namespace RooFit {
namespace TestStatistics {

class GradMinimizerFcn : public RooGradientFunction {
public:
   GradMinimizerFcn(RooAbsReal *funct, MinimizerGenericPtr context, bool verbose = false);

   GradMinimizerFcn(const GradMinimizerFcn &other);

   ROOT::Math::IMultiGradFunction *Clone() const override;

   void BackProp(const ROOT::Fit::FitResult &results);
   void ApplyCovarianceMatrix(TMatrixDSym &V);

   ROOT::Minuit2::MnStrategy get_strategy() const;
   double get_error_def() const;
   void set_strategy(int istrat);

   Bool_t Synchronize(std::vector<ROOT::Fit::ParameterSettings> &parameter_settings, Bool_t optConst = kTRUE,
                      Bool_t verbose = kFALSE);

private:
   std::vector<ROOT::Fit::ParameterSettings> &parameter_settings() const override;

   MinimizerGenericPtr _context;
};

} // namespace TestStatistics
} // namespace RooFit

#endif // ROOT_ROOFIT_TESTSTATISTICS_GRAD_MINIMIZER_FCN
#endif
