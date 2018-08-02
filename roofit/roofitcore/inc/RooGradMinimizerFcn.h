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

#ifndef __ROOFIT_NORooGradMinimizer

#ifndef ROO_GRAD_MINIMIZER_FCN
#define ROO_GRAD_MINIMIZER_FCN

#include "Fit/FitResult.h"
#include "Minuit2/MnStrategy.h"
#include "TMatrixDSym.h"
#include "RooGradientFunction.h"

#include "RooGradMinimizer.h"


class RooGradMinimizerFcn : public RooGradientFunction {
 public:
  RooGradMinimizerFcn(RooAbsReal *funct, RooMinimizerGenericPtr context,
                      bool verbose = false);

  RooGradMinimizerFcn(const RooGradMinimizerFcn &other);

  ROOT::Math::IMultiGradFunction* Clone() const override;

  void BackProp(const ROOT::Fit::FitResult &results);
  void ApplyCovarianceMatrix(TMatrixDSym &V);

  ROOT::Minuit2::MnStrategy get_strategy() const;
  double get_error_def() const;
  void set_strategy(int istrat);

  Bool_t Synchronize(std::vector<ROOT::Fit::ParameterSettings> &parameter_settings,
                     Bool_t optConst = kTRUE, Bool_t verbose = kFALSE);

private:
  std::vector<ROOT::Fit::ParameterSettings>& parameter_settings() const override;

  RooMinimizerGenericPtr _context;
};
#endif
#endif
