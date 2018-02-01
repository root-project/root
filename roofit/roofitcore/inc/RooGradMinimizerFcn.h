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
#include "TMatrixDSym.h"
#include "RooGradientFunction.h"

class RooGradMinimizer;


class RooGradMinimizerFcn : public RooGradientFunction {

public:

  RooGradMinimizerFcn(RooAbsReal *funct, RooGradMinimizer *context,
                      GradientCalculatorMode grad_mode = GradientCalculatorMode::ExactlyMinuit2,
                      bool verbose = false);

  RooGradMinimizerFcn(const RooGradMinimizerFcn &other);

  TObject* Clone() const override;

  void BackProp(const ROOT::Fit::FitResult &results);
  void ApplyCovarianceMatrix(TMatrixDSym &V);

  void synchronize_gradient_with_minimizer() const;

private:
  std::vector<ROOT::Fit::ParameterSettings>& parameter_settings() const override;

  RooGradMinimizer *_context;

  ClassDefOverride(RooGradMinimizerFcn,0)
};
#endif
#endif
