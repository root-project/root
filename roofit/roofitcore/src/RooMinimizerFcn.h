/// \cond ROOFIT_INTERNAL

/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   AL, Alfio Lazzaro,   INFN Milan,        alfio.lazzaro@mi.infn.it        *
 *   PB, Patrick Bos, Netherlands eScience Center, p.bos@esciencecenter.nl   *
 *                                                                           *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

#ifndef ROO_MINIMIZER_FCN
#define ROO_MINIMIZER_FCN

#include "Math/IFunction.h"

#include "RooAbsReal.h"
#include "RooArgList.h"

#include <Math/Minimizer.h>

#include <fstream>
#include <mutex>
#include <vector>

#include "RooAbsMinimizerFcn.h"

// forward declaration
class RooMinimizer;

class RooMinimizerFcn : public RooAbsMinimizerFcn {

public:
   RooMinimizerFcn(RooAbsReal *funct, RooMinimizer *context);

   /// Set this function on a ROOT::Math::Minimizer. Only used for minimizers
   /// other than Minuit2, which RooMinimizer drives directly via the
   /// ROOT::Minuit2::FCNBase interface.
   void initMinimizer(ROOT::Math::Minimizer &minim) const;

   std::string getFunctionName() const override;
   std::string getFunctionTitle() const override;

   void setOffsetting(bool flag) override;

   double operator()(const double *x) const;
   void evaluateGradient(const double *x, double *out) const;

   RooArgSet freezeDisconnectedParameters() const override;

   /// \name ROOT::Minuit2::FCNBase interface
   /// @{
   double operator()(std::vector<double> const &x) const override { return (*this)(x.data()); }
   bool HasGradient() const override { return _useGradient; }
   std::vector<double> Gradient(std::vector<double> const &x) const override;
   bool HasHessian() const override { return _useHessian; }
   std::vector<double> Hessian(std::vector<double> const &x) const override;
   bool SecondDerivativeAlwaysVanishes(unsigned int i, unsigned int j) const override;
   /// @}

private:
   void buildSecondDerivMask() const;

   RooAbsReal *_funct = nullptr;
   bool _useGradient = false; ///< Whether to provide the analytic gradient to the minimizer.
   bool _useHessian = false;  ///< Whether to provide the analytic Hessian to the minimizer.
   /// Adapter to ROOT::Math::Minimizer, only used for minimizers other than Minuit2.
   std::unique_ptr<ROOT::Math::IBaseFunctionMultiDim> _multiGenFcn;
   mutable std::vector<double> _gradientOutput;
   mutable std::vector<double> _hessianOutput;
   mutable std::vector<bool> _secondDerivMask; ///< Lazily built, see buildSecondDerivMask().
   mutable std::once_flag _secondDerivMaskOnce;
};

#endif

/// \endcond
