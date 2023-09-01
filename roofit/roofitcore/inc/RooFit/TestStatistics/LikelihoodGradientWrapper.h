/*
 * Project: RooFit
 * Authors:
 *   PB, Patrick Bos, Netherlands eScience Center, p.bos@esciencecenter.nl
 *
 * Copyright (c) 2021, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef ROOT_ROOFIT_TESTSTATISTICS_LikelihoodGradientWrapper
#define ROOT_ROOFIT_TESTSTATISTICS_LikelihoodGradientWrapper

#include <Fit/ParameterSettings.h>
#include <Math/IFunctionfwd.h>
#include "Math/MinimizerOptions.h"

#include <vector>
#include <memory> // shared_ptr

// forward declaration
class RooMinimizer;

namespace RooFit {
namespace TestStatistics {

// forward declaration
class RooAbsL;
struct WrapperCalculationCleanFlags;

enum class LikelihoodGradientMode { multiprocess };

class LikelihoodGradientWrapper {
public:
   LikelihoodGradientWrapper(std::shared_ptr<RooAbsL> likelihood,
                             std::shared_ptr<WrapperCalculationCleanFlags> calculation_is_clean, std::size_t N_dim,
                             RooMinimizer *minimizer);
   virtual ~LikelihoodGradientWrapper() = default;
   virtual LikelihoodGradientWrapper *clone() const = 0;

   static std::unique_ptr<LikelihoodGradientWrapper>
   create(LikelihoodGradientMode likelihoodGradientMode, std::shared_ptr<RooAbsL> likelihood,
          std::shared_ptr<WrapperCalculationCleanFlags> calculationIsClean, std::size_t nDim, RooMinimizer *minimizer);

   virtual void fillGradient(double *grad) = 0;
   virtual void
   fillGradientWithPrevResult(double *grad, double *previous_grad, double *previous_g2, double *previous_gstep) = 0;

   /// Synchronize minimizer settings with calculators in child classes.
   virtual void synchronizeWithMinimizer(const ROOT::Math::MinimizerOptions &options);
   virtual void synchronizeParameterSettings(const std::vector<ROOT::Fit::ParameterSettings> &parameter_settings);
   virtual void synchronizeParameterSettings(ROOT::Math::IMultiGenFunction *function,
                                             const std::vector<ROOT::Fit::ParameterSettings> &parameter_settings) = 0;
   /// Minuit passes in parameter values that may not conform to RooFit internal standards (like applying range
   /// clipping), but that the specific calculator does need. This function can be implemented to receive these
   /// Minuit-internal values.
   virtual void updateMinuitInternalParameterValues(const std::vector<double> &minuit_internal_x);
   virtual void updateMinuitExternalParameterValues(const std::vector<double> &minuit_external_x);

   /// \brief Implement usesMinuitInternalValues to return true when you want Minuit to send this class Minuit-internal
   /// values, or return false when you want "regular" Minuit-external values.
   ///
   /// Minuit internally uses a transformed parameter space to graciously handle externally mandated parameter range
   /// boundaries. Transformation from Minuit-internal to external (i.e. "regular") parameters is done using
   /// trigonometric functions that in some cases can cause a few bits of precision loss with respect to the original
   /// parameter values. To circumvent this, Minuit also allows external gradient providers (like
   /// LikelihoodGradientWrapper) to take the Minuit-internal parameter values directly, without transformation. This
   /// way, the gradient provider (e.g. the implementation of this class) can handle transformation manually, possibly
   /// with higher precision.
   virtual bool usesMinuitInternalValues() = 0;

   /// Reports whether or not the gradient is currently being calculated.
   ///
   /// This is used in MinuitFcnGrad to switch between LikelihoodWrapper implementations
   /// inside and outside of a LikelihoodGradientJob calculation when the LikelihoodWrapper
   /// used is LikelihoodJob. This is to prevent Jobs from being started within Jobs.
   virtual bool isCalculating() = 0;

protected:
   std::shared_ptr<RooAbsL> likelihood_;
   RooMinimizer *minimizer_;
   std::shared_ptr<WrapperCalculationCleanFlags> calculation_is_clean_;
};

} // namespace TestStatistics
} // namespace RooFit

#endif // ROOT_ROOFIT_TESTSTATISTICS_LikelihoodGradientWrapper
