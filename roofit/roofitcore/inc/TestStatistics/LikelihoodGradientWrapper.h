// Author: Patrick Bos, Netherlands eScience Center / NIKHEF 2021

/*****************************************************************************
 * RooFit
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2021, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

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

class LikelihoodGradientWrapper {
public:
   LikelihoodGradientWrapper(std::shared_ptr<RooAbsL> likelihood, std::shared_ptr<WrapperCalculationCleanFlags> calculation_is_clean, std::size_t N_dim, RooMinimizer* minimizer);
   virtual ~LikelihoodGradientWrapper() = default;
   virtual LikelihoodGradientWrapper* clone() const = 0;

   virtual void fillGradient(double *grad) = 0;

   // synchronize minimizer settings with calculators in child classes
   virtual void synchronizeWithMinimizer(const ROOT::Math::MinimizerOptions &options);
   virtual void synchronizeParameterSettings(const std::vector<ROOT::Fit::ParameterSettings> &parameter_settings);
   virtual void synchronizeParameterSettings(ROOT::Math::IMultiGenFunction* function, const std::vector<ROOT::Fit::ParameterSettings> &parameter_settings) = 0;
   // Minuit passes in parameter values that may not conform to RooFit internal standards (like applying range clipping),
   // but that the specific calculator does need. This function can be implemented to receive these Minuit-internal values:
   virtual void updateMinuitInternalParameterValues(const std::vector<double>& minuit_internal_x);
   virtual void updateMinuitExternalParameterValues(const std::vector<double>& minuit_external_x);

   // completely depends on the implementation, so pure virtual
   virtual bool usesMinuitInternalValues() = 0;

protected:
   std::shared_ptr<RooAbsL> likelihood_;
   RooMinimizer * minimizer_;
   std::shared_ptr<WrapperCalculationCleanFlags> calculation_is_clean_;
};

} // namespace TestStatistics
} // namespace RooFit

#endif // ROOT_ROOFIT_TESTSTATISTICS_LikelihoodGradientWrapper
