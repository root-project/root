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

#ifndef ROOT_ROOFIT_TESTSTATISTICS_MinuitFcnGrad
#define ROOT_ROOFIT_TESTSTATISTICS_MinuitFcnGrad

#include "RooArgList.h"
#include "RooRealVar.h"
#include "TestStatistics/RooAbsL.h"
#include "TestStatistics/LikelihoodWrapper.h"
#include "TestStatistics/LikelihoodGradientWrapper.h"
#include "RooAbsMinimizerFcn.h"

#include <Fit/ParameterSettings.h>
#include <Fit/Fitter.h>
#include "Math/IFunction.h" // ROOT::Math::IMultiGradFunction

// forward declaration
class RooAbsReal;
class RooMinimizer;

namespace RooFit {
namespace TestStatistics {

// forward declaration
class LikelihoodSerial;
class LikelihoodGradientSerial;

// -- for communication with wrappers: --
struct WrapperCalculationCleanFlags {
   // indicate whether that part has been calculated since the last parameter update
   bool likelihood = false;
   bool gradient = false;

   void set_all(bool value)
   {
      likelihood = value;
      gradient = value;
   }
};

class MinuitFcnGrad : public ROOT::Math::IMultiGradFunction, public RooAbsMinimizerFcn {
public:
   // factory
   template <typename LikelihoodWrapperT = RooFit::TestStatistics::LikelihoodSerial,
             typename LikelihoodGradientWrapperT = RooFit::TestStatistics::LikelihoodGradientSerial>
   static MinuitFcnGrad *
   create(const std::shared_ptr<RooFit::TestStatistics::RooAbsL> &likelihood, RooMinimizer *context,
          std::vector<ROOT::Fit::ParameterSettings> &parameters, bool verbose = false);

   inline ROOT::Math::IMultiGradFunction *Clone() const override { return new MinuitFcnGrad(*this); }

   // override to include gradient strategy synchronization:
   Bool_t Synchronize(std::vector<ROOT::Fit::ParameterSettings> &parameter_settings, Bool_t optConst,
                      Bool_t verbose = kFALSE) override;

   // used inside Minuit:
   inline bool returnsInMinuit2ParameterSpace() const override { return gradient->usesMinuitInternalValues(); }

   inline void setOptimizeConstOnFunction(RooAbsArg::ConstOpCode opcode, Bool_t doAlsoTrackingOpt) override
   {
      likelihood->constOptimizeTestStatistic(opcode, doAlsoTrackingOpt);
   }

private:
   // IMultiGradFunction overrides necessary for Minuit: DoEval, Gradient
   double DoEval(const double *x) const override;

public:
   void Gradient(const double *x, double *grad) const override;

   // part of IMultiGradFunction interface, used widely both in Minuit and in RooFit:
   inline unsigned int NDim() const override { return _nDim; }

   inline std::string getFunctionName() const override { return likelihood->GetName(); }

   inline std::string getFunctionTitle() const override { return likelihood->GetTitle(); }

   inline void setOffsetting(Bool_t flag) override { likelihood->enableOffsetting(flag); }

private:
   template <typename LikelihoodWrapperT /*= RooFit::TestStatistics::LikelihoodJob*/,
             typename LikelihoodGradientWrapperT /*= RooFit::TestStatistics::LikelihoodGradientJob*/>
   MinuitFcnGrad(
      const std::shared_ptr<RooFit::TestStatistics::RooAbsL> &_likelihood, RooMinimizer *context,
      std::vector<ROOT::Fit::ParameterSettings> &parameters, bool verbose,
      LikelihoodWrapperT * /* used only for template deduction */ = static_cast<LikelihoodWrapperT *>(nullptr),
      LikelihoodGradientWrapperT * /* used only for template deduction */ =
         static_cast<LikelihoodGradientWrapperT *>(nullptr));

   // The following three overrides will not actually be used in this class, so they will throw:
   double DoDerivative(const double *x, unsigned int icoord) const override;

   bool syncParameterValuesFromMinuitCalls(const double *x, bool minuit_internal) const;

   // members
   std::shared_ptr<LikelihoodWrapper> likelihood;
   std::shared_ptr<LikelihoodGradientWrapper> gradient;

public:
   mutable std::shared_ptr<WrapperCalculationCleanFlags> calculation_is_clean;

private:
   mutable std::vector<double> minuit_internal_x_;
   mutable std::vector<double> minuit_external_x_;

public:
   mutable bool minuit_internal_roofit_x_mismatch_ = false;
};


/// \param[in] context RooMinimizer that creates and owns this class.
/// \param[in] parameters The vector of ParameterSettings objects that describe the parameters used in the Minuit
/// Fitter. Note that these must match the set used in the Fitter used by \p context! It can be passed in from
/// RooMinimizer with fitter()->Config().ParamsSettings().
template <typename LikelihoodWrapperT, typename LikelihoodGradientWrapperT>
MinuitFcnGrad::MinuitFcnGrad(const std::shared_ptr<RooFit::TestStatistics::RooAbsL> &_likelihood, RooMinimizer *context,
                             std::vector<ROOT::Fit::ParameterSettings> &parameters, bool verbose,
                             LikelihoodWrapperT * /* value unused */, LikelihoodGradientWrapperT * /* value unused */)
   : RooAbsMinimizerFcn(RooArgList(*_likelihood->getParameters()), context, verbose), minuit_internal_x_(NDim(), 0),
     minuit_external_x_(NDim(), 0)
{
   synchronizeParameterSettings(parameters, kTRUE, verbose);

   calculation_is_clean = std::make_shared<WrapperCalculationCleanFlags>();
   likelihood = std::make_shared<LikelihoodWrapperT>(_likelihood, calculation_is_clean /*, _context*/);
   gradient = std::make_shared<LikelihoodGradientWrapperT>(_likelihood, calculation_is_clean, getNDim(), _context);

   likelihood->synchronizeParameterSettings(parameters);
   gradient->synchronizeParameterSettings(this, parameters);

   // Note: can be different than RooGradMinimizerFcn, where default options are passed
   // (ROOT::Math::MinimizerOptions::DefaultStrategy() and ROOT::Math::MinimizerOptions::DefaultErrorDef())
   likelihood->synchronizeWithMinimizer(ROOT::Math::MinimizerOptions());
   gradient->synchronizeWithMinimizer(ROOT::Math::MinimizerOptions());
}

// static function
template <typename LikelihoodWrapperT, typename LikelihoodGradientWrapperT>
MinuitFcnGrad *
MinuitFcnGrad::create(const std::shared_ptr<RooFit::TestStatistics::RooAbsL> &likelihood, RooMinimizer *context,
                      std::vector<ROOT::Fit::ParameterSettings> &parameters, bool verbose)
{
   return new MinuitFcnGrad(likelihood, context, parameters, verbose, static_cast<LikelihoodWrapperT *>(nullptr),
                            static_cast<LikelihoodGradientWrapperT *>(nullptr));
}

} // namespace TestStatistics
} // namespace RooFit

#endif // ROOT_ROOFIT_TESTSTATISTICS_MinuitFcnGrad
