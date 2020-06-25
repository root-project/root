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

#include <vector>

#include "Fit/FitResult.h"
#include "Minuit2/MnStrategy.h"
#include "TMatrixDSym.h"
#include "Math/IFunction.h"
#include "RooGradientFunction.h"
#include "Fit/ParameterSettings.h"

#include "RooAbsMinimizerFcn.h"

class RooGradMinimizerFcn : public ROOT::Math::IMultiGradFunction, public RooAbsMinimizerFcn {
public:
   RooGradMinimizerFcn(RooAbsReal *funct, RooMinimizer *context, bool verbose = false);
   RooGradMinimizerFcn(const RooGradMinimizerFcn& other);
   ROOT::Math::IMultiGradFunction *Clone() const override;

   ROOT::Minuit2::MnStrategy get_strategy() const;
   double get_error_def() const;
   void set_strategy(int istrat);

   Bool_t Synchronize(std::vector<ROOT::Fit::ParameterSettings> &parameter_settings, Bool_t optConst,
                      Bool_t verbose = kFALSE) override;

   void synchronize_gradient_parameter_settings(std::vector<ROOT::Fit::ParameterSettings> &parameter_settings) const;

   bool returnsInMinuit2ParameterSpace() const override;

   unsigned int NDim() const override;

   void set_step_tolerance(double step_tolerance) const;
   void set_grad_tolerance(double grad_tolerance) const;
   void set_ncycles(unsigned int ncycles) const;
   void set_error_level(double error_level) const;

   std::string getFunctionName() const override;
   std::string getFunctionTitle() const override;

   void setOptimizeConst(Int_t flag) override;

private:
   void run_derivator(unsigned int i_component) const;

   bool sync_parameter(double x, std::size_t ix) const;
   bool sync_parameters(const double *x) const;

   void optimizeConstantTerms(bool constStatChange, bool constValChange) override;

public:
   enum class GradientCalculatorMode {
      ExactlyMinuit2, AlmostMinuit2
   };

protected:
   // accessors for the const data members of _grad
   // TODO: find out why FunctionGradient keeps its data const.. but work around it in the meantime
   ROOT::Minuit2::MnAlgebraicVector& mutable_grad() const;
   ROOT::Minuit2::MnAlgebraicVector& mutable_g2() const;
   ROOT::Minuit2::MnAlgebraicVector& mutable_gstep() const;

private:
   // IMultiGradFunction overrides
   double DoEval(const double *x) const override;
   double DoDerivative(const double *x, unsigned int icoord) const override;
   bool hasG2ndDerivative() const override;
   double DoSecondDerivative(const double *x, unsigned int icoord) const override;
   bool hasGStepSize() const override;
   double DoStepSize(const double *x, unsigned int icoord) const override;

   // members
   // mutable because ROOT::Math::IMultiGradFunction::DoDerivative is const
protected:
   // CAUTION: do not move _grad below _gradf, as it is needed for _gradf construction
   mutable ROOT::Minuit2::FunctionGradient _grad;
   mutable std::vector<double> _grad_params;
private:
   // CAUTION: do not move _gradf above _grad, as it is needed for _gradf construction
   mutable RooFit::NumericalDerivatorMinuit2 _gradf;
   RooAbsReal *_funct;
   mutable std::vector<bool> has_been_calculated;
   mutable bool none_have_been_calculated = false;
};
#endif
#endif
