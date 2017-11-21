// @(#)root/mathcore:$Id$
// Authors: L. Moneta, J.T. Offermann, E.G.P. Bos    2013-2017
//
/**********************************************************************
 *                                                                    *
 * Copyright (c) 2013 , LCG ROOT MathLib Team                         *
 * Copyright (c) 2017 Patrick Bos, Netherlands eScience Center        *
 *                                                                    *
 **********************************************************************/
/*
 * NumericalDerivatorMinuit2.h
 *
 *  Original version (NumericalDerivator) created on: Aug 14, 2013
 *      Authors: L. Moneta, J. T. Offermann
 *  Modified version (NumericalDerivatorMinuit2) created on: Sep 27, 2017
 *      Author: E. G. P. Bos
 */

#ifndef RooFit_NumericalDerivatorMinuit2
#define RooFit_NumericalDerivatorMinuit2

#ifndef ROOT_Math_IFunctionfwd
#include <Math/IFunctionfwd.h>
#endif

#include <vector>
#include "Fit/ParameterSettings.h"
#include "Minuit2/SinParameterTransformation.h"
#include "Minuit2/SqrtUpParameterTransformation.h"
#include "Minuit2/SqrtLowParameterTransformation.h"
#include "Minuit2/MnMachinePrecision.h"

#include "Minuit2/FunctionGradient.h"


namespace RooFit {

  class NumericalDerivatorMinuit2 {
  public:

    NumericalDerivatorMinuit2();
    NumericalDerivatorMinuit2(const NumericalDerivatorMinuit2 &other);
    NumericalDerivatorMinuit2& operator=(const NumericalDerivatorMinuit2 &other);
    NumericalDerivatorMinuit2(const ROOT::Math::IBaseFunctionMultiDim &f, double step_tolerance, double grad_tolerance, unsigned int ncycles, double error_level);//, double precision);
    //   NumericalDerivatorMinuit2(const ROOT::Math::IBaseFunctionMultiDim &f, const ROOT::Fit::Fitter &fitter);
    //   NumericalDerivatorMinuit2(const ROOT::Math::IBaseFunctionMultiDim &f, const ROOT::Fit::Fitter &fitter, const ROOT::Minuit2::MnStrategy &strategy);
    virtual ~NumericalDerivatorMinuit2();

    ROOT::Minuit2::FunctionGradient Differentiate(const double* x, const std::vector<ROOT::Fit::ParameterSettings>& parameters);
    ROOT::Minuit2::FunctionGradient operator() (const double* x, const std::vector<ROOT::Fit::ParameterSettings>& parameters);
    double GetFValue() const {
      return fVal;
    }
    const double * GetG2() {
      return fG.G2().Data();
    }
    void SetStepTolerance(double value);
    void SetGradTolerance(double value);
    void SetNCycles(int value);

    double Int2ext(const ROOT::Fit::ParameterSettings& parameter, double val) const;
    double Ext2int(const ROOT::Fit::ParameterSettings& parameter, double val) const;
    double DInt2Ext(const ROOT::Fit::ParameterSettings& parameter, double val) const;
    double D2Int2Ext(const ROOT::Fit::ParameterSettings& parameter, double val) const;
    double GStepInt2Ext(const ROOT::Fit::ParameterSettings& parameter, double val) const;

    void SetInitialGradient(std::vector<ROOT::Fit::ParameterSettings>& parameters) const;
    void SetParameterHasLimits(std::vector<ROOT::Fit::ParameterSettings>& parameters) const;

  private:

    const ROOT::Math::IBaseFunctionMultiDim* fFunction;

    double fStepTolerance;
    double fGradTolerance;
    unsigned int fNCycles;
    double Up;
    double fVal;
    unsigned int fN;

    // these are mutable because SetInitialGradient must be const because it's called
    // from InitGradient which is const because DoDerivative must be const because the
    // ROOT::Math::IMultiGradFunction interface requires this
//    mutable std::vector <double> fGrd;
//    mutable std::vector <double> fG2;
//    mutable std::vector <double> fGstep;
    mutable ROOT::Minuit2::FunctionGradient fG;
    mutable ROOT::Minuit2::FunctionGradient fG_internal;

    // same story for SetParameterHasLimits
    mutable std::vector <bool> _parameter_has_limits;


    // MODIFIED: Minuit2 determines machine precision itself in MnMachinePrecision.cxx, but
    //           mathcore isn't linked with minuit, so easier to pass in the correct eps from RooFit.
    //           This means precision is the caller's responsibility, beware!
//  double eps;
//  double eps2;
    // MODIFIED: Minuit2 determines machine precision in a slightly different way than
    // std::numeric_limits<double>::epsilon()). We go with the Minuit2 one.
    ROOT::Minuit2::MnMachinePrecision precision;

    ROOT::Minuit2::SinParameterTransformation fDoubleLimTrafo;
    ROOT::Minuit2::SqrtUpParameterTransformation fUpperLimTrafo;
    ROOT::Minuit2::SqrtLowParameterTransformation fLowerLimTrafo;

  };

} // namespace RooFit

#endif /* NumericalDerivatorMinuit2_H_ */