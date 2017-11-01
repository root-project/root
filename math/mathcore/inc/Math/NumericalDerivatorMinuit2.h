// @(#)root/mathcore:$Id$
// Authors: L. Moneta, J.T. Offermann    08/2013 
//          E.G.P. Bos    09/2017
/**********************************************************************
 *                                                                    *
 * Copyright (c) 2013 , LCG ROOT MathLib Team                         *
 * Copyright (c) 2017 , Netherlands eScience Center                   *
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

#ifndef ROOT_Math_NumericalDerivatorMinuit2
#define ROOT_Math_NumericalDerivatorMinuit2

#ifndef ROOT_Math_IFunctionfwd
#include <Math/IFunctionfwd.h>
#endif

#include <vector>
#include "Fit/ParameterSettings.h"
#include "Minuit2/SinParameterTransformation.h"
#include "Minuit2/SqrtUpParameterTransformation.h"
#include "Minuit2/SqrtLowParameterTransformation.h"


namespace ROOT {
namespace Math {


class NumericalDerivatorMinuit2 {
public:

  NumericalDerivatorMinuit2();
  NumericalDerivatorMinuit2(const NumericalDerivatorMinuit2 &other);
  NumericalDerivatorMinuit2& operator=(const NumericalDerivatorMinuit2 &other);
  NumericalDerivatorMinuit2(const ROOT::Math::IBaseFunctionMultiDim &f, double step_tolerance, double grad_tolerance, unsigned int ncycles, double error_level, double precision);
  //   NumericalDerivatorMinuit2(const ROOT::Math::IBaseFunctionMultiDim &f, const ROOT::Fit::Fitter &fitter);
  //   NumericalDerivatorMinuit2(const ROOT::Math::IBaseFunctionMultiDim &f, const ROOT::Fit::Fitter &fitter, const ROOT::Minuit2::MnStrategy &strategy);
  virtual ~NumericalDerivatorMinuit2();

  std::vector<double> Differentiate(const double* x);
  std::vector<double> operator() (const double* x);
  double GetFValue() const {
    return fVal;
  }
  const double * GetG2() {
    return &fG2[0];
  }
  void SetStepTolerance(double value);
  void SetGradTolerance(double value);
  void SetNCycles(int value);

  void SetInitialValues(const double* g, const double* g2, const double* s);

  double Int2ext(const ROOT::Fit::ParameterSettings& parameter, double val) const;
  double Ext2int(const ROOT::Fit::ParameterSettings& parameter, double val) const;

  void SetInitialGradient(std::vector<ROOT::Fit::ParameterSettings>& parameters) const;
  void SetParameterHasLimits(std::vector<ROOT::Fit::ParameterSettings>& parameters) const;

private:

  // these are mutable because SetInitialGradient must be const because it's called
  // from InitGradient which is const because DoDerivative must be const because the
  // ROOT::Math::IMultiGradFunction interface requires this
  mutable std::vector <double> fGrd;
  mutable std::vector <double> fG2;
  mutable std::vector <double> fGstep;
  // same story for SetParameterHasLimits
  mutable std::vector <bool> _parameter_has_limits;

  const ROOT::Math::IBaseFunctionMultiDim* fFunction;
  double fStepTolerance;
  double fGradTolerance;
  double fNCycles;
  double fVal;
  unsigned int fN;
  double Up;

  // MODIFIED: Minuit2 determines machine precision itself in MnMachinePrecision.cxx, but
  //           mathcore isn't linked with minuit, so easier to pass in the correct eps from RooFit.
  //           This means precision is the caller's responsibility, beware!
  double eps;
  double eps2;

  SinParameterTransformation fDoubleLimTrafo;
  SqrtUpParameterTransformation fUpperLimTrafo;
  SqrtLowParameterTransformation fLowerLimTrafo;

};


} // namespace Math
} // namespace ROOT

#endif /* NumericalDerivatorMinuit2_H_ */