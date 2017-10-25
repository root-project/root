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
// MODIFIED: MnStrategy.h
//#include "Minuit2/MnStrategy.h"
// MODIFIED: Fitter.h
//#include <Fit/Fitter.h>
// MODIFIED: ParameterSettings.h
#include "Fit/ParameterSettings.h"


namespace ROOT {
namespace Math {


class NumericalDerivatorMinuit2 {
public:

  NumericalDerivatorMinuit2();
  NumericalDerivatorMinuit2(const NumericalDerivatorMinuit2 &other);
  NumericalDerivatorMinuit2& operator=(const NumericalDerivatorMinuit2 &other);
  NumericalDerivatorMinuit2(const ROOT::Math::IBaseFunctionMultiDim &f, double step_tolerance, double grad_tolerance, unsigned int ncycles, double error_level);
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

  void SetInitialGradient(std::vector<ROOT::Fit::ParameterSettings>& parameters) const;

private:

  // these are mutable because SetInitialGradient must be const because it's called
  // from InitGradient which is const because DoDerivative must be const because the
  // ROOT::Math::IMultiGradFunction interface requires this
  mutable std::vector <double> fGrd;
  mutable std::vector <double> fG2;
  mutable std::vector <double> fGstep;

  const ROOT::Math::IBaseFunctionMultiDim* fFunction;
  double fStepTolerance;
  double fGradTolerance;
  double fNCycles;
  double fVal;
  unsigned int fN;
  double Up;

  // DIFFERS: eps, eps2
  // Minuit2 determines machine precision itself in MnMachinePrecision.cxx
  double eps;
  double eps2;

};


} // namespace Math
} // namespace ROOT

#endif /* NumericalDerivatorMinuit2_H_ */