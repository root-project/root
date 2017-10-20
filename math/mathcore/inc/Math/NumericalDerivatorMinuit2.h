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
#include "Minuit2/MnStrategy.h"
// MODIFIED: Fitter.h
#include <Fit/Fitter.h>


namespace ROOT {
namespace Math {


class NumericalDerivatorMinuit2 {
public:

   NumericalDerivatorMinuit2();
   NumericalDerivatorMinuit2(const NumericalDerivatorMinuit2 &other);
   NumericalDerivatorMinuit2(const ROOT::Math::IBaseFunctionMultiDim &f, double step_tolerance, double grad_tolerance, unsigned int ncycles, double error_level);
   NumericalDerivatorMinuit2(const ROOT::Math::IBaseFunctionMultiDim &f, const ROOT::Fit::Fitter &fitter);
   NumericalDerivatorMinuit2(const ROOT::Math::IBaseFunctionMultiDim &f, const ROOT::Fit::Fitter &fitter, const ROOT::Minuit2::MnStrategy &strategy);
   virtual ~NumericalDerivatorMinuit2();
   const double* Differentiate(const double* x);
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
       
   void SetInitialGradient();

private:

    std::vector <double> fGrd;
    std::vector <double> fG2;
    std::vector <double> fGstep;
    const ROOT::Math::IBaseFunctionMultiDim* fFunction;
    ROOT::Minuit2::MnStrategy _strategy;
    double fStepTolerance;
    double fGradTolerance;
    double fNCycles;
    double fVal;
    unsigned int fN;
    const double Up;

};


} // namespace Math
} // namespace ROOT

#endif /* NumericalDerivatorMinuit2_H_ */