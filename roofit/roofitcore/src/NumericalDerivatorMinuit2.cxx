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
 * NumericalDerivatorMinuit2.cxx
 *
 *  Original version (NumericalDerivator) created on: Aug 14, 2013
 *      Authors: L. Moneta, J. T. Offermann
 *  Modified version (NumericalDerivatorMinuit2) created on: Sep 27, 2017
 *      Author: E. G. P. Bos
 *
 *      NumericalDerivator was essentially a slightly modified copy of code
 *      written by M. Winkler, F. James, L. Moneta, and A. Zsenei for Minuit2,
 *      Copyright (c) 2005 LCG ROOT Math team, CERN/PH-SFT.
 *
 *      This class attempts to more closely follow the Minuit2 implementation.
 *      Modified things (w.r.t. NumericalDerivator) are indicated by MODIFIED.
 */

#include "NumericalDerivatorMinuit2.h"
#include <cmath>
#include <algorithm>
#include <Math/IFunction.h>
#include <iostream>
#include <TMath.h>
#include <cassert>
#include "Fit/ParameterSettings.h"

#include <Math/Minimizer.h>  // needed here because in Fitter is only a forward declaration


namespace RooFit {

  NumericalDerivatorMinuit2::NumericalDerivatorMinuit2(const ROOT::Math::IBaseFunctionMultiDim &f, bool always_exactly_mimic_minuit2) :
      fFunction(&f),
      fN(f.NDim()),
      fG(f.NDim()),
      _always_exactly_mimic_minuit2(always_exactly_mimic_minuit2)
  {}


  NumericalDerivatorMinuit2::NumericalDerivatorMinuit2(const ROOT::Math::IBaseFunctionMultiDim &f, double step_tolerance, double grad_tolerance, unsigned int ncycles, double error_level, bool always_exactly_mimic_minuit2):
      fFunction(&f),
      fStepTolerance(step_tolerance),
      fGradTolerance(grad_tolerance),
      fNCycles(ncycles),
      Up(error_level),
      fN(f.NDim()),
      fG(f.NDim()),
      _always_exactly_mimic_minuit2(always_exactly_mimic_minuit2)
  {
    //number of dimensions, will look at vector size
    _parameter_has_limits.resize(f.NDim());
  }

// copy constructor
  NumericalDerivatorMinuit2::NumericalDerivatorMinuit2(const RooFit::NumericalDerivatorMinuit2 &other) :
      fFunction(other.fFunction),
      fStepTolerance(other.fStepTolerance),
      fGradTolerance(other.fGradTolerance),
      fNCycles(other.fNCycles),
      Up(other.Up),
      fVal(other.fVal),
      fN(other.fN),
      fG(other.fG),
      _parameter_has_limits(other._parameter_has_limits),
      precision(other.precision),
      _always_exactly_mimic_minuit2(other._always_exactly_mimic_minuit2)
  {}

  RooFit::NumericalDerivatorMinuit2& NumericalDerivatorMinuit2::operator=(const RooFit::NumericalDerivatorMinuit2 &other) {
    if(&other != this) {
      fG = other.fG;
      _parameter_has_limits = other._parameter_has_limits;
      fFunction = other.fFunction;
      fStepTolerance = other.fStepTolerance;
      fGradTolerance = other.fGradTolerance;
      fNCycles = other.fNCycles;
      fVal = other.fVal;
      fN = other.fN;
      Up = other.Up;
      precision = other.precision;
      _always_exactly_mimic_minuit2 = other._always_exactly_mimic_minuit2;
    }
    return *this;
  }

  void NumericalDerivatorMinuit2::SetStepTolerance(double value) {
    fStepTolerance = value;
  }

  void NumericalDerivatorMinuit2::SetGradTolerance(double value) {
    fGradTolerance = value;
  }

  void NumericalDerivatorMinuit2::SetNCycles(int value) {
    fNCycles = value;
  }

  NumericalDerivatorMinuit2::~NumericalDerivatorMinuit2() {
    // TODO Auto-generated destructor stub
  }

  ROOT::Minuit2::FunctionGradient NumericalDerivatorMinuit2::Differentiate(const double* cx,
                                                                           const std::vector<ROOT::Fit::ParameterSettings>& parameters) {
    assert(fFunction != 0);
    assert(fFunction->NDim() == fN);
    std::vector<double> vx(fFunction->NDim()), vx_external(fFunction->NDim());

    std::copy (cx, cx+fFunction->NDim(), vx.data());

    // convert to Minuit external parameters
    for (unsigned i = 0; i < fFunction->NDim(); i++) {
      vx_external[i] = Int2ext(parameters[i], vx[i]);
    }

    fVal = (*fFunction)(vx_external.data());  // value of function at given points

    ROOT::Minuit2::MnAlgebraicVector grad_vec(fG.Grad()),
                                     gr2_vec(fG.G2()),
                                     gstep_vec(fG.Gstep());

    // MODIFIED: Up
    // In Minuit2, this depends on the type of function to minimize, e.g.
    // chi-squared or negative log likelihood. It is set in the RooMinimizer
    // ctor and can be set in the Derivator in the ctor as well using
    // _theFitter->GetMinimizer()->ErrorDef() in the initialization call.
    // const double Up = 1;

    // MODIFIED: two redundant double casts removed, for dfmin and for epspri
    double dfmin = 8. * precision.Eps2() * (std::abs(fVal) + Up);
    double vrysml = 8. * precision.Eps() * precision.Eps();

    for (int i = 0; i < int(fN); i++) {
      double xtf = vx[i];
      double epspri = precision.Eps2() + std::abs(grad_vec(i) * precision.Eps2());
      double step_old = 0.;
      for (unsigned int j = 0; j < fNCycles; ++ j) {
        double optstp = std::sqrt(dfmin/(std::abs(gr2_vec(i))+epspri));
        double step = std::max(optstp, std::abs(0.1*gstep_vec(i)));

        // MODIFIED: in Minuit2 we have here the following condition:
        //   if(Trafo().Parameter(Trafo().ExtOfInt(i)).HasLimits()) {
        // We replaced it by this:
        if (parameters[i].IsBound()) {
          if(step > 0.5) step = 0.5;
        }
        // See the discussion above NumericalDerivatorMinuit2::SetInitialGradient
        // below on how to pass parameter information to this derivator.

        double stpmax = 10.*std::abs(gstep_vec(i));
        if (step > stpmax) step = stpmax;

        double stpmin = std::max(vrysml, 8.*std::abs(precision.Eps2() * vx[i]));
        if (step < stpmin) step = stpmin;
        if (std::abs((step-step_old)/step) < fStepTolerance) {
          break;
        }
        gstep_vec(i) = step;
        step_old = step;
        vx[i] = xtf + step;
        vx_external[i] = Int2ext(parameters[i], vx[i]);
        double fs1 = (*fFunction)(vx_external.data());
        vx[i] = xtf - step;
        vx_external[i] = Int2ext(parameters[i], vx[i]);
        double fs2 = (*fFunction)(vx_external.data());
        vx[i] = xtf;
        vx_external[i] = Int2ext(parameters[i], vx[i]);

        double fGrd_old = grad_vec(i);
        grad_vec(i) = 0.5*(fs1-fs2)/step;

        gr2_vec(i) = (fs1 + fs2 -2.*fVal)/step/step;

        // MODIFIED:
        // The condition below had a closing parenthesis differently than
        // Minuit. Fixed in this version.
        if (std::abs(fGrd_old - grad_vec(i))/(std::abs(grad_vec(i)) + dfmin/step) < fGradTolerance) {
          break;
        }
      }
    }

    fG = ROOT::Minuit2::FunctionGradient(grad_vec, gr2_vec, gstep_vec);
    return fG;
  }

  ROOT::Minuit2::FunctionGradient NumericalDerivatorMinuit2::operator()(const double* x, const std::vector<ROOT::Fit::ParameterSettings>& parameters) {
    return NumericalDerivatorMinuit2::Differentiate(x, parameters);
  }


  void NumericalDerivatorMinuit2::SetParameterHasLimits(std::vector<ROOT::Fit::ParameterSettings>& parameters) const {
    if (_parameter_has_limits.size() != fN) {
      _parameter_has_limits.resize(fN);
    }

    unsigned ix = 0;
    for (auto parameter = parameters.begin(); parameter != parameters.end(); ++parameter, ++ix) {
      _parameter_has_limits[ix] = parameter->IsBound();
    }
  }

  double NumericalDerivatorMinuit2::Int2ext(const ROOT::Fit::ParameterSettings& parameter, double val) const {
    // return external value from internal value for parameter i
    if(parameter.IsBound()) {
      if(parameter.IsDoubleBound()) {
        return fDoubleLimTrafo.Int2ext(val, parameter.UpperLimit(), parameter.LowerLimit());
      } else if (parameter.HasUpperLimit() && !parameter.HasLowerLimit()) {
        return fUpperLimTrafo.Int2ext(val, parameter.UpperLimit());
      } else {
        return fLowerLimTrafo.Int2ext(val, parameter.LowerLimit());
      }
    }

    return val;
  }

  double NumericalDerivatorMinuit2::Ext2int(const ROOT::Fit::ParameterSettings& parameter, double val) const {
    // return the internal value for parameter i with external value val

    if(parameter.IsBound()) {
      if(parameter.IsDoubleBound()) {
        return fDoubleLimTrafo.Ext2int(val, parameter.UpperLimit(), parameter.LowerLimit(), precision);
      } else if (parameter.HasUpperLimit() && !parameter.HasLowerLimit()) {
        return fUpperLimTrafo.Ext2int(val, parameter.UpperLimit(), precision);
      } else {
        return fLowerLimTrafo.Ext2int(val, parameter.LowerLimit(), precision);
      }
    }

    return val;
  }


  double NumericalDerivatorMinuit2::DInt2Ext(const ROOT::Fit::ParameterSettings& parameter, double val) const {
    // return the derivative of the int->ext transformation: dPext(i) / dPint(i)
    // for the parameter i with value val

    double dd = 1.;
    if(parameter.IsBound()) {
      if(parameter.IsDoubleBound()) {
        dd = fDoubleLimTrafo.DInt2Ext(val, parameter.UpperLimit(), parameter.LowerLimit());
      } else if(parameter.HasUpperLimit() && !parameter.HasLowerLimit()) {
        dd = fUpperLimTrafo.DInt2Ext(val, parameter.UpperLimit());
      } else {
        dd = fLowerLimTrafo.DInt2Ext(val, parameter.LowerLimit());
      }
    }

    return dd;
  }


  double NumericalDerivatorMinuit2::D2Int2Ext(const ROOT::Fit::ParameterSettings& parameter, double val) const {
    double dd = 1.;
    if(parameter.IsBound()) {
      if(parameter.IsDoubleBound()) {
        dd = fDoubleLimTrafo.D2Int2Ext(val, parameter.UpperLimit(), parameter.LowerLimit());
      } else if(parameter.HasUpperLimit() && !parameter.HasLowerLimit()) {
        dd = fUpperLimTrafo.D2Int2Ext(val, parameter.UpperLimit());
      } else {
        dd = fLowerLimTrafo.D2Int2Ext(val, parameter.LowerLimit());
      }
    }

    return dd;
  }


  double NumericalDerivatorMinuit2::GStepInt2Ext(const ROOT::Fit::ParameterSettings& parameter, double val) const {
    double dd = 1.;
    if(parameter.IsBound()) {
      if(parameter.IsDoubleBound()) {
        dd = fDoubleLimTrafo.GStepInt2Ext(val, parameter.UpperLimit(), parameter.LowerLimit());
      } else if(parameter.HasUpperLimit() && !parameter.HasLowerLimit()) {
        dd = fUpperLimTrafo.GStepInt2Ext(val, parameter.UpperLimit());
      } else {
        dd = fLowerLimTrafo.GStepInt2Ext(val, parameter.LowerLimit());
      }
    }

    return dd;
  }


  // MODIFIED:
// This function was not implemented as in Minuit2. Now it copies the behavior
// of InitialGradientCalculator. See https://github.com/roofit-dev/root/issues/10
  void NumericalDerivatorMinuit2::SetInitialGradient(std::vector<ROOT::Fit::ParameterSettings>& parameters) const {
    // set an initial gradient using some given steps
    // (used in the first iteration)

    assert(fFunction != 0);
    assert(fFunction->NDim() == fN);

    double eps2 = precision.Eps2();

    ROOT::Minuit2::MnAlgebraicVector grad_vec(fFunction->NDim()),
                                     gr2_vec(fFunction->NDim()),
                                     gstep_vec(fFunction->NDim());

    unsigned ix = 0;
    for (auto parameter = parameters.begin(); parameter != parameters.end(); ++parameter, ++ix) {
      // What Minuit2 calls "Error" is stepsize on the ROOT side.
      double werr = parameter->StepSize();

      // Actually, sav in Minuit2 is the external parameter value, so that is
      // what we called var before and var is unnecessary here.
      double sav = parameter->Value();

      // However, we do need var below, so let's calculate it using Ext2int:
      double var = Ext2int(*parameter, sav);

      if (_always_exactly_mimic_minuit2) {
        // this transformation can lose a few bits, but Minuit2 does it too
        sav = Int2ext(*parameter, var);
      }

      double sav2 = sav + werr;
      if(parameter->HasUpperLimit() && sav2 > parameter->UpperLimit()) {
        sav2 = parameter->UpperLimit();
      }

      double var2 = Ext2int(*parameter, sav2);
      double vplu = var2 - var;
      sav2 = sav - werr;
      if(parameter->HasLowerLimit() && sav2 < parameter->LowerLimit()) {
        sav2 = parameter->LowerLimit();
      }

      var2 = Ext2int(*parameter, sav2);
      double vmin = var2 - var;
      double gsmin = 8. * eps2 * (fabs(var) + eps2);
      // protect against very small step sizes which can cause dirin to zero and then nan values in grd
      double dirin = std::max(0.5*(fabs(vplu) + fabs(vmin)),  gsmin );
      double g2 = 2.0*Up/(dirin*dirin);
      double gstep = std::max(gsmin, 0.1*dirin);
      double grd = g2*dirin;

      if(parameter->IsBound()) {
        if(gstep > 0.5) gstep = 0.5;
      }

      grad_vec(ix) = grd;
      gr2_vec(ix) = g2;
      gstep_vec(ix) = gstep;

    }

    fG = ROOT::Minuit2::FunctionGradient(grad_vec, gr2_vec, gstep_vec);
  }

  bool NumericalDerivatorMinuit2::always_exactly_mimic_minuit2() const {
    return _always_exactly_mimic_minuit2;
  };

  void NumericalDerivatorMinuit2::set_step_tolerance(double step_tolerance) {
    fStepTolerance = step_tolerance;
  }
  void NumericalDerivatorMinuit2::set_grad_tolerance(double grad_tolerance) {
    fGradTolerance = grad_tolerance;
  }
  void NumericalDerivatorMinuit2::set_ncycles(unsigned int ncycles) {
    fNCycles = ncycles;
  }
  void NumericalDerivatorMinuit2::set_error_level(double error_level) {
    Up = error_level;
  }

  void NumericalDerivatorMinuit2::set_fN(unsigned int fN_new) {
    fN = fN_new;
  }

} // namespace RooFit


