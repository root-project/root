// @(#)root/mathcore:$Id$
// Authors: L. Moneta, J.T. Offermann, E.G.P. Bos    2013-2018
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

#include <RooTimer.h>
#include <RooMsgService.h>

//#include <MultiProcess/TaskManager.h>

namespace RooFit {

  NumericalDerivatorMinuit2::NumericalDerivatorMinuit2(const ROOT::Math::IBaseFunctionMultiDim *f, ROOT::Minuit2::FunctionGradient & grad, bool always_exactly_mimic_minuit2) :
      fFunction(f),
      fN(f->NDim()),
      fG(grad),
      _always_exactly_mimic_minuit2(always_exactly_mimic_minuit2)
  {}


  NumericalDerivatorMinuit2::NumericalDerivatorMinuit2(const ROOT::Math::IBaseFunctionMultiDim *f, ROOT::Minuit2::FunctionGradient & grad, double step_tolerance, double grad_tolerance, unsigned int ncycles, double error_level, bool always_exactly_mimic_minuit2):
      fFunction(f),
      fStepTolerance(step_tolerance),
      fGradTolerance(grad_tolerance),
      fNCycles(ncycles),
      Up(error_level),
      fN(f->NDim()),
      fG(grad),
      _always_exactly_mimic_minuit2(always_exactly_mimic_minuit2)
  {}

  // deep copy constructor
  NumericalDerivatorMinuit2::NumericalDerivatorMinuit2(const RooFit::NumericalDerivatorMinuit2 &other, ROOT::Minuit2::FunctionGradient & grad) :
      fFunction(other.fFunction),
      fStepTolerance(other.fStepTolerance),
      fGradTolerance(other.fGradTolerance),
      fNCycles(other.fNCycles),
      Up(other.Up),
      fVal(other.fVal),
      fN(other.fN),
      fG(grad),
      vx(other.vx),
      vx_external(other.vx_external),
      dfmin(other.dfmin),
      vrysml(other.vrysml),
      precision(other.precision),
      _always_exactly_mimic_minuit2(other._always_exactly_mimic_minuit2),
      vx_fVal_cache(other.vx_fVal_cache)
  {}

  // Almost deep copy constructor, except for fFunction.
  // This ctor is used for cloning when the fFunction has just been (deep)
  // copied and it must then be passed here from the initialization list.
  NumericalDerivatorMinuit2::NumericalDerivatorMinuit2(const RooFit::NumericalDerivatorMinuit2 &other, ROOT::Minuit2::FunctionGradient & grad, const ROOT::Math::IBaseFunctionMultiDim *f) :
      fFunction(f),
      fStepTolerance(other.fStepTolerance),
      fGradTolerance(other.fGradTolerance),
      fNCycles(other.fNCycles),
      Up(other.Up),
      fVal(other.fVal),
      fN(other.fN),
      fG(grad),
      vx(other.vx),
      vx_external(other.vx_external),
      dfmin(other.dfmin),
      vrysml(other.vrysml),
      precision(other.precision),
      _always_exactly_mimic_minuit2(other._always_exactly_mimic_minuit2),
      vx_fVal_cache(other.vx_fVal_cache)
  {}

  // an operator= copy ctor doesn't make sense with const members...
//  RooFit::NumericalDerivatorMinuit2& NumericalDerivatorMinuit2::operator=(const RooFit::NumericalDerivatorMinuit2 &other) {
//    if(&other != this) {
//      fG = other.fG;
//      _parameter_has_limits = other._parameter_has_limits;
//      fFunction = other.fFunction;
//      fStepTolerance = other.fStepTolerance;
//      fGradTolerance = other.fGradTolerance;
//      fNCycles = other.fNCycles;
//      fVal = other.fVal;
//      fN = other.fN;
//      Up = other.Up;
//      precision = other.precision;
//      _always_exactly_mimic_minuit2 = other._always_exactly_mimic_minuit2;
//    }
//    return *this;
//  }

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

  // This function sets internal state based on input parameters. This state
  // setup is used in the actual (partial) derivative calculations.
  void NumericalDerivatorMinuit2::setup_differentiate(const double* cx,
                                                      const std::vector<ROOT::Fit::ParameterSettings>& parameters) {
    assert(fFunction != 0);
    assert(fFunction->NDim() == fN);

    auto get_time = [](){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();};
    decltype(get_time()) t1, t2, t3, t4, t5, t6, t7, t8;

    t1 = get_time();
    if (vx.size() != fFunction->NDim()) {
      vx.resize(fFunction->NDim());
    }
    t2 = get_time();
    if (vx_external.size() != fFunction->NDim()) {
      vx_external.resize(fFunction->NDim());
    }
    t3 = get_time();
    if (vx_fVal_cache.size() != fFunction->NDim()) {
      vx_fVal_cache.resize(fFunction->NDim());
    }
    t4 = get_time();

    std::copy(cx, cx + fFunction->NDim(), vx.data());
    t5 = get_time();

    // convert to Minuit external parameters
    for (unsigned i = 0; i < fFunction->NDim(); i++) {
      vx_external[i] = Int2ext(parameters[i], vx[i]);
    }

    t6 = get_time();

    if (vx != vx_fVal_cache) {
//#ifndef NDEBUG
//      ++fVal_eval_counter;
//#endif
      vx_fVal_cache = vx;
      fVal = (*fFunction)(vx_external.data());  // value of function at given points
//#ifndef NDEBUG
//      std::cout << "NumericalDerivatorMinuit2::setup_differentiate, fVal evaluations: " << fVal_eval_counter << std::endl;
//#endif
    }
    t7 = get_time();

    dfmin = 8. * precision.Eps2() * (std::abs(fVal) + Up);
    vrysml = 8. * precision.Eps() * precision.Eps();

    t8 = get_time();

//    if (RooFit::MultiProcess::TaskManager::is_instantiated()) {
//      oocxcoutD((TObject *) nullptr, Benchmarking2) << "NumericalDerivatorMinuit2::setup_differentiate on worker "
//                                                    << MultiProcess::TaskManager::instance()->get_worker_id()
//                                                    << ", timestamps: " << t1 << " " << t2 << " " << t3 << " " << t4
//                                                    << " " << t5 << " " << t6 << " " << t7 << " " << t8 << std::endl;
//    }
  }

  std::tuple<double, double, double> NumericalDerivatorMinuit2::partial_derivative(const double *x, const std::vector<ROOT::Fit::ParameterSettings>& parameters, unsigned int i_component) {
    setup_differentiate(x, parameters);
    do_fast_partial_derivative(parameters, i_component);
    return {fG.Grad()(i_component), fG.G2()(i_component), fG.Gstep()(i_component)};
  }

  // leaves the parameter setup to the caller
  void NumericalDerivatorMinuit2::do_fast_partial_derivative(const std::vector<ROOT::Fit::ParameterSettings>& parameters, unsigned int ix) {
    double xtf = vx[ix];
    double epspri = precision.Eps2() + std::abs(fG.Grad()(ix) * precision.Eps2());
    double step_old = 0.;
    for (unsigned int j = 0; j < fNCycles; ++ j) {
      double optstp = std::sqrt(dfmin/(std::abs(fG.G2()(ix))+epspri));
      double step = std::max(optstp, std::abs(0.1*fG.Gstep()(ix)));

      if (parameters[ix].IsBound()) {
        if(step > 0.5) step = 0.5;
      }

      double stpmax = 10.*std::abs(fG.Gstep()(ix));
      if (step > stpmax) step = stpmax;

      double stpmin = std::max(vrysml, 8.*std::abs(precision.Eps2() * vx[ix]));
      if (step < stpmin) step = stpmin;
      if (std::abs((step-step_old)/step) < fStepTolerance) {
        break;
      }
      mutable_gstep()(ix) = step;
      step_old = step;
      vx[ix] = xtf + step;
      vx_external[ix] = Int2ext(parameters[ix], vx[ix]);
      double fs1 = (*fFunction)(vx_external.data());
      vx[ix] = xtf - step;
      vx_external[ix] = Int2ext(parameters[ix], vx[ix]);
      double fs2 = (*fFunction)(vx_external.data());
      vx[ix] = xtf;
      vx_external[ix] = Int2ext(parameters[ix], vx[ix]);

      double fGrd_old = fG.Grad()(ix);
      mutable_grad()(ix) = 0.5*(fs1-fs2)/step;

      mutable_g2()(ix) = (fs1 + fs2 -2.*fVal)/step/step;

      if (std::abs(fGrd_old - fG.Grad()(ix))/(std::abs(fG.Grad()(ix)) + dfmin/step) < fGradTolerance) {
        break;
      }
    }
  }

  std::tuple<double, double, double> NumericalDerivatorMinuit2::operator()(const double *x, const std::vector<ROOT::Fit::ParameterSettings>& parameters, unsigned int i_component) {
    return partial_derivative(x, parameters, i_component);
  }


  ROOT::Minuit2::FunctionGradient NumericalDerivatorMinuit2::Differentiate(const double* cx,
                                                                           const std::vector<ROOT::Fit::ParameterSettings>& parameters) {
    setup_differentiate(cx, parameters);

    for (int ix = 0; ix < int(fN); ++ix) {
      do_fast_partial_derivative(parameters, ix);
    }

    return fG;
  }

  ROOT::Minuit2::FunctionGradient NumericalDerivatorMinuit2::operator()(const double* x, const std::vector<ROOT::Fit::ParameterSettings>& parameters) {
    return NumericalDerivatorMinuit2::Differentiate(x, parameters);
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
  void NumericalDerivatorMinuit2::SetInitialGradient(std::vector<ROOT::Fit::ParameterSettings>& parameters) {
    // set an initial gradient using some given steps
    // (used in the first iteration)
    auto get_time = [](){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();};
    decltype(get_time()) t1, t2;

    RooWallTimer timer;
    t1 = get_time();

    assert(fFunction != 0);
    assert(fFunction->NDim() == fN);

    double eps2 = precision.Eps2();

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

      mutable_grad()(ix) = grd;
      mutable_g2()(ix) = g2;
      mutable_gstep()(ix) = gstep;
    }

    t2 = get_time();
    timer.stop();
    oocxcoutD((TObject*)nullptr,Benchmarking1) << "SetInitialGradient time: " << timer.timing_s() << "s (from " << t1 << " to " << t2 << "ns)" << std::endl;

  }

  bool NumericalDerivatorMinuit2::always_exactly_mimic_minuit2() const {
    return _always_exactly_mimic_minuit2;
  };

  void NumericalDerivatorMinuit2::set_always_exactly_mimic_minuit2(bool flag) {
    _always_exactly_mimic_minuit2 = flag;
  }

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

  ROOT::Minuit2::MnAlgebraicVector& NumericalDerivatorMinuit2::mutable_grad() const {
    return const_cast<ROOT::Minuit2::MnAlgebraicVector &>(fG.Grad());
  }
  ROOT::Minuit2::MnAlgebraicVector& NumericalDerivatorMinuit2::mutable_g2() const {
    return const_cast<ROOT::Minuit2::MnAlgebraicVector &>(fG.G2());
  }
  ROOT::Minuit2::MnAlgebraicVector& NumericalDerivatorMinuit2::mutable_gstep() const {
    return const_cast<ROOT::Minuit2::MnAlgebraicVector &>(fG.Gstep());
  }

} // namespace RooFit


