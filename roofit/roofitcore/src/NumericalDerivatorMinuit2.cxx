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

  NumericalDerivatorMinuit2::NumericalDerivatorMinuit2() :
      fFunction(0),
      fStepTolerance(0.5),
      fGradTolerance(0.1),
      fNCycles(2),
      Up(1),
      fVal(0),
      fN(0),
      fG(0)
//      fG_internal(0)
//   eps(std::numeric_limits<double>::epsilon()),
//   eps2(2 * std::sqrt(eps))
  {}


  NumericalDerivatorMinuit2::NumericalDerivatorMinuit2(const ROOT::Math::IBaseFunctionMultiDim &f, double step_tolerance, double grad_tolerance, unsigned int ncycles, double error_level)://, double precision):
      fFunction(&f),
      fStepTolerance(step_tolerance),
      fGradTolerance(grad_tolerance),
      fNCycles(ncycles),
      Up(error_level),
      fVal(0),
      fN(f.NDim()),
      fG(f.NDim())
//      fG_internal(f.NDim())
//   eps(precision),
//   eps2(2 * std::sqrt(eps))
  {
    // constructor with function, and tolerances (coordinates must be specified for differentiate function, not constructor)
//    fStepTolerance=step_tolerance;
//    fGradTolerance=grad_tolerance;
//    fFunction=&f;

    ; //number of dimensions, will look at vector size
//    fGrd.resize(fN);
//    fGstep.resize(fN);
//    fG2.resize(fN);
    _parameter_has_limits.resize(fN);

//    for (unsigned int i = 0; i<fN; i++) {
//      fGrd[i]=0.1;
//      fG2[i]=0.1;
//      fGstep[i]=0.001;
//    }
//    fVal = 0;
  }

// copy constructor
  NumericalDerivatorMinuit2::NumericalDerivatorMinuit2(const RooFit::NumericalDerivatorMinuit2 &other) :
//      fGrd(other.fGrd),
//      fG2(other.fG2),
//      fGstep(other.fGstep),
      fFunction(other.fFunction),
      fStepTolerance(other.fStepTolerance),
      fGradTolerance(other.fGradTolerance),
      fNCycles(other.fNCycles),
      Up(other.Up),
      fVal(other.fVal),
      fN(other.fN),
      fG(other.fG),
//      fG_internal(other.fG_internal),
      _parameter_has_limits(other._parameter_has_limits),
      precision(other.precision)
//    eps(other.eps),
//    eps2(other.eps2)
  {}

  RooFit::NumericalDerivatorMinuit2& NumericalDerivatorMinuit2::operator=(const RooFit::NumericalDerivatorMinuit2 &other) {
    if(&other != this) {
//      fGrd = other.fGrd;
//      fG2 = other.fG2;
//      fGstep = other.fGstep;
      fG = other.fG;
//      fG_internal = other.fG_internal;
      _parameter_has_limits = other._parameter_has_limits;
      fFunction = other.fFunction;
      fStepTolerance = other.fStepTolerance;
      fGradTolerance = other.fGradTolerance;
      fNCycles = other.fNCycles;
      fVal = other.fVal;
      fN = other.fN;
      Up = other.Up;
      precision = other.precision;
//    eps = other.eps;
//    eps2 = other.eps2;
    }
    return *this;
  }


  // MODIFIED: ctors with higher level arguments
// The parameters can be extracted from a ROOT::Fit::Fitter object, for
// simpler initialization.
//NumericalDerivatorMinuit2::NumericalDerivatorMinuit2(const ROOT::Math::IBaseFunctionMultiDim &f, const ROOT::Fit::Fitter &fitter) :
//    NumericalDerivatorMinuit2::NumericalDerivatorMinuit2(f, fitter,  ROOT::Minuit2::MnStrategy(fitter.GetMinimizer()->Strategy()))
//{}
//
//NumericalDerivatorMinuit2::NumericalDerivatorMinuit2(const ROOT::Math::IBaseFunctionMultiDim &f, const ROOT::Fit::Fitter &fitter, const ROOT::Minuit2::MnStrategy &strategy) :
//    NumericalDerivatorMinuit2::NumericalDerivatorMinuit2(f,
//                                                         strategy.GradientStepTolerance(),
//                                                         strategy.GradientTolerance(),
//                                                         strategy.GradientNCycles(),
//                                                         fitter.GetMinimizer()->ErrorDef()
//    )
//{}


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

  ROOT::Minuit2::FunctionGradient NumericalDerivatorMinuit2::Differentiate(const double* cx, const std::vector<ROOT::Fit::ParameterSettings>& parameters) {
//    std::cout << std::endl << "Start:" << std::endl;
//    for (unsigned int i = 0; i<fN; i++) {
//      std::cout << "fGrd[" << i <<"] = " << fGrd[i] << "\t";
//      std::cout << "fG2[" << i <<"] = " << fG2[i] << "\t";
//      std::cout << "fGstep[" << i <<"] = " << fGstep[i] << std::endl;
//    }

//    std::cout << "########### NumericalDerivatorMinuit2::Differentiate()" <<std::endl;



    assert(fFunction != 0);
    std::vector<double> vx(fFunction->NDim()), vx_external(fFunction->NDim());
    assert (vx.size() > 0);

//    double *x = &vx[0];
    std::copy (cx, cx+fFunction->NDim(), vx.data());
    std::copy (cx, cx+fFunction->NDim(), vx_external.data());

//    std::cout << "cx: ("; EGP
//    for (int i = 0; i < int(fN); i++) {
//      std::cout << "ext: " << cx[i] << "; ";
//    }
//    std::cout << std::endl;

    // convert to Minuit external parameters
//    std::cout << "vx_external: ("; EGP
    for (int i = 0; i < int(fN); i++) {
//      std::cout << "int: " << vx_external[i] << "\t";
      vx_external[i] = Int2ext(parameters[i], vx[i]);
//      std::cout << "ext: " << vx_external[i] << "; ";
    }
//    std::cout << std::endl;

    double step_tolerance = fStepTolerance;
    double grad_tolerance = fGradTolerance;
    const ROOT::Math::IBaseFunctionMultiDim &f = *fFunction;
    fVal = f(vx_external.data()); //value of function at given points

//    std::cout << std::hexfloat << "fval= " << fVal << std::endl; EGP


    ROOT::Minuit2::MnAlgebraicVector grad_vec(fG.Grad()),
                                     gr2_vec(fG.G2()),
                                     gstep_vec(fG.Gstep());
//    ROOT::Minuit2::MnAlgebraicVector grad_vec_internal(fG_internal.Grad()),
//                                     gr2_vec_internal(fG_internal.G2()),
//                                     gstep_vec_internal(fG_internal.Gstep());

    // MODIFIED: Up
    // In Minuit2, this depends on the type of function to minimize, e.g.
    // chi-squared or negative log likelihood. It is set in the RooMinimizer
    // ctor and can be set in the Derivator in the ctor as well using
    // _theFitter->GetMinimizer()->ErrorDef() in the initialization call.
    // const double Up = 1;

    double eps = precision.Eps();
    double eps2 = precision.Eps2();

//    std::cout<< std::hexfloat<<"eps= "<<eps<<std::endl;
//    std::cout<< std::hexfloat<<"eps2= "<<eps2<<std::endl; EGP

    // MODIFIED: two redundant double casts removed, for dfmin and for epspri
    double dfmin = 8. * eps2 * (std::abs(fVal) + Up);
    double vrysml = 8.*eps*eps;
    unsigned int ncycle = fNCycles;

//    std::cout<< std::hexfloat<<"dfmin= "<<dfmin<<std::endl;
//    std::cout<< std::hexfloat<<"vrysml= "<<vrysml<<std::endl;
//    std::cout << " ncycle " << ncycle << std::endl; EGP

//    for (int i = 0; i < fN; ++i) {
//      std::cout << std::hexfloat << "x=("<< vx[i] << ",\t";
//    }
//    std::cout << ")" << std::endl; EGP

    for (int i = 0; i < int(fN); i++) {

//      std::cout << "BEFORE, EXTERNAL: fGrd[" << i <<"] = " << grad_vec(i) << "\t";
//      std::cout << "fG2[" << i <<"] = " << gr2_vec(i) << "\t";
//      std::cout << "fGstep[" << i <<"] = " << gstep_vec(i) << "\t";
//      std::cout << "x[" << i << "] = " << vx_external[i] << "\t";
//      std::cout << "fVal = " << fVal << "\t";
//      std::cout << std::endl;

//      std::cout << "BEFORE, INTERNAL: fGrd[" << i <<"] = " << grad_vec_internal(i) << "\t";
//      std::cout << "fG2[" << i <<"] = " << gr2_vec_internal(i) << "\t";
//      std::cout << "fGstep[" << i <<"] = " << gstep_vec_internal(i) << "\t";
//      std::cout << "x[" << i << "] = " << vx[i] << "\t";
//      std::cout << "fVal = " << fVal << "\t";
//      std::cout << std::endl;

      double xtf = vx[i];
//      double epspri = eps2 + std::abs(grad_vec_internal(i) * eps2);
      double epspri = eps2 + std::abs(grad_vec(i) * eps2);
      double step_old = 0.;
      for (unsigned int j = 0; j < ncycle; ++ j) {

//        double optstp = std::sqrt(dfmin/(std::abs(gr2_vec_internal(i))+epspri));
        double optstp = std::sqrt(dfmin/(std::abs(gr2_vec(i))+epspri));
//        double step = std::max(optstp, std::abs(0.1*gstep_vec_internal(i)));
        double step = std::max(optstp, std::abs(0.1*gstep_vec(i)));

        // MODIFIED: in Minuit2 we have here the following condition:
        //   if(Trafo().Parameter(Trafo().ExtOfInt(i)).HasLimits()) {
        // We replaced it by this:
        if (parameters[i].IsBound()) {
          if(step > 0.5) step = 0.5;
        }
        // See the discussion above NumericalDerivatorMinuit2::SetInitialGradient
        // below on how to pass parameter information to this derivator.

//        double stpmax = 10.*std::abs(gstep_vec_internal(i));
        double stpmax = 10.*std::abs(gstep_vec(i));
        if (step > stpmax) step = stpmax;

        double stpmin = std::max(vrysml, 8.*std::abs(eps2*vx[i])); //8.*std::abs(double(eps2*x[i]))
        if (step < stpmin) step = stpmin;
        if (std::abs((step-step_old)/step) < step_tolerance) {
          //answer = fGrd[i];
          break;
        }
//        gstep_vec_internal(i) = step;
        gstep_vec(i) = step;
//        std::cout<< std::hexfloat<<"step= "<<step<<std::endl; EGP
        step_old = step;
        // std::cout << "step = " << step << std::endl;
        vx[i] = xtf + step;
//        std::cout<< std::hexfloat<<"x(i)= "<<vx[i]<<std::endl; EGP
        vx_external[i] = Int2ext(parameters[i], vx[i]);
        //std::cout << "x[" << i << "] = " << x[i] <<std::endl;
        double fs1 = f(vx_external.data());
//        std::cout<< std::hexfloat<<"fs1= "<<fs1<<std::endl;
        //std::cout << "xtf + step = " << x[i] << ", fs1 = " << fs1 << std::endl;
        vx[i] = xtf - step;
//        std::cout<< std::hexfloat<<"x(i)= "<<vx[i]<<std::endl;
        vx_external[i] = Int2ext(parameters[i], vx[i]);
        double fs2 = f(vx_external.data());
//        std::cout<< std::hexfloat<<"fs2= "<<fs2<<std::endl;
        //std::cout << "xtf - step = " << x[i] << ", fs2 = " << fs2 << std::endl;
        vx[i] = xtf;
//        std::cout<< std::hexfloat<<"x(i)= "<<vx[i]<<std::endl;
        vx_external[i] = Int2ext(parameters[i], vx[i]);

//        double fGrd_old = grad_vec_internal(i);
        double fGrd_old = grad_vec(i);
//        std::cout<< std::hexfloat<<"grdb4= "<<fGrd_old<<std::endl;
//        grad_vec_internal(i) = 0.5*(fs1-fs2)/step;
        grad_vec(i) = 0.5*(fs1-fs2)/step;

//        std::cout<< std::hexfloat<<"grd(i)= "<<grad_vec_internal(i)<<std::endl;
//            std::cout << "int i = " << i << std::endl;
//            std::cout << "fs1 = " << fs1 << std::endl;
//            std::cout << "fs2 = " << fs2 << std::endl;
//            std::cout << "fVal = " << fVal << std::endl;
//            std::cout << "step^2 = " << (step*step) << std::endl;
//            std::cout << std::endl;
//        gr2_vec_internal(i) = (fs1 + fs2 -2.*fVal)/step/step;
        gr2_vec(i) = (fs1 + fs2 -2.*fVal)/step/step;
//        std::cout<< std::hexfloat<<"g2(i)= "<<gr2_vec_internal(i)<<std::endl;

        // MODIFIED:
        // The condition below had a closing parenthesis differently than
        // Minuit. Fixed in this version.
        // if (std::abs(fGrd_old-fGrd[i])/(std::abs(fGrd[i]+dfmin/step)) < grad_tolerance)
//        if (std::abs(fGrd_old - grad_vec_internal(i))/(std::abs(grad_vec_internal(i)) + dfmin/step) < grad_tolerance)
        if (std::abs(fGrd_old - grad_vec(i))/(std::abs(grad_vec(i)) + dfmin/step) < grad_tolerance)
        {
          //answer = fGrd[i];
          break;
        }
      }

//      std::cout << "AFTER, INTERNAL:  fGrd[" << i <<"] = " << grad_vec_internal(i) << "\t";
//      std::cout << "fG2[" << i <<"] = " << gr2_vec_internal(i) << "\t";
//      std::cout << "fGstep[" << i <<"] = " << gstep_vec_internal(i) << "\t";
//      std::cout << "x[" << i << "] = " << xtf << "\t";
//      std::cout << "fVal = " << fVal << "\t";
//      std::cout << std::endl;

//      grad_vec(i)  = grad_vec_internal(i) / DInt2Ext(parameters[i], xtf);
//      gr2_vec(i)   = gr2_vec_internal(i) / D2Int2Ext(parameters[i], xtf);
//      gstep_vec(i) = gstep_vec_internal(i) / GStepInt2Ext(parameters[i], xtf);

//      std::cout << "AFTER, EXTERNAL:  fGrd[" << i <<"] = " << grad_vec(i) << "\t";
//      std::cout << "fG2[" << i <<"] = " << gr2_vec(i) << "\t";
//      std::cout << "fGstep[" << i <<"] = " << gstep_vec(i) << "\t";
//      std::cout << "x[" << i << "] = " << vx_external[i] << "\t";
//      std::cout << "fVal = " << fVal << "\t";
//      std::cout << std::endl;

    }

//    std::cout << std::endl <<"End:" << std::endl;
//    for (unsigned int i = 0; i<fN; i++) {
//      std::cout << "fGrd[" << i <<"] = " << fGrd[i] << "\t";
//      std::cout << "fG2[" << i <<"] = " << fG2[i] << "\t";
//      std::cout << "fGstep[" << i <<"] = " << fGstep[i] << std::endl;
//    }

    fG = ROOT::Minuit2::FunctionGradient(grad_vec, gr2_vec, gstep_vec);
//    fG_internal = ROOT::Minuit2::FunctionGradient(grad_vec_internal, gr2_vec_internal, gstep_vec_internal);
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

    double eps2 = precision.Eps2();

    ROOT::Minuit2::MnAlgebraicVector grad_vec(fFunction->NDim()),
                                     gr2_vec(fFunction->NDim()),
                                     gstep_vec(fFunction->NDim());

    unsigned ix = 0;
    for (auto parameter = parameters.begin(); parameter != parameters.end(); ++parameter, ++ix) {
      // I'm guessing par.Vec()(i) should give the value of variable i...
//    double var = parameter->Value();

      // Judging by the ParameterSettings.h constructor argument name "err",
      // I'm guessing what MINUIT calls "Error" is stepsize on the ROOT side.
      double werr = parameter->StepSize();

      // Actually, sav in Minuit2 is the external parameter value, so that is
      // what we called var before and var is unnecessary here.
      double sav = parameter->Value();

      // However, we do need var below, so let's calculate it using Ext2int:
      double var = Ext2int(*parameter, sav);

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

//      std::cout << "fGrd[" << ix <<"] = " << grd << "\t";
//      std::cout << "fG2[" << ix <<"] = " << g2 << "\t";
//      std::cout << "fGstep[" << ix <<"] = " << gstep << std::endl;

    }

    fG = ROOT::Minuit2::FunctionGradient(grad_vec, gr2_vec, gstep_vec);
  }

} // namespace RooFit


