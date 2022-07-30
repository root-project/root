/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

/**
\file RooIntegrator1D.cxx
\class RooIntegrator1D
\ingroup Roofitcore

RooIntegrator1D implements an adaptive one-dimensional
numerical integration algorithm.
**/


#include "Riostream.h"

#include "TClass.h"
#include "RooIntegrator1D.h"
#include "RooArgSet.h"
#include "RooRealVar.h"
#include "RooNumber.h"
#include "RooIntegratorBinding.h"
#include "RooNumIntConfig.h"
#include "RooNumIntFactory.h"
#include "RooMsgService.h"

#include <assert.h>



using namespace std;

ClassImp(RooIntegrator1D);
;

// Register this class with RooNumIntConfig

////////////////////////////////////////////////////////////////////////////////
/// Register RooIntegrator1D, is parameters and capabilities with RooNumIntFactory

void RooIntegrator1D::registerIntegrator(RooNumIntFactory& fact)
{
  RooCategory sumRule("sumRule","Summation Rule") ;
  sumRule.defineType("Trapezoid",RooIntegrator1D::Trapezoid) ;
  sumRule.defineType("Midpoint",RooIntegrator1D::Midpoint) ;
  sumRule.setLabel("Trapezoid") ;
  RooCategory extrap("extrapolation","Extrapolation procedure") ;
  extrap.defineType("None",0) ;
  extrap.defineType("Wynn-Epsilon",1) ;
  extrap.setLabel("Wynn-Epsilon") ;
  RooRealVar maxSteps("maxSteps","Maximum number of steps",20) ;
  RooRealVar minSteps("minSteps","Minimum number of steps",999) ;
  RooRealVar fixSteps("fixSteps","Fixed number of steps",0) ;

  RooIntegrator1D* proto = new RooIntegrator1D() ;
  fact.storeProtoIntegrator(proto,RooArgSet(sumRule,extrap,maxSteps,minSteps,fixSteps)) ;
  RooNumIntConfig::defaultConfig().method1D().setLabel(proto->ClassName()) ;
}


////////////////////////////////////////////////////////////////////////////////
/// Construct integrator on given function binding, using specified summation
/// rule, maximum number of steps and conversion tolerance. The integration
/// limits are taken from the function binding

RooIntegrator1D::RooIntegrator1D(const RooAbsFunc& function, SummationRule rule,
             Int_t maxSteps, double eps) :
  RooAbsIntegrator(function), _rule(rule), _maxSteps(maxSteps),  _minStepsZero(999), _fixSteps(0), _epsAbs(eps), _epsRel(eps), _doExtrap(true)
{
  _useIntegrandLimits= true;
  _valid= initialize();
}


////////////////////////////////////////////////////////////////////////////////
/// Construct integrator on given function binding for given range,
/// using specified summation rule, maximum number of steps and
/// conversion tolerance. The integration limits are taken from the
/// function binding

RooIntegrator1D::RooIntegrator1D(const RooAbsFunc& function, double xmin, double xmax,
             SummationRule rule, Int_t maxSteps, double eps) :
  RooAbsIntegrator(function),
  _rule(rule),
  _maxSteps(maxSteps),
  _minStepsZero(999),
  _fixSteps(0),
  _epsAbs(eps),
  _epsRel(eps),
  _doExtrap(true)
{
  _useIntegrandLimits= false;
  _xmin= xmin;
  _xmax= xmax;
  _valid= initialize();
}


////////////////////////////////////////////////////////////////////////////////
/// Construct integrator on given function binding, using specified
/// configuration object. The integration limits are taken from the
/// function binding

RooIntegrator1D::RooIntegrator1D(const RooAbsFunc& function, const RooNumIntConfig& config) :
  RooAbsIntegrator(function,config.printEvalCounter()),
  _epsAbs(config.epsAbs()),
  _epsRel(config.epsRel())
{
  // Extract parameters from config object
  const RooArgSet& configSet = config.getConfigSection(ClassName()) ;
  _rule = (SummationRule) configSet.getCatIndex("sumRule",Trapezoid) ;
  _maxSteps = (Int_t) configSet.getRealValue("maxSteps",20) ;
  _minStepsZero = (Int_t) configSet.getRealValue("minSteps",999) ;
  _fixSteps = (Int_t) configSet.getRealValue("fixSteps",0) ;
  _doExtrap = (bool) configSet.getCatIndex("extrapolation",1) ;

  if (_fixSteps>_maxSteps) {
    oocoutE(nullptr,Integration) << "RooIntegrator1D::ctor() ERROR: fixSteps>maxSteps, fixSteps set to maxSteps" << endl ;
    _fixSteps = _maxSteps ;
  }

  _useIntegrandLimits= true;
  _valid= initialize();
}



////////////////////////////////////////////////////////////////////////////////
/// Construct integrator on given function binding, using specified
/// configuration object and integration range

RooIntegrator1D::RooIntegrator1D(const RooAbsFunc& function, double xmin, double xmax,
            const RooNumIntConfig& config) :
  RooAbsIntegrator(function,config.printEvalCounter()),
  _epsAbs(config.epsAbs()),
  _epsRel(config.epsRel())
{
  // Extract parameters from config object
  const RooArgSet& configSet = config.getConfigSection(ClassName()) ;
  _rule = (SummationRule) configSet.getCatIndex("sumRule",Trapezoid) ;
  _maxSteps = (Int_t) configSet.getRealValue("maxSteps",20) ;
  _minStepsZero = (Int_t) configSet.getRealValue("minSteps",999) ;
  _fixSteps = (Int_t) configSet.getRealValue("fixSteps",0) ;
  _doExtrap = (bool) configSet.getCatIndex("extrapolation",1) ;

  _useIntegrandLimits= false;
  _xmin= xmin;
  _xmax= xmax;
  _valid= initialize();
}



////////////////////////////////////////////////////////////////////////////////
/// Clone integrator with new function binding and configuration. Needed by RooNumIntFactory

RooAbsIntegrator* RooIntegrator1D::clone(const RooAbsFunc& function, const RooNumIntConfig& config) const
{
  return new RooIntegrator1D(function,config) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Initialize the integrator

bool RooIntegrator1D::initialize()
{
  // apply defaults if necessary
  if(_maxSteps <= 0) {
    _maxSteps= (_rule == Trapezoid) ? 20 : 14;
  }

  if(_epsRel <= 0) _epsRel= 1e-6;
  if(_epsAbs <= 0) _epsAbs= 1e-6;

  // check that the integrand is a valid function
  if(!isValid()) {
    oocoutE(nullptr,Integration) << "RooIntegrator1D::initialize: cannot integrate invalid function" << endl;
    return false;
  }

  // Allocate coordinate buffer size after number of function dimensions
  _x.resize(_function->getDimension());


  // Allocate workspace for numerical integration engine
  _h.resize(_maxSteps + 2);
  _s.resize(_maxSteps + 2);
  _c.resize(_nPoints + 1);
  _d.resize(_nPoints + 1);

  return checkLimits();
}


////////////////////////////////////////////////////////////////////////////////
/// Change our integration limits. Return true if the new limits are
/// ok, or otherwise false. Always returns false and does nothing
/// if this object was constructed to always use our integrand's limits.

bool RooIntegrator1D::setLimits(double *xmin, double *xmax)
{
  if(_useIntegrandLimits) {
    oocoutE(nullptr,Integration) << "RooIntegrator1D::setLimits: cannot override integrand's limits" << endl;
    return false;
  }
  _xmin= *xmin;
  _xmax= *xmax;
  return checkLimits();
}


////////////////////////////////////////////////////////////////////////////////
/// Check that our integration range is finite and otherwise return false.
/// Update the limits from the integrand if requested.

bool RooIntegrator1D::checkLimits() const
{
  if(_useIntegrandLimits) {
    assert(0 != integrand() && integrand()->isValid());
    const_cast<double&>(_xmin) = integrand()->getMinLimit(0);
    const_cast<double&>(_xmax) = integrand()->getMaxLimit(0);
  }
  const_cast<double&>(_range) = _xmax - _xmin;
  if (_range < 0.) {
    oocoutE(nullptr,Integration) << "RooIntegrator1D::checkLimits: bad range with min > max (_xmin = " << _xmin << " _xmax = " << _xmax << ")" << endl;
    return false;
  }
  return (RooNumber::isInfinite(_xmin) || RooNumber::isInfinite(_xmax)) ? false : true;
}


////////////////////////////////////////////////////////////////////////////////
/// Calculate numeric integral at given set of function binding parameters

double RooIntegrator1D::integral(const double *yvec)
{
  assert(isValid());

  if (_range == 0.)
    return 0.;

  // Copy yvec to xvec if provided
  if (yvec) {
    for (UInt_t i = 0 ; i<_function->getDimension()-1 ; i++) {
      _x[i+1] = yvec[i] ;
    }
  }


  _h[1]=1.0;
  double zeroThresh = _epsAbs/_range ;
  for(Int_t j = 1; j <= _maxSteps; ++j) {
    // refine our estimate using the appropriate summation rule
    _s[j]= (_rule == Trapezoid) ? addTrapezoids(j) : addMidpoints(j);

    if (j >= _minStepsZero) {
      bool allZero(true) ;
      for (int jj=0 ; jj<=j ; jj++) {
        if (_s[j]>=zeroThresh) {
          allZero=false ;
        }
      }
      if (allZero) {
        //cout << "Roo1DIntegrator(" << this << "): zero convergence at step " << j << ", value = " << 0 << endl ;
        return 0;
      }
    }

    if (_fixSteps>0) {

      // Fixed step mode, return result after fixed number of steps
      if (j==_fixSteps) {
        //cout << "returning result at fixed step " << j << endl ;
        return _s[j];
      }

    } else  if(j >= _nPoints) {

      // extrapolate the results of recent refinements and check for a stable result
      if (_doExtrap) {
        extrapolate(j);
      } else {
        _extrapValue = _s[j] ;
        _extrapError = _s[j]-_s[j-1] ;
      }

      if(fabs(_extrapError) <= _epsRel*fabs(_extrapValue)) {
        return _extrapValue;
      }
      if(fabs(_extrapError) <= _epsAbs) {
        return _extrapValue ;
      }

    }
    // update the step size for the next refinement of the summation
    _h[j+1]= (_rule == Trapezoid) ? _h[j]/4. : _h[j]/9.;
  }

  oocoutW(nullptr,Integration) << "RooIntegrator1D::integral: integral of " << _function->getName() << " over range (" << _xmin << "," << _xmax << ") did not converge after "
      << _maxSteps << " steps" << endl;
  for(Int_t j = 1; j <= _maxSteps; ++j) {
    ooccoutW(nullptr,Integration) << "   [" << j << "] h = " << _h[j] << " , s = " << _s[j] << endl;
  }

  return _s[_maxSteps] ;
}


////////////////////////////////////////////////////////////////////////////////
/// Calculate the n-th stage of refinement of the Second Euler-Maclaurin
/// summation rule which has the useful property of not evaluating the
/// integrand at either of its endpoints but requires more function
/// evaluations than the trapezoidal rule. This rule can be used with
/// a suitable change of variables to estimate improper integrals.

double RooIntegrator1D::addMidpoints(Int_t n)
{
  double x,tnm,sum,del,ddel;
  Int_t it,j;

  if(n == 1) {
    double xmid= 0.5*(_xmin + _xmax);
    return (_savedResult= _range*integrand(xvec(xmid)));
  }
  else {
    for(it=1, j=1; j < n-1; j++) it*= 3;
    tnm= it;
    del= _range/(3.*tnm);
    ddel= del+del;
    x= _xmin + 0.5*del;
    for(sum= 0, j= 1; j <= it; j++) {
      sum+= integrand(xvec(x));
      x+= ddel;
      sum+= integrand(xvec(x));
      x+= del;
    }
    return (_savedResult= (_savedResult + _range*sum/tnm)/3.);
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Calculate the n-th stage of refinement of the extended trapezoidal
/// summation rule. This is the most efficient rule for a well behaved
/// integrands that can be evaluated over its entire range, including the
/// endpoints.

double RooIntegrator1D::addTrapezoids(Int_t n)
{
  if (n == 1) {
    // use a single trapezoid to cover the full range
    return (_savedResult= 0.5*_range*(integrand(xvec(_xmin)) + integrand(xvec(_xmax))));
  }
  else {
    // break the range down into several trapezoids using 2**(n-2)
    // equally-spaced interior points
    const int nInt = 1 << (n-2);
    const double del = _range/nInt;
    const double xmin = _xmin;

    double sum = 0.;
    // TODO Replace by batch computation
    for (int j=0; j<nInt; ++j) {
      double x = xmin + (0.5+j)*del;
      sum += integrand(xvec(x));
    }

    return (_savedResult= 0.5*(_savedResult + _range*sum/nInt));
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Extrapolate result to final value

void RooIntegrator1D::extrapolate(Int_t n)
{
  double *xa= &_h[n-_nPoints];
  double *ya= &_s[n-_nPoints];
  Int_t i,m,ns=1;
  double den,dif,dift,ho,hp,w;

  dif=fabs(xa[1]);
  for (i= 1; i <= _nPoints; i++) {
    if ((dift=fabs(xa[i])) < dif) {
      ns=i;
      dif=dift;
    }
    _c[i]=ya[i];
    _d[i]=ya[i];
  }
  _extrapValue= ya[ns--];
  for(m= 1; m < _nPoints; m++) {
    for(i= 1; i <= _nPoints-m; i++) {
      ho=xa[i];
      hp=xa[i+m];
      w=_c[i+1]-_d[i];
      if((den=ho-hp) == 0.0) {
   oocoutE(nullptr,Integration) << "RooIntegrator1D::extrapolate: internal error" << endl;
      }
      den=w/den;
      _d[i]=hp*den;
      _c[i]=ho*den;
    }
    _extrapValue += (_extrapError=(2*ns < (_nPoints-m) ? _c[ns+1] : _d[ns--]));
  }
}
