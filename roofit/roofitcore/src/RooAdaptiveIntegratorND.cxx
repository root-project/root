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
\file RooAdaptiveIntegratorND.cxx
\class RooAdaptiveIntegratorND
\ingroup Roofitcore

Adaptive one-dimensional numerical integration algorithm.
**/


#include "Riostream.h"

#include "TClass.h"
#include "RooAdaptiveIntegratorND.h"
#include "RooFunctor.h"
#include "RooArgSet.h"
#include "RooRealVar.h"
#include "RooNumber.h"
#include "RooMsgService.h"
#include "RooNumIntFactory.h"
#include "Math/AdaptiveIntegratorMultiDim.h"
#include "Math/Functor.h"

#include <cassert>

using std::endl, std::string;

// Register this class with RooNumIntConfig

////////////////////////////////////////////////////////////////////////////////
/// Register RooAdaptiveIntegratorND, its parameters, dependencies and capabilities with RooNumIntFactory

void RooAdaptiveIntegratorND::registerIntegrator(RooNumIntFactory& fact)
{
  RooRealVar maxEval2D("maxEval2D","Max number of function evaluations for 2-dim integrals",100000) ;
  RooRealVar maxEval3D("maxEval3D","Max number of function evaluations for 3-dim integrals",1000000) ;
  RooRealVar maxEvalND("maxEvalND","Max number of function evaluations for >3-dim integrals",10000000) ;
  RooRealVar maxWarn("maxWarn","Max number of warnings on precision not reached that is printed",5) ;

   auto creator = [](const RooAbsFunc &function, const RooNumIntConfig &config) {
      return std::make_unique<RooAdaptiveIntegratorND>(function, config);
   };

   fact.registerPlugin("RooAdaptiveIntegratorND", creator, {maxEval2D,maxEval3D,maxEvalND,maxWarn},
                     /*canIntegrate1D=*/false,
                     /*canIntegrate2D=*/true,
                     /*canIntegrateND=*/true,
                     /*canIntegrateOpenEnded=*/false);
}



////////////////////////////////////////////////////////////////////////////////
/// Constructor of integral on given function binding and with given configuration. The
/// integration limits are taken from the definition in the function binding
///_func = function.

RooAdaptiveIntegratorND::RooAdaptiveIntegratorND(const RooAbsFunc &function, const RooNumIntConfig &config)
   : RooAbsIntegrator(function),
     _nWarn(static_cast<Int_t>(config.getConfigSection("RooAdaptiveIntegratorND").getRealValue("maxWarn")))
{

  _rooFunctor = std::make_unique<RooFunctor>(function);
  _func = std::make_unique<ROOT::Math::Functor>(*_rooFunctor, static_cast<unsigned int>(_rooFunctor->nObs()));

  switch (_func->NDim()) {
  case 1: throw string(Form("RooAdaptiveIntegratorND::ctor ERROR dimension of function must be at least 2")) ;
  case 2: _nmax = static_cast<Int_t>(config.getConfigSection("RooAdaptiveIntegratorND").getRealValue("maxEval2D")) ; break ;
  case 3: _nmax = static_cast<Int_t>(config.getConfigSection("RooAdaptiveIntegratorND").getRealValue("maxEval3D")) ; break ;
  default: _nmax = static_cast<Int_t>(config.getConfigSection("RooAdaptiveIntegratorND").getRealValue("maxEvalND")) ; break ;
  }
  // by default do not use absolute tolerance (see https://root.cern/phpBB3/viewtopic.php?f=15&t=20071 )
  _epsAbs = 0.0;
  _epsRel = config.epsRel();
  _integrator = new ROOT::Math::AdaptiveIntegratorMultiDim(_epsAbs,_epsRel,_nmax) ;
  _integrator->SetFunction(*_func) ;
  _useIntegrandLimits=true ;

  _nError = 0 ;
  _nWarn = 0 ;
  checkLimits() ;
  _intName = function.getName() ;
}


////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooAdaptiveIntegratorND::~RooAdaptiveIntegratorND()
{
  delete _integrator ;
  if (_nError>_nWarn) {
    oocoutW(nullptr, NumIntegration) << "RooAdaptiveIntegratorND::dtor(" << _intName
           << ") WARNING: Number of suppressed warningings about integral evaluations where target precision was not reached is " << _nError-_nWarn << std::endl;
  }

}



////////////////////////////////////////////////////////////////////////////////
/// Check that our integration range is finite and otherwise return false.
/// Update the limits from the integrand if requested.

bool RooAdaptiveIntegratorND::checkLimits() const
{
  if (_xmin.empty()) {
    _xmin.resize(_func->NDim());
    _xmax.resize(_func->NDim());
  }

  if (_useIntegrandLimits) {
    for (UInt_t i=0 ; i<_func->NDim() ; i++) {
      _xmin[i]= integrand()->getMinLimit(i);
      _xmax[i]= integrand()->getMaxLimit(i);
    }
  }

  return true ;
}


////////////////////////////////////////////////////////////////////////////////
/// Change our integration limits. Return true if the new limits are
/// ok, or otherwise false. Always returns false and does nothing
/// if this object was constructed to always use our integrand's limits.

bool RooAdaptiveIntegratorND::setLimits(double *xmin, double *xmax)
{
  if(_useIntegrandLimits) {
    oocoutE(nullptr,Integration) << "RooAdaptiveIntegratorND::setLimits: cannot override integrand's limits" << std::endl;
    return false;
  }
  for (UInt_t i=0 ; i<_func->NDim() ; i++) {
    _xmin[i]= xmin[i];
    _xmax[i]= xmax[i];
  }

  return checkLimits();
}




////////////////////////////////////////////////////////////////////////////////
/// Evaluate integral at given function binding parameter values

double RooAdaptiveIntegratorND::integral(const double* /*yvec*/)
{
  double ret = _integrator->Integral(_xmin.data(),_xmax.data());
  if (_integrator->Status()==1) {
    _nError++ ;
    if (_nError<=_nWarn) {
      oocoutW(nullptr, NumIntegration) << "RooAdaptiveIntegratorND::integral(" << integrand()->getName() << ") WARNING: target rel. precision not reached due to nEval limit of "
             << _nmax << ", estimated rel. precision is " << Form("%3.1e",_integrator->RelError()) << std::endl ;
    }
    if (_nError==_nWarn) {
      oocoutW(nullptr, NumIntegration) << "RooAdaptiveIntegratorND::integral(" << integrand()->getName()
             << ") Further warnings on target precision are suppressed conform specification in integrator specification" << std::endl ;
    }
  }
  return ret ;
}

