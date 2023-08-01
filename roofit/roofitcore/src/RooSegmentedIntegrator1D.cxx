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
\file RooSegmentedIntegrator1D.cxx
\class RooSegmentedIntegrator1D
\ingroup Roofitcore

RooSegmentedIntegrator1D implements an adaptive one-dimensional
numerical integration algorithm.
**/

#include "Riostream.h"

#include "TClass.h"
#include "RooSegmentedIntegrator1D.h"
#include "RooArgSet.h"
#include "RooRealVar.h"
#include "RooNumber.h"
#include "RooMsgService.h"
#include "RooNumIntFactory.h"

#include <assert.h>



using namespace std;

ClassImp(RooSegmentedIntegrator1D);

// Register this class with RooNumIntConfig

////////////////////////////////////////////////////////////////////////////////
/// Register RooSegmentedIntegrator1D, its parameters, dependencies and capabilities with RooNumIntFactory

void RooSegmentedIntegrator1D::registerIntegrator(RooNumIntFactory& fact)
{
  RooRealVar numSeg("numSeg","Number of segments",3) ;

  auto creator = [](const RooAbsFunc &function, const RooNumIntConfig &config) {
     return std::make_unique<RooSegmentedIntegrator1D>(function, config);
  };

  fact.registerPlugin("RooSegmentedIntegrator1D", creator, {numSeg},
                    /*canIntegrate1D=*/true,
                    /*canIntegrate2D=*/false,
                    /*canIntegrateND=*/false,
                    /*canIntegrateOpenEnded=*/false,
                    /*depName=*/"RooIntegrator1D");
}


RooSegmentedIntegrator1D::~RooSegmentedIntegrator1D() = default;


////////////////////////////////////////////////////////////////////////////////
/// Constructor of integral on given function binding and with given configuration. The
/// integration limits are taken from the definition in the function binding

RooSegmentedIntegrator1D::RooSegmentedIntegrator1D(const RooAbsFunc& function, const RooNumIntConfig& config) :
  RooAbsIntegrator(function), _config(config)
{
  _nseg = (Int_t) config.getConfigSection(ClassName()).getRealValue("numSeg",3) ;
  _useIntegrandLimits= true;

  _valid= initialize();
}



////////////////////////////////////////////////////////////////////////////////
/// Constructor integral on given function binding, with given configuration and
/// explicit definition of integration range

RooSegmentedIntegrator1D::RooSegmentedIntegrator1D(const RooAbsFunc& function, double xmin, double xmax,
                     const RooNumIntConfig& config) :
  RooAbsIntegrator(function), _config(config)
{
  _nseg = (Int_t) config.getConfigSection(ClassName()).getRealValue("numSeg",3) ;
  _useIntegrandLimits= false;
  _xmin= xmin;
  _xmax= xmax;

  _valid= initialize();
}


////////////////////////////////////////////////////////////////////////////////
/// One-time integrator initialization

bool RooSegmentedIntegrator1D::initialize()
{
  _array.clear();

  bool limitsOK = checkLimits();
  if (!limitsOK) return false ;

  // Make array of integrators for each segment
  _array.resize(_nseg);

  Int_t i ;

  double segSize = (_xmax - _xmin) / _nseg ;

  // Adjust integrator configurations for reduced intervals
  _config.setEpsRel(_config.epsRel()/sqrt(1.*_nseg)) ;
  _config.setEpsAbs(_config.epsAbs()/sqrt(1.*_nseg)) ;

  for (i=0 ; i<_nseg ; i++) {
    _array[i] = std::make_unique<RooIntegrator1D>(*_function,_xmin+i*segSize,_xmin+(i+1)*segSize,_config) ;
  }

  return true ;
}


////////////////////////////////////////////////////////////////////////////////
/// Change our integration limits. Return true if the new limits are
/// ok, or otherwise false. Always returns false and does nothing
/// if this object was constructed to always use our integrand's limits.

bool RooSegmentedIntegrator1D::setLimits(double* xmin, double* xmax)
{
  if(_useIntegrandLimits) {
    oocoutE(nullptr,InputArguments) << "RooSegmentedIntegrator1D::setLimits: cannot override integrand's limits" << endl;
    return false;
  }
  _xmin= *xmin;
  _xmax= *xmax;
  return checkLimits();
}



////////////////////////////////////////////////////////////////////////////////
/// Check that our integration range is finite and otherwise return false.
/// Update the limits from the integrand if requested.

bool RooSegmentedIntegrator1D::checkLimits() const
{
  if(_useIntegrandLimits) {
    assert(nullptr != integrand() && integrand()->isValid());
    _xmin= integrand()->getMinLimit(0);
    _xmax= integrand()->getMaxLimit(0);
  }
  _range= _xmax - _xmin;
  if(_range <= 0) {
    oocoutE(nullptr,InputArguments) << "RooIntegrator1D::checkLimits: bad range with min >= max" << endl;
    return false;
  }
  bool ret =  (RooNumber::isInfinite(_xmin) || RooNumber::isInfinite(_xmax)) ? false : true;

  // Adjust component integrators, if already created
  if (!_array.empty() && ret) {
    double segSize = (_xmax - _xmin) / _nseg ;
    Int_t i ;
    for (i=0 ; i<_nseg ; i++) {
      _array[i]->setLimits(_xmin+i*segSize,_xmin+(i+1)*segSize) ;
    }
  }

  return ret ;
}




////////////////////////////////////////////////////////////////////////////////
/// Evaluate integral at given function binding parameter values

double RooSegmentedIntegrator1D::integral(const double *yvec)
{
  assert(isValid());

  Int_t i ;
  double result(0) ;
  for (i=0 ; i<_nseg ; i++) {
    result += _array[i]->integral(yvec) ;
  }

  return result;
}
