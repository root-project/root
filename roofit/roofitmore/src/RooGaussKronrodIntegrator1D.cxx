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
\file RooGaussKronrodIntegrator1D.cxx
\class RooGaussKronrodIntegrator1D
\ingroup Roofitcore

Implements the Gauss-Kronrod integration algorithm.

An Gaussian quadrature method for numerical integration in which
error is estimation based on evaluation at special points known as
"Kronrod points."  By suitably picking these points, abscissas from
previous iterations can be reused as part of the new set of points,
whereas usual Gaussian quadrature would require recomputation of
all abscissas at each iteration.

This class automatically handles (-inf,+inf) integrals by dividing
the integration in three regions (-inf,-1), (-1,1), (1,inf) and
calculating the 1st and 3rd term using a x -> 1/x coordinate
transformation

This class embeds the Gauss-Kronrod integrator from the GNU
Scientific Library version 1.5 and applies the 10-, 21-, 43- and
87-point rule in succession until the required target precision is
reached
**/

#include "RooGaussKronrodIntegrator1D.h"

#include <RooArgSet.h>
#include <RooMsgService.h>
#include <RooNumIntFactory.h>
#include <RooNumber.h>
#include <RooRealVar.h>

#include <Riostream.h>

#include <TMath.h>

#include <gsl/gsl_integration.h>

#include <cassert>
#include <cfloat>
#include <cmath>

using std::endl;

/// \cond ROOFIT_INTERNAL

// register integrator class
// create a derived class in order to call the protected method of the
// RoodaptiveGaussKronrodIntegrator1D
namespace RooFit_internal {
struct Roo_internal_GKInteg1D : public RooGaussKronrodIntegrator1D {

   static void registerIntegrator()
   {
      auto &intFactory = RooNumIntFactory::instance();
      RooGaussKronrodIntegrator1D::registerIntegrator(intFactory);
   }
};
// class used to register integrator at loafing time
struct Roo_reg_GKInteg1D {
   Roo_reg_GKInteg1D() { Roo_internal_GKInteg1D::registerIntegrator(); }
};

static Roo_reg_GKInteg1D instance;
} // namespace RooFit_internal

/// \endcond


////////////////////////////////////////////////////////////////////////////////
/// Register RooGaussKronrodIntegrator1D, its parameters and capabilities with RooNumIntConfig

void RooGaussKronrodIntegrator1D::registerIntegrator(RooNumIntFactory &fact)
{
   auto creator = [](const RooAbsFunc &function, const RooNumIntConfig &config) {
      return std::make_unique<RooGaussKronrodIntegrator1D>(function, config);
   };

   fact.registerPlugin("RooGaussKronrodIntegrator1D", creator, {},
                       /*canIntegrate1D=*/true,
                       /*canIntegrate2D=*/false,
                       /*canIntegrateND=*/false,
                       /*canIntegrateOpenEnded=*/true);

   oocoutI(nullptr, Integration) << "RooGaussKronrodIntegrator1D has been registered" << std::endl;
}



////////////////////////////////////////////////////////////////////////////////
/// Construct integral on 'function' using given configuration object. The integration
/// range is taken from the definition in the function binding

RooGaussKronrodIntegrator1D::RooGaussKronrodIntegrator1D(const RooAbsFunc &function, const RooNumIntConfig &config)
   : RooAbsIntegrator(function), _useIntegrandLimits(true), _epsAbs(config.epsRel()), _epsRel(config.epsAbs())
{

  _valid= initialize();
}



////////////////////////////////////////////////////////////////////////////////
/// Construct integral on 'function' using given configuration object in the given range

RooGaussKronrodIntegrator1D::RooGaussKronrodIntegrator1D(const RooAbsFunc &function, double xmin, double xmax,
                                                         const RooNumIntConfig &config)
   : RooAbsIntegrator(function),
     _useIntegrandLimits(false),
     _epsAbs(config.epsRel()),
     _epsRel(config.epsAbs()),
     _xmin(xmin),
     _xmax(xmax)
{
  _valid= initialize();
}


////////////////////////////////////////////////////////////////////////////////
/// Perform one-time initialization of integrator

bool RooGaussKronrodIntegrator1D::initialize()
{
  // Allocate coordinate buffer size after number of function dimensions
  _x.resize(_function->getDimension());

  return checkLimits();
}



////////////////////////////////////////////////////////////////////////////////
/// Change our integration limits. Return true if the new limits are
/// ok, or otherwise false. Always returns false and does nothing
/// if this object was constructed to always use our integrand's limits.

bool RooGaussKronrodIntegrator1D::setLimits(double* xmin, double* xmax)
{
  if(_useIntegrandLimits) {
    oocoutE(nullptr,Eval) << "RooGaussKronrodIntegrator1D::setLimits: cannot override integrand's limits" << std::endl;
    return false;
  }
  _xmin= *xmin;
  _xmax= *xmax;
  return checkLimits();
}



////////////////////////////////////////////////////////////////////////////////
/// Check that our integration range is finite and otherwise return false.
/// Update the limits from the integrand if requested.

bool RooGaussKronrodIntegrator1D::checkLimits() const
{
  if(_useIntegrandLimits) {
    assert(nullptr != integrand() && integrand()->isValid());
    _xmin= integrand()->getMinLimit(0);
    _xmax= integrand()->getMaxLimit(0);
  }
  return true ;
}



double RooGaussKronrodIntegrator1D_GSL_GlueFunction(double x, void *data)
{
  auto instance = reinterpret_cast<RooGaussKronrodIntegrator1D*>(data);
  return instance->integrand(instance->xvec(x)) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Calculate and return integral

double RooGaussKronrodIntegrator1D::integral(const double *yvec)
{
  assert(isValid());

  // Copy yvec to xvec if provided
  if (yvec) {
    UInt_t i ; for (i=0 ; i<_function->getDimension()-1 ; i++) {
      _x[i+1] = yvec[i] ;
    }
  }

  // Setup glue function
  gsl_function F;
  F.function = &RooGaussKronrodIntegrator1D_GSL_GlueFunction ;
  F.params = this ;

  // Return values
  double result;
  double error;
  size_t neval = 0 ;

  // Call GSL implementation of integeator
  gsl_integration_qng (&F, _xmin, _xmax, _epsAbs, _epsRel, &result, &error, &neval);

  return result;
}
