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
\file RooBinIntegrator.cxx
\class RooBinIntegrator
\ingroup Roofitcore

Computes the integral over a binned distribution by summing the bin contents of all bins.
**/

#include "RooBinIntegrator.h"

#include "RooArgSet.h"
#include "RooRealVar.h"
#include "RooNumber.h"
#include "RooNumIntConfig.h"
#include "RooNumIntFactory.h"
#include "RooMsgService.h"
#include "RooRealBinding.h"

#include <cassert>
#include <memory>


using std::endl, std::list;

// Register this class with RooNumIntConfig

////////////////////////////////////////////////////////////////////////////////
/// Register RooBinIntegrator, is parameters and capabilities with RooNumIntFactory

void RooBinIntegrator::registerIntegrator(RooNumIntFactory& fact)
{
   RooRealVar numBins("numBins","Number of bins in range",100) ;

   std::string name = "RooBinIntegrator";

   auto creator = [](const RooAbsFunc &function, const RooNumIntConfig &config) {
      return std::make_unique<RooBinIntegrator>(function, config);
   };

   fact.registerPlugin(name, creator, {numBins},
                     /*canIntegrate1D=*/true,
                     /*canIntegrate2D=*/true,
                     /*canIntegrateND=*/true,
                     /*canIntegrateOpenEnded=*/false);

  RooNumIntConfig::defaultConfig().method1D().setLabel(name);
}


////////////////////////////////////////////////////////////////////////////////
/// Construct integrator on given function binding binding

RooBinIntegrator::RooBinIntegrator(const RooAbsFunc &function, int numBins)
   : RooAbsIntegrator(function), _useIntegrandLimits(true)
{
  assert(_function && _function->isValid());

  // Allocate coordinate buffer size after number of function dimensions
  _x.resize(_function->getDimension());
  _numBins = numBins;

  _xmin.resize(_function->getDimension()) ;
  _xmax.resize(_function->getDimension()) ;

  for (UInt_t i=0 ; i<_function->getDimension() ; i++) {
    _xmin[i]= _function->getMinLimit(i);
    _xmax[i]= _function->getMaxLimit(i);

    // Retrieve bin configuration from integrand
    std::unique_ptr<list<double>> tmp{ _function->binBoundaries(i) };
    if (!tmp) {
      oocoutW(nullptr,Integration) << "RooBinIntegrator::RooBinIntegrator WARNING: integrand provide no binning definition observable #"
          << i << " substituting default binning of " << _numBins << " bins" << endl ;
      tmp = std::make_unique<list<double>>( );
      for (Int_t j=0 ; j<=_numBins ; j++) {
        tmp->push_back(_xmin[i]+j*(_xmax[i]-_xmin[i])/_numBins) ;
      }
    }
    _binb.emplace_back(tmp->begin(), tmp->end());

  }
  checkLimits();

}


////////////////////////////////////////////////////////////////////////////////
/// Construct integrator on given function binding binding

RooBinIntegrator::RooBinIntegrator(const RooAbsFunc& function, const RooNumIntConfig& config) :
  RooBinIntegrator(function, static_cast<int>(config.getConfigSection("RooBinIntegrator").getRealValue("numBins")))
{
}


////////////////////////////////////////////////////////////////////////////////
/// Change our integration limits. Return true if the new limits are
/// ok, or otherwise false. Always returns false and does nothing
/// if this object was constructed to always use our integrand's limits.

bool RooBinIntegrator::setLimits(double *xmin, double *xmax)
{
  if(_useIntegrandLimits) {
    oocoutE(nullptr,Integration) << "RooBinIntegrator::setLimits: cannot override integrand's limits" << endl;
    return false;
  }
  _xmin[0]= *xmin;
  _xmax[0]= *xmax;
  return checkLimits();
}


////////////////////////////////////////////////////////////////////////////////
/// Check that our integration range is finite and otherwise return false.
/// Update the limits from the integrand if requested.

bool RooBinIntegrator::checkLimits() const
{
  if(_useIntegrandLimits) {
    assert(nullptr != integrand() && integrand()->isValid());
    _xmin.resize(_function->getDimension()) ;
    _xmax.resize(_function->getDimension()) ;
    for (UInt_t i=0 ; i<_function->getDimension() ; i++) {
      _xmin[i]= integrand()->getMinLimit(i);
      _xmax[i]= integrand()->getMaxLimit(i);
    }
  }
  for (UInt_t i=0 ; i<_function->getDimension() ; i++) {
    if (_xmax[i]<=_xmin[i]) {
      oocoutE(nullptr,Integration) << "RooBinIntegrator::checkLimits: bad range with min >= max (_xmin = " << _xmin[i] << " _xmax = " << _xmax[i] << ")" << endl;
      return false;
    }
    if (RooNumber::isInfinite(_xmin[i]) || RooNumber::isInfinite(_xmax[i])) {
      return false ;
    }
  }

  return true;
}


////////////////////////////////////////////////////////////////////////////////
/// Calculate numeric integral at given set of function binding parameters,
/// for any number of dimensions.
double RooBinIntegrator::integral(const double *)
{
   assert(isValid());
   if (_function->getDimension() < 1) return 0.;
   
   ROOT::Math::KahanSum<double> sum;
   
   recursive_integration(0,1.,sum);
   
   return sum.Sum();
}

/**
 * @brief It performs recursively for loops to calculate N-dimensional integration
 * @param d the current recursivity depth (dimension currently being for-looped)
 * @param delta the (d-1)-dimensional bin width/area/volume/hypervolume...
 * @param sum the resulting integral where to accumulate the integral, passed by reference
 */
void RooBinIntegrator::recursive_integration(const UInt_t d, const double delta, ROOT::Math::KahanSum<double>& sum) {
   const std::vector<double>& binb = _binb[d];
   const bool isLastDim = d+1 == _function->getDimension();
   for (unsigned int ibin=0; ibin < binb.size() - 1; ++ibin) {
      const double hi = binb[ibin + 1];
      const double lo = binb[ibin];
      const double mid = (hi+lo)/2.;
      _x[d] = mid;
      if (isLastDim) {
         const double binInt = integrand(_x.data())*(hi-lo)*delta;
         sum += binInt ;
      }
      else {
         recursive_integration(d+1, (hi-lo)*delta, sum);
      }
   }
}
