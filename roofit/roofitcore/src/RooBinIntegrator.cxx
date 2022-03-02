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

RooBinIntegrator computes the integral over a binned distribution by summing the bin
contents of all bins.
**/

#include "RooBinIntegrator.h"

#include "RooArgSet.h"
#include "RooRealVar.h"
#include "RooNumber.h"
#include "RooIntegratorBinding.h"
#include "RooNumIntConfig.h"
#include "RooNumIntFactory.h"
#include "RooMsgService.h"
#include "RunContext.h"
#include "RooRealBinding.h"

#include "TClass.h"
#include "Math/Util.h"

#include <assert.h>



using namespace std;

ClassImp(RooBinIntegrator);
;

// Register this class with RooNumIntConfig

////////////////////////////////////////////////////////////////////////////////
/// Register RooBinIntegrator, is parameters and capabilities with RooNumIntFactory

void RooBinIntegrator::registerIntegrator(RooNumIntFactory& fact)
{
  RooRealVar numBins("numBins","Number of bins in range",100) ;
  RooBinIntegrator* proto = new RooBinIntegrator() ;
  fact.storeProtoIntegrator(proto,RooArgSet(numBins)) ;
  RooNumIntConfig::defaultConfig().method1D().setLabel(proto->IsA()->GetName()) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Default constructor

RooBinIntegrator::RooBinIntegrator() : _numBins(0), _useIntegrandLimits(kFALSE), _x(0)
{
}


////////////////////////////////////////////////////////////////////////////////
/// Construct integrator on given function binding binding

RooBinIntegrator::RooBinIntegrator(const RooAbsFunc& function) :
  RooAbsIntegrator(function)
{
  _useIntegrandLimits= kTRUE;
  assert(_function && _function->isValid());

  // Allocate coordinate buffer size after number of function dimensions
  _x = new Double_t[_function->getDimension()] ;
  _numBins = 100 ;

  _xmin.resize(_function->getDimension()) ;
  _xmax.resize(_function->getDimension()) ;

  auto realBinding = dynamic_cast<const RooRealBinding*>(_function);
  if (realBinding) {
    _evalData.reset(new RooBatchCompute::RunContext());
    _evalDataOrig.reset(new RooBatchCompute::RunContext());
  }

  for (UInt_t i=0 ; i<_function->getDimension() ; i++) {
    _xmin[i]= _function->getMinLimit(i);
    _xmax[i]= _function->getMaxLimit(i);

    // Retrieve bin configuration from integrand
    std::unique_ptr<list<Double_t>> tmp{ _function->binBoundaries(i) };
    if (!tmp) {
      oocoutW((TObject*)0,Integration) << "RooBinIntegrator::RooBinIntegrator WARNING: integrand provide no binning definition observable #"
          << i << " substituting default binning of " << _numBins << " bins" << endl ;
      tmp.reset( new list<Double_t> );
      for (Int_t j=0 ; j<=_numBins ; j++) {
        tmp->push_back(_xmin[i]+j*(_xmax[i]-_xmin[i])/_numBins) ;
      }
    }
    _binb.emplace_back(tmp->begin(), tmp->end());

    if (realBinding) {
      const std::vector<double>& binb = _binb.back();
      RooSpan<double> binCentres = _evalDataOrig->makeBatch(realBinding->observable(i), binb.size() - 1);
      for (unsigned int ibin = 0; ibin < binb.size() - 1; ++ibin) {
        binCentres[ibin] = (binb[ibin + 1] + binb[ibin]) / 2.;
      }
    }
  }
  checkLimits();

}


////////////////////////////////////////////////////////////////////////////////
/// Construct integrator on given function binding binding

RooBinIntegrator::RooBinIntegrator(const RooAbsFunc& function, const RooNumIntConfig& config) :
  RooAbsIntegrator(function)
{
  const RooArgSet& configSet = config.getConfigSection(IsA()->GetName()) ;
  _useIntegrandLimits= kTRUE;
  _numBins = (Int_t) configSet.getRealValue("numBins") ;
  assert(_function && _function->isValid());

  // Allocate coordinate buffer size after number of function dimensions
  _x = new Double_t[_function->getDimension()] ;

  auto realBinding = dynamic_cast<const RooRealBinding*>(_function);
  if (realBinding) {
    _evalData.reset(new RooBatchCompute::RunContext());
    _evalDataOrig.reset(new RooBatchCompute::RunContext());
  }

  for (UInt_t i=0 ; i<_function->getDimension() ; i++) {
    _xmin.push_back(_function->getMinLimit(i));
    _xmax.push_back(_function->getMaxLimit(i));

    // Retrieve bin configuration from integrand
    std::unique_ptr<list<Double_t>> tmp{ _function->binBoundaries(i) };
    if (!tmp) {
      oocoutW((TObject*)0,Integration) << "RooBinIntegrator::RooBinIntegrator WARNING: integrand provide no binning definition observable #"
          << i << " substituting default binning of " << _numBins << " bins" << endl ;
      tmp.reset( new list<Double_t> );
      for (Int_t j=0 ; j<=_numBins ; j++) {
        tmp->push_back(_xmin[i]+j*(_xmax[i]-_xmin[i])/_numBins) ;
      }
    }
    _binb.emplace_back(tmp->begin(), tmp->end());

    if (realBinding) {
      const std::vector<double>& binb = _binb.back();
      RooSpan<double> binCentres = _evalDataOrig->makeBatch(realBinding->observable(i), binb.size() - 1);
      for (unsigned int ibin = 0; ibin < binb.size() - 1; ++ibin) {
        binCentres[ibin] = (binb[ibin + 1] + binb[ibin]) / 2.;
      }
    }
  }

  checkLimits();
}


////////////////////////////////////////////////////////////////////////////////
/// Clone integrator with new function binding and configuration. Needed by RooNumIntFactory

RooAbsIntegrator* RooBinIntegrator::clone(const RooAbsFunc& function, const RooNumIntConfig& config) const
{
  return new RooBinIntegrator(function,config) ;
}





////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooBinIntegrator::~RooBinIntegrator()
{
  if(_x) delete[] _x;
}


////////////////////////////////////////////////////////////////////////////////
/// Change our integration limits. Return kTRUE if the new limits are
/// ok, or otherwise kFALSE. Always returns kFALSE and does nothing
/// if this object was constructed to always use our integrand's limits.

Bool_t RooBinIntegrator::setLimits(Double_t *xmin, Double_t *xmax)
{
  if(_useIntegrandLimits) {
    oocoutE((TObject*)0,Integration) << "RooBinIntegrator::setLimits: cannot override integrand's limits" << endl;
    return kFALSE;
  }
  _xmin[0]= *xmin;
  _xmax[0]= *xmax;
  return checkLimits();
}


////////////////////////////////////////////////////////////////////////////////
/// Check that our integration range is finite and otherwise return kFALSE.
/// Update the limits from the integrand if requested.

Bool_t RooBinIntegrator::checkLimits() const
{
  if(_useIntegrandLimits) {
    assert(0 != integrand() && integrand()->isValid());
    _xmin.resize(_function->getDimension()) ;
    _xmax.resize(_function->getDimension()) ;
    for (UInt_t i=0 ; i<_function->getDimension() ; i++) {
      _xmin[i]= integrand()->getMinLimit(i);
      _xmax[i]= integrand()->getMaxLimit(i);
    }
  }
  for (UInt_t i=0 ; i<_function->getDimension() ; i++) {
    if (_xmax[i]<=_xmin[i]) {
      oocoutE((TObject*)0,Integration) << "RooBinIntegrator::checkLimits: bad range with min >= max (_xmin = " << _xmin[i] << " _xmax = " << _xmax[i] << ")" << endl;
      return kFALSE;
    }
    if (RooNumber::isInfinite(_xmin[i]) || RooNumber::isInfinite(_xmax[i])) {
      return kFALSE ;
    }
  }

  return kTRUE;
}


////////////////////////////////////////////////////////////////////////////////
/// Calculate numeric integral at given set of function binding parameters.
Double_t RooBinIntegrator::integral(const Double_t *)
{
  assert(isValid());

  ROOT::Math::KahanSum<double> sum;

  if (_function->getDimension() == 1) {
    const std::vector<double>& binb = _binb[0];

    if (_evalData) {
      // Real bindings support batch evaluations. Can fast track now.
      auto realBinding = static_cast<const RooRealBinding*>(integrand());

      // Reset computation results to only contain known bin centres, and keep all memory intact:
      _evalData->spans = _evalDataOrig->spans;
      auto results = realBinding->getValuesOfBoundFunction(*_evalData);
      assert(results.size() == binb.size() - 1);

      for (unsigned int ibin = 0; ibin < binb.size() - 1; ++ibin) {
        const double width = binb[ibin + 1] - binb[ibin];
        sum += results[ibin] * width;
      }
    } else {
      // Need to use single-value interface
      for (unsigned int ibin=0; ibin < binb.size() - 1; ++ibin) {
        const double xhi = binb[ibin + 1];
        const double xlo = binb[ibin];
        const double xcenter = (xhi+xlo)/2.;
        const double binInt = integrand(xvec(xcenter))*(xhi-xlo) ;
        sum += binInt ;
      }
    }
  } else if (_function->getDimension() == 2) {
    const std::vector<double>& binbx = _binb[0];
    const std::vector<double>& binby = _binb[1];

    for (unsigned int ibin1=0; ibin1 < binbx.size() - 1; ++ibin1) {
      const double x1hi = binbx[ibin1 + 1];
      const double x1lo = binbx[ibin1];
      Double_t x1center = (x1hi+x1lo)/2 ;

      for (unsigned int ibin2=0; ibin2 < binby.size() - 1; ++ibin2) {
        const double x2hi = binby[ibin2 + 1];
        const double x2lo = binby[ibin2];
        const double x2center = (x2hi+x2lo)/2.;

        const double binInt = integrand(xvec(x1center,x2center))*(x1hi-x1lo)*(x2hi-x2lo) ;
        sum += binInt ;
      }
    }
  } else if (_function->getDimension() == 3) {
    const std::vector<double>& binbx = _binb[0];
    const std::vector<double>& binby = _binb[1];
    const std::vector<double>& binbz = _binb[2];

    for (unsigned int ibin1=0; ibin1 < binbx.size() - 1; ++ibin1) {
      const double x1hi = binbx[ibin1 + 1];
      const double x1lo = binbx[ibin1];
      Double_t x1center = (x1hi+x1lo)/2 ;

      for (unsigned int ibin2=0; ibin2 < binby.size() - 1; ++ibin2) {
        const double x2hi = binby[ibin2 + 1];
        const double x2lo = binby[ibin2];
        const double x2center = (x2hi+x2lo)/2.;

        for (unsigned int ibin3=0; ibin3 < binbz.size() - 1; ++ibin3) {
          const double x3hi = binbz[ibin3 + 1];
          const double x3lo = binbz[ibin3];
          const double x3center = (x3hi+x3lo)/2.;

          const double binInt = integrand(xvec(x1center,x2center,x3center))*(x1hi-x1lo)*(x2hi-x2lo)*(x3hi-x3lo);
          sum += binInt ;
        }
      }
    }
  }

  return sum;
}


