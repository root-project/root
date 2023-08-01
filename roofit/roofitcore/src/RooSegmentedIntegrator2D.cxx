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
\file RooSegmentedIntegrator2D.cxx
\class RooSegmentedIntegrator2D
\ingroup Roofitcore

RooSegmentedIntegrator2D implements an adaptive one-dimensional
numerical integration algorithm.
**/


#include "Riostream.h"

#include "TClass.h"
#include "RooSegmentedIntegrator2D.h"
#include "RooArgSet.h"
#include "RooIntegratorBinding.h"
#include "RooRealVar.h"
#include "RooNumber.h"
#include "RooNumIntFactory.h"
#include "RooMsgService.h"

#include <assert.h>



using namespace std;

ClassImp(RooSegmentedIntegrator2D);


////////////////////////////////////////////////////////////////////////////////
/// Register RooSegmentedIntegrator2D, its parameters, dependencies and capabilities with RooNumIntFactory

void RooSegmentedIntegrator2D::registerIntegrator(RooNumIntFactory& fact)
{
  auto creator = [](const RooAbsFunc &function, const RooNumIntConfig &config) {
     return std::make_unique<RooSegmentedIntegrator2D>(function, config);
  };

  fact.registerPlugin("RooSegmentedIntegrator2D", creator, {},
                    /*canIntegrate1D=*/false,
                    /*canIntegrate2D=*/true,
                    /*canIntegrateND=*/false,
                    /*canIntegrateOpenEnded=*/false,
                    /*depName=*/"RooSegmentedIntegrator1D");
}


RooSegmentedIntegrator2D::~RooSegmentedIntegrator2D() = default;


////////////////////////////////////////////////////////////////////////////////
/// Constructor of integral on given function binding and with given configuration. The
/// integration limits are taken from the definition in the function binding

RooSegmentedIntegrator2D::RooSegmentedIntegrator2D(const RooAbsFunc &function, const RooNumIntConfig &config)
   : RooSegmentedIntegrator1D(*(new RooIntegratorBinding(std::make_unique<RooSegmentedIntegrator1D>(function, config))),
                              config),
     _xint{integrand()}
{
}


////////////////////////////////////////////////////////////////////////////////
/// Constructor integral on given function binding, with given configuration and
/// explicit definition of integration range

RooSegmentedIntegrator2D::RooSegmentedIntegrator2D(const RooAbsFunc &function, double xmin, double xmax, double ymin,
                                                   double ymax, const RooNumIntConfig &config)
   : RooSegmentedIntegrator1D(
        *(new RooIntegratorBinding(std::make_unique<RooSegmentedIntegrator1D>(function, ymin, ymax, config))), xmin,
        xmax, config),
     _xint{integrand()}
{
}


////////////////////////////////////////////////////////////////////////////////
/// Check that our integration range is finite and otherwise return false.
/// Update the limits from the integrand if requested.

bool RooSegmentedIntegrator2D::checkLimits() const
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
