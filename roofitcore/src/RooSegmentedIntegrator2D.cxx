/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooSegmentedIntegrator2D.cc,v 1.10 2005/06/20 15:45:14 wverkerke Exp $
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

// -- CLASS DESCRIPTION [AUX] --
// RooSegmentedIntegrator2D implements an adaptive one-dimensional 
// numerical integration algorithm.


#include "RooFitCore/RooFit.hh"

#include "RooFitCore/RooSegmentedIntegrator2D.hh"
#include "RooFitCore/RooSegmentedIntegrator2D.hh"
#include "RooFitCore/RooArgSet.hh"
#include "RooFitCore/RooIntegratorBinding.hh"
#include "RooFitCore/RooRealVar.hh"
#include "RooFitCore/RooNumber.hh"
#include "RooFitCore/RooNumIntFactory.hh"

#include <assert.h>

ClassImp(RooSegmentedIntegrator2D)
;

// Register this class with RooNumIntConfig
void RooSegmentedIntegrator2D::registerIntegrator(RooNumIntFactory& fact)
{
  fact.storeProtoIntegrator(new RooSegmentedIntegrator2D(),RooArgSet(),RooSegmentedIntegrator1D::Class()->GetName()) ;
}


RooSegmentedIntegrator2D::RooSegmentedIntegrator2D()
{
}

RooSegmentedIntegrator2D::RooSegmentedIntegrator2D(const RooAbsFunc& function, const RooNumIntConfig& config) :
  RooSegmentedIntegrator1D(*(_xint=new RooIntegratorBinding(*(_xIntegrator=new RooSegmentedIntegrator1D(function,config)))),config)
{
} 

RooSegmentedIntegrator2D::RooSegmentedIntegrator2D(const RooAbsFunc& function, Double_t xmin, Double_t xmax,
				 Double_t ymin, Double_t ymax,
				 const RooNumIntConfig& config) :
  RooSegmentedIntegrator1D(*(_xint=new RooIntegratorBinding(*(_xIntegrator=new RooSegmentedIntegrator1D(function,ymin,ymax,config)))),xmin,xmax,config)
{
} 

RooAbsIntegrator* RooSegmentedIntegrator2D::clone(const RooAbsFunc& function, const RooNumIntConfig& config) const
{
  return new RooSegmentedIntegrator2D(function,config) ;
}


RooSegmentedIntegrator2D::~RooSegmentedIntegrator2D() 
{
  delete _xint ;
  delete _xIntegrator ;
}



Bool_t RooSegmentedIntegrator2D::checkLimits() const {
  // Check that our integration range is finite and otherwise return kFALSE.
  // Update the limits from the integrand if requested.

  if(_useIntegrandLimits) {
    assert(0 != integrand() && integrand()->isValid());
    _xmin= integrand()->getMinLimit(0);
    _xmax= integrand()->getMaxLimit(0);
  }
  _range= _xmax - _xmin;
  if(_range <= 0) {
    cout << "RooIntegrator1D::checkLimits: bad range with min >= max" << endl;
    return kFALSE;
  }
  Bool_t ret =  (RooNumber::isInfinite(_xmin) || RooNumber::isInfinite(_xmax)) ? kFALSE : kTRUE;

  // Adjust component integrators, if already created
  if (_array && ret) {
    Double_t segSize = (_xmax - _xmin) / _nseg ;
    Int_t i ;
    for (i=0 ; i<_nseg ; i++) {
      _array[i]->setLimits(_xmin+i*segSize,_xmin+(i+1)*segSize) ;
    }
  }

  return ret ;
}



