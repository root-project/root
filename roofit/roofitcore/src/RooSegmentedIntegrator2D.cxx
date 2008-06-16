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

//////////////////////////////////////////////////////////////////////////////
//
// BEGIN_HTML
// RooSegmentedIntegrator2D implements an adaptive one-dimensional 
// numerical integration algorithm.
// END_HTML
//


#include "RooFit.h"
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



ClassImp(RooSegmentedIntegrator2D)
;


//_____________________________________________________________________________
void RooSegmentedIntegrator2D::registerIntegrator(RooNumIntFactory& fact)
{
  // Register RooSegmentedIntegrator2D, its parameters, dependencies and capabilities with RooNumIntFactory

  fact.storeProtoIntegrator(new RooSegmentedIntegrator2D(),RooArgSet(),RooSegmentedIntegrator1D::Class()->GetName()) ;
}



//_____________________________________________________________________________
RooSegmentedIntegrator2D::RooSegmentedIntegrator2D() :
  _xIntegrator(0), _xint(0)
{
  // Default constructor
}


//_____________________________________________________________________________
RooSegmentedIntegrator2D::RooSegmentedIntegrator2D(const RooAbsFunc& function, const RooNumIntConfig& config) :
  RooSegmentedIntegrator1D(*(_xint=new RooIntegratorBinding(*(_xIntegrator=new RooSegmentedIntegrator1D(function,config)))),config)
{
  // Constructor of integral on given function binding and with given configuration. The
  // integration limits are taken from the definition in the function binding
} 


//_____________________________________________________________________________
RooSegmentedIntegrator2D::RooSegmentedIntegrator2D(const RooAbsFunc& function, Double_t xmin, Double_t xmax,
				 Double_t ymin, Double_t ymax,
				 const RooNumIntConfig& config) :
  RooSegmentedIntegrator1D(*(_xint=new RooIntegratorBinding(*(_xIntegrator=new RooSegmentedIntegrator1D(function,ymin,ymax,config)))),xmin,xmax,config)
{
  // Constructor integral on given function binding, with given configuration and
  // explicit definition of integration range
} 


//_____________________________________________________________________________
RooAbsIntegrator* RooSegmentedIntegrator2D::clone(const RooAbsFunc& function, const RooNumIntConfig& config) const
{
  // Virtual constructor with given function and configuration. Needed by RooNumIntFactory

  return new RooSegmentedIntegrator2D(function,config) ;
}



//_____________________________________________________________________________
RooSegmentedIntegrator2D::~RooSegmentedIntegrator2D() 
{
  // Destructor

  delete _xint ;
  delete _xIntegrator ;
}



//_____________________________________________________________________________
Bool_t RooSegmentedIntegrator2D::checkLimits() const 
{
  // Check that our integration range is finite and otherwise return kFALSE.
  // Update the limits from the integrand if requested.

  if(_useIntegrandLimits) {
    assert(0 != integrand() && integrand()->isValid());
    _xmin= integrand()->getMinLimit(0);
    _xmax= integrand()->getMaxLimit(0);
  }
  _range= _xmax - _xmin;
  if(_range <= 0) {
    oocoutE((TObject*)0,InputArguments) << "RooIntegrator1D::checkLimits: bad range with min >= max" << endl;
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



