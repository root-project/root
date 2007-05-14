/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Name:  $:$Id: RooNumIntFactory.cxx,v 1.10 2007/05/11 09:11:58 verkerke Exp $
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

// -- CLASS DESCRIPTION [MISC] --
// RooNumIntFactory holds the configuration parameters of the various
// numeric integrators used by RooRealIntegral. RooRealIntegral and RooAbsPdf
// use this class in the (normalization) integral configuration interface

#include "TClass.h"

#include "RooFit.h"

#include "RooNumIntFactory.h"
#include "RooNumIntFactory.h"
#include "RooArgSet.h"
#include "RooAbsFunc.h"
#include "RooNumIntConfig.h"
#include "RooNumber.h"

#include "RooIntegrator1D.h"
#include "RooIntegrator2D.h"
#include "RooSegmentedIntegrator1D.h"
#include "RooSegmentedIntegrator2D.h"
#include "RooImproperIntegrator1D.h"
#include "RooMCIntegrator.h"
#include "RooGaussKronrodIntegrator1D.h"
#include "RooAdaptiveGaussKronrodIntegrator1D.h"


ClassImp(RooNumIntFactory)
;

RooNumIntFactory::RooNumIntFactory()
{
  RooIntegrator1D::registerIntegrator(*this) ;
  RooIntegrator2D::registerIntegrator(*this) ;
  RooSegmentedIntegrator1D::registerIntegrator(*this) ;
  RooSegmentedIntegrator2D::registerIntegrator(*this) ;
  RooImproperIntegrator1D::registerIntegrator(*this) ;
  RooMCIntegrator::registerIntegrator(*this) ;
  RooAdaptiveGaussKronrodIntegrator1D::registerIntegrator(*this) ;
  RooGaussKronrodIntegrator1D::registerIntegrator(*this) ;  
}


RooNumIntFactory::~RooNumIntFactory()
{
}

RooNumIntFactory::RooNumIntFactory(const RooNumIntFactory& other) : TObject(other)
{
}

RooNumIntFactory& RooNumIntFactory::instance()
{
  static RooNumIntFactory* _instance = 0 ;
  if (_instance==0) {
    _instance = new RooNumIntFactory ;
  } 
  return *_instance ;
}


Bool_t RooNumIntFactory::storeProtoIntegrator(RooAbsIntegrator* proto, const RooArgSet& defConfig, const char* depName) 
{
  TString name = proto->IsA()->GetName() ;

  if (getProtoIntegrator(name)) {
    //cout << "RooNumIntFactory::storeIntegrator() ERROR: integrator '" << name << "' already registered" << endl ;
    return kTRUE ;
  }

  // Add to factory 
  _integratorList.Add(proto) ;
  _nameList.Add(new TNamed(name,name)) ;
  _depList.Add(new TNamed(depName,depName)) ;

  // Add default config to master config
  RooNumIntConfig::defaultConfig().addConfigSection(proto,defConfig) ;
  
  return kFALSE ;
}


const RooAbsIntegrator* RooNumIntFactory::getProtoIntegrator(const char* name) 
{
  TObject* theNameObj = _nameList.FindObject(name) ;
  if (!theNameObj) return 0 ;
  Int_t index = _nameList.IndexOf(theNameObj) ;
  if (index<0) return 0 ;
  return (RooAbsIntegrator*) _integratorList.At(index) ;
}


const char* RooNumIntFactory::getDepIntegratorName(const char* name) 
{
  TObject* theNameObj = _nameList.FindObject(name) ;
  if (!theNameObj) return 0 ;
  Int_t index = _nameList.IndexOf(theNameObj) ;
  if (index<0) return 0 ;
  return  _depList.At(index)->GetName() ;
}

RooAbsIntegrator* RooNumIntFactory::createIntegrator(RooAbsFunc& func, const RooNumIntConfig& config, Int_t ndimPreset) 
{
  // First determine dimensionality and domain of integrand  
  Int_t ndim = ndimPreset>0 ? ndimPreset : func.getDimension() ;

  Bool_t openEnded = kFALSE ;
  Int_t i ;
  for (i=0 ; i<ndim ; i++) {
    if(RooNumber::isInfinite(func.getMinLimit(i)) ||
       RooNumber::isInfinite(func.getMaxLimit(i))) {
      openEnded = kTRUE ;
    }
  }

  // Find method defined configuration
  TString method ;
  switch(ndim) {
  case 1:
    method = openEnded ? config.method1DOpen().getLabel() : config.method1D().getLabel() ;
    break ;

  case 2:
    method = openEnded ? config.method2DOpen().getLabel() : config.method2D().getLabel() ;
    break ;

  default:
    method = openEnded ? config.methodNDOpen().getLabel() : config.methodND().getLabel() ;
    break ;
  }

  // Check that a method was defined for this case
  if (!method.CompareTo("N/A")) {
    cout << "RooNumIntFactory::createIntegrator: No integration method has been defined for " 
	 << (openEnded?"an open ended ":"a ") << ndim << "-dimensional integral" << endl ;
    return 0 ;    
  }

  // Retrieve proto integrator and return clone configured for the requested integration task
  const RooAbsIntegrator* proto = getProtoIntegrator(method) ;  
  RooAbsIntegrator* engine =  proto->clone(func,config) ;
  if (config.printEvalCounter()) {
    engine->setPrintEvalCounter(kTRUE) ;
  }
  return engine ;
}
