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
// RooNumIntConfig holds the configuration parameters of the various
// numeric integrators used by RooRealIntegral. RooRealIntegral and RooAbsPdf
// use this class in the (normalization) integral configuration interface
// END_HTML
//

#include "RooFit.h"
#include "Riostream.h"

#include "RooNumIntConfig.h"
#include "RooArgSet.h"
#include "RooAbsIntegrator.h"
#include "RooNumIntFactory.h"
#include "RooMsgService.h"

#include "TClass.h"



ClassImp(RooNumIntConfig)
;

RooNumIntConfig* RooNumIntConfig::_default = 0 ;


//_____________________________________________________________________________
void RooNumIntConfig::cleanup()
{
  // Function called by atexit() handler installed by RooSentinel to
  // cleanup global objects at end of job
  if (_default) {
    delete _default ;
    _default = 0 ;
  }
}



//_____________________________________________________________________________
RooNumIntConfig& RooNumIntConfig::defaultConfig() 
{
  // Return reference to instance of default numeric integrator configuration object
  
  // Instantiate object if it doesn't exist yet
  if (_default==0) {
    _default = new RooNumIntConfig ;    
    RooNumIntFactory::instance() ;
  }
  return *_default ;
}



//_____________________________________________________________________________
RooNumIntConfig::RooNumIntConfig() : 
  _epsAbs(1e-7),
  _epsRel(1e-7),
  _printEvalCounter(kFALSE),
  _method1D("method1D","1D integration method"),
  _method2D("method2D","2D integration method"),
  _methodND("methodND","ND integration method"),
  _method1DOpen("method1DOpen","1D integration method in open domain"),
  _method2DOpen("method2DOpen","2D integration method in open domain"),
  _methodNDOpen("methodNDOpen","ND integration method in open domain")
{
  // Constructor 

  // Set all methods to undefined
  // Defined methods will be registered by static initialization routines
  // of the various numeric integrator engines
  _method1D.defineType("N/A",0) ;
  _method2D.defineType("N/A",0) ;
  _methodND.defineType("N/A",0) ;
  _method1DOpen.defineType("N/A",0) ;
  _method2DOpen.defineType("N/A",0) ;
  _methodNDOpen.defineType("N/A",0) ;
}


//_____________________________________________________________________________
RooNumIntConfig::~RooNumIntConfig()
{
  // Destructor

  // Delete all configuration data
  _configSets.Delete() ;
}


//_____________________________________________________________________________
RooNumIntConfig::RooNumIntConfig(const RooNumIntConfig& other) :
  TObject(other), RooPrintable(other),
  _epsAbs(other._epsAbs),
  _epsRel(other._epsRel),
  _printEvalCounter(other._printEvalCounter),
  _method1D(other._method1D),
  _method2D(other._method2D),
  _methodND(other._methodND),
  _method1DOpen(other._method1DOpen),
  _method2DOpen(other._method2DOpen),
  _methodNDOpen(other._methodNDOpen)
{
  // Copy constructor
  
  // Clone all configuration dat
  TIterator* iter = other._configSets.MakeIterator() ;
  RooArgSet* set ;
  while((set=(RooArgSet*)iter->Next())) {
    RooArgSet* setCopy = (RooArgSet*) set->snapshot() ;
    setCopy->setName(set->GetName()) ;
   _configSets.Add(setCopy);
  }
  delete iter ;
}


//_____________________________________________________________________________
RooNumIntConfig& RooNumIntConfig::operator=(const RooNumIntConfig& other) 
{
  // Assignment operator from other RooNumIntConfig

  // Prevent self-assignment 
  if (&other==this) {
    return *this ;
  }

  // Copy common properties
  _epsAbs = other._epsAbs ;
  _epsRel = other._epsRel ;
  _method1D.setIndex(other._method1D.getIndex()) ;
  _method2D.setIndex(other._method2D.getIndex()) ;
  _methodND.setIndex(other._methodND.getIndex()) ;
  _method1DOpen.setIndex(other._method1DOpen.getIndex()) ;
  _method2DOpen.setIndex(other._method2DOpen.getIndex()) ;
  _methodNDOpen.setIndex(other._methodNDOpen.getIndex()) ;

  // Delete old integrator-specific configuration data
  _configSets.Delete() ;

  // Copy new integrator-specific data
  TIterator* iter = other._configSets.MakeIterator() ;
  RooArgSet* set ;
  while((set=(RooArgSet*)iter->Next())) {
    RooArgSet* setCopy = (RooArgSet*) set->snapshot() ;
    setCopy->setName(set->GetName()) ;
   _configSets.Add(setCopy);
  }
  delete iter ;

  return *this ;
}



//_____________________________________________________________________________
Bool_t RooNumIntConfig::addConfigSection(const RooAbsIntegrator* proto, const RooArgSet& inDefaultConfig)
{
  // Add a configuration section for a particular integrator. Integrator name and capabilities are
  // automatically determined from instance passed as 'proto'. The defaultConfig object is associated
  // as the default configuration for the integrator. 

  TString name = proto->IsA()->GetName() ;

  // Register integrator for appropriate dimensionalities
  if (proto->canIntegrate1D()) {
    _method1D.defineType(name) ;
    if (proto->canIntegrateOpenEnded()) {
      _method1DOpen.defineType(name) ;
    }
  }

  if (proto->canIntegrate2D()) {
    _method2D.defineType(name) ;
    if (proto->canIntegrateOpenEnded()) {
      _method2DOpen.defineType(name) ;
    }
  }

  if (proto->canIntegrateND()) {
    _methodND.defineType(name) ;
    if (proto->canIntegrateOpenEnded()) {
      _methodNDOpen.defineType(name) ;
    }
  }
  
  // Store default configuration parameters
  RooArgSet* config = (RooArgSet*) inDefaultConfig.snapshot() ;
  config->setName(name) ;
  _configSets.Add(config) ;

  return kFALSE ;
}



//_____________________________________________________________________________
RooArgSet& RooNumIntConfig::getConfigSection(const char* name)  
{
  // Return section with configuration parameters for integrator with given (class) name

  return const_cast<RooArgSet&>((const_cast<const RooNumIntConfig*>(this)->getConfigSection(name))) ;
}


//_____________________________________________________________________________
const RooArgSet& RooNumIntConfig::getConfigSection(const char* name) const
{
  // Retrieve configuration information specific to integrator with given name

  static RooArgSet dummy ;
  RooArgSet* config = (RooArgSet*) _configSets.FindObject(name) ;
  if (!config) {
    oocoutE((TObject*)0,InputArguments) << "RooNumIntConfig::getIntegrator: ERROR: no configuration stored for integrator '" << name << "'" << endl ;
    return dummy ;
  }
  return *config ;
}



//_____________________________________________________________________________
void RooNumIntConfig::setEpsAbs(Double_t newEpsAbs)
{
  // Set absolute convergence criteria (convergence if abs(Err)<newEpsAbs)

  if (newEpsAbs<=0) {
    oocoutE((TObject*)0,InputArguments) << "RooNumIntConfig::setEpsAbs: ERROR: target absolute precision must be greater than zero" << endl ;
    return ;
  }
  _epsAbs = newEpsAbs ;
}


RooPrintable::StyleOption RooNumIntConfig::defaultPrintStyle(Option_t* opt) const 
{ 
  if (!opt) {
    return kStandard ;
  }

  TString o(opt) ;
  o.ToLower() ;

  if (o.Contains("v")) {
    return kVerbose ;
  }
  return kStandard ; 
}



//_____________________________________________________________________________
void RooNumIntConfig::setEpsRel(Double_t newEpsRel) 
{
  // Set relative convergence criteria (convergence if abs(Err)/abs(Int)<newEpsRel)

  if (newEpsRel<=0) {
    oocoutE((TObject*)0,InputArguments) << "RooNumIntConfig::setEpsRel: ERROR: target absolute precision must be greater than zero" << endl ;
    return ;
  }
  _epsRel = newEpsRel ;
}



//_____________________________________________________________________________
void RooNumIntConfig::printMultiline(ostream &os, Int_t /*content*/, Bool_t verbose, TString indent) const
{
  // Detailed printing interface

  os << indent << "Requested precision: " << _epsAbs << " absolute, " << _epsRel << " relative" << endl << endl ;
  if (_printEvalCounter) {
    os << indent << "Printing of function evaluation counter for each integration enabled" << endl << endl ;
  }
  
  os << indent << "1-D integration method: " << _method1D.getLabel() ;
  if (_method1DOpen.getIndex()!=_method1D.getIndex()) {
    os << " (" << _method1DOpen.getLabel() << " if open-ended)" << endl ;
  } else {
    os << endl ;
  }
  os << indent << "2-D integration method: " << _method2D.getLabel() ;
  if (_method2DOpen.getIndex()!=_method2D.getIndex()) {
    os << " (" << _method2DOpen.getLabel() << " if open-ended)" << endl ;
  } else {
    os << endl ;
  }
  os << indent << "N-D integration method: " << _methodND.getLabel() ;
  if (_methodNDOpen.getIndex()!=_methodND.getIndex()) {
    os << " (" << _methodNDOpen.getLabel() << " if open-ended)" << endl ;
  } else {
    os << endl ;
  }
  
  if (verbose) {

    os << endl << "Available integration methods:" << endl << endl ;
    TIterator* cIter = _configSets.MakeIterator() ;
    RooArgSet* configSet ;
    while ((configSet=(RooArgSet*)cIter->Next())) {

      os << indent << "*** " << configSet->GetName() << " ***" << endl ;
      os << indent << "Capabilities: " ;
      const RooAbsIntegrator* proto = RooNumIntFactory::instance().getProtoIntegrator(configSet->GetName()) ;
      if (proto->canIntegrate1D()) os << "[1-D] " ;
      if (proto->canIntegrate2D()) os << "[2-D] " ;
      if (proto->canIntegrateND()) os << "[N-D] " ;
      if (proto->canIntegrateOpenEnded()) os << "[OpenEnded] " ;
      os << endl ;

      os << "Configuration: " << endl ;
      configSet->printMultiline(os,kName|kValue) ;
      //configSet->writeToStream(os,kFALSE) ;

      const char* depName = RooNumIntFactory::instance().getDepIntegratorName(configSet->GetName()) ;
      if (strlen(depName)>0) {
	os << indent << "(Depends on '" << depName << "')" << endl ;
      }
      os << endl ;

    }

    delete cIter ;
  }
}
