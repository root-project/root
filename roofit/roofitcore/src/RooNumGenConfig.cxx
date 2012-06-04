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
// RooNumGenConfig holds the configuration parameters of the various
// numeric integrators used by RooRealIntegral. RooRealIntegral and RooAbsPdf
// use this class in the (normalization) integral configuration interface
// END_HTML
//

#include "RooFit.h"
#include "Riostream.h"

#include "RooNumGenConfig.h"
#include "RooArgSet.h"
#include "RooAbsNumGenerator.h"
#include "RooNumGenFactory.h"
#include "RooMsgService.h"

#include "TClass.h"



using namespace std;

ClassImp(RooNumGenConfig)
;

RooNumGenConfig* RooNumGenConfig::_default = 0 ;


//_____________________________________________________________________________
void RooNumGenConfig::cleanup()
{
  // Function called by atexit() handler installed by RooSentinel to
  // cleanup global objects at end of job
  if (_default) {
    delete _default ;
    _default = 0 ;
  }
}



//_____________________________________________________________________________
RooNumGenConfig& RooNumGenConfig::defaultConfig() 
{
  // Return reference to instance of default numeric integrator configuration object
  
  // Instantiate object if it doesn't exist yet
  if (_default==0) {
    _default = new RooNumGenConfig ;    
    RooNumGenFactory::instance() ;
  }
  return *_default ;
}



//_____________________________________________________________________________
RooNumGenConfig::RooNumGenConfig() : 
  _method1D("method1D","1D sampling method"),
  _method1DCat("method1DCat","1D sampling method for pdfs with categories"),
  _method1DCond("method1DCond","1D sampling method for conditional pfs"),
  _method1DCondCat("method1DCond","1D sampling method for conditional pfs with categories"),
  _method2D("method2D","2D sampling method"),
  _method2DCat("method2DCat","2D sampling method for pdfs with categories"),
  _method2DCond("method2DCond","2D sampling method for conditional pfs"),
  _method2DCondCat("method2DCond","2D sampling method for conditional pfs with categories"),
  _methodND("methodND","ND sampling method"),
  _methodNDCat("methodNDCat","ND sampling method for pdfs with categories"),
  _methodNDCond("methodNDCond","ND sampling method for conditional pfs"),
  _methodNDCondCat("methodNDCond","ND sampling method for conditional pfs with categories")
{
  // Constructor 

  // Set all methods to undefined
  // Defined methods will be registered by static initialization routines
  // of the various numeric integrator engines
  _method1D.defineType("N/A",0) ;
  _method1DCat.defineType("N/A",0) ;
  _method1DCond.defineType("N/A",0) ;
  _method1DCondCat.defineType("N/A",0) ;

  _method2D.defineType("N/A",0) ;
  _method2DCat.defineType("N/A",0) ;
  _method2DCond.defineType("N/A",0) ;
  _method2DCondCat.defineType("N/A",0) ;

  _methodND.defineType("N/A",0) ;
  _methodNDCat.defineType("N/A",0) ;
  _methodNDCond.defineType("N/A",0) ;
  _methodNDCondCat.defineType("N/A",0) ;
}


//_____________________________________________________________________________
RooNumGenConfig::~RooNumGenConfig()
{
  // Destructor

  // Delete all configuration data
  _configSets.Delete() ;
}


//_____________________________________________________________________________
RooNumGenConfig::RooNumGenConfig(const RooNumGenConfig& other) :
  TObject(other), RooPrintable(other),
  _method1D(other._method1D),
  _method1DCat(other._method1DCat),
  _method1DCond(other._method1DCond),
  _method1DCondCat(other._method1DCondCat),
  _method2D(other._method2D),
  _method2DCat(other._method2DCat),
  _method2DCond(other._method2DCond),
  _method2DCondCat(other._method2DCondCat),
  _methodND(other._methodND),
  _methodNDCat(other._methodNDCat),
  _methodNDCond(other._methodNDCond),
  _methodNDCondCat(other._methodNDCondCat)
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
RooNumGenConfig& RooNumGenConfig::operator=(const RooNumGenConfig& other) 
{
  // Assignment operator from other RooNumGenConfig

  // Prevent self-assignment 
  if (&other==this) {
    return *this ;
  }

  // Copy common properties
  _method1D.setIndex(other._method1D.getIndex()) ;
  _method1DCat.setIndex(other._method1DCat.getIndex()) ;
  _method1DCond.setIndex(other._method1DCond.getIndex()) ;
  _method1DCondCat.setIndex(other._method1DCondCat.getIndex()) ;

  _method2D.setIndex(other._method2D.getIndex()) ;
  _method2DCat.setIndex(other._method2DCat.getIndex()) ;
  _method2DCond.setIndex(other._method2DCond.getIndex()) ;
  _method2DCondCat.setIndex(other._method2DCondCat.getIndex()) ;

  _methodND.setIndex(other._methodND.getIndex()) ;
  _methodNDCat.setIndex(other._methodNDCat.getIndex()) ;
  _methodNDCond.setIndex(other._methodNDCond.getIndex()) ;
  _methodNDCondCat.setIndex(other._methodNDCondCat.getIndex()) ;

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
RooCategory& RooNumGenConfig::method1D(Bool_t cond, Bool_t cat) 
{
  if (cond && cat) return _method1DCondCat ;
  if (cond) return _method1DCond ;
  if (cat) return _method1DCat ;
  return _method1D ;
}



//_____________________________________________________________________________
RooCategory& RooNumGenConfig::method2D(Bool_t cond, Bool_t cat) 
{
  if (cond && cat) return _method2DCondCat ;
  if (cond) return _method2DCond ;
  if (cat) return _method2DCat ;
  return _method2D ;
}



//_____________________________________________________________________________
RooCategory& RooNumGenConfig::methodND(Bool_t cond, Bool_t cat) 
{
  if (cond && cat) return _methodNDCondCat ;
  if (cond) return _methodNDCond ;
  if (cat) return _methodNDCat ;
  return _methodND ;
}



//_____________________________________________________________________________
const RooCategory& RooNumGenConfig::method1D(Bool_t cond, Bool_t cat) const 
{
  return const_cast<RooNumGenConfig*>(this)->method1D(cond,cat) ;
}



//_____________________________________________________________________________
const RooCategory& RooNumGenConfig::method2D(Bool_t cond, Bool_t cat) const 
{
  return const_cast<RooNumGenConfig*>(this)->method2D(cond,cat) ;
}



//_____________________________________________________________________________
const RooCategory& RooNumGenConfig::methodND(Bool_t cond, Bool_t cat) const 
{
  return const_cast<RooNumGenConfig*>(this)->methodND(cond,cat) ;
}



//_____________________________________________________________________________
Bool_t RooNumGenConfig::addConfigSection(const RooAbsNumGenerator* proto, const RooArgSet& inDefaultConfig)
{
  // Add a configuration section for a particular integrator. Integrator name and capabilities are
  // automatically determined from instance passed as 'proto'. The defaultConfig object is associated
  // as the default configuration for the integrator. 

  TString name = proto->IsA()->GetName() ;

  // Register integrator for appropriate dimensionalities
  
  _method1D.defineType(name) ;
  _method2D.defineType(name) ; 
  _methodND.defineType(name) ;

  if (proto->canSampleConditional()) {
    _method1DCond.defineType(name) ;
    _method2DCond.defineType(name) ;
    _methodNDCond.defineType(name) ;
  }
  if (proto->canSampleCategories()) {
    _method1DCat.defineType(name) ;
    _method2DCat.defineType(name) ;
    _methodNDCat.defineType(name) ;
  }

  if (proto->canSampleConditional() && proto->canSampleCategories()) {
    _method1DCondCat.defineType(name) ;
    _method2DCondCat.defineType(name) ;
    _methodNDCondCat.defineType(name) ;
  }
  
  // Store default configuration parameters
  RooArgSet* config = (RooArgSet*) inDefaultConfig.snapshot() ;
  config->setName(name) ;
  _configSets.Add(config) ;

  return kFALSE ;
}



//_____________________________________________________________________________
RooArgSet& RooNumGenConfig::getConfigSection(const char* name)  
{
  // Return section with configuration parameters for integrator with given (class) name

  return const_cast<RooArgSet&>((const_cast<const RooNumGenConfig*>(this)->getConfigSection(name))) ;
}


//_____________________________________________________________________________
const RooArgSet& RooNumGenConfig::getConfigSection(const char* name) const
{
  // Retrieve configuration information specific to integrator with given name

  static RooArgSet dummy ;
  RooArgSet* config = (RooArgSet*) _configSets.FindObject(name) ;
  if (!config) {
    oocoutE((TObject*)0,InputArguments) << "RooNumGenConfig::getIntegrator: ERROR: no configuration stored for integrator '" << name << "'" << endl ;
    return dummy ;
  }
  return *config ;
}


//_____________________________________________________________________________
RooPrintable::StyleOption RooNumGenConfig::defaultPrintStyle(Option_t* opt) const
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
void RooNumGenConfig::printMultiline(ostream &os, Int_t /*content*/, Bool_t verbose, TString indent) const
{
  // Detailed printing interface
  os << endl ;
  os << indent << "1-D sampling method: " << _method1D.getLabel() << endl ;
  if (_method1DCat.getIndex()!=_method1D.getIndex()) {
    os << " (" << _method1DCat.getLabel() << " if with categories)" << endl ;
  }
  if (_method1DCond.getIndex()!=_method1D.getIndex()) {
    os << " (" << _method1DCond.getLabel() << " if conditional)" << endl ;
  }
  if (_method1DCondCat.getIndex()!=_method1D.getIndex()) {
    os << " (" << _method1DCondCat.getLabel() << " if conditional with categories)" << endl ;    
  }
  os << endl ;

  os << indent << "2-D sampling method: " << _method2D.getLabel() << endl ;
  if (_method2DCat.getIndex()!=_method2D.getIndex()) {
    os << " (" << _method2DCat.getLabel() << " if with categories)" << endl ;
  }
  if (_method2DCond.getIndex()!=_method2D.getIndex()) {
    os << " (" << _method2DCond.getLabel() << " if conditional)" << endl ;
  }
  if (_method2DCondCat.getIndex()!=_method2D.getIndex()) {
    os << " (" << _method2DCondCat.getLabel() << " if conditional with categories)" << endl ;
  }
  os << endl ;

  os << indent << "N-D sampling method: " << _methodND.getLabel() << endl ;
  if (_methodNDCat.getIndex()!=_methodND.getIndex()) {
    os << " (" << _methodNDCat.getLabel() << " if with categories)" << endl ;
  }
  if (_methodNDCond.getIndex()!=_methodND.getIndex()) {
    os << " (" << _methodNDCond.getLabel() << " if conditional)" << endl ;
  }
  if (_methodNDCondCat.getIndex()!=_methodND.getIndex()) {
    os << " (" << _methodNDCondCat.getLabel() << " if conditional with categories)" << endl ;
  }
  os << endl ;
   
  if (verbose) {

    os << endl << "Available sampling methods:" << endl << endl ;
    TIterator* cIter = _configSets.MakeIterator() ;
    RooArgSet* configSet ;
    while ((configSet=(RooArgSet*)cIter->Next())) {

      os << indent << "*** " << configSet->GetName() << " ***" << endl ;
      os << indent << "Capabilities: " ;
      const RooAbsNumGenerator* proto = RooNumGenFactory::instance().getProtoSampler(configSet->GetName()) ;
     if (proto->canSampleConditional()) os << "[Conditional] " ;
     if (proto->canSampleCategories()) os << "[Categories] " ;
      os << endl ;

      os << "Configuration: " << endl ;
      configSet->printMultiline(os,kName|kValue|kTitle) ;
      os << endl ;

    }

    delete cIter ;
  }
}
