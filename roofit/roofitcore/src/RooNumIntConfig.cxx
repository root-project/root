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
\file RooNumIntConfig.cxx
\class RooNumIntConfig
\ingroup Roofitcore

Holds the configuration parameters of the various
numeric integrators used by RooRealIntegral. RooRealIntegral and RooAbsPdf
use this class in the (normalization) integral configuration interface
**/

#include "Riostream.h"

#include "RooNumIntConfig.h"
#include "RooArgSet.h"
#include "RooAbsIntegrator.h"
#include "RooNumIntFactory.h"
#include "RooMsgService.h"


using std::endl, std::ostream;

ClassImp(RooNumIntConfig)


////////////////////////////////////////////////////////////////////////////////
/// Return reference to instance of default numeric integrator configuration object

RooNumIntConfig& RooNumIntConfig::defaultConfig()
{
  static RooNumIntConfig theConfig;
  static bool initStarted = false;

  if (!initStarted) {
    // This is needed to break a deadlock. We need the RooNumIntFactory constructor
    // to initialise us, but this constructor will call back to us again.
    // Here, we ensure that we can return the instance to the factory constructor by
    // flipping the bool, but we only return to the outside world when the factory
    // is done constructing (i.e. we leave this block).
    initStarted = true;
    RooNumIntFactory::instance();
  }

  return theConfig;
}



////////////////////////////////////////////////////////////////////////////////
/// Constructor

RooNumIntConfig::RooNumIntConfig() :
  _epsAbs(1e-7),
  _epsRel(1e-7),
  _printEvalCounter(false),
  _method1D("method1D","1D integration method"),
  _method2D("method2D","2D integration method"),
  _methodND("methodND","ND integration method"),
  _method1DOpen("method1DOpen","1D integration method in open domain"),
  _method2DOpen("method2DOpen","2D integration method in open domain"),
  _methodNDOpen("methodNDOpen","ND integration method in open domain")
{
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


////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooNumIntConfig::~RooNumIntConfig()
{
  // Delete all configuration data
  _configSets.Delete() ;
}


////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

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
  // Clone all configuration dat
  for(auto * set : static_range_cast<RooArgSet*>(other._configSets)) {
    RooArgSet* setCopy = new RooArgSet;
    set->snapshot(*setCopy);
    setCopy->setName(set->GetName()) ;
   _configSets.Add(setCopy);
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Assignment operator from other RooNumIntConfig

RooNumIntConfig& RooNumIntConfig::operator=(const RooNumIntConfig& other)
{
  // Prevent self-assignment
  if (&other==this) {
    return *this ;
  }

  // Copy common properties
  _epsAbs = other._epsAbs ;
  _epsRel = other._epsRel ;
  _method1D.setIndex(other._method1D.getCurrentIndex()) ;
  _method2D.setIndex(other._method2D.getCurrentIndex()) ;
  _methodND.setIndex(other._methodND.getCurrentIndex()) ;
  _method1DOpen.setIndex(other._method1DOpen.getCurrentIndex()) ;
  _method2DOpen.setIndex(other._method2DOpen.getCurrentIndex()) ;
  _methodNDOpen.setIndex(other._methodNDOpen.getCurrentIndex()) ;

  // Delete old integrator-specific configuration data
  _configSets.Delete() ;

  // Copy new integrator-specific data
  for(auto * set : static_range_cast<RooArgSet*>(other._configSets)) {
    RooArgSet* setCopy = new RooArgSet;
    set->snapshot(*setCopy);
    setCopy->setName(set->GetName()) ;
   _configSets.Add(setCopy);
  }

  return *this ;
}



////////////////////////////////////////////////////////////////////////////////
/// Add a configuration section for a particular integrator. Integrator name and capabilities are
/// automatically determined from instance passed as 'proto'. The defaultConfig object is associated
/// as the default configuration for the integrator.

bool RooNumIntConfig::addConfigSection(std::string const &name, const RooArgSet &inDefaultConfig, bool canIntegrate1D,
                                       bool canIntegrate2D, bool canIntegrateND, bool canIntegrateOpenEnded)
{
  // Register integrator for appropriate dimensionalities
  if (canIntegrate1D) {
    _method1D.defineType(name) ;
    if (canIntegrateOpenEnded) {
      _method1DOpen.defineType(name) ;
    }
  }

  if (canIntegrate2D) {
    _method2D.defineType(name) ;
    if (canIntegrateOpenEnded) {
      _method2DOpen.defineType(name) ;
    }
  }

  if (canIntegrateND) {
    _methodND.defineType(name) ;
    if (canIntegrateOpenEnded) {
      _methodNDOpen.defineType(name) ;
    }
  }

  // Store default configuration parameters
  RooArgSet* config = new RooArgSet;
  inDefaultConfig.snapshot(*config);
  config->setName(name.c_str());
  _configSets.Add(config) ;

  return false ;
}



////////////////////////////////////////////////////////////////////////////////
/// Return section with configuration parameters for integrator with given (class) name

RooArgSet& RooNumIntConfig::getConfigSection(const char* name)
{
  return const_cast<RooArgSet&>((const_cast<const RooNumIntConfig*>(this)->getConfigSection(name))) ;
}


////////////////////////////////////////////////////////////////////////////////
/// Retrieve configuration information specific to integrator with given name

const RooArgSet& RooNumIntConfig::getConfigSection(const char* name) const
{
  static RooArgSet dummy ;
  RooArgSet* config = static_cast<RooArgSet*>(_configSets.FindObject(name)) ;
  if (!config) {
    oocoutE(nullptr,InputArguments) << "RooNumIntConfig::getConfigSection: ERROR: no configuration stored for integrator '" << name << "'" << std::endl ;
    return dummy ;
  }
  return *config ;
}



////////////////////////////////////////////////////////////////////////////////
/// Set absolute convergence criteria (convergence if std::abs(Err)<newEpsAbs)

void RooNumIntConfig::setEpsAbs(double newEpsAbs)
{
  if (newEpsAbs<0) {
    oocoutE(nullptr,InputArguments) << "RooNumIntConfig::setEpsAbs: ERROR: target absolute precision must be greater or equal than zero" << std::endl ;
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



////////////////////////////////////////////////////////////////////////////////
/// Set relative convergence criteria (convergence if std::abs(Err)/abs(Int)<newEpsRel)

void RooNumIntConfig::setEpsRel(double newEpsRel)
{
  if (newEpsRel<0) {
    oocoutE(nullptr,InputArguments) << "RooNumIntConfig::setEpsRel: ERROR: target absolute precision must be greater or equal than zero" << std::endl ;
    return ;
  }
  _epsRel = newEpsRel ;
}



////////////////////////////////////////////////////////////////////////////////
/// Detailed printing interface

void RooNumIntConfig::printMultiline(ostream &os, Int_t /*content*/, bool verbose, TString indent) const
{
  os << indent << "Requested precision: " << _epsAbs << " absolute, " << _epsRel << " relative" << std::endl << std::endl ;
  if (_printEvalCounter) {
    os << indent << "Printing of function evaluation counter for each integration enabled" << std::endl << std::endl ;
  }

  os << indent << "1-D integration method: " << _method1D.getCurrentLabel() ;
  if (_method1DOpen.getCurrentIndex()!=_method1D.getCurrentIndex()) {
    os << " (" << _method1DOpen.getCurrentLabel() << " if open-ended)" << std::endl ;
  } else {
    os << std::endl ;
  }
  os << indent << "2-D integration method: " << _method2D.getCurrentLabel() ;
  if (_method2DOpen.getCurrentIndex()!=_method2D.getCurrentIndex()) {
    os << " (" << _method2DOpen.getCurrentLabel() << " if open-ended)" << std::endl ;
  } else {
    os << std::endl ;
  }
  os << indent << "N-D integration method: " << _methodND.getCurrentLabel() ;
  if (_methodNDOpen.getCurrentIndex()!=_methodND.getCurrentIndex()) {
    os << " (" << _methodNDOpen.getCurrentLabel() << " if open-ended)" << std::endl ;
  } else {
    os << std::endl ;
  }

  if (verbose) {

    os << std::endl << "Available integration methods:" << std::endl << std::endl ;
    for(auto * configSet : static_range_cast<RooArgSet*>(_configSets)) {

      auto const& info = *RooNumIntFactory::instance().getPluginInfo(configSet->GetName());

      os << indent << "*** " << configSet->GetName() << " ***" << std::endl ;
      os << indent << "Capabilities: " ;
      if (info.canIntegrate1D) os << "[1-D] " ;
      if (info.canIntegrate2D) os << "[2-D] " ;
      if (info.canIntegrateND) os << "[N-D] " ;
      if (info.canIntegrateOpenEnded) os << "[OpenEnded] " ;
      os << std::endl ;

      os << "Configuration: " << std::endl ;
      configSet->printMultiline(os,kName|kValue) ;
      //configSet->writeToStream(os,false) ;

      if (!info.depName.empty()) {
   os << indent << "(Depends on '" << info.depName << "')" << std::endl ;
      }
      os << std::endl ;

    }
  }
}
