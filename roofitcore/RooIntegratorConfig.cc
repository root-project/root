/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2002, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

// -- CLASS DESCRIPTION [MISC] --
// RooIntegratorConfig holds the configuration parameters of the various
// numeric integrators used by RooRealIntegral. RooRealIntegral and RooAbsPdf
// use this class in the (normalization) integral configuration interface

#include "RooFitCore/RooIntegratorConfig.hh"

ClassImp(RooIntegratorConfig)
;

RooIntegratorConfig::RooIntegratorConfig()
{
  // 1D integrator
  _rule = RooIntegrator1D::Trapezoid ;
  _maxSteps = 20 ;
  _epsRel = 1e-6 ;
  _epsAbs = 1e-6 ;

  // MC Integrator
  _mode = RooMCIntegrator::Importance ;
  _genType = RooMCIntegrator::QuasiRandom ;
  _verboseMC = kFALSE ;
  _alpha = 1.5  ;
  _nRefineIter = 5  ;
  _nRefinePerDim = 1000 ;
  _nIntegratePerDim = 5000 ;
}

RooIntegratorConfig::~RooIntegratorConfig()
{
}

RooIntegratorConfig::RooIntegratorConfig(const RooIntegratorConfig& other) 
{
  // 1D integrator
  _rule = other._rule ;
  _maxSteps = other._maxSteps ;
  _epsRel = other._epsRel ;
  _epsAbs = other._epsAbs ;

  // MC Integrator
  _mode = other._mode ;
  _genType = other._genType ;
  _verboseMC = other._verboseMC ;
  _alpha = other._alpha ;
  _nRefineIter = other._nRefineIter ;
  _nRefinePerDim = other._nRefinePerDim ;
  _nIntegratePerDim = other._nIntegratePerDim ;
}
