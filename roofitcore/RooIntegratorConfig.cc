/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   DK, David Kirkby, UC Irvine, davidk@slac.stanford.edu
 * History:
 *   18-Mar-2002 WV Created initial version
 *
 * Copyright (C) 2002 University of California
 *****************************************************************************/

#include "RooFitCore/RooIntegratorConfig.hh"

ClassImp(RooIntegratorConfig)
;

RooIntegratorConfig::RooIntegratorConfig()
{
  // 1D integrator
  _rule = RooIntegrator1D::Trapezoid ;
  _maxSteps = 20 ;
  _eps = 1e-6 ;

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
  _eps = other._eps ;

  // MC Integrator
  _mode = other._mode ;
  _genType = other._genType ;
  _verboseMC = other._verboseMC ;
  _alpha = other._alpha ;
  _nRefineIter = other._nRefineIter ;
  _nRefinePerDim = other._nRefinePerDim ;
  _nIntegratePerDim = other._nIntegratePerDim ;
}
