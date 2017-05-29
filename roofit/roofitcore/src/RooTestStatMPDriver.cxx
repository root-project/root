/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,        dkirkby@uci.edu                  *
 *   PB, Patrick Bos,     Netherlands eScience Center,                       *
 *                                          p.bos@esciencecenter.nl          *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University.                         *
 *               2016,      Netherlands eScience Center.                     *
 *                          All rights reserved.                             *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

/**
\file RooTestStatMPDriver.cxx
\class RooTestStatMPDriver
\ingroup Roofitcore

RooTestStatMPDriver calculates the sum of a set of RooAbsReal terms, or
when constructed with two sets, it sums the product of the terms
in the two sets. This class does not (yet) do any smart handling of integrals, 
i.e. all integrals of the product are handled numerically
**/


#include "RooFit.h"

#include "Riostream.h"
#include <math.h>
#include <memory>
#include <list>
#include <algorithm>
using namespace std ;

#include "RooTestStatMPDriver.h"
#include "RooProduct.h"
#include "RooAbsReal.h"
#include "RooErrorHandler.h"
#include "RooArgSet.h"
#include "RooNameReg.h"
#include "RooNLLVar.h"
#include "RooChi2Var.h"
#include "RooMsgService.h"

ClassImp(RooTestStatMPDriver)
;


////////////////////////////////////////////////////////////////////////////////

RooTestStatMPDriver::RooTestStatMPDriver()
{
}



////////////////////////////////////////////////////////////////////////////////
/// Constructor with a single set of RooAbsReals. The value of the function will be
/// the sum of the values in sumSet. If takeOwnership is true the RooTestStatMPDriver object
/// will take ownership of the arguments in sumSet

RooTestStatMPDriver::RooTestStatMPDriver(const char* name, const char* title, RooAbsPdf& pdf, RooAbsData& data, int nCPU) 
  : RooAbsReal(name, title), _nll("nll","NLL implementation",this), _nCPU(nCPU)
{
  // Create NLL implementation
  RooAbsReal* nll = pdf.createNLL(data) ;
  
  nll->Print() ;

  // Save NLL in proxy
  _nll.setArg(*nll) ;

  // Take ownership of nll object
  addOwnedComponents(*nll) ;

  return ;
}

////////////////////////////////////////////////////////////////////////////////
/// Initialization

void RooTestStatMPDriver::init() const 
{
  RooAbsTestStatistic* nll = (RooAbsTestStatistic*) _nll.absArg() ;

  _mpfeArray = new pRooRealMPFE[_nCPU];  

  for (Int_t i = 0; i < _nCPU; ++i) {

    // The MP set determines what fraction of the data to process - default strategy is 'bulk partition'
    nll->setMPSet(i,_nCPU);

    // Rename object to encode process number in object name - no cloning of nll is needed - since fork effectively clones
    nll->SetName(Form("%s_NLL%d",GetName(),i));
    nll->SetTitle(Form("%s_NLL%d",GetTitle(),i));    
    _mpfeArray[i] = new RooRealMPFE(Form("%s_%lx_MPFE%d",GetName(),(ULong_t)this,i),Form("%s_%lx_MPFE%d",GetTitle(),(ULong_t)this,i),*nll, i, _nCPU, false);

    // Initialize MP frontend - fork server process etc
    _mpfeArray[i]->initialize();
    // _mpfeArray[i]->setVerbose() ; // uncomment this to see verbose messaging of client-server communications

    // Don't let each front-end independently figure out if parameters have changed. If MPFE[0] thinks it is needed,
    // all other MPs will follow its lead
    if (i > 0) {
      _mpfeArray[i]->followAsSlave(*_mpfeArray[0]);
    }
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooTestStatMPDriver::RooTestStatMPDriver(const RooTestStatMPDriver& other, const char* name) 
  : RooAbsReal(other, name), _nll("nll",this,other._nll), _nCPU(other._nCPU)
{
  // Don't copy MPFE array - but allow to reinitialize when copy is first used
}


////////////////////////////////////////////////////////////////////////////////

RooTestStatMPDriver::~RooTestStatMPDriver() 
{ // Destructor
  if (_mpfeArray) {
    for (Int_t i = 0; i < _nCPU; ++i) delete _mpfeArray[i];
    delete[] _mpfeArray ;
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Calculate and return current value of self

Double_t RooTestStatMPDriver::evaluate() const 
{
  if (_nCPU>0) {

    // Multiprocess mode of evaluation

    if (!_mpfeArray) init() ;
    
    // Start calculations in parallel
    for (Int_t i = 0; i < _nCPU; ++i) {
      // Non-blocking call to calculate() will start NLL calculation on remote process
      _mpfeArray[i]->calculate();
    }
    
    double sum(0) ;
    for (Int_t i = 0; i < _nCPU; ++i) {
      // Blocking call to getValV() will wait for answer of remote process to become available
      double tmp = _mpfeArray[i]->getValV(); 
      sum += tmp ;
    }
    
    return sum  ;

  } else{

    // Simple mode of evaluation
    return _nll ;

  }
}


