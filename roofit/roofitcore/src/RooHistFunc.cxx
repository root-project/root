/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofit:$Id$
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

// -- CLASS DESCRIPTION [PDF] --
// RooHistFunc implements a probablity density function sample from a 
// multidimensional histogram. The histogram distribution is explicitly
// normalized by RooHistFunc and can have an arbitrary number of real or 
// discrete dimensions.

#include "RooFit.h"
#include "Riostream.h"

#include "RooHistFunc.h"
#include "RooDataHist.h"
#include "RooMsgService.h"
#include "RooRealVar.h"
#include "RooCategory.h"



ClassImp(RooHistFunc)
;


RooHistFunc::RooHistFunc() : _dataHist(0), _totVolume(0)
{
}

RooHistFunc::RooHistFunc(const char *name, const char *title, const RooArgSet& vars, 
		       const RooDataHist& dhist, Int_t intOrder) :
  RooAbsReal(name,title), 
  _depList("depList","List of dependents",this),
  _dataHist((RooDataHist*)&dhist), 
  _codeReg(10),
  _intOrder(intOrder),
  _cdfBoundaries(kFALSE),
  _totVolume(0),
  _unitNorm(kFALSE)
{
  // Constructor from a RooDataHist. The variable listed in 'vars' control the dimensionality of the
  // PDF. Any additional dimensions present in 'dhist' will be projected out. RooDataHist dimensions
  // can be either real or discrete. See RooDataHist::RooDataHist for details on the binning.
  // RooHistFunc neither owns or clone 'dhist' and the user must ensure the input histogram exists
  // for the entire life span of this PDF.

  _depList.add(vars) ;

  // Verify that vars and dhist.get() have identical contents
  const RooArgSet* dvars = dhist.get() ;
  if (vars.getSize()!=dvars->getSize()) {
    coutE(InputArguments) << "RooHistFunc::ctor(" << GetName() 
			  << ") ERROR variable list and RooDataHist must contain the same variables." << endl ;
    assert(0) ;
  }
  TIterator* iter = vars.createIterator() ;
  RooAbsArg* arg ;
  while((arg=(RooAbsArg*)iter->Next())) {
    if (!dvars->find(arg->GetName())) {
      coutE(InputArguments) << "RooHistFunc::ctor(" << GetName() 
			    << ") ERROR variable list and RooDataHist must contain the same variables." << endl ;
      assert(0) ;
    }
  }
  delete iter ;
}


RooHistFunc::RooHistFunc(const RooHistFunc& other, const char* name) :
  RooAbsReal(other,name), 
  _depList("depList",this,other._depList),
  _dataHist(other._dataHist),
  _codeReg(other._codeReg),
  _intOrder(other._intOrder),
  _cdfBoundaries(other._cdfBoundaries),
  _totVolume(other._totVolume),
  _unitNorm(other._unitNorm)
{
  // Copy constructor
}


Double_t RooHistFunc::evaluate() const
{
  // Return the current value: The value of the bin enclosing the current coordinates
  // of the dependents, normalized by the histograms contents  
  Double_t ret =  _dataHist->weight(_depList,_intOrder,kFALSE,_cdfBoundaries) ;  
  return ret ;
}

Double_t RooHistFunc::totVolume() const
{
  // Return previously calculated value, if any
  if (_totVolume>0) {
    return _totVolume ;
  }
  _totVolume = 1. ;
  TIterator* iter = _depList.createIterator() ;
  RooAbsArg* arg ;
  while((arg=(RooAbsArg*)iter->Next())) {
    RooRealVar* real = dynamic_cast<RooRealVar*>(arg) ;
    if (real) {
      _totVolume *= (real->getMax()-real->getMin()) ;
    } else {
      RooCategory* cat = dynamic_cast<RooCategory*>(arg) ;
      if (cat) {
	_totVolume *= cat->numTypes() ;
      }
    }
  }
  delete iter ;
  return _totVolume ;
}


Int_t RooHistFunc::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName) const 
{
  // Determine integration scenario. RooHistFunc can perform all integrals over 
  // its dependents analytically via partial or complete summation of the input histogram.

  // Only analytical integrals over the full range are defined
  if (rangeName!=0) {
    return 0 ;
  }

  // Simplest scenario, integrate over all dependents
  RooAbsCollection *allVarsCommon = allVars.selectCommon(_depList) ;  
  Bool_t intAllObs = (allVarsCommon->getSize()==_depList.getSize()) ;
  if (intAllObs && matchArgs(allVars,analVars,_depList)) {
    return 1000 ;
  }

  // Disable partial analytical integrals if interpolation is used
  if (_intOrder>0) {
    return 0 ;
  }

  // Find subset of _depList that integration is requested over
  RooArgSet* allVarsSel = (RooArgSet*) allVars.selectCommon(_depList) ;
  if (allVarsSel->getSize()==0) {
    delete allVarsSel ;
    return 0 ;
  }

  // Partial integration scenarios.
  // Build unique code from bit mask of integrated variables in depList
  Int_t code(0),n(0) ;
  TIterator* iter = _depList.createIterator() ;
  RooAbsArg* arg ;
  while((arg=(RooAbsArg*)iter->Next())) {
    if (allVars.find(arg->GetName())) code |= (1<<n) ;
    n++ ;
  }
  delete iter ;
  analVars.add(*allVarsSel) ;

  return code ;

}



Double_t RooHistFunc::analyticalIntegral(Int_t code, const char* /*rangeName*/) const 
{
  // Return integral identified by 'code'. The actual integration
  // is deferred to RooDataHist::sum() which implements partial
  // or complete summation over the histograms contents

  // WVE needs adaptation for rangeName feature

  // Simplest scenario, integration over all dependents
  if (code==1000) {
    return _dataHist->sum(kTRUE) ;
  }

  // Partial integration scenario, retrieve set of variables, calculate partial sum

  RooArgSet intSet ;
  TIterator* iter = _depList.createIterator() ;
  RooAbsArg* arg ;
  Int_t n(0) ;
  while((arg=(RooAbsArg*)iter->Next())) {
    if (code & (1<<n)) {
      intSet.add(*arg) ;
    }
    n++ ;
  }
  delete iter ;

  Double_t ret =  _dataHist->sum(intSet,_depList,kTRUE) ;
  return ret ;
}

