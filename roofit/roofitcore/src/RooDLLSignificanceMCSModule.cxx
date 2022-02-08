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
\file RooDLLSignificanceMCSModule.cxx
\class RooDLLSignificanceMCSModule
\ingroup Roofitcore

RooDLLSignificanceMCSModule is an add-on modules to RooMCStudy that
calculates the significance of a signal by comparing the likelihood of
a fit fit with a given parameter floating with a fit with that given
parameter fixed to a nominal value (usually zero). The difference in
the -log(L) of those two fits can be interpreted as the probability
that a statistical background fluctation may result in a signal as large
or larger than the signal observed. This interpretation is contingent
on underlying normal sampling distributions and a MC study is a good way
to test that assumption.
**/

#include "Riostream.h"

#include "RooDataSet.h"
#include "RooRealVar.h"
#include "TString.h"
#include "RooFit.h"
#include "RooFitResult.h"
#include "RooDLLSignificanceMCSModule.h"
#include "RooMsgService.h"



using namespace std;

ClassImp(RooDLLSignificanceMCSModule);
  ;



////////////////////////////////////////////////////////////////////////////////
/// Constructor of module with parameter to be interpreted as nSignal and the value of the
/// null hypothesis for nSignal (usually zero)

RooDLLSignificanceMCSModule::RooDLLSignificanceMCSModule(const RooRealVar& param, Double_t nullHypoValue) :
  RooAbsMCStudyModule(Form("RooDLLSignificanceMCSModule_%s",param.GetName()),Form("RooDLLSignificanceMCSModule_%s",param.GetName())),
  _parName(param.GetName()),
  _data(0), _nll0h(0), _dll0h(0), _sig0h(0), _nullValue(nullHypoValue)
{
}



////////////////////////////////////////////////////////////////////////////////
/// Constructor of module with parameter name to be interpreted as nSignal and the value of the
/// null hypothesis for nSignal (usually zero)

RooDLLSignificanceMCSModule::RooDLLSignificanceMCSModule(const char* parName, Double_t nullHypoValue) :
  RooAbsMCStudyModule(Form("RooDLLSignificanceMCSModule_%s",parName),Form("RooDLLSignificanceMCSModule_%s",parName)),
  _parName(parName),
  _data(0), _nll0h(0), _dll0h(0), _sig0h(0), _nullValue(nullHypoValue)
{
}



////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooDLLSignificanceMCSModule::RooDLLSignificanceMCSModule(const RooDLLSignificanceMCSModule& other) :
  RooAbsMCStudyModule(other),
  _parName(other._parName),
  _data(0), _nll0h(0), _dll0h(0), _sig0h(0), _nullValue(other._nullValue)
{
}



////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooDLLSignificanceMCSModule:: ~RooDLLSignificanceMCSModule()
{
  if (_nll0h) {
    delete _nll0h ;
  }
  if (_dll0h) {
    delete _dll0h ;
  }
  if (_sig0h) {
    delete _sig0h ;
  }
  if (_data) {
    delete _data ;
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Initialize module after attachment to RooMCStudy object

Bool_t RooDLLSignificanceMCSModule::initializeInstance()
{
  // Check that parameter is also present in fit parameter list of RooMCStudy object
  if (!fitParams()->find(_parName.c_str())) {
    coutE(InputArguments) << "RooDLLSignificanceMCSModule::initializeInstance:: ERROR: No parameter named " << _parName << " in RooMCStudy!" << endl ;
    return kFALSE ;
  }

  // Construct variable that holds -log(L) fit with null hypothesis for given parameter
  TString nll0hName = Form("nll_nullhypo_%s",_parName.c_str()) ;
  TString nll0hTitle = Form("-log(L) with null hypothesis for param %s",_parName.c_str()) ;
  _nll0h = new RooRealVar(nll0hName.Data(),nll0hTitle.Data(),0) ;

  // Construct variable that holds -log(L) fit with null hypothesis for given parameter
  TString dll0hName = Form("dll_nullhypo_%s",_parName.c_str()) ;
  TString dll0hTitle = Form("-log(L) difference w.r.t null hypo for param %s",_parName.c_str()) ;
  _dll0h = new RooRealVar(dll0hName.Data(),dll0hTitle.Data(),0) ;

  // Construct variable that holds significance corresponding to delta(-log(L)) w.r.t to null hypothesis for given parameter
  TString sig0hName = Form("significance_nullhypo_%s",_parName.c_str()) ;
  TString sig0hTitle = Form("Gaussian signficiance of Delta(-log(L)) w.r.t null hypo for param %s",_parName.c_str()) ;
  _sig0h = new RooRealVar(sig0hName.Data(),sig0hTitle.Data(),-10,100) ;

  // Create new dataset to be merged with RooMCStudy::fitParDataSet
  _data = new RooDataSet("DeltaLLSigData","Additional data for Delta(-log(L)) study",RooArgSet(*_nll0h,*_dll0h,*_sig0h)) ;

  return kTRUE ;
}



////////////////////////////////////////////////////////////////////////////////
/// Initialize module at beginning of RooCMStudy run

Bool_t RooDLLSignificanceMCSModule::initializeRun(Int_t /*numSamples*/)
{
  _data->reset() ;
  return kTRUE ;
}



////////////////////////////////////////////////////////////////////////////////
/// Return auxiliary dataset with results of delta(-log(L))
/// calculations of this module so that it is merged with
/// RooMCStudy::fitParDataSet() by RooMCStudy

RooDataSet* RooDLLSignificanceMCSModule::finalizeRun()
{
  return _data ;
}



////////////////////////////////////////////////////////////////////////////////
/// Save likelihood from nominal fit, fix chosen parameter to its
/// null hypothesis value and rerun fit Save difference in likelihood
/// and associated Gaussian significance in auxilary dataset

Bool_t RooDLLSignificanceMCSModule::processAfterFit(Int_t /*sampleNum*/)
{
  RooRealVar* par = static_cast<RooRealVar*>(fitParams()->find(_parName.c_str())) ;
  par->setVal(_nullValue) ;
  par->setConstant(kTRUE) ;
  RooFitResult* frnull = refit() ;
  par->setConstant(kFALSE) ;

  _nll0h->setVal(frnull->minNll()) ;

  Double_t deltaLL = (frnull->minNll() - nllVar()->getVal()) ;
  Double_t signif = deltaLL>0 ? sqrt(2*deltaLL) : -sqrt(-2*deltaLL) ;
  _sig0h->setVal(signif) ;
  _dll0h->setVal(deltaLL) ;


  _data->add(RooArgSet(*_nll0h,*_dll0h,*_sig0h)) ;

  delete frnull ;

  return kTRUE ;
}
