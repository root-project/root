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

Add-on module to RooMCStudy that
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
#include "RooFitResult.h"
#include "RooDLLSignificanceMCSModule.h"
#include "RooMsgService.h"



using std::endl;

ClassImp(RooDLLSignificanceMCSModule);


////////////////////////////////////////////////////////////////////////////////
/// Constructor of module with parameter to be interpreted as nSignal and the value of the
/// null hypothesis for nSignal (usually zero)

RooDLLSignificanceMCSModule::RooDLLSignificanceMCSModule(const RooRealVar& param, double nullHypoValue) :
  RooAbsMCStudyModule(Form("RooDLLSignificanceMCSModule_%s",param.GetName()),Form("RooDLLSignificanceMCSModule_%s",param.GetName())),
  _parName(param.GetName()),
  _nullValue(nullHypoValue)
{
}



////////////////////////////////////////////////////////////////////////////////
/// Constructor of module with parameter name to be interpreted as nSignal and the value of the
/// null hypothesis for nSignal (usually zero)

RooDLLSignificanceMCSModule::RooDLLSignificanceMCSModule(const char* parName, double nullHypoValue) :
  RooAbsMCStudyModule(Form("RooDLLSignificanceMCSModule_%s",parName),Form("RooDLLSignificanceMCSModule_%s",parName)),
  _parName(parName),
  _nullValue(nullHypoValue)
{
}



////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooDLLSignificanceMCSModule::RooDLLSignificanceMCSModule(const RooDLLSignificanceMCSModule& other) :
  RooAbsMCStudyModule(other),
  _parName(other._parName),
  _nullValue(other._nullValue)
{
}

RooDLLSignificanceMCSModule::~RooDLLSignificanceMCSModule() = default;

////////////////////////////////////////////////////////////////////////////////
/// Initialize module after attachment to RooMCStudy object

bool RooDLLSignificanceMCSModule::initializeInstance()
{
  // Check that parameter is also present in fit parameter list of RooMCStudy object
  if (!fitParams()->find(_parName.c_str())) {
    coutE(InputArguments) << "RooDLLSignificanceMCSModule::initializeInstance:: ERROR: No parameter named " << _parName << " in RooMCStudy!" << std::endl ;
    return false ;
  }

  // Construct variable that holds -log(L) fit with null hypothesis for given parameter
  std::string nll0hName = "nll_nullhypo_" + _parName;
  std::string nll0hTitle = "-log(L) with null hypothesis for param " + _parName;
  _nll0h = std::make_unique<RooRealVar>(nll0hName.c_str(),nll0hTitle.c_str(),0) ;

  // Construct variable that holds -log(L) fit with null hypothesis for given parameter
  std::string dll0hName = "dll_nullhypo_" + _parName;
  std::string dll0hTitle = "-log(L) difference w.r.t null hypo for param " + _parName;
  _dll0h = std::make_unique<RooRealVar>(dll0hName.c_str(),dll0hTitle.c_str(),0) ;

  // Construct variable that holds significance corresponding to delta(-log(L)) w.r.t to null hypothesis for given parameter
  std::string sig0hName = "significance_nullhypo_" + _parName;
  std::string sig0hTitle = "Gaussian signficiance of Delta(-log(L)) w.r.t null hypo for param " + _parName;
  _sig0h = std::make_unique<RooRealVar>(sig0hName.c_str(),sig0hTitle.c_str(),-10,100) ;

  // Create new dataset to be merged with RooMCStudy::fitParDataSet
  _data = std::make_unique<RooDataSet>("DeltaLLSigData","Additional data for Delta(-log(L)) study",RooArgSet(*_nll0h,*_dll0h,*_sig0h)) ;

  return true ;
}



////////////////////////////////////////////////////////////////////////////////
/// Initialize module at beginning of RooCMStudy run

bool RooDLLSignificanceMCSModule::initializeRun(Int_t /*numSamples*/)
{
  _data->reset() ;
  return true ;
}



////////////////////////////////////////////////////////////////////////////////
/// Return auxiliary dataset with results of delta(-log(L))
/// calculations of this module so that it is merged with
/// RooMCStudy::fitParDataSet() by RooMCStudy

RooDataSet* RooDLLSignificanceMCSModule::finalizeRun()
{
  return _data.get();
}



////////////////////////////////////////////////////////////////////////////////
/// Save likelihood from nominal fit, fix chosen parameter to its
/// null hypothesis value and rerun fit Save difference in likelihood
/// and associated Gaussian significance in auxiliary dataset

bool RooDLLSignificanceMCSModule::processAfterFit(Int_t /*sampleNum*/)
{
  RooRealVar* par = static_cast<RooRealVar*>(fitParams()->find(_parName.c_str())) ;
  par->setVal(_nullValue) ;
  par->setConstant(true) ;
  std::unique_ptr<RooFitResult> frnull{refit()};
  par->setConstant(false) ;

  _nll0h->setVal(frnull->minNll()) ;

  double deltaLL = (frnull->minNll() - nllVar()->getVal()) ;
  double signif = deltaLL>0 ? sqrt(2*deltaLL) : -sqrt(-2*deltaLL) ;
  _sig0h->setVal(signif) ;
  _dll0h->setVal(deltaLL) ;


  _data->add(RooArgSet(*_nll0h,*_dll0h,*_sig0h)) ;

  return true ;
}
