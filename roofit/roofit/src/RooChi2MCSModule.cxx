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

/** \class RooChi2MCSModule
    \ingroup Roofit

RooChi2MCSModule is an add-on module to RooMCStudy that
calculates the chi-squared of fitted p.d.f with respect to a binned
version of the data. For each fit the chi-squared, the reduced chi-squared
the number of degrees of freedom and the probability of the chi-squared
is store in the summary dataset.
**/

#include "Riostream.h"

#include "RooAbsPdf.h"
#include "RooDataSet.h"
#include "RooRealVar.h"
#include "RooFitResult.h"
#include "RooChi2MCSModule.h"
#include "RooMsgService.h"
#include "RooDataHist.h"
#include "TMath.h"
#include "RooGlobalFunc.h"

ClassImp(RooChi2MCSModule);

////////////////////////////////////////////////////////////////////////////////

RooChi2MCSModule::RooChi2MCSModule() : RooAbsMCStudyModule("RooChi2MCSModule", "RooChi2Module") {}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooChi2MCSModule::RooChi2MCSModule(const RooChi2MCSModule &other) : RooAbsMCStudyModule(other) {}

////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooChi2MCSModule::~RooChi2MCSModule() = default;

////////////////////////////////////////////////////////////////////////////////
/// Initialize module after attachment to RooMCStudy object

bool RooChi2MCSModule::initializeInstance()
{
   // Construct variable that holds -log(L) fit with null hypothesis for given parameter
   _chi2 = std::make_unique<RooRealVar>("chi2", "chi^2", 0);
   _ndof = std::make_unique<RooRealVar>("ndof", "number of degrees of freedom", 0);
   _chi2red = std::make_unique<RooRealVar>("chi2red", "reduced chi^2", 0);
   _prob = std::make_unique<RooRealVar>("prob", "prob(chi2,ndof)", 0);

   // Create new dataset to be merged with RooMCStudy::fitParDataSet
   _data = std::make_unique<RooDataSet>("Chi2Data", "Additional data for Chi2 study",
                                        RooArgSet(*_chi2, *_ndof, *_chi2red, *_prob));

   return true;
}

////////////////////////////////////////////////////////////////////////////////
/// Initialize module at beginning of RooCMStudy run

bool RooChi2MCSModule::initializeRun(Int_t /*numSamples*/)
{
  _data->reset() ;
  return true ;
}

////////////////////////////////////////////////////////////////////////////////
/// Return auxiliary dataset with results of chi2 analysis
/// calculations of this module so that it is merged with
/// RooMCStudy::fitParDataSet() by RooMCStudy

RooDataSet *RooChi2MCSModule::finalizeRun()
{
   return _data.get();
}

////////////////////////////////////////////////////////////////////////////////
/// Bin dataset and calculate chi2 of p.d.f w.r.t binned dataset

bool RooChi2MCSModule::processAfterFit(Int_t /*sampleNum*/)
{
  RooAbsData* data = genSample() ;
  std::unique_ptr<RooDataHist> binnedDataOwned;
  RooDataHist* binnedData = dynamic_cast<RooDataHist*>(data) ;
  if (!binnedData) {
    binnedDataOwned = std::unique_ptr<RooDataHist>{static_cast<RooDataSet*>(data)->binnedClone()};
    binnedData = binnedDataOwned.get();
  }

  std::unique_ptr<RooAbsReal> chi2Var{fitModel()->createChi2(*binnedData,RooFit::Extended(extendedGen()),RooFit::DataError(RooAbsData::SumW2))};

  std::unique_ptr<RooArgSet> floatPars{fitParams()->selectByAttrib("Constant",false)};

  _chi2->setVal(chi2Var->getVal()) ;
  _ndof->setVal(binnedData->numEntries()-floatPars->size()-1) ;
  _chi2red->setVal(_chi2->getVal()/_ndof->getVal()) ;
  _prob->setVal(TMath::Prob(_chi2->getVal(),static_cast<int>(_ndof->getVal()))) ;

  _data->add(RooArgSet(*_chi2,*_ndof,*_chi2red,*_prob)) ;

  return true ;
}
