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

// -- CLASS DESCRIPTION [PDF] --
// Class RooDataWeightedAverage implements a a -log(likelihood) calculation from a dataset
// and a PDF. The NLL is calculated as 
//
//  Sum[data] -log( pdf(x_data) )
//
// In extended mode, a (Nexpect - Nobserved*log(NExpected) term is added

#include "RooFit.h"
#include "Riostream.h"

#include "RooDataWeightedAverage.h"
#include "RooAbsData.h"
#include "RooAbsPdf.h"
#include "RooCmdConfig.h"
#include "RooMsgService.h"



ClassImp(RooDataWeightedAverage)
;

RooDataWeightedAverage::RooDataWeightedAverage(const char *name, const char *title, RooAbsReal& pdf, RooAbsData& data,
					       Int_t nCPU, Bool_t interleave, Bool_t showProgress, Bool_t verbose) : 
  RooAbsOptTestStatistic(name,title,pdf,data,RooArgSet(),0,0,nCPU,interleave,verbose,kFALSE),
  _showProgress(showProgress)
{
  if (_showProgress) {
    coutI(Plotting) << "RooDataWeightedAverage::ctor(" << GetName() << ") constructing data weighted average of function " << pdf.GetName() 
		    << " over " << data.numEntries() << " data points of " << *(data.get()) << " with a total weight of " << data.numEntries(kTRUE) << endl ;
  }
  _sumWeight = data.numEntries(kTRUE) ;
}

RooDataWeightedAverage::RooDataWeightedAverage(const RooDataWeightedAverage& other, const char* name) : 
  RooAbsOptTestStatistic(other,name),
  _sumWeight(other._sumWeight),
  _showProgress(other._showProgress)
{
}


RooDataWeightedAverage::~RooDataWeightedAverage()
{
}


Double_t RooDataWeightedAverage::globalNormalization() const 
{
  return _sumWeight ;
}


Double_t RooDataWeightedAverage::evaluatePartition(Int_t firstEvent, Int_t lastEvent, Int_t stepSize) const 
{
  Int_t i ;
  Double_t result(0) ;

  if (setNum()==0 && _showProgress) {
    ccoutP(Plotting) << "." ;
    cout.flush() ;
  }

  for (i=firstEvent ; i<lastEvent ; i+=stepSize) {
    
    // get the data values for this event
    _dataClone->get(i);
    if (_dataClone->weight()==0) continue ;

    Double_t term = _dataClone->weight() * _funcClone->getVal(_normSet);
    result += term;
  }
  
  return result  ;
}



