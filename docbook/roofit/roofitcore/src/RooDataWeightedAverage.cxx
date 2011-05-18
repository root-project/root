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
// Class RooDataWeightedAverage calculate a weighted
// average of a function or p.d.f given a dataset with observable
// values, i.e. DWA(f(x),D(x)) = sum_i f(x_i) where x_i is draw from
// D(i). This class is an implementation of RooAbsOptTestStatistics 
// can make use of the optimization and parallization infrastructure
// of that base class. The main use of RooDataWeightedAverage is
// to calculate curves in RooPlots that are added with ProjWData()
// plot option.
//
// END_HTML
//

#include "RooFit.h"
#include "Riostream.h"

#include "RooDataWeightedAverage.h"
#include "RooAbsData.h"
#include "RooAbsPdf.h"
#include "RooCmdConfig.h"
#include "RooMsgService.h"



ClassImp(RooDataWeightedAverage)
;


//_____________________________________________________________________________
RooDataWeightedAverage::RooDataWeightedAverage(const char *name, const char *title, RooAbsReal& pdf, RooAbsData& indata, 
					       const RooArgSet& projdeps, Int_t nCPU, Bool_t interleave, Bool_t showProgress, Bool_t verbose) : 
  RooAbsOptTestStatistic(name,title,pdf,indata,projdeps,0,0,nCPU,interleave,verbose,kFALSE),
  _showProgress(showProgress)
{
  // Constructor of data weighted average of given p.d.f over given data. If nCPU>1 the calculation is parallelized
  // over multuple processes. If showProgress is true a progress indicator printing a single dot for each evaluation
  // is shown. If interleave is true, the dataset split over multiple processes is done with an interleave pattern
  // rather than a bulk-split pattern.

  if (_showProgress) {
    coutI(Plotting) << "RooDataWeightedAverage::ctor(" << GetName() << ") constructing data weighted average of function " << pdf.GetName() 
		    << " over " << indata.numEntries() << " data points of " << *(indata.get()) << " with a total weight of " << indata.sumEntries() << endl ;
  }
  _sumWeight = indata.sumEntries() ;
}


//_____________________________________________________________________________
RooDataWeightedAverage::RooDataWeightedAverage(const RooDataWeightedAverage& other, const char* name) : 
  RooAbsOptTestStatistic(other,name),
  _sumWeight(other._sumWeight),
  _showProgress(other._showProgress)
{
  // Copy constructor
}



//_____________________________________________________________________________
RooDataWeightedAverage::~RooDataWeightedAverage()
{
  // Destructor
}



//_____________________________________________________________________________
Double_t RooDataWeightedAverage::globalNormalization() const 
{
  // Return global normalization term by which raw (combined) test statistic should
  // be defined to obtain final test statistic. For a data weighted avarage this
  // the the sum of all weights

  return _sumWeight ;
}



//_____________________________________________________________________________
Double_t RooDataWeightedAverage::evaluatePartition(Int_t firstEvent, Int_t lastEvent, Int_t stepSize) const 
{
  // Calculate the data weighted average for events [firstEVent,lastEvent] with step size stepSize

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



