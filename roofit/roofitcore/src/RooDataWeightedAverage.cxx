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
\file RooDataWeightedAverage.cxx
\class RooDataWeightedAverage
\ingroup Roofitcore

Class RooDataWeightedAverage calculate a weighted
average of a function or p.d.f given a dataset with observable
values, i.e. DWA(f(x),D(x)) = sum_i f(x_i) where x_i is draw from
D(i). This class is an implementation of RooAbsOptTestStatistics
can make use of the optimization and parallization infrastructure
of that base class. The main use of RooDataWeightedAverage is
to calculate curves in RooPlots that are added with ProjWData()
plot option.

**/

#include "Riostream.h"

#include "RooDataWeightedAverage.h"
#include "RooAbsData.h"
#include "RooAbsPdf.h"
#include "RooCmdConfig.h"
#include "RooMsgService.h"
#include "RooAbsDataStore.h"



using namespace std;

ClassImp(RooDataWeightedAverage);
;


////////////////////////////////////////////////////////////////////////////////
/// Constructor of data weighted average of given p.d.f over given data. If nCPU>1 the calculation is parallelized
/// over multuple processes. If showProgress is true a progress indicator printing a single dot for each evaluation
/// is shown. If interleave is true, the dataset split over multiple processes is done with an interleave pattern
/// rather than a bulk-split pattern.

RooDataWeightedAverage::RooDataWeightedAverage(const char *name, const char *title, RooAbsReal& pdf, RooAbsData& indata,
                                               const RooArgSet& projdeps, RooAbsTestStatistic::Configuration const& cfg,
                                               bool showProgress) :
  RooAbsOptTestStatistic(name,title,pdf,indata,projdeps,cfg),
  _showProgress(showProgress)
{
  if (_showProgress) {
    coutI(Plotting) << "RooDataWeightedAverage::ctor(" << GetName() << ") constructing data weighted average of function " << pdf.GetName()
          << " over " << indata.numEntries() << " data points of " << *(indata.get()) << " with a total weight of " << indata.sumEntries() << endl ;
  }
  _sumWeight = indata.sumEntries() ;
}


////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooDataWeightedAverage::RooDataWeightedAverage(const RooDataWeightedAverage& other, const char* name) :
  RooAbsOptTestStatistic(other,name),
  _sumWeight(other._sumWeight),
  _showProgress(other._showProgress)
{
}



////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooDataWeightedAverage::~RooDataWeightedAverage()
{
}



////////////////////////////////////////////////////////////////////////////////
/// Return global normalization term by which raw (combined) test statistic should
/// be defined to obtain final test statistic. For a data weighted avarage this
/// the sum of all weights

double RooDataWeightedAverage::globalNormalization() const
{
  return _sumWeight ;
}



////////////////////////////////////////////////////////////////////////////////
/// Calculate the data weighted average for events [firstEVent,lastEvent] with step size stepSize

double RooDataWeightedAverage::evaluatePartition(std::size_t firstEvent, std::size_t lastEvent, std::size_t stepSize) const
{
  double result(0) ;

  _dataClone->store()->recalculateCache( _projDeps, firstEvent, lastEvent, stepSize,false) ;

  if (setNum()==0 && _showProgress) {
    ccoutP(Plotting) << "." ;
    cout.flush() ;
  }

  for (auto i=firstEvent ; i<lastEvent ; i+=stepSize) {

    // get the data values for this event
    _dataClone->get(i);
    if (_dataClone->weight()==0) continue ;

    double term = _dataClone->weight() * _funcClone->getVal(_normSet);
    result += term;
  }

  return result  ;
}



