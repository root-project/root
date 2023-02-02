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
\file RooAcceptReject.cxx
\class RooAcceptReject
\ingroup Roofitcore

Class RooAcceptReject is a generic toy monte carlo generator implement
the accept/reject sampling technique on any positively valued function.
The RooAcceptReject generator is used by the various generator context
classes to take care of generation of observables for which p.d.fs
do not define internal methods
**/

#include "Riostream.h"

#include "RooAcceptReject.h"
#include "RooAbsReal.h"
#include "RooCategory.h"
#include "RooRealVar.h"
#include "RooDataSet.h"
#include "RooRandom.h"
#include "RooErrorHandler.h"
#include "RooPrintable.h"
#include "RooMsgService.h"
#include "RooRealBinding.h"
#include "RooNumGenFactory.h"
#include "RooNumGenConfig.h"

#include "TFoam.h"
#include "TNamed.h"

#include <assert.h>

using namespace std;


////////////////////////////////////////////////////////////////////////////////
/// Register RooIntegrator1D, is parameters and capabilities with RooNumIntFactory

void RooAcceptReject::registerSampler(RooNumGenFactory& fact)
{
  RooRealVar nTrial0D("nTrial0D","Number of trial samples for cat-only generation",100,0,1e9) ;
  RooRealVar nTrial1D("nTrial1D","Number of trial samples for 1-dim generation",1000,0,1e9) ;
  RooRealVar nTrial2D("nTrial2D","Number of trial samples for 2-dim generation",100000,0,1e9) ;
  RooRealVar nTrial3D("nTrial3D","Number of trial samples for N-dim generation",10000000,0,1e9) ;

  RooAcceptReject* proto = new RooAcceptReject ;
  fact.storeProtoSampler(proto,RooArgSet(nTrial0D,nTrial1D,nTrial2D,nTrial3D)) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Initialize an accept-reject generator for the specified distribution function,
/// which must be non-negative but does not need to be normalized over the
/// variables to be generated, genVars. The function and its dependents are
/// cloned and so will not be disturbed during the generation process.

RooAcceptReject::RooAcceptReject(const RooAbsReal &func, const RooArgSet &genVars, const RooNumGenConfig& config, bool verbose, const RooAbsReal* maxFuncVal) :
  RooAbsNumGenerator(func,genVars,verbose,maxFuncVal)
{
  _minTrialsArray[0] = static_cast<Int_t>(config.getConfigSection("RooAcceptReject").getRealValue("nTrial0D")) ;
  _minTrialsArray[1] = static_cast<Int_t>(config.getConfigSection("RooAcceptReject").getRealValue("nTrial1D")) ;
  _minTrialsArray[2] = static_cast<Int_t>(config.getConfigSection("RooAcceptReject").getRealValue("nTrial2D")) ;
  _minTrialsArray[3] = static_cast<Int_t>(config.getConfigSection("RooAcceptReject").getRealValue("nTrial3D")) ;

  _realSampleDim = _realVars.getSize() ;
  _catSampleMult = 1 ;
  for (auto * cat : static_range_cast<RooAbsCategory*>(_catVars)) {
    _catSampleMult *=  cat->numTypes() ;
  }


  // calculate the minimum number of trials needed to estimate our integral and max value
  if (!_funcMaxVal) {

    if(_realSampleDim > 3) {
      _minTrials= _minTrialsArray[3]*_catSampleMult;
      oocoutW(nullptr, Generation) << _funcClone->GetName() << "::RooAcceptReject" << ": WARNING: generating " << _realSampleDim
         << " variables with accept-reject may not be accurate" << endl;
    }
    else {
      _minTrials= _minTrialsArray[_realSampleDim]*_catSampleMult;
    }
    if (_realSampleDim > 1) {
       oocoutW(nullptr, Generation) << "RooAcceptReject::ctor(" << _funcClone->GetName()
                         << ") WARNING: performing accept/reject sampling on a p.d.f in "
                         << _realSampleDim << " dimensions without prior knowledge on maximum value "
                         << "of p.d.f. Determining maximum value by taking " << _minTrials
                         << " trial samples. If p.d.f contains sharp peaks smaller than average "
                         << "distance between trial sampling points these may be missed and p.d.f. "
                         << "may be sampled incorrectly." << endl ;
    }
  } else {
    // No trials needed if we know the maximum a priori
    _minTrials=0 ;
  }

  // Need to fix some things here
  if (_minTrials>10000) {
    oocoutW(nullptr, Generation) << "RooAcceptReject::ctor(" << func.GetName() << "): WARNING: " << _minTrials << " trial samples requested by p.d.f for "
            << _realSampleDim << "-dimensional accept/reject sampling, this may take some time" << endl ;
  }

  // print a verbose summary of our configuration, if requested
  if(_verbose) {
    oocoutI(nullptr, Generation) << func.GetName() << "::RooAcceptReject" << ":" << endl
            << "  Initializing accept-reject generator for" << endl << "    ";
    _funcClone->printStream(ooccoutI(nullptr, Generation),RooPrintable::kName,RooPrintable::kSingleLine);
    if (_funcMaxVal) {
      ooccoutI(nullptr, Generation) << "  Function maximum provided, no trial sampling performed" << endl ;
    } else {
      ooccoutI(nullptr, Generation) << "  Real sampling dimension is " << _realSampleDim << endl;
      ooccoutI(nullptr, Generation) << "  Category sampling multiplier is " << _catSampleMult << endl ;
      ooccoutI(nullptr, Generation) << "  Min sampling trials is " << _minTrials << endl;
    }
    if (_catVars.getSize()>0) {
      ooccoutI(nullptr, Generation) << "  Will generate category vars "<< _catVars << endl ;
    }
    if (_realVars.getSize()>0) {
      ooccoutI(nullptr, Generation) << "  Will generate real vars " << _realVars << endl ;
    }
  }

  // initialize our statistics
  _maxFuncVal= 0;
  _funcSum= 0;
  _totalEvents= 0;
  _eventsUsed= 0;
}


////////////////////////////////////////////////////////////////////////////////
/// Return a pointer to a generated event. The caller does not own the event and it
/// will be overwritten by a subsequent call. The input parameter 'remaining' should
/// contain your best guess at the total number of subsequent events you will request.

const RooArgSet *RooAcceptReject::generateEvent(UInt_t remaining, double& resampleRatio)
{
  // are we actually generating anything? (the cache always contains at least our function value)
  const RooArgSet *event= _cache->get();
  if(event->getSize() == 1) return event;

  if (!_funcMaxVal) {
    // Generation with empirical maximum determination

    // first generate enough events to get reasonable estimates for the integral and
    // maximum function value

    while(_totalEvents < _minTrials) {
      addEventToCache();

      // Limit cache size to 1M events
      if (_cache->numEntries()>1000000) {
   oocoutI(nullptr, Generation) << "RooAcceptReject::generateEvent: resetting event cache" << std::endl;
   _cache->reset() ;
   _eventsUsed = 0 ;
      }
    }

    event= 0;
    double oldMax2(_maxFuncVal);
    while(0 == event) {
      // Use any cached events first
      if (_maxFuncVal>oldMax2) {
   oocxcoutD(static_cast<TObject*>(nullptr), Generation) << "RooAcceptReject::generateEvent maxFuncVal has changed, need to resample already accepted events by factor"
             << oldMax2 << "/" << _maxFuncVal << "=" << oldMax2/_maxFuncVal << endl ;
   resampleRatio=oldMax2/_maxFuncVal ;
      }
      event= nextAcceptedEvent();
      if(event) break;
      // When we have used up the cache, start a new cache and add
      // some more events to it.
      _cache->reset();
      _eventsUsed= 0;
      // Calculate how many more events to generate using our best estimate of our efficiency.
      // Always generate at least one more event so we don't get stuck.
      if(_totalEvents*_maxFuncVal <= 0) {
   oocoutE(nullptr, Generation) << "RooAcceptReject::generateEvent: cannot estimate efficiency...giving up" << endl;
   return 0;
      }

      double eff= _funcSum/(_totalEvents*_maxFuncVal);
      Long64_t extra= 1 + (Long64_t)(1.05*remaining/eff);
      oocxcoutD(static_cast<TObject*>(nullptr), Generation) << "RooAcceptReject::generateEvent: adding " << extra << " events to the cache, eff = " << eff << endl;
      double oldMax(_maxFuncVal);
      while(extra--) {
   addEventToCache();
   if((_maxFuncVal > oldMax)) {
     oocxcoutD(static_cast<TObject*>(nullptr), Generation) << "RooAcceptReject::generateEvent: estimated function maximum increased from "
               << oldMax << " to " << _maxFuncVal << endl;
     oldMax = _maxFuncVal ;
     // Trim cache here
   }
      }
    }

    // Limit cache size to 1M events
    if (_eventsUsed>1000000) {
      _cache->reset() ;
      _eventsUsed = 0 ;
    }

  } else {
    // Generation with a priori maximum knowledge
    _maxFuncVal = _funcMaxVal->getVal() ;

    // Generate enough trials to produce a single accepted event
    event = 0 ;
    while(0==event) {
      addEventToCache() ;
      event = nextAcceptedEvent() ;
    }

  }
  return event;
}



////////////////////////////////////////////////////////////////////////////////
/// Scan through events in the cache which have not been used yet,
/// looking for the first accepted one which is added to the specified
/// container. Return a pointer to the accepted event, or else zero
/// if we use up the cache before we accept an event. The caller does
/// not own the event and it will be overwritten by a subsequent call.

const RooArgSet *RooAcceptReject::nextAcceptedEvent()
{
  const RooArgSet *event = 0;
  while((event= _cache->get(_eventsUsed))) {
    _eventsUsed++ ;
    // accept this cached event?
    double r= RooRandom::uniform();
    if(r*_maxFuncVal > _funcValPtr->getVal()) {
      //cout << " event number " << _eventsUsed << " has been rejected" << endl ;
      continue;
    }
    //cout << " event number " << _eventsUsed << " has been accepted" << endl ;
    // copy this event into the output container
    if(_verbose && (_eventsUsed%1000==0)) {
      cerr << "RooAcceptReject: accepted event (used " << _eventsUsed << " of "
      << _cache->numEntries() << " so far)" << endl;
    }
    break;
  }
  //cout << "accepted event " << _eventsUsed << " of " << _cache->numEntries() << endl ;
  return event;
}



////////////////////////////////////////////////////////////////////////////////
/// Add a trial event to our cache and update our estimates
/// of the function maximum value and integral.

void RooAcceptReject::addEventToCache()
{
  // randomize each discrete argument
  for(auto * cat : static_range_cast<RooCategory*>(_catVars)) cat->randomize();

  // randomize each real argument
  for(auto * real : static_range_cast<RooRealVar*>(_realVars)) real->randomize();

  // calculate and store our function value at this new point
  double val= _funcClone->getVal();
  _funcValPtr->setVal(val);

  // Update the estimated integral and maximum value. Increase our
  // maximum estimate slightly to give a safety margin with a
  // corresponding loss of efficiency.
  if(val > _maxFuncVal) _maxFuncVal= 1.05*val;
  _funcSum+= val;

  // fill a new entry in our cache dataset for this point
  _cache->fill();
  _totalEvents++;

  if (_verbose &&_totalEvents%10000==0) {
    cerr << "RooAcceptReject: generated " << _totalEvents << " events so far." << endl ;
  }

}

double RooAcceptReject::getFuncMax()
{
  // Empirically determine maximum value of function by taking a large number
  // of samples. The actual number depends on the number of dimensions in which
  // the sampling occurs

  // Generate the minimum required number of samples for a reliable maximum estimate
  while(_totalEvents < _minTrials) {
    addEventToCache();

    // Limit cache size to 1M events
    if (_cache->numEntries()>1000000) {
      oocoutI(nullptr, Generation) << "RooAcceptReject::getFuncMax: resetting event cache" << endl ;
      _cache->reset() ;
      _eventsUsed = 0 ;
    }
  }

  return _maxFuncVal ;
}

std::string const& RooAcceptReject::generatorName() const {
   static const std::string name = "RooAcceptReject";
   return name;
}
