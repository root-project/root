/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsPdf.cc,v 1.6 2001/05/14 05:22:54 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

#include <iostream.h>
#include <math.h>
#include "TObjString.h"
#include "TH1.h"
#include "RooFitCore/RooAbsPdf.hh"
#include "RooFitCore/RooDataSet.hh"
#include "RooFitCore/RooArgSet.hh"
#include "RooFitCore/RooArgProxy.hh"
#include "RooFitCore/RooFitContext.hh"
#include "RooFitCore/RooRealVar.hh"

ClassImp(RooAbsPdf) 
;


Bool_t RooAbsPdf::_verboseEval(kFALSE) ;


RooAbsPdf::RooAbsPdf(const char *name, const char *title, const char *unit) : 
  RooAbsReal(name,title,unit), _norm(0), _lastDataSet(0)
{

  resetErrorCounters() ;
  setTraceCounter(0) ;
}


RooAbsPdf::RooAbsPdf(const char *name, const char *title, 
				 Double_t plotMin, Double_t plotMax, const char *unit) :
  RooAbsReal(name,title,plotMin,plotMax,unit), _norm(0), _lastDataSet(0)
{
  resetErrorCounters() ;
  setTraceCounter(0) ;
}



RooAbsPdf::RooAbsPdf(const RooAbsPdf& other, const char* name) : 
  RooAbsReal(other,name), _norm(0), _lastDataSet(0)
{
  resetErrorCounters() ;
  setTraceCounter(0) ;
}




RooAbsPdf::~RooAbsPdf()
{
  if (_norm) delete _norm ;
}


Double_t RooAbsPdf::getVal(const RooDataSet* dset) const
{
  // Unnormalized values are not cached
  // Doing so would be complicated as _norm->getVal() could
  // spoil the cache and interfere with returning the cached
  // return value. Since unnormalized calls are typically
  // done in integration calls, there is no performance hit.
  if (!dset) {
    return traceEval() ;
  }

  // Process change in last data set used
  if (dset != _lastDataSet) {
    if (_verboseEval) cout << "RooAbsPdf:getVal(" << GetName() << ") recalculating normalization" << endl ;
    _lastDataSet = (RooDataSet*) dset ;
 
    // Update dataset pointers of proxies
    for (int i=0 ; i<numProxies() ; i++) {
      getProxy(i).changeDataSet(dset) ;
    }
 
    RooArgSet* depList = getDependents(dset) ;

    // Destroy old normalization & create new
    if (_norm) delete _norm ;

    if (selfNormalized(*depList)) {
      _norm = new RooRealVar(TString(GetName()).Append("Norm"),
			     TString(GetTitle()).Append(" Unit Normalization"),1) ;
    } else {
      _norm = new RooRealIntegral(TString(GetName()).Append("Norm"),
				  TString(GetTitle()).Append(" Integral"),*this,*depList) ;
    }
    delete depList ;
  }

  // Return value of object. Calculated if dirty, otherwise cached value is returned.
  if (isValueDirty() || _norm->isValueDirty() || dset != _lastDataSet) {
    if (_verboseEval) cout << "RooAbsPdf::getVal(" << GetName() << "): recalculating value" << endl ;
    _value = traceEval() / _norm->getVal() ;
  }

  return _value ;
}



Bool_t RooAbsPdf::traceEvalHook(Double_t value) const 
{
  // check for a math error or negative value
  Bool_t error= isnan(value) || (value < 0);

  // do nothing if we are no longer tracing evaluations and there was no error
  if(!error && _traceCount <= 0) return error ;

  // otherwise, print out this evaluations input values and result
  if(error && ++_errorCount <= 10) {
    cout << "*** Evaluation Error " << _errorCount << " ";
    if(_errorCount == 10) cout << "(no more will be printed) ";
  }
  else if(_traceCount > 0) {
    cout << '[' << _traceCount-- << "] ";
  }
  else {
    return error ;
  }

  Print() ;

  return error ;
}




void RooAbsPdf::resetErrorCounters(Int_t resetValue)
{
  _errorCount = resetValue ;
  _negCount   = resetValue ;
}



void RooAbsPdf::setTraceCounter(Int_t value)
{
  _traceCount = value ;
}



void RooAbsPdf::attachDataSet(const RooDataSet* set) 
{
  if(0 != set) recursiveRedirectServers(*set->get(),kFALSE) ;
}




Double_t RooAbsPdf::getLogVal(const RooDataSet* dset) const 
{
  Double_t prob = getVal(dset) ;
  if(prob <= 0) {

    if (_negCount-- > 0) {
      cout << "RooAbsPdf:" << fName << ": calculated prob = " << prob
           << " using" << endl;
      
      cout << "dset ptr = " << (void*)dset << endl ;
      cout << "raw Value = " << getVal(0) << endl ;
      RooArgSet* params = getParameters(dset) ;
      RooArgSet* depends = getDependents(dset) ;	 
      params->Print("v") ;
      depends->Print("v") ;
      delete params ;
      delete depends ;

      if(_negCount == 0) cout << "(no more such warnings will be printed) "<<endl;
    }
    return 0;
  }
  return log(prob);
}



Double_t RooAbsPdf::extendedTerm(UInt_t observed) const {
  // check if this PDF supports extended maximum likelihood fits
  if(!canBeExtended()) {
    cout << fName << ": this PDF does not support extended maximum likelihood"
         << endl;
    return 0;
  }

  Double_t expected= expectedEvents();
  if(expected < 0) {
    cout << fName << ": calculated negative expected events: " << expected
         << endl;
    return 0;
  }

  // calculate and return the negative log-likelihood of the Poisson
  // factor for this dataset, dropping the constant log(observed!)
  Double_t extra= expected - observed*log(expected);

  Bool_t trace(kFALSE) ;
  if(trace) {
    cout << fName << "::extendedTerm: expected " << expected << " events, got "
         << observed << " events. extendedTerm = " << extra << endl;
  }
  return extra;
}



Double_t RooAbsPdf::nLogLikelihood(const RooDataSet* dset, Bool_t extended) const
{
  Double_t result(0);
  const RooArgSet *values(0);
  Stat_t events= dset->GetEntries();

  for(Int_t index= 0; index<events; index++) {

    // get the data values for this event
    values=dset->get(index);
    if(!values) {
      cout << dset->GetName() << "::nLogLikelihood: cannot get values for event "
           << index << endl;
      return 0.0;
    }

    Double_t term = getLogVal(dset);
    if(term == 0) return 0;

    result-= term;
  }

  // include the extended maximum likelihood term, if requested
  if(extended) {
    result+= extendedTerm(events);
  }

  return result;
}



Int_t RooAbsPdf::fitTo(RooDataSet& data, Option_t *options = "", Double_t *minValue= 0) 
{
  RooFitContext context(&data,this) ;
  return context.fit(options,minValue) ;
}

void RooAbsPdf::printToStream(ostream& os, PrintOption opt, TString indent) const
{
  // Print info about this object to the specified stream. In addition to the info
  // from RooAbsArg::printToStream() we add:
  //
  //     Shape : value, units, plot range
  //   Verbose : default binning and print label

  RooAbsArg::printToStream(os,opt,indent);

  if(opt >= Verbose) {
    os << indent << "--- RooAbsPdf ---" << endl;
    os << " Normalization integral: " << endl ;
    _norm->printToStream(os,Standard,TString(indent).Append("  ")) ;
  }
}

RooDataSet *RooAbsPdf::generate(const RooArgSet &whatVars, Int_t nEvents= 0) const {
  // Generate a new dataset containing the specified variables with
  // events sampled from our distribution. Generate the specified
  // number of events or else try to use expectedEvents() if nEvents <= 0.
  // Any variables of this PDF that are not in whatVars will use their
  // current values and be treated as fixed parameters. Returns zero
  // in case of an error. The caller takes ownership of the returned
  // dataset.

  // Initialize an empty dataset with the specified variables.
  RooAbsPdf *pdfClone(0);
  RooArgSet *cloneSet(0);
  RooDataSet *data= initGeneratedDataset(whatVars, cloneSet, pdfClone);
  if(0 == data || 0 == cloneSet || 0 == pdfClone) {
    cout << fName << "::" << ClassName() << ":generate: unable to initialize dataset" << endl;
    delete cloneSet;
    return 0;
  }

  // Calculate the expected number of events if requested
  if(nEvents <= 0) {
    nEvents= expectedEvents() + 0.5;
    if(nEvents <= 0) {
      cout << fName << "::" << ClassName()
	   << ":generate: cannot calculate expected number of events" << endl;
      return 0;
    }
  }

  // Loop over events to generate using our clone
  for(Int_t evt= 0; evt < nEvents; evt++) pdfClone->generateEvent(whatVars);

  delete cloneSet;
  return data;
}

RooDataSet *RooAbsPdf::generate(const RooArgSet &whatVars, const RooDataSet &prototype) const {
  // Generate a new dataset with values of the whatVars variables
  // sampled from our distribution. Use the specified existing dataset
  // as a prototype: the new dataset will contain the same number of
  // events as the prototype, and any prototype variables not in
  // whatVars will be copied into the new dataset for each generated
  // event and also used to set our PDF parameters. The result is a
  // copy of the prototype dataset with only variables in whatVars
  // randomized. Variables in whatVars that are not in the prototype
  // will be added as new columns to the generated dataset.  Returns
  // zero in case of an error. The caller takes ownership of the
  // returned dataset.

  return 0;
}

RooDataSet *RooAbsPdf::initGeneratedDataset(const RooArgSet &vars, RooArgSet *&cloneSet,
					    RooAbsPdf *&pdfClone) const {
  // create a new empty dataset using the specified variables
  TString name(GetName()),title(GetTitle());
  name.Append("Data");
  title.Prepend("Generated From ");
  RooDataSet *data= new RooDataSet(name.Data(),title.Data(),vars);

  // Make a deep-copy clone of ourself
  RooArgSet tmp("PdfBranchNodeList") ;
  branchNodeServerList(&tmp) ;
  cloneSet= tmp.snapshot(kFALSE) ;

  // Find our clone in the snapshot list
  pdfClone = (RooAbsPdf*)cloneSet->FindObject(GetName()) ;
  
  // Attach our clone to the new data set
  pdfClone->attachDataSet(data) ;

  // Reset our clone's error counters
  pdfClone->resetErrorCounters() ;

  return data;
}

void RooAbsPdf::generateEvent(const RooArgSet &vars) {
}
