/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsPdf.cc,v 1.24 2001/08/18 02:13:10 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION --
// RooAbsPdf is the abstract interface for all probability density functions
// The class provides hybrid analytical/numerical normalization for its implementations,
// error tracing and a MC generator interface.
//
// Implementations need to provide the evaluate() member, which returns the (unnormalized)
// PDF value, and optionally indicate support for analytical integration of certain
// variables by reimplementing the getAnalyticalIntegral/analyticalIntegral members.
//
// Implementation should not attempt to perform normalization internally, since they
// do not have the information to do it correctly: integrated dependents may be derived
// and have jacobian terms that are invisible from within the class.
// 

#include <iostream.h>
#include <math.h>
#include "TObjString.h"
#include "TList.h"
#include "TH1.h"
#include "RooFitCore/RooAbsPdf.hh"
#include "RooFitCore/RooDataSet.hh"
#include "RooFitCore/RooArgSet.hh"
#include "RooFitCore/RooArgProxy.hh"
#include "RooFitCore/RooRealProxy.hh"
#include "RooFitCore/RooFitContext.hh"
#include "RooFitCore/RooRealVar.hh"
#include "RooFitCore/RooGenContext.hh"
#include "RooFitCore/RooPlot.hh"

ClassImp(RooAbsPdf) 
;


Int_t RooAbsPdf::_verboseEval(0) ;


RooAbsPdf::RooAbsPdf(const char *name, const char *title) : 
  RooAbsReal(name,title), _norm(0), _lastNormSet(0)
{
  // Constructor with name and title only
  resetErrorCounters() ;
  setTraceCounter(0) ;
}


RooAbsPdf::RooAbsPdf(const char *name, const char *title, 
		     Double_t plotMin, Double_t plotMax) :
  RooAbsReal(name,title,plotMin,plotMax), _norm(0), _lastNormSet(0)
{
  // Constructor with name, title, and plot range
  resetErrorCounters() ;
  setTraceCounter(0) ;
}



RooAbsPdf::RooAbsPdf(const RooAbsPdf& other, const char* name) : 
  RooAbsReal(other,name), _norm(0), _lastNormSet(0)
{
  // Copy constructor
  resetErrorCounters() ;
  setTraceCounter(0) ;
}




RooAbsPdf::~RooAbsPdf()
{
  // Destructor
  if (_norm) delete _norm ;
}


Double_t RooAbsPdf::getVal(const RooArgSet* nset) const
{
  // Return current value with normalization appropriate for given dataset.
  // A null data set pointer will return the unnormalized value

  // Unnormalized values are not cached
  // Doing so would be complicated as _norm->getVal() could
  // spoil the cache and interfere with returning the cached
  // return value. Since unnormalized calls are typically
  // done in integration calls, there is no performance hit.
  if (!nset) return traceEval(nset) ;

  // Process change in last data set used
  Bool_t nsetChanged = (nset != _lastNormSet) ;
  if (nsetChanged) syncNormalization(nset) ;

  // Return value of object. Calculated if dirty, otherwise cached value is returned.
  if ((isValueDirty() || _norm->isValueDirty() || nsetChanged) && operMode()!=AClean) {

//     startTimer() ;
//     _nDirtyCacheHits++ ;

    Double_t rawVal = evaluate(nset) ;
    _value = rawVal / _norm->getVal() ;
    traceEvalPdf(rawVal) ; // Error checking and printing

    if (_verboseEval>1) cout << "RooAbsPdf::getVal(" << GetName() << "): value = " 
			     << rawVal << " / " << _norm->getVal() << " = " << _value << endl ;

    clearValueDirty() ; //setValueDirty(kFALSE) ;
    clearShapeDirty() ; //setShapeDirty(kFALSE) ;    
//     stopTimer() ;

//   } else {
//     _nCleanCacheHits++ ;
  }

  return _value ;
}


void RooAbsPdf::traceEvalPdf(Double_t value) const
{
  // check for a math error or negative value
  Bool_t error= isnan(value) || (value < 0);

  // do nothing if we are no longer tracing evaluations and there was no error
  if(!error && _traceCount <= 0) return ;

  // otherwise, print out this evaluations input values and result
  if(error && ++_errorCount <= 10) {
    cout << "*** Evaluation Error " << _errorCount << " ";
    if(_errorCount == 10) cout << "(no more will be printed) ";
  }
  else if(_traceCount > 0) {
    cout << '[' << _traceCount-- << "] ";
  }
  else {
    return  ;
  }

  Print() ;
}



Double_t RooAbsPdf::getNorm(const RooArgSet* nset) const
{
  if (!nset) return 1 ;

  syncNormalization(nset) ;
  if (_verboseEval>1) cout << "RooAbsPdf::getNorm(" << GetName() << "): norm(" << _norm << ") = " << _norm->getVal() << endl ;
  return _norm->getVal() ;
}




void RooAbsPdf::syncNormalization(const RooArgSet* nset) const
{
  // Check if data sets are identical
  if (nset == _lastNormSet) return ;

  // Check if data sets have identical contents
  if (_lastNormSet) {
    RooNameSet newNames(*nset) ;
    if (newNames==_lastNameSet) {
      if (_verboseEval>1) {
	cout << "RooAbsPdf::syncNormalization(" << GetName() << ") new data and old data sets are identical" << endl ;
      }
      return ;
    }
  }

  if (_verboseEval>0) cout << "RooAbsPdf:syncNormalization(" << GetName() 
			 << ") recreating normalization integral(" 
			 << _lastNormSet << " -> " << nset << ")" << endl ;
  _lastNormSet = (RooArgSet*) nset ;
  _lastNameSet.refill(*nset) ;

  // Update dataset pointers of proxies
  ((RooAbsPdf*) this)->setProxyNormSet(nset) ;
  
  // Allow optional post-processing
  Bool_t fullNorm = syncNormalizationPreHook(_norm,nset) ;
  RooArgSet* depList = fullNorm ? ((RooArgSet*)nset) : getDependents(nset) ;

  // Destroy old normalization & create new
  if (_norm) delete _norm ;
    
  TString nname(GetName()) ; nname.Append("Norm") ;
  if (selfNormalized() || !dependsOn(*depList)) {
    TString ntitle(GetTitle()) ; ntitle.Append(" Unit Normalization") ;
    _norm = new RooRealVar(nname.Data(),ntitle.Data(),1) ;
  } else {
    TString ntitle(GetTitle()) ; ntitle.Append(" Integral") ;
    _norm = new RooRealIntegral(nname.Data(),ntitle.Data(),*this,*depList) ;
  }

  // Allow optional post-processing
  syncNormalizationPostHook(_norm,nset) ;
 
  if (!fullNorm) delete depList ;
}



Bool_t RooAbsPdf::traceEvalHook(Double_t value) const 
{
  // Floating point error checking and tracing for given float value

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
  // Reset error counter to given value 
  _errorCount = resetValue ;
  _negCount   = resetValue ;
}



void RooAbsPdf::setTraceCounter(Int_t value)
{
  // Reset trace counter to given value
  _traceCount = value ;
}

void RooAbsPdf::operModeHook() 
{
//   if (operMode()==AClean) {
//     delete _norm ;
//     _norm = 0 ;
//     _lastNormSet=0 ;
//   }
}



Bool_t RooAbsPdf::matchArgs(const RooArgSet& allDeps, RooArgSet& analDeps, 
			      const RooArgProxy& a) const
{
  // Wrapper function for matchArgsByName()
  TList nameList ;
  nameList.Add(new TObjString(a.absArg()->GetName())) ;
  Bool_t result = matchArgsByName(allDeps,analDeps,nameList) ;
  nameList.Delete() ;
  return result ;
}



Bool_t RooAbsPdf::matchArgs(const RooArgSet& allDeps, RooArgSet& analDeps, 
			      const RooArgProxy& a, const RooArgProxy& b) const
{
  // Wrapper function for matchArgsByName()
  TList nameList ;
  nameList.Add(new TObjString(a.absArg()->GetName())) ;
  nameList.Add(new TObjString(b.absArg()->GetName())) ;  
  Bool_t result = matchArgsByName(allDeps,analDeps,nameList) ;
  nameList.Delete() ;
  return result ;
}



Bool_t RooAbsPdf::matchArgs(const RooArgSet& allDeps, RooArgSet& analDeps, 
			      const RooArgProxy& a, const RooArgProxy& b,
			      const RooArgProxy& c) const
{
  // Wrapper function for matchArgsByName()
  TList nameList ;
  nameList.Add(new TObjString(a.absArg()->GetName())) ;
  nameList.Add(new TObjString(b.absArg()->GetName())) ;
  nameList.Add(new TObjString(c.absArg()->GetName())) ;
  Bool_t result = matchArgsByName(allDeps,analDeps,nameList) ;
  nameList.Delete() ;
  return result ;
}



Bool_t RooAbsPdf::matchArgs(const RooArgSet& allDeps, RooArgSet& analDeps, 
			      const RooArgProxy& a, const RooArgProxy& b,
			      const RooArgProxy& c, const RooArgProxy& d) const
{
  // Wrapper function for matchArgsByName()
  TList nameList ;
  nameList.Add(new TObjString(a.absArg()->GetName())) ;
  nameList.Add(new TObjString(b.absArg()->GetName())) ;
  nameList.Add(new TObjString(c.absArg()->GetName())) ;
  nameList.Add(new TObjString(d.absArg()->GetName())) ;
  Bool_t result = matchArgsByName(allDeps,analDeps,nameList) ;
  nameList.Delete() ;
  return result ;
}


Bool_t RooAbsPdf::matchArgs(const RooArgSet& allDeps, RooArgSet& analDeps, 
			    const RooArgSet& set) const 
{
  // Wrapper function for matchArgsByName()
  TList nameList ;
  TIterator* iter = set.MakeIterator() ;
  RooAbsArg* arg ;
  while (arg=(RooAbsArg*)iter->Next()) {
    nameList.Add(new TObjString(arg->GetName())) ;    
  }
  delete iter ;

  Bool_t result = matchArgsByName(allDeps,analDeps,nameList) ;
  nameList.Delete() ;
  return result ;
}



Bool_t RooAbsPdf::matchArgsByName(const RooArgSet &allArgs, RooArgSet &matchedArgs,
				  const TList &nameList) const {
  // Check if allArgs contains matching elements for each name in nameList. If it does,
  // add the corresponding args from allArgs to matchedArgs and return kTRUE. Otherwise
  // return kFALSE and do not change matchedArgs.

  RooArgSet matched("matched");
  TIterator *iterator= nameList.MakeIterator();
  TObjString *name(0);
  Bool_t isMatched(kTRUE);
  while(isMatched && (name= (TObjString*)iterator->Next())) {
    RooAbsArg *found= allArgs.find(name->String().Data());
    if(found) {
      matched.add(*found);
    }
    else {
      isMatched= kFALSE;
    }
  }
  delete iterator;
  if(isMatched) matchedArgs.add(matched);
  return isMatched;
}

Double_t RooAbsPdf::getLogVal(const RooArgSet* nset) const 
{
  // Return the log of the current value 
  Double_t prob = getVal(nset) ;
  if(prob <= 0) {

    if (_negCount-- > 0) {
      cout << "RooAbsPdf:" << fName << ": calculated prob = " << prob
           << " using" << endl;
      
      cout << "nset ptr = " << (void*)nset << endl ;
      cout << "raw Value = " << getVal(0) << endl ;
      RooArgSet* params = getParameters(nset) ;
      RooArgSet* depends = getDependents(nset) ;	 
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



Double_t RooAbsPdf::extendedTerm(UInt_t observed) const 
{
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




const RooFitResult* RooAbsPdf::fitTo(RooDataSet& data, Option_t *fitOpt, Option_t *optOpt) 
{
  // Fit this PDF to given data set
  RooFitContext context(&data,this) ;
  return context.fit(fitOpt,optOpt) ;
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
    os << indent << "Cached value = " << _value << endl ;
    if (_norm) {
      os << " Normalization integral: " << endl ;
      TString moreIndent(indent) ; moreIndent.Append("   ") ;
      _norm->printToStream(os,Verbose,moreIndent.Data()) ;
      _norm->printToStream(os,Standard,moreIndent.Data()) ;
    }
  }
}

RooDataSet *RooAbsPdf::generate(const RooArgSet &whatVars, Int_t nEvents) const {
  // Generate a new dataset containing the specified variables with
  // events sampled from our distribution. Generate the specified
  // number of events or else try to use expectedEvents() if nEvents <= 0.
  // Any variables of this PDF that are not in whatVars will use their
  // current values and be treated as fixed parameters. Returns zero
  // in case of an error. The caller takes ownership of the returned
  // dataset.

  RooDataSet *generated(0);
  RooGenContext *context= new RooGenContext(*this, whatVars);
  if(context->isValid()) generated= context->generate(nEvents);
  delete context;
  return generated;
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

void RooAbsPdf::generateEvent(const RooArgSet &vars, Int_t maxTrials) {
  // Set the values of the specified subset of our dependents to
  // generate a new "event" according to our PDF model. Use an accept/reject
  // algorithm with at most maxTrials trials.

  // loop over accept/reject trials
  Int_t trial(0);
  while(trial++ < maxTrials) {
    // generate an event according to an envelope function
    Double_t envelopeProb= generateEnvelope(vars);
    // reject this event?
    if(envelopeProb > 0 && RooGenContext::uniform() > envelopeProb) continue;
    // apply resolution smearing
    applyResolution(vars);
    // test if the generated event is within the allowed range for each generated variable
    //if(!vars.areValid()) continue;
  }
  if(trial >= maxTrials) {
    cout << fName << "::" << ClassName() << ":generateEvent: giving up after "
	 << maxTrials << " trials" << endl;
  }
}

Double_t RooAbsPdf::generateEnvelope(const RooArgSet &vars) {
  // Set the values of the specified variables by sampling them from
  // an envelope model whose value is always >= our PDF value with
  // the current settings of our servers that are not in vars. Return
  // zero if the envelope function is exact, or else return the probability
  // (target-prob)/(envelope-prob) that the generated point should be
  // accepted in order to recover the target generator model. The generated
  // value for real variables does not necessarily need to lie within the
  // allowed range of each variable in vars, but this will reduce the efficiency
  // of the generator. The target model is not necessarily the same as our model
  // since resolution effects can be applied as a separate step using
  // the applyResolution() method.

  return 0;
}

Bool_t RooAbsPdf::applyResolution(const RooArgSet &vars) {
  // Apply any resolution smearing to the specified variables, calculated
  // using the current settings of our servers that are not in vars. Return
  // kTRUE if any smearing has been applied, or otherwise kFALSE.

  return kFALSE;
}

Int_t RooAbsPdf::getGenerator(const RooArgSet &directVars, RooArgSet &generatedVars) const {
  // Load generatedVars with the subset of directVars that we can generate events for,
  // and return a code that specifies the generator algorithm we will use. A code of
  // zero indicates that we cannot generate any of the directVars (in this case, nothing
  // should be added to generatedVars). Any non-zero codes will be passed to our generateEvent()
  // implementation, but otherwise its value is arbitrary. The default implemetation of
  // this method returns zero. Subclasses will usually implement this method using the
  // matchArgs() methods to advertise the algorithms they provide.

  return 0;
}

void RooAbsPdf::generateEvent(Int_t code) {
  // Generate an event using the algorithm corresponding to the specified code. The
  // meaning of each code is defined by the getGenerator() implementation. The default
  // implementation does nothing.
}
