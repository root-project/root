/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsPdf.cc,v 1.29 2001/08/29 19:14:20 bevan Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   AB, Adrian Bevan, Liverpool University, bevan@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *   25-Aug-2001 AB Added TH2F * plot methods
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
#include "TH2.h"
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
  // Return current value, normalizated by integrating over
  // all dependents in nset. A null data set pointer will return 
  // the unnormalized value

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

    Double_t rawVal = evaluate() ;
    _value = rawVal / _norm->getVal() ;
    traceEvalPdf(rawVal) ; // Error checking and printing

    if (_verboseEval>1) cout << "RooAbsPdf::getVal(" << GetName() << "): value = " 
			     << rawVal << " / " << _norm->getVal() << " = " << _value << endl ;

    clearValueDirty() ; //setValueDirty(kFALSE) ;
    clearShapeDirty() ; //setShapeDirty(kFALSE) ;    
  }

  return _value ;
}


void RooAbsPdf::traceEvalPdf(Double_t value) const
{
  // Check that passed value is positive and not 'not-a-number'.
  // If not, print an error, until the error counter reaches
  // its set maximum.

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
  // Return the normalization factor for this PDF consisting of the
  // integral over all dependents listed in nset

  if (!nset) return 1 ;

  syncNormalization(nset) ;
  if (_verboseEval>1) cout << "RooAbsPdf::getNorm(" << GetName() << "): norm(" << _norm << ") = " << _norm->getVal() << endl ;
  return _norm->getVal() ;
}




void RooAbsPdf::syncNormalization(const RooArgSet* nset) const
{
  // Synchronize internal caches to hold values appropriate for
  // integration over the dependents in nset

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
  // WVE 08/21/01 Probably obsolete now.

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
  // WVE 08/21/01 Probably obsolete now
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
  TIterator* iter = set.createIterator() ;
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

// get a histogram normalized to the integral of a reference histogram.  
// The new histogram has the same dimensions as the reference one.
TH2F * RooAbsPdf::plot(TH2F & hist, RooAbsReal & varX, RooAbsReal & varY, const char * name)
{
  return plot(&hist, &varX, &varY, name);
}

TH2F * RooAbsPdf::plot(TH2F * hist, RooAbsReal * varX, RooAbsReal * varY, const char * name)
{
  TString histName(name);
  histName.Prepend("_");
  histName.Prepend(fName);

  RooRealVar * var1  = (RooRealVar*)varX;
  RooRealVar * var2  = (RooRealVar*)varY;

  TH2F * histogram = 0;
  if(!var1)
  {
    cout << "RooAbsPdf::" << fName << ", unable to make histogram"<<endl;
  }
  if(!var2)
  {
    cout << "RooAbsPdf::" << fName << ", unable to make histogram"<<endl;
  }
  if(var1 && var2)
  {
    histogram = var1->createHistogram(histName.Data(), var1->GetName(), var2->GetName(), 
                                      hist->GetXaxis()->GetXmin(), hist->GetXaxis()->GetXmax(), hist->GetNbinsX(),
                                      hist->GetYaxis()->GetXmin(), hist->GetYaxis()->GetXmax(), hist->GetNbinsY());

    Double_t binWidthX = (hist->GetXaxis()->GetXmax() - hist->GetXaxis()->GetXmin())/hist->GetNbinsX();
    Double_t binWidthY = (hist->GetYaxis()->GetXmax() - hist->GetYaxis()->GetXmin())/hist->GetNbinsY();
    Double_t prob; 
    //dump into the histogram
    for(Int_t ix = 0; ix < var1->getPlotBins()-1; ix++)
    {
      for(Int_t iy = 0; iy < var2->getPlotBins()-1; iy++)
      {
        var1->setVal( (ix+0.5)*binWidthX + hist->GetXaxis()->GetXmin() );
        var2->setVal( (iy+0.5)*binWidthY + hist->GetYaxis()->GetXmin() );
        prob = evaluate();
        histogram->Fill( var1->getVal() , var2->getVal() , prob );
      }
    }
    if( histogram->Integral() != 0.0 ) histogram->Scale( hist->Integral()/ histogram->Integral());
  }
  return histogram;
}

// get a histogram normalized to the newIntegral given an argset or roorealvars
TH2F * RooAbsPdf::plot(const char * name1, const char * name2, const RooArgSet * nset, const char * name, const Double_t newIntegral)
{
  return plot((RooRealVar*)(nset->find(name1)), (RooRealVar*)(nset->find(name2)), name, newIntegral);
}

TH2F * RooAbsPdf::plot(const char * name1, const char * name2, const RooArgSet& nset, const char * name, const Double_t newIntegral)
{
  return plot(name1, name2, &nset, name, newIntegral);
}

TH2F * RooAbsPdf::plot(RooAbsReal & varX, RooAbsReal & varY, const char * name, const Double_t newIntegral)
{
  return plot(&varX, &varY, name, newIntegral);
}

TH2F * RooAbsPdf::plot(RooAbsReal & varX, RooAbsReal & varY, const Double_t newIntegral, int nX, int nY)
{
  Int_t origNx = varX.getPlotBins();
  Int_t origNy = varY.getPlotBins();
  varX.setPlotBins(nX);
  varY.setPlotBins(nY);
  TH2F * newHist = plot(varX, varY, " ", newIntegral);
  varX.setPlotBins(origNx);
  varY.setPlotBins(origNy);
  return newHist;
}

TH2F * RooAbsPdf::plot(RooAbsReal * varX, RooAbsReal * varY, const char * name, const Double_t newIntegral)
{
  TString histName(name);
  histName.Prepend("_");
  histName.Prepend(fName);

  RooRealVar * var1 = (RooRealVar*)varX;
  RooRealVar * var2 = (RooRealVar*)varY;

  TH2F * histogram = 0;
  if(!var1)
  {
    cout << "RooAbsPdf::" << fName << ", unable to make histogram"<<endl;
  }
  if(!var2)
  {
    cout << "RooAbsPdf::" << fName << ", unable to make histogram"<<endl;
  }
  if(var1 && var2)
  {
    histogram = var1->createHistogram(histName.Data(), *var2);

    Double_t binWidthX = (var1->getPlotMax() - var1->getPlotMin())/var1->getPlotBins();
    Double_t binWidthY = (var2->getPlotMax() - var2->getPlotMin())/var2->getPlotBins();
    Double_t prob; 
    //dump into the histogram
    for(Int_t ix = 0; ix < var1->getPlotBins()-1; ix++)
    {
      for(Int_t iy = 0; iy < var2->getPlotBins()-1; iy++)
      {
        var1->setVal( (ix+0.5)*binWidthX + var1->getPlotMin() );
        var2->setVal( (iy+0.5)*binWidthY + var2->getPlotMin() );
        prob = evaluate();
        histogram->Fill( var1->getVal() , var2->getVal() , prob );
      }
    }
    if( histogram->Integral() != 0.0 ) histogram->Scale( newIntegral/ histogram->Integral());
  }
  return histogram;
}
