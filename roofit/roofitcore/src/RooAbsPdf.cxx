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
// RooAbsPdf is the abstract interface for all probability density
// functions The class provides hybrid analytical/numerical
// normalization for its implementations, error tracing and a MC
// generator interface.
//
// A minimal implementation of a PDF class derived from RooAbsPdf
// should overload the evaluate() function. This functions should
// return PDFs value.
//
//
// [Normalization/Integration]
//
// Although the normalization of a PDF is an integral part of a
// probability density function, normalization is treated separately
// in RooAbsPdf. The reason is that a RooAbsPdf object is more than a
// PDF: it can be a building block for a more complex, composite PDF
// if any of its variables are functions instead of variables. In
// such cases the normalization of the composite may not be simply the
// integral over the dependents of the top level PDF as these are
// functions with potentially non-trivial Jacobian terms themselves.
// Therefore 
//
// --> No explicit attempt should be made to normalize 
//     the functions output in evaluate(). 
//
// In addition, RooAbsPdf objects do not have a static concept of what
// variables are parameters and what variables are dependents (which
// need to be integrated over for a correct PDF normalization). 
// Instead the choice of normalization is always specified each time a
// normalized values is requested from the PDF via the getVal()
// method.
//
// RooAbsPdf manages the entire normalization logic of each PDF with
// help of a RooRealIntegral object, which coordinates the integration
// of a given choice of normalization. By default, RooRealIntegral will
// perform a fully numeric integration of all dependents. However,
// PDFs can advertise one or more (partial) analytical integrals of
// their function, and these will be used by RooRealIntegral, if it
// determines that this is safe (i.e. no hidden Jacobian terms,
// multiplication with other PDFs that have one or more dependents in
// commen etc)
//
// To implement analytical integrals, two functions must be implemented. First,
//
// Int_t getAnalyticalIntegral(const RooArgSet& integSet, RooArgSet& anaIntSet)
// 
// advertises the analytical integrals that are supported. 'integSet'
// is the set of dependents for which integration is requested. The
// function should copy the subset of dependents it can analytically
// integrate to anaIntSet and return a unique identification code for
// this integration configuration.  If no integration can be
// performed, zero should be returned.  Second,
//
// Double_t analyticalIntegral(Int_t code)
//
// Implements the actual analytical integral(s) advertised by
// getAnalyticalIntegral.  This functions will only be called with
// codes returned by getAnalyticalIntegral, except code zero.
//
// The integration range for real each dependent to be integrated can
// be obtained from the dependents' proxy functions min() and
// max(). Never call these proxy functions for any proxy not known to
// be a dependent via the integration code.  Doing so may be
// ill-defined, e.g. in case the proxy holds a function, and will
// trigger an assert. Integrated category dependents should always be
// summed over all of their states.
//
//
//
// [Direct generation of observables]
//
// Any PDF dependent can be generated with the accept/reject method,
// but for certain PDFs more efficient methods may be implemented. To
// implement direct generation of one or more observables, two
// functions need to be implemented, similar to those for analytical
// integrals:
//
// Int_t getGenerator(const RooArgSet& generateVars, RooArgSet& directVars) and
// void generateEvent(Int_t code)
//
// The first function advertises observables that can be generated,
// similar to the way analytical integrals are advertised. The second
// function implements the generator for the advertised observables
//
// The generated dependent values should be store in the proxy
// objects. For this the assignment operator can be used (i.e. xProxy
// = 3.0 ). Never call assign to any proxy not known to be a dependent
// via the generation code.  Doing so may be ill-defined, e.g. in case
// the proxy holds a function, and will trigger an assert


#include "RooFit.h"
#include "RooMsgService.h" 

#include "TClass.h"
#include "Riostream.h"
#include "TMath.h"
#include "TObjString.h"
#include "TPaveText.h"
#include "TList.h"
#include "TH1.h"
#include "TH2.h"
#include "TMatrixD.h"
#include "TMatrixDSym.h"
#include "RooAbsPdf.h"
#include "RooDataSet.h"
#include "RooArgSet.h"
#include "RooArgProxy.h"
#include "RooRealProxy.h"
#include "RooRealVar.h"
#include "RooGenContext.h"
#include "RooPlot.h"
#include "RooCurve.h"
#include "RooNLLVar.h"
#include "RooMinuit.h"
#include "RooCategory.h"
#include "RooNameReg.h"
#include "RooCmdConfig.h"
#include "RooGlobalFunc.h"
#include "RooAddition.h"
#include "RooRandom.h"
#include "RooNumIntConfig.h"
#include "RooProjectedPdf.h"
#include "RooInt.h"
#include "RooCustomizer.h"
#include "RooConstraintSum.h"
#include "RooParamBinning.h"
#include "RooNumCdf.h"
#include "RooFitResult.h"
#include "RooNumGenConfig.h"
#include "RooCachedReal.h"
#include "RooXYChi2Var.h"
#include "RooChi2Var.h"
#include "RooMinimizer.h"
#include "RooRealIntegral.h"
#include <string>

ClassImp(RooAbsPdf) 
;


Int_t RooAbsPdf::_verboseEval = 0;
Bool_t RooAbsPdf::_evalError = kFALSE ;
TString RooAbsPdf::_normRangeOverride ;

//_____________________________________________________________________________
RooAbsPdf::RooAbsPdf() : _norm(0), _normSet(0), _minDimNormValueCache(999), _valueCacheIntOrder(2), _specGeneratorConfig(0)
{
  // Default constructor
  _errorCount = 0 ;
  _negCount = 0 ;
  _rawValue = 0 ;
  _selectComp = kFALSE ;
  _traceCount = 0 ;
}



//_____________________________________________________________________________
RooAbsPdf::RooAbsPdf(const char *name, const char *title) : 
  RooAbsReal(name,title), _norm(0), _normSet(0), _minDimNormValueCache(999), _valueCacheIntOrder(2), _normMgr(this,10), _selectComp(kTRUE), _specGeneratorConfig(0)
{
  // Constructor with name and title only
  resetErrorCounters() ;
  setTraceCounter(0) ;
}



//_____________________________________________________________________________
RooAbsPdf::RooAbsPdf(const char *name, const char *title, 
		     Double_t plotMin, Double_t plotMax) :
  RooAbsReal(name,title,plotMin,plotMax), _norm(0), _normSet(0), _minDimNormValueCache(999), _valueCacheIntOrder(2), _normMgr(this,10), _selectComp(kTRUE), _specGeneratorConfig(0)
{
  // Constructor with name, title, and plot range
  resetErrorCounters() ;
  setTraceCounter(0) ;
}



//_____________________________________________________________________________
RooAbsPdf::RooAbsPdf(const RooAbsPdf& other, const char* name) : 
  RooAbsReal(other,name), _norm(0), _normSet(0), _minDimNormValueCache(other._minDimNormValueCache), _valueCacheIntOrder(other._valueCacheIntOrder),
  _normMgr(other._normMgr,this), _selectComp(other._selectComp), _normRange(other._normRange)
{
  // Copy constructor
  resetErrorCounters() ;
  setTraceCounter(other._traceCount) ;

  if (other._specGeneratorConfig) {
    _specGeneratorConfig = new RooNumGenConfig(*other._specGeneratorConfig) ;
  } else {
    _specGeneratorConfig = 0 ;
  }
}



//_____________________________________________________________________________
RooAbsPdf::~RooAbsPdf()
{
  // Destructor

  if (_specGeneratorConfig) delete _specGeneratorConfig ;
}



//_____________________________________________________________________________
Double_t RooAbsPdf::getVal(const RooArgSet* nset) const
{
  // Return current value, normalizated by integrating over
  // the observables in 'nset'. If 'nset' is 0, the unnormalized value. 
  // is returned. All elements of 'nset' must be lvalues
  //
  // Unnormalized values are not cached
  // Doing so would be complicated as _norm->getVal() could
  // spoil the cache and interfere with returning the cached
  // return value. Since unnormalized calls are typically
  // done in integration calls, there is no performance hit.

  if (!nset) {
    RooArgSet* tmp = _normSet ;
    _normSet = 0 ;
    Double_t val = evaluate() ;
    _normSet = tmp ;
    Bool_t error = traceEvalPdf(val) ;
    cxcoutD(Tracing) << IsA()->GetName() << "::getVal(" << GetName() 
		     << "): value = " << val << " (unnormalized)" << endl ;
    if (error) {
      raiseEvalError() ;
      return 0 ;
    }
    return val ;
  }

  // Process change in last data set used
  Bool_t nsetChanged(kFALSE) ;
  if (nset!=_normSet || _norm==0) {
    nsetChanged = syncNormalization(nset) ;
  }

  // Return value of object. Calculated if dirty, otherwise cached value is returned.
  if ((isValueDirty() || nsetChanged || _norm->isValueDirty()) && operMode()!=AClean) {

    // Evaluate numerator
    Double_t rawVal = evaluate() ;
    Bool_t error = traceEvalPdf(rawVal) ; // Error checking and printing

    // Evaluate denominator
    Double_t normVal(_norm->getVal()) ;
    
    Double_t normError(kFALSE) ;
    if (normVal==0.) {
      normError=kTRUE ;
      logEvalError("p.d.f normalization integral is zero") ;  
    }

    // Raise global error flag if problems occur
    if (normError||error) raiseEvalError() ;

    _value = normError ? 0 : (rawVal / normVal) ;

    cxcoutD(Tracing) << "RooAbsPdf::getVal(" << GetName() << ") new value with norm " << _norm->GetName() << " = " << rawVal << "/" << normVal << " = " << _value << endl ;

    clearValueDirty() ; //setValueDirty(kFALSE) ;
    clearShapeDirty() ; //setShapeDirty(kFALSE) ;    
  } 

  if (_traceCount>0) {
    cxcoutD(Tracing) << "[" << _traceCount << "] " ;
    Int_t tmp = _traceCount ;
    _traceCount = 0 ;
    printStream(ccoutD(Tracing),kName|kValue|kArgs,kSingleLine) ;
    _traceCount = tmp-1  ;
  }

  return _value ;
}



//_____________________________________________________________________________
Double_t RooAbsPdf::analyticalIntegralWN(Int_t code, const RooArgSet* normSet, const char* rangeName) const
{
  // Analytical integral with normalization (see RooAbsReal::analyticalIntegralWN() for further information)
  //
  // This function applies the normalization specified by 'normSet' to the integral returned
  // by RooAbsReal::analyticalIntegral(). The passthrough scenario (code=0) is also changed
  // to return a normalized answer

  cxcoutD(Eval) << "RooAbsPdf::analyticalIntegralWN(" << GetName() << ") code = " << code << " normset = " << (normSet?*normSet:RooArgSet()) << endl ;


  if (code==0) return getVal(normSet) ;
  if (normSet) {
    return analyticalIntegral(code,rangeName) / getNorm(normSet) ;
  } else {
    return analyticalIntegral(code,rangeName) ;
  }
}



//_____________________________________________________________________________
Bool_t RooAbsPdf::traceEvalPdf(Double_t value) const
{
  // Check that passed value is positive and not 'not-a-number'.  If
  // not, print an error, until the error counter reaches its set
  // maximum.

  // check for a math error or negative value
  Bool_t error= isnan(value) || (value < 0);
  if (isnan(value)) {
    logEvalError(Form("p.d.f value is Not-a-Number (%f), forcing value to zero",value)) ;
  }
  if (value<0) {
    logEvalError(Form("p.d.f value is less than zero (%f), forcing value to zero",value)) ;
  }

  // do nothing if we are no longer tracing evaluations and there was no error
  if(!error) return error ;

  // otherwise, print out this evaluations input values and result
  if(++_errorCount <= 10) {
    cxcoutD(Tracing) << "*** Evaluation Error " << _errorCount << " ";
    if(_errorCount == 10) cxcoutD(Tracing) << "(no more will be printed) ";
  }
  else {
    return error  ;
  }

  Print() ;
  return error ;
}




//_____________________________________________________________________________
Double_t RooAbsPdf::getNorm(const RooArgSet* nset) const
{
  // Return the integral of this PDF over all observables listed in 'nset'. 

  if (!nset) return 1 ;

  syncNormalization(nset,kTRUE) ;
  if (_verboseEval>1) cxcoutD(Tracing) << IsA()->GetName() << "::getNorm(" << GetName() << "): norm(" << _norm << ") = " << _norm->getVal() << endl ;

  Double_t ret = _norm->getVal() ;
//   cout << "RooAbsPdf::getNorm(" << GetName() << ") norm obj = " << _norm->GetName() << endl ;
  if (ret==0.) {
    if(++_errorCount <= 10) {
      coutW(Eval) << "RooAbsPdf::getNorm(" << GetName() << ":: WARNING normalization is zero, nset = " ;  nset->Print("1") ;
      if(_errorCount == 10) coutW(Eval) << "RooAbsPdf::getNorm(" << GetName() << ") INFO: no more messages will be printed " << endl ;
    }
  }

  return ret ;
}



//_____________________________________________________________________________
const RooAbsReal* RooAbsPdf::getNormObj(const RooArgSet* nset, const RooArgSet* iset, const TNamed* rangeName) const 
{
  // Return pointer to RooAbsReal object that implements calculation of integral over observables iset in range
  // rangeName, optionally taking the integrand normalized over observables nset


  // Check normalization is already stored
  CacheElem* cache = (CacheElem*) _normMgr.getObj(nset,iset,0,rangeName) ;
  if (cache) {
    return cache->_norm ;
  }

  // If not create it now
  RooArgSet* depList = getObservables(iset) ;
  RooAbsReal* norm = createIntegral(*depList,*nset, *getIntegratorConfig(), RooNameReg::str(rangeName)) ;
  delete depList ;

  // Store it in the cache
  cache = new CacheElem(*norm) ;
  _normMgr.setObj(nset,iset,cache,rangeName) ;

  // And return the newly created integral
  return norm ;
}



//_____________________________________________________________________________
Bool_t RooAbsPdf::syncNormalization(const RooArgSet* nset, Bool_t adjustProxies) const
{
  // Verify that the normalization integral cached with this PDF
  // is valid for given set of normalization observables
  //
  // If not, the cached normalization integral (if any) is deleted
  // and a new integral is constructed for use with 'nset'
  // Elements in 'nset' can be discrete and real, but must be lvalues
  //
  // For functions that declare to be self-normalized by overloading the
  // selfNormalized() function, a unit normalization is always constructed


  //cout << IsA()->GetName() << "::syncNormalization(" << GetName() << ") nset = " << nset << " = " << (nset?*nset:RooArgSet()) << endl ;

  _normSet = (RooArgSet*) nset ;

  // Check if data sets are identical
  CacheElem* cache = (CacheElem*) _normMgr.getObj(nset) ;
  if (cache) {

    Bool_t nsetChanged = (_norm!=cache->_norm) ;
    _norm = cache->_norm ;


//     cout << "returning existing object " << _norm->GetName() << endl ;

    if (nsetChanged && adjustProxies) {
      // Update dataset pointers of proxies
      ((RooAbsPdf*) this)->setProxyNormSet(nset) ;
    }
  
    return nsetChanged ;
  }
    
  // Update dataset pointers of proxies
  if (adjustProxies) {
    ((RooAbsPdf*) this)->setProxyNormSet(nset) ;
  }
  
  RooArgSet* depList = getObservables(nset) ;

  if (_verboseEval>0) {
    if (!selfNormalized()) {
      cxcoutD(Tracing) << IsA()->GetName() << "::syncNormalization(" << GetName() 
	   << ") recreating normalization integral " << endl ;
      if (depList) depList->printStream(ccoutD(Tracing),kName|kValue|kArgs,kSingleLine) ; else ccoutD(Tracing) << "<none>" << endl ;
    } else {
      cxcoutD(Tracing) << IsA()->GetName() << "::syncNormalization(" << GetName() << ") selfNormalized, creating unit norm" << endl;
    }
  }

  // Destroy old normalization & create new
  if (selfNormalized() || !dependsOn(*depList)) {    
    TString ntitle(GetTitle()) ; ntitle.Append(" Unit Normalization") ;
    TString nname(GetName()) ; nname.Append("_UnitNorm") ;
    _norm = new RooRealVar(nname.Data(),ntitle.Data(),1) ;
  } else {    
    const char* nr = (_normRangeOverride.Length()>0 ? _normRangeOverride.Data() : (_normRange.Length()>0 ? _normRange.Data() : 0)) ;

//     cout << "RooAbsPdf::syncNormalization(" << GetName() << ") rangeName for normalization is " << (nr?nr:"<null>") << endl ;
    RooRealIntegral* normInt = (RooRealIntegral*) createIntegral(*depList,*getIntegratorConfig(),nr) ;
    normInt->getVal() ;
//     cout << "resulting normInt = " << normInt->GetName() << endl ;

    RooArgSet* normParams = normInt->getVariables() ;
    if (normParams->getSize()>0 && normParams->getSize()<3 && normInt->numIntRealVars().getSize()>=_minDimNormValueCache) {
      coutI(Caching) << "RooAbsPdf::syncNormalization(" << GetName() << ") INFO: constructing " << normParams->getSize() 
		     << "-dim value cache for normalization integral over " << *depList << endl ;
      string name = Form("%s_CACHE_[%s]",normInt->GetName(),normParams->contentsString().c_str()) ;
      RooCachedReal* cachedNorm = new RooCachedReal(name.c_str(),name.c_str(),*normInt,*normParams) ;     
      cachedNorm->setInterpolationOrder(_valueCacheIntOrder) ;
      cachedNorm->addOwnedComponents(*normInt) ;
      _norm = cachedNorm ;
    } else {
      _norm = normInt ;
    } 
    delete normParams ;
  }



  // Register new normalization with manager (takes ownership)
  cache = new CacheElem(*_norm) ;
  _normMgr.setObj(nset,cache) ;

//     cout << "making new object " << _norm->GetName() << endl ;

  delete depList ;
  return kTRUE ;
}



//_____________________________________________________________________________
void RooAbsPdf::setNormValueCaching(Int_t minNumIntDim, Int_t ipOrder) 
{ 
  // Activate caching of normalization integral values in a interpolated histogram 
  // for integrals that exceed the specified minimum number of numerically integrated
  // dimensions, _and_ of which the integral has at most 2 parameters. 
  //
  // The cache is scanned with a granularity defined by a binning named "cache" in the 
  // scanned integral parameters and is interpolated to given order.
  // The cache values are kept for the livetime of the ROOT session/application
  // and are persisted along with the object in case the p.d.f. is persisted
  // in a RooWorkspace
  // 
  // This feature can substantially speed up fits and improve convergence with slow 
  // multi-dimensional integrals whose value varies slowly with the parameters so that the
  // an interpolated histogram is a good approximation of the true integral value.
  // The improved convergence behavior is a result of making the value of the normalization
  // integral deterministic for each value of the parameters. If (multi-dimensional) numeric
  // integrals are calculated at insufficient precision (>=1e-7) MINUIT convergence may
  // be impaired by the effects numerical noise that can cause that subsequent evaluations
  // of an integral at the same point in parameter space can give slightly different answers.

  _minDimNormValueCache = minNumIntDim ; 
  _valueCacheIntOrder = ipOrder ; 
}




//_____________________________________________________________________________
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
    cxcoutD(Tracing) << "*** Evaluation Error " << _errorCount << " ";
    if(_errorCount == 10) ccoutD(Tracing) << "(no more will be printed) ";
  }
  else if(_traceCount > 0) {
    ccoutD(Tracing) << '[' << _traceCount-- << "] ";
  }
  else {
    return error ;
  }

  Print() ;

  return error ;
}




//_____________________________________________________________________________
void RooAbsPdf::resetErrorCounters(Int_t resetValue)
{
  // Reset error counter to given value, limiting the number
  // of future error messages for this pdf to 'resetValue'

  _errorCount = resetValue ;
  _negCount   = resetValue ;
}



//_____________________________________________________________________________
void RooAbsPdf::setTraceCounter(Int_t value, Bool_t allNodes)
{
  // Reset trace counter to given value, limiting the
  // number of future trace messages for this pdf to 'value'

  if (!allNodes) {
    _traceCount = value ;
    return ; 
  } else {
    RooArgList branchList ;
    branchNodeServerList(&branchList) ;
    TIterator* iter = branchList.createIterator() ;
    RooAbsArg* arg ;
    while((arg=(RooAbsArg*)iter->Next())) {
      RooAbsPdf* pdf = dynamic_cast<RooAbsPdf*>(arg) ;
      if (pdf) pdf->setTraceCounter(value,kFALSE) ;
    }
    delete iter ;
  }

}




//_____________________________________________________________________________
Double_t RooAbsPdf::getLogVal(const RooArgSet* nset) const 
{
  // Return the log of the current value with given normalization
  // An error message is printed if the argument of the log is negative.

  Double_t prob = getVal(nset) ;
  if(prob < 0) {

    logEvalError("getLogVal() top-level p.d.f evaluates to a negative number") ;

    return 0;
  }
  if(prob == 0) {

    logEvalError("getLogVal() top-level p.d.f evaluates to zero") ;

    return log((double)0);
  }
  return log(prob);
}



//_____________________________________________________________________________
Double_t RooAbsPdf::extendedTerm(UInt_t observed, const RooArgSet* nset) const 
{
  // Returned the extended likelihood term (Nexpect - Nobserved*log(NExpected)
  // of this PDF for the given number of observed events
  //
  // For successfull operation the PDF implementation must indicate
  // it is extendable by overloading canBeExtended() and must
  // implemented the expectedEvents() function.

  // check if this PDF supports extended maximum likelihood fits
  if(!canBeExtended()) {
    coutE(InputArguments) << fName << ": this PDF does not support extended maximum likelihood"
         << endl;
    return 0;
  }

  Double_t expected= expectedEvents(nset);
  if(expected < 0) {
    coutE(InputArguments) << fName << ": calculated negative expected events: " << expected
         << endl;
    return 0;
  }

  // calculate and return the negative log-likelihood of the Poisson
  // factor for this dataset, dropping the constant log(observed!)
  Double_t extra= expected - observed*log(expected);

//   cout << "RooAbsPdf::extendedTerm(" << GetName() << ") observed = " << observed << " expected = " << expected << endl ;

  Bool_t trace(kFALSE) ;
  if(trace) {
    cxcoutD(Tracing) << fName << "::extendedTerm: expected " << expected << " events, got "
		     << observed << " events. extendedTerm = " << extra << endl;
  }
  return extra;
}



//_____________________________________________________________________________
RooAbsReal* RooAbsPdf::createNLL(RooAbsData& data, const RooCmdArg& arg1, const RooCmdArg& arg2, const RooCmdArg& arg3, const RooCmdArg& arg4, 
                                             const RooCmdArg& arg5, const RooCmdArg& arg6, const RooCmdArg& arg7, const RooCmdArg& arg8) 
{
  // Construct representation of -log(L) of PDFwith given dataset. If dataset is unbinned, an unbinned likelihood is constructed. If the dataset
  // is binned, a binned likelihood is constructed. 
  //
  // The following named arguments are supported
  //
  // ConditionalObservables(const RooArgSet& set) -- Do not normalize PDF over listed observables
  // Extended(Bool_t flag)           -- Add extended likelihood term, off by default
  // Range(const char* name)         -- Fit only data inside range with given name
  // Range(Double_t lo, Double_t hi) -- Fit only data inside given range. A range named "fit" is created on the fly on all observables.
  //                                    Multiple comma separated range names can be specified.
  // SumCoefRange(const char* name)  -- Set the range in which to interpret the coefficients of RooAddPdf components 
  // NumCPU(int num)                 -- Parallelize NLL calculation on num CPUs
  // Optimize(Bool_t flag)           -- Activate constant term optimization (on by default)
  // SplitRange(Bool_t flag)         -- Use separate fit ranges in a simultaneous fit. Actual range name for each
  //                                    subsample is assumed to by rangeName_{indexState} where indexState
  //                                    is the state of the master index category of the simultaneous fit
  // Constrain(const RooArgSet&pars) -- For p.d.f.s that contain internal parameter constraint terms, only apply constraints to given subset of parameters
  // ExternalConstraints(const RooArgSet& ) -- Include given external constraints to likelihood
  // Verbose(Bool_t flag)           -- Constrols RooFit informational messages in likelihood construction
  // CloneData(Bool flag)           -- Use clone of dataset in NLL (default is true)
  // 
  // 
  
  RooLinkedList l ;
  l.Add((TObject*)&arg1) ;  l.Add((TObject*)&arg2) ;  
  l.Add((TObject*)&arg3) ;  l.Add((TObject*)&arg4) ;
  l.Add((TObject*)&arg5) ;  l.Add((TObject*)&arg6) ;  
  l.Add((TObject*)&arg7) ;  l.Add((TObject*)&arg8) ;
  return createNLL(data,l) ;
}




//_____________________________________________________________________________
RooAbsReal* RooAbsPdf::createNLL(RooAbsData& data, const RooLinkedList& cmdList) 
{
  // Construct representation of -log(L) of PDFwith given dataset. If dataset is unbinned, an unbinned likelihood is constructed. If the dataset
  // is binned, a binned likelihood is constructed. 
  //
  // See RooAbsPdf::createNLL(RooAbsData& data, RooCmdArg arg1, RooCmdArg arg2, RooCmdArg arg3, RooCmdArg arg4, 
  //                                    RooCmdArg arg5, RooCmdArg arg6, RooCmdArg arg7, RooCmdArg arg8) 
  //
  // for documentation of options


  // Select the pdf-specific commands 
  RooCmdConfig pc(Form("RooAbsPdf::createNLL(%s)",GetName())) ;

  pc.defineString("rangeName","RangeWithName",0,"",kTRUE) ;
  pc.defineString("addCoefRange","SumCoefRange",0,"") ;
  pc.defineDouble("rangeLo","Range",0,-999.) ;
  pc.defineDouble("rangeHi","Range",1,-999.) ;
  pc.defineInt("splitRange","SplitRange",0,0) ;
  pc.defineInt("ext","Extended",0,2) ;
  pc.defineInt("numcpu","NumCPU",0,1) ;
  pc.defineInt("verbose","Verbose",0,0) ;
  pc.defineInt("optConst","Optimize",0,0) ;
  pc.defineInt("cloneData","CloneData",2,0) ;
  pc.defineSet("projDepSet","ProjectedObservables",0,0) ;
  pc.defineSet("cPars","Constrain",0,0) ;
  pc.defineInt("constrAll","Constrained",0,0) ;
  pc.defineSet("extCons","ExternalConstraints",0,0) ;
  pc.defineMutex("Range","RangeWithName") ;
  pc.defineMutex("Constrain","Constrained") ;
  
  // Process and check varargs 
  pc.process(cmdList) ;
  if (!pc.ok(kTRUE)) {
    return 0 ;
  }

  // Decode command line arguments
  const char* rangeName = pc.getString("rangeName",0,kTRUE) ;
  const char* addCoefRangeName = pc.getString("addCoefRange",0,kTRUE) ;
  Int_t ext      = pc.getInt("ext") ;
  Int_t numcpu   = pc.getInt("numcpu") ;
  Int_t splitr   = pc.getInt("splitRange") ;
  Bool_t verbose = pc.getInt("verbose") ;
  Int_t optConst = pc.getInt("optConst") ;
  Int_t cloneData = pc.getInt("cloneData") ;
  
  // If no explicit cloneData command is specified, cloneData is set to true if optimization is activated
  if (cloneData==2) {
    cloneData = optConst ;
  }
  RooArgSet* cPars = pc.getSet("cPars") ;
  Bool_t doStripDisconnected=kFALSE ;

  // If no explicit list of parameters to be constrained is specified apply default algorithm
  // All terms of RooProdPdfs that do not contain observables and share a parameters with one or more
  // terms that do contain observables are added as constraints.
  if (!cPars) {    
    cPars = getParameters(data,kFALSE) ;
    doStripDisconnected=kTRUE ;
  }
  const RooArgSet* extCons = pc.getSet("extCons") ;

  // Process automatic extended option
  if (ext==2) {
    ext = ((extendMode()==CanBeExtended || extendMode()==MustBeExtended)) ? 1 : 0 ;
    if (ext) {
      coutI(Minimization) << "p.d.f. provides expected number of events, including extended term in likelihood." << endl ;
    }
  }

  if (pc.hasProcessed("Range")) {
    Double_t rangeLo = pc.getDouble("rangeLo") ;
    Double_t rangeHi = pc.getDouble("rangeHi") ;
   
    // Create range with name 'fit' with above limits on all observables
    RooArgSet* obs = getObservables(&data) ;
    TIterator* iter = obs->createIterator() ;
    RooAbsArg* arg ;
    while((arg=(RooAbsArg*)iter->Next())) {
      RooRealVar* rrv =  dynamic_cast<RooRealVar*>(arg) ;
      if (rrv) rrv->setRange("fit",rangeLo,rangeHi) ;
    }
    // Set range name to be fitted to "fit"
    rangeName = "fit" ;
  }

  RooArgSet projDeps ;
  RooArgSet* tmp = pc.getSet("projDepSet") ;  
  if (tmp) {
    projDeps.add(*tmp) ;
  }

  // Construct NLL
  RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::CollectErrors) ;
  RooAbsReal* nll ;
  string baseName = Form("nll_%s_%s",GetName(),data.GetName()) ;
  if (!rangeName || strchr(rangeName,',')==0) {
    // Simple case: default range, or single restricted range
    //cout<<"FK: Data test 1: "<<data.sumEntries()<<endl;

    nll = new RooNLLVar(baseName.c_str(),"-log(likelihood)",*this,data,projDeps,ext,rangeName,addCoefRangeName,numcpu,kFALSE,verbose,splitr,cloneData) ;

  } else {
    // Composite case: multiple ranges
    RooArgList nllList ;
    char* buf = new char[strlen(rangeName)+1] ;
    strlcpy(buf,rangeName,strlen(rangeName)+1) ;
    char* token = strtok(buf,",") ;
    while(token) {
      RooAbsReal* nllComp = new RooNLLVar(Form("%s_%s",baseName.c_str(),token),"-log(likelihood)",*this,data,projDeps,ext,token,addCoefRangeName,numcpu,kFALSE,verbose,splitr,cloneData) ;
      nllList.add(*nllComp) ;
      token = strtok(0,",") ;
    }
    delete[] buf ;
    nll = new RooAddition(baseName.c_str(),"-log(likelihood)",nllList,kTRUE) ;
  }
  RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::PrintErrors) ;
  
  // Collect internal and external constraint specifications
  RooArgSet allConstraints ;
  if (cPars && cPars->getSize()>0) {
    RooArgSet* constraints = getAllConstraints(*data.get(),*cPars,doStripDisconnected) ;
    allConstraints.add(*constraints) ;
    delete constraints ;
    
  }
  if (extCons) {
    allConstraints.add(*extCons) ;
  }

  // Include constraints, if any, in likelihood
  RooAbsReal* nllCons(0) ;
  if (allConstraints.getSize()>0 && cPars) {   

    coutI(Minimization) << " Including the following contraint terms in minimization: " << allConstraints << endl ;
    
    nllCons = new RooConstraintSum(Form("%s_constr",baseName.c_str()),"nllCons",allConstraints,*cPars) ;
    RooAbsReal* orignll = nll ;

    nll = new RooAddition(Form("%s_with_constr",baseName.c_str()),"nllWithCons",RooArgSet(*nll,*nllCons)) ;
    nll->addOwnedComponents(RooArgSet(*orignll,*nllCons)) ;
  }

  if (optConst) {

    nll->constOptimizeTestStatistic(RooAbsArg::Activate) ;
  }

  if (doStripDisconnected) {
    delete cPars ;
  }
  return nll ;
}






//_____________________________________________________________________________
RooFitResult* RooAbsPdf::fitTo(RooAbsData& data, const RooCmdArg& arg1, const RooCmdArg& arg2, const RooCmdArg& arg3, const RooCmdArg& arg4, 
                                                 const RooCmdArg& arg5, const RooCmdArg& arg6, const RooCmdArg& arg7, const RooCmdArg& arg8) 
{
  // Fit PDF to given dataset. If dataset is unbinned, an unbinned maximum likelihood is performed. If the dataset
  // is binned, a binned maximum likelihood is performed. By default the fit is executed through the MINUIT
  // commands MIGRAD, HESSE and MINOS in succession.
  //
  // The following named arguments are supported
  //
  // Options to control construction of -log(L)
  // ------------------------------------------
  // ConditionalObservables(const RooArgSet& set) -- Do not normalize PDF over listed observables
  // Extended(Bool_t flag)           -- Add extended likelihood term, off by default
  // Range(const char* name)         -- Fit only data inside range with given name
  // Range(Double_t lo, Double_t hi) -- Fit only data inside given range. A range named "fit" is created on the fly on all observables.
  //                                    Multiple comma separated range names can be specified.
  // SumCoefRange(const char* name)  -- Set the range in which to interpret the coefficients of RooAddPdf components 
  // NumCPU(int num)                 -- Parallelize NLL calculation on num CPUs
  // SplitRange(Bool_t flag)         -- Use separate fit ranges in a simultaneous fit. Actual range name for each
  //                                    subsample is assumed to by rangeName_{indexState} where indexState
  //                                    is the state of the master index category of the simultaneous fit
  // Constrained()                   -- Apply all constrained contained in the p.d.f. in the likelihood 
  // Contrain(const RooArgSet&pars)  -- Apply constraints to listed parameters in likelihood using internal constrains in p.d.f
  // ExternalConstraints(const RooArgSet& ) -- Include given external constraints to likelihood
  //
  // Options to control flow of fit procedure
  // ----------------------------------------
  //
  // Minimizer(type,algo)           -- Choose minimization package and algorithm to use. Default is MINUIT/MIGRAD through the RooMinuit
  //                                   interface, but others can be specified (through RooMinimizer interface)
  //
  //                                          Type         Algorithm
  //                                          ------       ---------
  //                                          Minuit       migrad, simplex, minimize (=migrad+simplex), migradimproved (=migrad+improve)
  //                                          Minuit2      migrad, simplex, minimize, scan
  //                                          GSLMultiMin  conjugatefr, conjugatepr, bfgs, bfgs2, steepestdescent
  //                                          GSLSimAn     -
  //
  // 
  // InitialHesse(Bool_t flag)      -- Flag controls if HESSE before MIGRAD as well, off by default
  // Optimize(Bool_t flag)          -- Activate constant term optimization of test statistic during minimization (on by default)
  // Hesse(Bool_t flag)             -- Flag controls if HESSE is run after MIGRAD, on by default
  // Minos(Bool_t flag)             -- Flag controls if MINOS is run after HESSE, on by default
  // Minos(const RooArgSet& set)    -- Only run MINOS on given subset of arguments
  // Save(Bool_t flag)              -- Flac controls if RooFitResult object is produced and returned, off by default
  // Strategy(Int_t flag)           -- Set Minuit strategy (0 through 2, default is 1)
  // FitOptions(const char* optStr) -- Steer fit with classic options string (for backward compatibility). Use of this option
  //                                   excludes use of any of the new style steering options.
  //
  // SumW2Error(Bool_t flag)        -- Apply correaction to errors and covariance matrix using sum-of-weights covariance matrix
  //                                   to obtain correct error for weighted likelihood fits. If this option is activated the
  //                                   corrected covariance matrix is calculated as Vcorr = V C-1 V, where V is the original 
  //                                   covariance matrix and C is the inverse of the covariance matrix calculated using the
  //                                   weights squared
  //
  // Options to control informational output
  // ---------------------------------------
  // Verbose(Bool_t flag)           -- Flag controls if verbose output is printed (NLL, parameter changes during fit
  // Timer(Bool_t flag)             -- Time CPU and wall clock consumption of fit steps, off by default
  // PrintLevel(Int_t level)        -- Set Minuit print level (-1 through 3, default is 1). At -1 all RooFit informational 
  //                                   messages are suppressed as well
  // Warnings(Bool_t flag)          -- Enable or disable MINUIT warnings (enabled by default)
  // PrintEvalErrors(Int_t numErr)  -- Control number of p.d.f evaluation errors printed per likelihood evaluation. A negative
  //                                   value suppress output completely, a zero value will only print the error count per p.d.f component,
  //                                   a positive value is will print details of each error up to numErr messages per p.d.f component.
  // 
  // 
  
  RooLinkedList l ;
  l.Add((TObject*)&arg1) ;  l.Add((TObject*)&arg2) ;  
  l.Add((TObject*)&arg3) ;  l.Add((TObject*)&arg4) ;
  l.Add((TObject*)&arg5) ;  l.Add((TObject*)&arg6) ;  
  l.Add((TObject*)&arg7) ;  l.Add((TObject*)&arg8) ;
  return fitTo(data,l) ;
}



//_____________________________________________________________________________
RooFitResult* RooAbsPdf::fitTo(RooAbsData& data, const RooLinkedList& cmdList) 
{
  // Fit PDF to given dataset. If dataset is unbinned, an unbinned maximum likelihood is performed. If the dataset
  // is binned, a binned maximum likelihood is performed. By default the fit is executed through the MINUIT
  // commands MIGRAD, HESSE and MINOS in succession.
  //
  // See RooAbsPdf::fitTo(RooAbsData& data, RooCmdArg arg1, RooCmdArg arg2, RooCmdArg arg3, RooCmdArg arg4, 
  //                                         RooCmdArg arg5, RooCmdArg arg6, RooCmdArg arg7, RooCmdArg arg8) 
  //
  // for documentation of options


  // Select the pdf-specific commands 
  RooCmdConfig pc(Form("RooAbsPdf::fitTo(%s)",GetName())) ;

  RooLinkedList fitCmdList(cmdList) ;
  RooLinkedList nllCmdList = pc.filterCmdList(fitCmdList,"ProjectedObservables,Extended,Range,RangeWithName,SumCoefRange,NumCPU,SplitRange,Constrained,Constrain,ExternalConstraints,CloneData") ;

  pc.defineString("fitOpt","FitOptions",0,"") ;
  pc.defineInt("optConst","Optimize",0,1) ;
  pc.defineInt("verbose","Verbose",0,0) ;
  pc.defineInt("doSave","Save",0,0) ;
  pc.defineInt("doTimer","Timer",0,0) ;
  pc.defineInt("plevel","PrintLevel",0,1) ;
  pc.defineInt("strat","Strategy",0,1) ;
  pc.defineInt("initHesse","InitialHesse",0,0) ;
  pc.defineInt("hesse","Hesse",0,1) ;
  pc.defineInt("minos","Minos",0,0) ;
  pc.defineInt("ext","Extended",0,2) ;
  pc.defineInt("numcpu","NumCPU",0,1) ;
  pc.defineInt("numee","PrintEvalErrors",0,10) ;
  pc.defineInt("doEEWall","EvalErrorWall",0,1) ;
  pc.defineInt("doWarn","Warnings",0,1) ;
  pc.defineInt("doSumW2","SumW2Error",0,-1) ;
  pc.defineString("mintype","Minimizer",0,"") ;
  pc.defineString("minalg","Minimizer",1,"") ;
  pc.defineObject("minosSet","Minos",0,0) ;
  pc.defineSet("cPars","Constrain",0,0) ;
  pc.defineSet("extCons","ExternalConstraints",0,0) ;
  pc.defineMutex("FitOptions","Verbose") ;
  pc.defineMutex("FitOptions","Save") ;
  pc.defineMutex("FitOptions","Timer") ;
  pc.defineMutex("FitOptions","Strategy") ;
  pc.defineMutex("FitOptions","InitialHesse") ;
  pc.defineMutex("FitOptions","Hesse") ;
  pc.defineMutex("FitOptions","Minos") ;
  pc.defineMutex("Range","RangeWithName") ;
  pc.defineMutex("InitialHesse","Minimizer") ;
  
  // Process and check varargs 
  pc.process(fitCmdList) ;
  if (!pc.ok(kTRUE)) {
    return 0 ;
  }

  // Decode command line arguments
  const char* fitOpt = pc.getString("fitOpt",0,kTRUE) ;
  Int_t optConst = pc.getInt("optConst") ;
  Int_t verbose  = pc.getInt("verbose") ;
  Int_t doSave   = pc.getInt("doSave") ;
  Int_t doTimer  = pc.getInt("doTimer") ;
  Int_t plevel    = pc.getInt("plevel") ;
  Int_t strat    = pc.getInt("strat") ;
  Int_t initHesse= pc.getInt("initHesse") ;
  Int_t hesse    = pc.getInt("hesse") ;
  Int_t minos    = pc.getInt("minos") ;
  Int_t numee    = pc.getInt("numee") ;
  Int_t doEEWall = pc.getInt("doEEWall") ;
  Int_t doWarn   = pc.getInt("doWarn") ;
  Int_t doSumW2  = pc.getInt("doSumW2") ;
  const RooArgSet* minosSet = static_cast<RooArgSet*>(pc.getObject("minosSet")) ;
#ifdef __ROOFIT_NOROOMINIMIZER
  const char* minType =0 ;
#else
  const char* minType = pc.getString("mintype","",kTRUE) ;
  const char* minAlg = pc.getString("minalg","",kTRUE) ;
#endif

  // Determine if the dataset has weights  
  Bool_t weightedData = data.isNonPoissonWeighted() ;

  // Warn user that a SumW2Error() argument should be provided if weighted data is offered
  if (weightedData && doSumW2==-1) {
    coutW(InputArguments) << "RooAbsPdf::fitTo(" << GetName() << ") WARNING: a likelihood fit is request of what appears to be weighted data. " << endl
                          << "       While the estimated values of the parameters will always be calculated taking the weights into account, " << endl 
			  << "       there are multiple ways to estimate the errors on these parameter values. You are advised to make an " << endl 
			  << "       explicit choice on the error calculation: " << endl
			  << "           - Either provide SumW2Error(kTRUE), to calculate a sum-of-weights corrected HESSE error matrix " << endl
			  << "             (error will be proportional to the number of events)" << endl 
			  << "           - Or provide SumW2Error(kFALSE), to return errors from original HESSE error matrix" << endl 
			  << "             (which will be proportional to the sum of the weights)" << endl 
			  << "       If you want the errors to reflect the information contained in the provided dataset, choose kTRUE. " << endl
			  << "       If you want the errors to reflect the precision you would be able to obtain with an unweighted dataset " << endl 
			  << "       with 'sum-of-weights' events, choose kFALSE." << endl ;
  }


  // Warn user that sum-of-weights correction does not apply to MINOS errrors
  if (doSumW2==1 && minos) {
    coutW(InputArguments) << "RooAbsPdf::fitTo(" << GetName() << ") WARNING: sum-of-weights correction does not apply to MINOS errors" << endl ;
  }
    
  RooAbsReal* nll = createNLL(data,nllCmdList) ;  

  RooFitResult *ret = 0 ;    

  // Instantiate MINUIT

  if (minType) {

#ifndef __ROOFIT_NOROOMINIMIZER
    RooMinimizer m(*nll) ;

    m.setMinimizerType(minType) ;
    
    m.setEvalErrorWall(doEEWall) ;
    if (doWarn==0) {
      // m.setNoWarn() ; WVE FIX THIS
    }
    
    m.setPrintEvalErrors(numee) ;
    if (plevel!=1) {
      m.setPrintLevel(plevel) ;
    }
    
    if (optConst) {
      // Activate constant term optimization
      m.optimizeConst(1) ;
    }
    
    if (fitOpt) {
      
      // Play fit options as historically defined
      ret = m.fit(fitOpt) ;
      
    } else {
      
      if (verbose) {
	// Activate verbose options
	m.setVerbose(1) ;
      }
      if (doTimer) {
	// Activate timer options
	m.setProfile(1) ;
      }
      
      if (strat!=1) {
	// Modify fit strategy
	m.setStrategy(strat) ;
      }
      
      if (initHesse) {
	// Initialize errors with hesse
	m.hesse() ;
      }
      
      // Minimize using chosen algorithm
      m.minimize(minType,minAlg) ;
      
      if (hesse) {
	// Evaluate errors with Hesse
	m.hesse() ;
      }
      
      if (doSumW2==1) {
	
	// Make list of RooNLLVar components of FCN
	list<RooNLLVar*> nllComponents ;
	RooArgSet* comps = nll->getComponents() ;
	RooAbsArg* arg ;
	TIterator* citer = comps->createIterator() ;
	while((arg=(RooAbsArg*)citer->Next())) {
	  RooNLLVar* nllComp = dynamic_cast<RooNLLVar*>(arg) ;
	  if (nllComp) {
	    nllComponents.push_back(nllComp) ;
	  }
	}
	delete citer ;
	delete comps ;  
	
	// Calculated corrected errors for weighted likelihood fits
	RooFitResult* rw = m.save() ;
	for (list<RooNLLVar*>::iterator iter1=nllComponents.begin() ; iter1!=nllComponents.end() ; iter1++) {
	  (*iter1)->applyWeightSquared(kTRUE) ;
	}
	coutI(Fitting) << "RooAbsPdf::fitTo(" << GetName() << ") Calculating sum-of-weights-squared correction matrix for covariance matrix" << endl ;
	m.hesse() ;
	RooFitResult* rw2 = m.save() ;
	for (list<RooNLLVar*>::iterator iter2=nllComponents.begin() ; iter2!=nllComponents.end() ; iter2++) {
	  (*iter2)->applyWeightSquared(kFALSE) ;
	}
	
	// Apply correction matrix
	const TMatrixDSym& V = rw->covarianceMatrix() ;
	TMatrixDSym  C = rw2->covarianceMatrix() ;
	
	// Invert C
	Double_t det(0) ;
	C.Invert(&det) ;
	if (det==0) {
	  coutE(Fitting) << "RooAbsPdf::fitTo(" << GetName() 
			 << ") ERROR: Cannot apply sum-of-weights correction to covariance matrix: correction matrix calculated with weight-squared is singular" <<endl ;
	} else {
	  
	  // Calculate corrected covariance matrix = V C-1 V
	  TMatrixD VCV(V,TMatrixD::kMult,TMatrixD(C,TMatrixD::kMult,V)) ; 
	  
	  // Make matrix explicitly symmetric
	  Int_t n = VCV.GetNrows() ;
	  TMatrixDSym VCVsym(n) ;
	  for (Int_t i=0 ; i<n ; i++) {
	    for (Int_t j=i ; j<n ; j++) {
	      if (i==j) {
		VCVsym(i,j) = VCV(i,j) ;
	      }
	      if (i!=j) {
		Double_t deltaRel = (VCV(i,j)-VCV(j,i))/sqrt(VCV(i,i)*VCV(j,j)) ;
		if (fabs(deltaRel)>1e-3) {
		  coutW(Fitting) << "RooAbsPdf::fitTo(" << GetName() << ") WARNING: Corrected covariance matrix is not (completely) symmetric: V[" << i << "," << j << "] = " 
				 << VCV(i,j) << " V[" << j << "," << i << "] = " << VCV(j,i) << " explicitly restoring symmetry by inserting average value" << endl ;
		}
		VCVsym(i,j) = (VCV(i,j)+VCV(j,i))/2 ;
	      }
	    }
	  }
	  
	  // Propagate corrected errors to parameters objects
	  m.applyCovarianceMatrix(VCVsym) ;
	}
	
	delete rw ;
	delete rw2 ;
      }
      
      if (minos) {
	// Evaluate errs with Minos
	if (minosSet) {
	  m.minos(*minosSet) ;
	} else {
	  m.minos() ;
	}
      }
      
      // Optionally return fit result
      if (doSave) {
	string name = Form("fitresult_%s_%s",GetName(),data.GetName()) ;
	string title = Form("Result of fit of p.d.f. %s to dataset %s",GetName(),data.GetName()) ;
	ret = m.save(name.c_str(),title.c_str()) ;
      } 
      
    }
    if (optConst) {
      m.optimizeConst(0) ;
    }

#endif

  } else {

    RooMinuit m(*nll) ;
    
    m.setEvalErrorWall(doEEWall) ;
    if (doWarn==0) {
      m.setNoWarn() ;
    }
    
    m.setPrintEvalErrors(numee) ;
    if (plevel!=1) {
      m.setPrintLevel(plevel) ;
    }
    
    if (optConst) {
      // Activate constant term optimization
      m.optimizeConst(1) ;
    }
    
    if (fitOpt) {
      
      // Play fit options as historically defined
      ret = m.fit(fitOpt) ;
      
    } else {
      
      if (verbose) {
	// Activate verbose options
	m.setVerbose(1) ;
      }
      if (doTimer) {
	// Activate timer options
	m.setProfile(1) ;
      }
      
      if (strat!=1) {
	// Modify fit strategy
	m.setStrategy(strat) ;
      }
      
      if (initHesse) {
	// Initialize errors with hesse
	m.hesse() ;
      }
      
      // Minimize using migrad
      m.migrad() ;
      
      if (hesse) {
	// Evaluate errors with Hesse
	m.hesse() ;
      }
      
      if (doSumW2==1) {
	
	// Make list of RooNLLVar components of FCN
	list<RooNLLVar*> nllComponents ;
	RooArgSet* comps = nll->getComponents() ;
	RooAbsArg* arg ;
	TIterator* citer = comps->createIterator() ;
	while((arg=(RooAbsArg*)citer->Next())) {
	  RooNLLVar* nllComp = dynamic_cast<RooNLLVar*>(arg) ;
	  if (nllComp) {
	    nllComponents.push_back(nllComp) ;
	  }
	}
	delete citer ;
	delete comps ;  
	
	// Calculated corrected errors for weighted likelihood fits
	RooFitResult* rw = m.save() ;
	for (list<RooNLLVar*>::iterator iter1=nllComponents.begin() ; iter1!=nllComponents.end() ; iter1++) {
	  (*iter1)->applyWeightSquared(kTRUE) ;
	}
	coutI(Fitting) << "RooAbsPdf::fitTo(" << GetName() << ") Calculating sum-of-weights-squared correction matrix for covariance matrix" << endl ;
	m.hesse() ;
	RooFitResult* rw2 = m.save() ;
	for (list<RooNLLVar*>::iterator iter2=nllComponents.begin() ; iter2!=nllComponents.end() ; iter2++) {
	  (*iter2)->applyWeightSquared(kFALSE) ;
	}
	
	// Apply correction matrix
	const TMatrixDSym& V = rw->covarianceMatrix() ;
	TMatrixDSym  C = rw2->covarianceMatrix() ;
	
	// Invert C
	Double_t det(0) ;
	C.Invert(&det) ;
	if (det==0) {
	  coutE(Fitting) << "RooAbsPdf::fitTo(" << GetName() 
			 << ") ERROR: Cannot apply sum-of-weights correction to covariance matrix: correction matrix calculated with weight-squared is singular" <<endl ;
	} else {
	  
	  // Calculate corrected covariance matrix = V C-1 V
	  TMatrixD VCV(V,TMatrixD::kMult,TMatrixD(C,TMatrixD::kMult,V)) ; 
	  
	  // Make matrix explicitly symmetric
	  Int_t n = VCV.GetNrows() ;
	  TMatrixDSym VCVsym(n) ;
	  for (Int_t i=0 ; i<n ; i++) {
	    for (Int_t j=i ; j<n ; j++) {
	      if (i==j) {
		VCVsym(i,j) = VCV(i,j) ;
	      }
	      if (i!=j) {
		Double_t deltaRel = (VCV(i,j)-VCV(j,i))/sqrt(VCV(i,i)*VCV(j,j)) ;
		if (fabs(deltaRel)>1e-3) {
		  coutW(Fitting) << "RooAbsPdf::fitTo(" << GetName() << ") WARNING: Corrected covariance matrix is not (completely) symmetric: V[" << i << "," << j << "] = " 
				 << VCV(i,j) << " V[" << j << "," << i << "] = " << VCV(j,i) << " explicitly restoring symmetry by inserting average value" << endl ;
		}
		VCVsym(i,j) = (VCV(i,j)+VCV(j,i))/2 ;
	      }
	    }
	  }
	  
	  // Propagate corrected errors to parameters objects
	  m.applyCovarianceMatrix(VCVsym) ;
	}
	
	delete rw ;
	delete rw2 ;
      }
      
      if (minos) {
	// Evaluate errs with Minos
	if (minosSet) {
	  m.minos(*minosSet) ;
	} else {
	  m.minos() ;
	}
      }
      
      // Optionally return fit result
      if (doSave) {
	string name = Form("fitresult_%s_%s",GetName(),data.GetName()) ;
	string title = Form("Result of fit of p.d.f. %s to dataset %s",GetName(),data.GetName()) ;
	ret = m.save(name.c_str(),title.c_str()) ;
      } 
      
    }

    if (optConst) {
      m.optimizeConst(0) ;
    }
    
  }


  
  // Cleanup
  delete nll ;
  return ret ;
}



//_____________________________________________________________________________
RooFitResult* RooAbsPdf::chi2FitTo(RooDataHist& data, const RooLinkedList& cmdList) 
{
  // Internal back-end function to steer chi2 fits

  // Select the pdf-specific commands 
  RooCmdConfig pc(Form("RooAbsPdf::chi2FitTo(%s)",GetName())) ;

  // Pull arguments to be passed to chi2 construction from list
  RooLinkedList fitCmdList(cmdList) ;
  RooLinkedList chi2CmdList = pc.filterCmdList(fitCmdList,"Range,RangeWithName,NumCPU,Optimize,ProjectedObservables,AddCoefRange,SplitRange") ;

  RooAbsReal* chi2 = createChi2(data,chi2CmdList) ;
  RooFitResult* ret = chi2FitDriver(*chi2,fitCmdList) ;
  
  // Cleanup
  delete chi2 ;
  return ret ;
}




//_____________________________________________________________________________
RooAbsReal* RooAbsPdf::createChi2(RooDataHist& data, const RooCmdArg& arg1,  const RooCmdArg& arg2,  
				   const RooCmdArg& arg3,  const RooCmdArg& arg4, const RooCmdArg& arg5,  
				   const RooCmdArg& arg6,  const RooCmdArg& arg7, const RooCmdArg& arg8) 
{
  // Create a chi-2 from a histogram and this function.
  //
  // The following named arguments are supported
  //
  //  Options to control construction of the chi^2
  //  ------------------------------------------
  //  Extended()   -- Use expected number of events of an extended p.d.f as normalization 
  //  DataError()  -- Choose between Poisson errors and Sum-of-weights errors
  //  NumCPU()     -- Activate parallel processing feature
  //  Range()      -- Fit only selected region
  //  SumCoefRange() -- Set the range in which to interpret the coefficients of RooAddPdf components 
  //  SplitRange() -- Fit range is split by index catory of simultaneous PDF
  //  ConditionalObservables() -- Define projected observables 

  string name = Form("chi2_%s_%s",GetName(),data.GetName()) ;
 
  return new RooChi2Var(name.c_str(),name.c_str(),*this,data,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8) ;
}




//_____________________________________________________________________________
RooAbsReal* RooAbsPdf::createChi2(RooDataSet& data, const RooLinkedList& cmdList) 
{
  // Internal back-end function to create a chi^2 from a p.d.f. and a dataset

  // Select the pdf-specific commands 
  RooCmdConfig pc(Form("RooAbsPdf::fitTo(%s)",GetName())) ;

  pc.defineInt("integrate","Integrate",0,0) ;
  pc.defineObject("yvar","YVar",0,0) ;
  
  // Process and check varargs 
  pc.process(cmdList) ;
  if (!pc.ok(kTRUE)) {
    return 0 ;
  }

  // Decode command line arguments 
  Bool_t integrate = pc.getInt("integrate") ;
  RooRealVar* yvar = (RooRealVar*) pc.getObject("yvar") ;

  string name = Form("chi2_%s_%s",GetName(),data.GetName()) ;
 
  if (yvar) {
    return new RooXYChi2Var(name.c_str(),name.c_str(),*this,data,*yvar,integrate) ;
  } else {
    return new RooXYChi2Var(name.c_str(),name.c_str(),*this,data,integrate) ;
  }  
}




//_____________________________________________________________________________
void RooAbsPdf::printValue(ostream& os) const
{
  // Print value of p.d.f, also print normalization integral that was last used, if any

  getVal() ;

  if (_norm) {
    os << evaluate() << "/" << _norm->getVal() ;
  } else {
    os << evaluate() ;
  }
}



//_____________________________________________________________________________
void RooAbsPdf::printMultiline(ostream& os, Int_t contents, Bool_t verbose, TString indent) const
{
  // Print multi line detailed information of this RooAbsPdf

  RooAbsReal::printMultiline(os,contents,verbose,indent);
  os << indent << "--- RooAbsPdf ---" << endl;
  os << indent << "Cached value = " << _value << endl ;
  if (_norm) {
    os << indent << " Normalization integral: " << endl ;
    TString moreIndent(indent) ; moreIndent.Append("   ") ;
    _norm->printStream(os,kName|kAddress|kTitle|kValue|kArgs,kSingleLine,moreIndent.Data()) ;
  }
}



//_____________________________________________________________________________
RooAbsGenContext* RooAbsPdf::genContext(const RooArgSet &vars, const RooDataSet *prototype, 
					const RooArgSet* auxProto, Bool_t verbose) const 
{
  // Interface function to create a generator context from a p.d.f. This default
  // implementation returns a 'standard' context that works for any p.d.f
  return new RooGenContext(*this,vars,prototype,auxProto,verbose) ;
}



//_____________________________________________________________________________
RooDataSet *RooAbsPdf::generate(const RooArgSet& whatVars, Int_t nEvents, const RooCmdArg& arg1,
				const RooCmdArg& arg2, const RooCmdArg& arg3,const RooCmdArg& arg4, const RooCmdArg& arg5) 
{
  // Generate a new dataset containing the specified variables with events sampled from our distribution. 
  // Generate the specified number of events or expectedEvents() if not specified.
  //
  // Any variables of this PDF that are not in whatVars will use their
  // current values and be treated as fixed parameters. Returns zero
  // in case of an error. The caller takes ownership of the returned
  // dataset.
  //
  // The following named arguments are supported
  //
  // Name(const char* name)             -- Name of the output dataset
  // Verbose(Bool_t flag)               -- Print informational messages during event generation
  // Extended()                         -- The actual number of events generated will be sampled from a Poisson distribution
  //                                       with mu=nevt. For use with extended maximum likelihood fits
  // ProtoData(const RooDataSet& data,  -- Use specified dataset as prototype dataset. If randOrder is set to true
  //                 Bool_t randOrder)     the order of the events in the dataset will be read in a random order
  //                                       if the requested number of events to be generated does not match the
  //                                       number of events in the prototype dataset
  //                                        
  // If ProtoData() is used, the specified existing dataset as a prototype: the new dataset will contain 
  // the same number of events as the prototype (unless otherwise specified), and any prototype variables not in
  // whatVars will be copied into the new dataset for each generated event and also used to set our PDF parameters. 
  // The user can specify a  number of events to generate that will override the default. The result is a
  // copy of the prototype dataset with only variables in whatVars randomized. Variables in whatVars that 
  // are not in the prototype will be added as new columns to the generated dataset.  
  return generate(whatVars,RooFit::NumEvents(nEvents),arg1,arg2,arg3,arg4,arg5) ;
}



//_____________________________________________________________________________
RooDataSet *RooAbsPdf::generate(const RooArgSet& whatVars, const RooCmdArg& arg1,const RooCmdArg& arg2,
				const RooCmdArg& arg3,const RooCmdArg& arg4, const RooCmdArg& arg5,const RooCmdArg& arg6) 
{
  // Generate a new dataset containing the specified variables with events sampled from our distribution. 
  // Generate the specified number of events or expectedEvents() if not specified.
  //
  // Any variables of this PDF that are not in whatVars will use their
  // current values and be treated as fixed parameters. Returns zero
  // in case of an error. The caller takes ownership of the returned
  // dataset.
  //
  // The following named arguments are supported
  //
  // Name(const char* name)             -- Name of the output dataset
  // Verbose(Bool_t flag)               -- Print informational messages during event generation
  // NumEvent(int nevt)                 -- Generate specified number of events
  // Extended()                         -- The actual number of events generated will be sampled from a Poisson distribution
  //                                       with mu=nevt. For use with extended maximum likelihood fits
  // ProtoData(const RooDataSet& data,  -- Use specified dataset as prototype dataset. If randOrder is set to true
  //                 Bool_t randOrder,     the order of the events in the dataset will be read in a random order
  //                 Bool_t resample)      if the requested number of events to be generated does not match the
  //                                       number of events in the prototype dataset. If resample is also set to 
  //                                       true, the prototype dataset will be resampled rather than be strictly
  //                                       reshuffled. In this mode events of the protodata may be used more than
  //                                       once.
  //
  // If ProtoData() is used, the specified existing dataset as a prototype: the new dataset will contain 
  // the same number of events as the prototype (unless otherwise specified), and any prototype variables not in
  // whatVars will be copied into the new dataset for each generated event and also used to set our PDF parameters. 
  // The user can specify a  number of events to generate that will override the default. The result is a
  // copy of the prototype dataset with only variables in whatVars randomized. Variables in whatVars that 
  // are not in the prototype will be added as new columns to the generated dataset.  

  // Select the pdf-specific commands 
  RooCmdConfig pc(Form("RooAbsPdf::generate(%s)",GetName())) ;
  pc.defineObject("proto","PrototypeData",0,0) ;
  pc.defineString("dsetName","Name",0,"") ;
  pc.defineInt("randProto","PrototypeData",0,0) ;
  pc.defineInt("resampleProto","PrototypeData",1,0) ;
  pc.defineInt("verbose","Verbose",0,0) ;
  pc.defineInt("extended","Extended",0,0) ;
  pc.defineInt("nEvents","NumEvents",0,0) ;
  
  
  // Process and check varargs 
  pc.process(arg1,arg2,arg3,arg4,arg5,arg6) ;
  if (!pc.ok(kTRUE)) {
    return 0 ;
  }

  // Decode command line arguments
  RooDataSet* protoData = static_cast<RooDataSet*>(pc.getObject("proto",0)) ;
  const char* dsetName = pc.getString("dsetName") ;
  Int_t nEvents = pc.getInt("nEvents") ;
  Bool_t verbose = pc.getInt("verbose") ;
  Bool_t randProto = pc.getInt("randProto") ;
  Bool_t resampleProto = pc.getInt("resampleProto") ;
  Bool_t extended = pc.getInt("extended") ;

  if (extended) {
    nEvents = RooRandom::randomGenerator()->Poisson(nEvents==0?expectedEvents(&whatVars):nEvents) ;
    cxcoutI(Generation) << " Extended mode active, number of events generated (" << nEvents << ") is Poisson fluctuation on " 
			  << GetName() << "::expectedEvents() = " << nEvents << endl ;
    // If Poisson fluctuation results in zero events, stop here
    if (nEvents==0) {
      return new RooDataSet("emptyData","emptyData",whatVars) ;
    }
  } else if (nEvents==0) {
    cxcoutI(Generation) << "No number of events specified , number of events generated is " 
			  << GetName() << "::expectedEvents() = " << expectedEvents(&whatVars)<< endl ;
  }

  if (extended && protoData && !randProto) {
    cxcoutI(Generation) << "WARNING Using generator option Extended() (Poisson distribution of #events) together "
			  << "with a prototype dataset implies incomplete sampling or oversampling of proto data. " 
			  << "Set randomize flag in ProtoData() option to randomize prototype dataset order and thus "
			  << "to randomize the set of over/undersampled prototype events for each generation cycle." << endl ;
  }


  // Forward to appropiate implementation
  RooDataSet* data ;
  if (protoData) {
    data = generate(whatVars,*protoData,nEvents,verbose,randProto,resampleProto) ;
  } else {
    data = generate(whatVars,nEvents,verbose) ;
  }

  // Rename dataset to given name if supplied
  if (dsetName && strlen(dsetName)>0) {
    data->SetName(dsetName) ;
  }

  return data ;
}





//_____________________________________________________________________________
RooAbsPdf::GenSpec* RooAbsPdf::prepareMultiGen(const RooArgSet &whatVars,  
					       const RooCmdArg& arg1,const RooCmdArg& arg2,
					       const RooCmdArg& arg3,const RooCmdArg& arg4,
					       const RooCmdArg& arg5,const RooCmdArg& arg6) 
{
  // Prepare GenSpec configuration object for efficient generation of multiple datasets from idetical specification
  // This method does not perform any generation. To generate according to generations specification call RooAbsPdf::generate(RooAbsPdf::GenSpec&)
  //
  // Generate the specified number of events or expectedEvents() if not specified.
  //
  // Any variables of this PDF that are not in whatVars will use their
  // current values and be treated as fixed parameters. Returns zero
  // in case of an error. The caller takes ownership of the returned
  // dataset.
  //
  // The following named arguments are supported
  //
  // Name(const char* name)             -- Name of the output dataset
  // Verbose(Bool_t flag)               -- Print informational messages during event generation
  // NumEvent(int nevt)                 -- Generate specified number of events
  // Extended()                         -- The actual number of events generated will be sampled from a Poisson distribution
  //                                       with mu=nevt. For use with extended maximum likelihood fits
  // ProtoData(const RooDataSet& data,  -- Use specified dataset as prototype dataset. If randOrder is set to true
  //                 Bool_t randOrder,     the order of the events in the dataset will be read in a random order
  //                 Bool_t resample)      if the requested number of events to be generated does not match the
  //                                       number of events in the prototype dataset. If resample is also set to 
  //                                       true, the prototype dataset will be resampled rather than be strictly
  //                                       reshuffled. In this mode events of the protodata may be used more than
  //                                       once.
  //
  // If ProtoData() is used, the specified existing dataset as a prototype: the new dataset will contain 
  // the same number of events as the prototype (unless otherwise specified), and any prototype variables not in
  // whatVars will be copied into the new dataset for each generated event and also used to set our PDF parameters. 
  // The user can specify a  number of events to generate that will override the default. The result is a
  // copy of the prototype dataset with only variables in whatVars randomized. Variables in whatVars that 
  // are not in the prototype will be added as new columns to the generated dataset.  

  // Select the pdf-specific commands 
  RooCmdConfig pc(Form("RooAbsPdf::generate(%s)",GetName())) ;
  pc.defineObject("proto","PrototypeData",0,0) ;
  pc.defineString("dsetName","Name",0,"") ;
  pc.defineInt("randProto","PrototypeData",0,0) ;
  pc.defineInt("resampleProto","PrototypeData",1,0) ;
  pc.defineInt("verbose","Verbose",0,0) ;
  pc.defineInt("extended","Extended",0,0) ;
  pc.defineInt("nEvents","NumEvents",0,0) ;
  
  
  // Process and check varargs 
  pc.process(arg1,arg2,arg3,arg4,arg5,arg6) ;
  if (!pc.ok(kTRUE)) {
    return 0 ;
  }

  // Decode command line arguments
  RooDataSet* protoData = static_cast<RooDataSet*>(pc.getObject("proto",0)) ;
  const char* dsetName = pc.getString("dsetName") ;
  Int_t nEvents = pc.getInt("nEvents") ;
  Bool_t verbose = pc.getInt("verbose") ;
  Bool_t randProto = pc.getInt("randProto") ;
  Bool_t resampleProto = pc.getInt("resampleProto") ;
  Bool_t extended = pc.getInt("extended") ;

  return new GenSpec(genContext(whatVars,protoData,0,verbose),whatVars,protoData,nEvents,extended,randProto,resampleProto,dsetName) ;  
}


//_____________________________________________________________________________
RooDataSet *RooAbsPdf::generate(RooAbsPdf::GenSpec& spec) const
{
  // Generate data according to a pre-configured specification created by
  // RooAbsPdf::prepareMultiGen(). If many identical generation requests
  // are needed, e.g. in toy MC studies, it is more efficient to use the prepareMultiGen()/generate()
  // combination than calling the standard generate() multiple times as 
  // initialization overhead is only incurred once.

  Int_t nEvt = spec._extended ? RooRandom::randomGenerator()->Poisson(spec._nGen) : spec._nGen ;

  return generate(*spec._genContext,spec._whatVars,spec._protoData,
		  nEvt,kFALSE,spec._randProto,spec._resampleProto) ;
}





//_____________________________________________________________________________
RooDataSet *RooAbsPdf::generate(const RooArgSet &whatVars, Int_t nEvents, Bool_t verbose) const 
{
  // Generate a new dataset containing the specified variables with
  // events sampled from our distribution. Generate the specified
  // number of events or else try to use expectedEvents() if nEvents <= 0.
  // Any variables of this PDF that are not in whatVars will use their
  // current values and be treated as fixed parameters. Returns zero
  // in case of an error. The caller takes ownership of the returned
  // dataset.

  if (nEvents==0 && extendMode()==CanNotBeExtended) {
    return new RooDataSet("emptyData","emptyData",whatVars) ;
  }

  RooDataSet *generated = 0;
  RooAbsGenContext *context= genContext(whatVars,0,0,verbose);
  if(0 != context && context->isValid()) {
    generated= context->generate(nEvents);
  }
  else {
    coutE(Generation)  << "RooAbsPdf::generate(" << GetName() << ") cannot create a valid context" << endl;
  }
  if(0 != context) delete context;
  return generated;
}




//_____________________________________________________________________________
RooDataSet *RooAbsPdf::generate(RooAbsGenContext& context, const RooArgSet &whatVars, const RooDataSet *prototype,
				Int_t nEvents, Bool_t /*verbose*/, Bool_t randProtoOrder, Bool_t resampleProto) const 
{
  // Internal method  
  if (nEvents==0 && (prototype==0 || prototype->numEntries()==0)) {
    return new RooDataSet("emptyData","emptyData",whatVars) ;
  }


  RooDataSet *generated = 0;

  // Resampling implies reshuffling in the implementation
  if (resampleProto) {
    randProtoOrder=kTRUE ;
  }

  if (randProtoOrder && prototype && prototype->numEntries()!=nEvents) {
    coutI(Generation) << "RooAbsPdf::generate (Re)randomizing event order in prototype dataset (Nevt=" << nEvents << ")" << endl ;
    Int_t* newOrder = randomizeProtoOrder(prototype->numEntries(),nEvents,resampleProto) ;
    context.setProtoDataOrder(newOrder) ;
    delete[] newOrder ;
  }

  if(context.isValid()) {
    generated= context.generate(nEvents);
  }
  else {
    coutE(Generation) << "RooAbsPdf::generate(" << GetName() << ") do not have a valid generator context" << endl;
  }
  return generated;
}




//_____________________________________________________________________________
RooDataSet *RooAbsPdf::generate(const RooArgSet &whatVars, const RooDataSet& prototype,
				Int_t nEvents, Bool_t verbose, Bool_t randProtoOrder, Bool_t resampleProto) const 
{
  // Generate a new dataset with values of the whatVars variables
  // sampled from our distribution. Use the specified existing dataset
  // as a prototype: the new dataset will contain the same number of
  // events as the prototype (by default), and any prototype variables not in
  // whatVars will be copied into the new dataset for each generated
  // event and also used to set our PDF parameters. The user can specify a
  // number of events to generate that will override the default. The result is a
  // copy of the prototype dataset with only variables in whatVars
  // randomized. Variables in whatVars that are not in the prototype
  // will be added as new columns to the generated dataset.  Returns
  // zero in case of an error. The caller takes ownership of the
  // returned dataset.

  RooAbsGenContext *context= genContext(whatVars,&prototype,0,verbose);
  if (context) {
    RooDataSet* data =  generate(*context,whatVars,&prototype,nEvents,verbose,randProtoOrder,resampleProto) ;
    delete context ;
    return data ;
  } else {
    coutE(Generation) << "RooAbsPdf::generate(" << GetName() << ") ERROR creating generator context" << endl ;
    return 0 ;
  }
}



//_____________________________________________________________________________
Int_t* RooAbsPdf::randomizeProtoOrder(Int_t nProto, Int_t, Bool_t resampleProto) const
{
  // Return lookup table with randomized access order for prototype events,
  // given nProto prototype data events and nGen events that will actually
  // be accessed

  // Make unsorted linked list of indeces
  RooLinkedList l ;
  Int_t i ;
  for (i=0 ; i<nProto ; i++) {
    l.Add(new RooInt(i)) ;
  }

  // Make output list
  Int_t* lut = new Int_t[nProto] ;

  // Randomly samply input list into output list
  if (!resampleProto) {
    // In this mode, randomization is a strict reshuffle of the order
    for (i=0 ; i<nProto ; i++) {
      Int_t iran = RooRandom::integer(nProto-i) ;
      RooInt* sample = (RooInt*) l.At(iran) ;
      lut[i] = *sample ;
      l.Remove(sample) ;
      delete sample ;
    }
  } else {
    // In this mode, we resample, i.e. events can be used more than once
    for (i=0 ; i<nProto ; i++) {
      lut[i] = RooRandom::integer(nProto);
    }
  }


  return lut ;
}



//_____________________________________________________________________________
Int_t RooAbsPdf::getGenerator(const RooArgSet &/*directVars*/, RooArgSet &/*generatedVars*/, Bool_t /*staticInitOK*/) const 
{
  // Load generatedVars with the subset of directVars that we can generate events for,
  // and return a code that specifies the generator algorithm we will use. A code of
  // zero indicates that we cannot generate any of the directVars (in this case, nothing
  // should be added to generatedVars). Any non-zero codes will be passed to our generateEvent()
  // implementation, but otherwise its value is arbitrary. The default implemetation of
  // this method returns zero. Subclasses will usually implement this method using the
  // matchArgs() methods to advertise the algorithms they provide.


  return 0 ;
}



//_____________________________________________________________________________
void RooAbsPdf::initGenerator(Int_t /*code*/) 
{  
  // Interface for one-time initialization to setup the generator for the specified code.
}



//_____________________________________________________________________________
void RooAbsPdf::generateEvent(Int_t /*code*/) 
{
  // Interface for generation of anan event using the algorithm
  // corresponding to the specified code. The meaning of each code is
  // defined by the getGenerator() implementation. The default
  // implementation does nothing.
}



//_____________________________________________________________________________
Bool_t RooAbsPdf::isDirectGenSafe(const RooAbsArg& arg) const 
{
  // Check if given observable can be safely generated using the
  // pdfs internal generator mechanism (if that existsP). Observables
  // on which a PDF depends via more than route are not safe
  // for use with internal generators because they introduce
  // correlations not known to the internal generator

  // Arg must be direct server of self
  if (!findServer(arg.GetName())) return kFALSE ;

  // There must be no other dependency routes
  TIterator* sIter = serverIterator() ;
  const RooAbsArg *server = 0;
  while((server=(const RooAbsArg*)sIter->Next())) {
    if(server == &arg) continue;
    if(server->dependsOn(arg)) {
      delete sIter ;
      return kFALSE ;
    }
  }
  delete sIter ;
  return kTRUE ;
}




//_____________________________________________________________________________
RooDataHist *RooAbsPdf::generateBinned(const RooArgSet& whatVars, Double_t nEvents, const RooCmdArg& arg1,
				       const RooCmdArg& arg2, const RooCmdArg& arg3,const RooCmdArg& arg4, const RooCmdArg& arg5) 
{
  // Generate a new dataset containing the specified variables with events sampled from our distribution. 
  // Generate the specified number of events or expectedEvents() if not specified.
  //
  // Any variables of this PDF that are not in whatVars will use their
  // current values and be treated as fixed parameters. Returns zero
  // in case of an error. The caller takes ownership of the returned
  // dataset.
  //
  // The following named arguments are supported
  //
  // Name(const char* name)             -- Name of the output dataset
  // Verbose(Bool_t flag)               -- Print informational messages during event generation
  // Extended()                         -- The actual number of events generated will be sampled from a Poisson distribution
  //                                       with mu=nevt. For use with extended maximum likelihood fits
  // ExpectedData()                     -- Return a binned dataset _without_ statistical fluctuations (also aliased as Asimov())
  return generateBinned(whatVars,RooFit::NumEvents(nEvents),arg1,arg2,arg3,arg4,arg5) ;
}



//_____________________________________________________________________________
RooDataHist *RooAbsPdf::generateBinned(const RooArgSet& whatVars, const RooCmdArg& arg1,const RooCmdArg& arg2,
				       const RooCmdArg& arg3,const RooCmdArg& arg4, const RooCmdArg& arg5,const RooCmdArg& arg6) 
{
  // Generate a new dataset containing the specified variables with events sampled from our distribution. 
  // Generate the specified number of events or expectedEvents() if not specified.
  //
  // Any variables of this PDF that are not in whatVars will use their
  // current values and be treated as fixed parameters. Returns zero
  // in case of an error. The caller takes ownership of the returned
  // dataset.
  //
  // The following named arguments are supported
  //
  // Name(const char* name)             -- Name of the output dataset
  // Verbose(Bool_t flag)               -- Print informational messages during event generation
  // NumEvent(int nevt)                 -- Generate specified number of events
  // Extended()                         -- The actual number of events generated will be sampled from a Poisson distribution
  //                                       with mu=nevt. For use with extended maximum likelihood fits
  // ExpectedData()                     -- Return a binned dataset _without_ statistical fluctuations (also aliased as Asimov())
  

  // Select the pdf-specific commands 
  RooCmdConfig pc(Form("RooAbsPdf::generate(%s)",GetName())) ;
  pc.defineString("dsetName","Name",0,"") ;
  pc.defineInt("verbose","Verbose",0,0) ;
  pc.defineInt("extended","Extended",0,0) ;
  pc.defineInt("nEvents","NumEvents",0,0) ;
  pc.defineDouble("nEventsD","NumEventsD",0,-1.) ;
  pc.defineInt("expectedData","ExpectedData",0,0) ;
  
  // Process and check varargs 
  pc.process(arg1,arg2,arg3,arg4,arg5,arg6) ;
  if (!pc.ok(kTRUE)) {
    return 0 ;
  }

  // Decode command line arguments
  Double_t nEvents = pc.getDouble("nEventsD") ;
  if (nEvents<0) {
    nEvents = pc.getInt("nEvents") ;
  }
  //Bool_t verbose = pc.getInt("verbose") ;
  Bool_t extended = pc.getInt("extended") ;
  Bool_t expectedData = pc.getInt("expectedData") ;
  const char* dsetName = pc.getString("dsetName") ;

  if (extended) {
    nEvents = (nEvents==0?Int_t(expectedEvents(&whatVars)+0.5):nEvents) ;
    cxcoutI(Generation) << " Extended mode active, number of events generated (" << nEvents << ") is Poisson fluctuation on " 
			<< GetName() << "::expectedEvents() = " << nEvents << endl ;
    // If Poisson fluctuation results in zero events, stop here
    if (nEvents==0) {
      return 0 ;
    }
  } else if (nEvents==0) {
    cxcoutI(Generation) << "No number of events specified , number of events generated is " 
			<< GetName() << "::expectedEvents() = " << expectedEvents(&whatVars)<< endl ;
  }

  // Forward to appropiate implementation
  RooDataHist* data = generateBinned(whatVars,nEvents,expectedData,extended) ;

  // Rename dataset to given name if supplied
  if (dsetName && strlen(dsetName)>0) {
    data->SetName(dsetName) ;
  }

  return data ;
}




//_____________________________________________________________________________
RooDataHist *RooAbsPdf::generateBinned(const RooArgSet &whatVars, Double_t nEvents, Bool_t expectedData, Bool_t extended) const 
{
  // Generate a new dataset containing the specified variables with
  // events sampled from our distribution. Generate the specified
  // number of events or else try to use expectedEvents() if nEvents <= 0.
  //
  // If expectedData is kTRUE (it is kFALSE by default), the returned histogram returns the 'expected'
  // data sample, i.e. no statistical fluctuations are present.
  //
  // Any variables of this PDF that are not in whatVars will use their
  // current values and be treated as fixed parameters. Returns zero
  // in case of an error. The caller takes ownership of the returned
  // dataset.

  // Create empty RooDataHist
  RooDataHist* hist = new RooDataHist("genData","genData",whatVars) ;

  // Scale to number of events and introduce Poisson fluctuations
  if (nEvents<=0) {
    if (!canBeExtended()) {
      coutE(InputArguments) << "RooAbsPdf::generateBinned(" << GetName() << ") ERROR: No event count provided and p.d.f does not provide expected number of events" << endl ;
      delete hist ;
      return 0 ;
    } else {
      // Don't round in expectedData mode
      if (expectedData) {
	nEvents = expectedEvents(&whatVars) ;
      } else {
	nEvents = Int_t(expectedEvents(&whatVars)+0.5) ;
      }
    }
  } 
  
  // Sample p.d.f. distribution
  fillDataHist(hist,&whatVars,1,kTRUE) ;  

  vector<int> histOut(hist->numEntries()) ;
  Double_t histMax(-1) ;
  Int_t histOutSum(0) ;
  for (int i=0 ; i<hist->numEntries() ; i++) {
    hist->get(i) ;
    if (expectedData) {

      // Expected data, multiply p.d.f by nEvents
      Double_t w=hist->weight()*nEvents ;
      hist->set(w,sqrt(w)) ;

    } else if (extended) {

      // Extended mode, set contents to Poisson(pdf*nEvents)
      Double_t w = RooRandom::randomGenerator()->Poisson(hist->weight()*nEvents) ;
      hist->set(w,sqrt(w)) ;

    } else {

      // Regular mode, fill array of weights with Poisson(pdf*nEvents), but to not fill
      // histogram yet.
      if (hist->weight()>histMax) {
	histMax = hist->weight() ;
      }
      histOut[i] = RooRandom::randomGenerator()->Poisson(hist->weight()*nEvents) ;
      histOutSum += histOut[i] ;
    }
  }


  if (!expectedData && !extended) {

    // Second pass for regular mode - Trim/Extend dataset to exact number of entries

    // Calculate difference between what is generated so far and what is requested
    Int_t nEvtExtra = abs(Int_t(nEvents)-histOutSum) ;
    Int_t wgt = (histOutSum>nEvents) ? -1 : 1 ;

    // Perform simple binned accept/reject procedure to get to exact event count
    while(nEvtExtra>0) {

      Int_t ibinRand = RooRandom::randomGenerator()->Integer(hist->numEntries()) ;
      hist->get(ibinRand) ;
      Double_t ranY = RooRandom::randomGenerator()->Uniform(histMax) ;

      if (ranY<hist->weight()) {
	if (wgt==1) {
	  histOut[ibinRand]++ ;
	} else {
	  // If weight is negative, prior bin content must be at least 1
	  if (histOut[ibinRand]>0) {
	    histOut[ibinRand]-- ;
	  } else {
	    continue ;
	  }
	}
	nEvtExtra-- ;
      }
    }

    // Transfer working array to histogram
    for (int i=0 ; i<hist->numEntries() ; i++) {
      hist->get(i) ;
      hist->set(histOut[i],sqrt(1.0*histOut[i])) ;
    }    

  } else if (expectedData) {

    // Second pass for expectedData mode -- Normalize to exact number of requested events
    // Minor difference may be present in first round due to difference between 
    // bin average and bin integral in sampling bins
    Double_t corr = nEvents/hist->sumEntries() ;
    for (int i=0 ; i<hist->numEntries() ; i++) {
      hist->get(i) ;
      hist->set(hist->weight()*corr,sqrt(hist->weight()*corr)) ;
    }

  }

  return hist;
}



//_____________________________________________________________________________
RooDataSet* RooAbsPdf::generateSimGlobal(const RooArgSet& whatVars, Int_t nEvents) 
{
  // Special generator interface for generation of 'global observables' -- for RooStats tools

  return generate(whatVars,nEvents) ;
}



//_____________________________________________________________________________
RooPlot* RooAbsPdf::plotOn(RooPlot* frame, RooLinkedList& cmdList) const
{
  // Plot (project) PDF on specified frame. If a PDF is plotted in an empty frame, it
  // will show a unit normalized curve in the frame variable, taken at the present value 
  // of other observables defined for this PDF
  //
  // If a PDF is plotted in a frame in which a dataset has already been plotted, it will
  // show a projected curve integrated over all variables that were present in the shown
  // dataset except for the one on the x-axis. The normalization of the curve will also
  // be adjusted to the event count of the plotted dataset. An informational message
  // will be printed for each projection step that is performed
  //
  // This function takes the following named arguments
  //
  // Projection control
  // ------------------
  // Slice(const RooArgSet& set)     -- Override default projection behaviour by omittting observables listed 
  //                                    in set from the projection, resulting a 'slice' plot. Slicing is usually
  //                                    only sensible in discrete observables
  // Project(const RooArgSet& set)   -- Override default projection behaviour by projecting over observables
  //                                    given in set and complete ignoring the default projection behavior. Advanced use only.
  // ProjWData(const RooAbsData& d)  -- Override default projection _technique_ (integration). For observables present in given dataset
  //                                    projection of PDF is achieved by constructing an average over all observable values in given set.
  //                                    Consult RooFit plotting tutorial for further explanation of meaning & use of this technique
  // ProjWData(const RooArgSet& s,   -- As above but only consider subset 's' of observables in dataset 'd' for projection through data averaging
  //           const RooAbsData& d)
  // ProjectionRange(const char* rn) -- Override default range of projection integrals to a different range speficied by given range name.
  //                                    This technique allows you to project a finite width slice in a real-valued observable
  // NormRange(const char* name)     -- Calculate curve normalization w.r.t. only in specified ranges. NB: A Range() by default implies a NormRange()
  //                                    on the same range, but this option allows to override the default, or specify a normalization ranges
  //                                    when the full curve is to be drawn
  // 
  // Misc content control
  // --------------------
  // Normalization(Double_t scale,   -- Adjust normalization by given scale factor. Interpretation of number depends on code: Relative:
  //                ScaleType code)     relative adjustment factor, NumEvent: scale to match given number of events.
  // Name(const chat* name)          -- Give curve specified name in frame. Useful if curve is to be referenced later
  // Asymmetry(const RooCategory& c) -- Show the asymmetry of the PDF in given two-state category [F(+)-F(-)] / [F(+)+F(-)] rather than
  //                                    the PDF projection. Category must have two states with indices -1 and +1 or three states with
  //                                    indeces -1,0 and +1.
  // ShiftToZero(Bool_t flag)        -- Shift entire curve such that lowest visible point is at exactly zero. Mostly useful when
  //                                    plotting -log(L) or chi^2 distributions
  // AddTo(const char* name,         -- Add constructed projection to already existing curve with given name and relative weight factors
  //       double_t wgtSelf, double_t wgtOther)
  //
  // Plotting control 
  // ----------------
  // LineStyle(Int_t style)          -- Select line style by ROOT line style code, default is solid
  // LineColor(Int_t color)          -- Select line color by ROOT color code, default is blue
  // LineWidth(Int_t width)          -- Select line with in pixels, default is 3
  // FillStyle(Int_t style)          -- Select fill style, default is not filled. If a filled style is selected, also use VLines()
  //                                    to add vertical downward lines at end of curve to ensure proper closure
  // FillColor(Int_t color)          -- Select fill color by ROOT color code
  // Range(const char* name)         -- Only draw curve in range defined by given name
  // Range(double lo, double hi)     -- Only draw curve in specified range
  // VLines()                        -- Add vertical lines to y=0 at end points of curve
  // Precision(Double_t eps)         -- Control precision of drawn curve w.r.t to scale of plot, default is 1e-3. Higher precision
  //                                    will result in more and more densely spaced curve points
  // Invisble(Bool_t flag)           -- Add curve to frame, but do not display. Useful in combination AddTo()


  // Pre-processing if p.d.f. contains a fit range and there is no command specifying one,
  // add a fit range as default range
  RooCmdArg* plotRange(0) ;
  RooCmdArg* normRange2(0) ;  
  if (getStringAttribute("fitrange") && !cmdList.FindObject("Range") && 
      !cmdList.FindObject("RangeWithName")) {
    plotRange = (RooCmdArg*) RooFit::Range(getStringAttribute("fitrange")).Clone() ;    
    cmdList.Add(plotRange) ;
  }

  if (getStringAttribute("fitrange") && !cmdList.FindObject("NormRange")) {
    normRange2 = (RooCmdArg*) RooFit::NormRange(getStringAttribute("fitrange")).Clone() ;    
    cmdList.Add(normRange2) ;
  }

  if (plotRange || normRange2) {
    coutI(Plotting) << "RooAbsPdf::plotOn(" << GetName() << ") p.d.f was fitted in range and no explicit " 
		    << (plotRange?"plot":"") << ((plotRange&&normRange2)?",":"")
		    << (normRange2?"norm":"") << " range was specified, using fit range as default" << endl ;
  }

  // Sanity checks
  if (plotSanityChecks(frame)) return frame ;

  // Select the pdf-specific commands 
  RooCmdConfig pc(Form("RooAbsPdf::plotOn(%s)",GetName())) ;
  pc.defineDouble("scaleFactor","Normalization",0,1.0) ;
  pc.defineInt("scaleType","Normalization",0,Relative) ;  
  pc.defineObject("compSet","SelectCompSet",0) ;
  pc.defineString("compSpec","SelectCompSpec",0) ;
  pc.defineObject("asymCat","Asymmetry",0) ;
  pc.defineDouble("rangeLo","Range",0,-999.) ;
  pc.defineDouble("rangeHi","Range",1,-999.) ;
  pc.defineString("rangeName","RangeWithName",0,"") ;
  pc.defineString("normRangeName","NormRange",0,"") ;
  pc.defineInt("rangeAdjustNorm","Range",0,0) ;
  pc.defineInt("rangeWNAdjustNorm","RangeWithName",0,0) ;
  pc.defineMutex("SelectCompSet","SelectCompSpec") ;
  pc.defineMutex("Range","RangeWithName") ;
  pc.allowUndefined() ; // unknowns may be handled by RooAbsReal

  // Process and check varargs 
  pc.process(cmdList) ;
  if (!pc.ok(kTRUE)) {
    return frame ;
  }

  // Decode command line arguments
  ScaleType stype = (ScaleType) pc.getInt("scaleType") ;
  Double_t scaleFactor = pc.getDouble("scaleFactor") ;
  const RooAbsCategoryLValue* asymCat = (const RooAbsCategoryLValue*) pc.getObject("asymCat") ;
  const char* compSpec = pc.getString("compSpec") ;
  const RooArgSet* compSet = (const RooArgSet*) pc.getObject("compSet") ;
  Bool_t haveCompSel = ( (compSpec && strlen(compSpec)>0) || compSet) ;

  // Suffix for curve name
  TString nameSuffix ;
  if (compSpec && strlen(compSpec)>0) {
    nameSuffix.Append("_Comp[") ;
    nameSuffix.Append(compSpec) ;
    nameSuffix.Append("]") ;    
  } else if (compSet) {
    nameSuffix.Append("_Comp[") ;
    nameSuffix.Append(compSet->contentsString().c_str()) ;
    nameSuffix.Append("]") ;    
  }

  // Remove PDF-only commands from command list
  pc.stripCmdList(cmdList,"SelectCompSet,SelectCompSpec") ;
  
  // Adjust normalization, if so requested
  if (asymCat) {
    RooCmdArg cnsuffix("CurveNameSuffix",0,0,0,0,nameSuffix.Data(),0,0,0) ;
    cmdList.Add(&cnsuffix);
    return  RooAbsReal::plotOn(frame,cmdList) ;
  }

  // More sanity checks
  Double_t nExpected(1) ;
  if (stype==RelativeExpected) {
    if (!canBeExtended()) {
      coutE(Plotting) << "RooAbsPdf::plotOn(" << GetName() 
		      << "): ERROR the 'Expected' scale option can only be used on extendable PDFs" << endl ;
      return frame ;
    }
    nExpected = expectedEvents(frame->getNormVars()) ;
  }

  if (stype != Raw) {    

    if (frame->getFitRangeNEvt() && stype==Relative) {

      Bool_t hasCustomRange(kFALSE), adjustNorm(kFALSE) ;

      list<pair<Double_t,Double_t> > rangeLim ;

      // Retrieve plot range to be able to adjust normalization to data
      if (pc.hasProcessed("Range")) {

	Double_t rangeLo = pc.getDouble("rangeLo") ;
	Double_t rangeHi = pc.getDouble("rangeHi") ;
	rangeLim.push_back(make_pair(rangeLo,rangeHi)) ;
	adjustNorm = pc.getInt("rangeAdjustNorm") ;
	hasCustomRange = kTRUE ;

	coutI(Plotting) << "RooAbsPdf::plotOn(" << GetName() << ") only plotting range [" 
			<< rangeLo << "," << rangeHi << "]" ;
	if (!pc.hasProcessed("NormRange")) {	  
	  ccoutI(Plotting) << ", curve is normalized to data in " << (adjustNorm?"given":"full") << " given range" << endl ;
	} else {
	  ccoutI(Plotting) << endl ;
	}

	nameSuffix.Append(Form("_Range[%f_%f]",rangeLo,rangeHi)) ;

      } else if (pc.hasProcessed("RangeWithName")) {    

	char tmp[1024] ;
	strlcpy(tmp,pc.getString("rangeName",0,kTRUE),1024) ;
	char* rangeNameToken = strtok(tmp,",") ;
	while(rangeNameToken) {
	  Double_t rangeLo = frame->getPlotVar()->getMin(rangeNameToken) ;
	  Double_t rangeHi = frame->getPlotVar()->getMax(rangeNameToken) ;
	  rangeLim.push_back(make_pair(rangeLo,rangeHi)) ;
	  rangeNameToken = strtok(0,",") ;
	}
	adjustNorm = pc.getInt("rangeWNAdjustNorm") ;
	hasCustomRange = kTRUE ;

	coutI(Plotting) << "RooAbsPdf::plotOn(" << GetName() << ") only plotting range '" << pc.getString("rangeName",0,kTRUE) << "'" ;
	if (!pc.hasProcessed("NormRange")) {	  
	  ccoutI(Plotting) << ", curve is normalized to data in " << (adjustNorm?"given":"full") << " given range" << endl ;
	} else {
	  ccoutI(Plotting) << endl ;
	}

	nameSuffix.Append(Form("_Range[%s]",pc.getString("rangeName"))) ;
      } 
      // Specification of a normalization range override those in a regular ranage
      if (pc.hasProcessed("NormRange")) {    
	char tmp[1024] ;
	strlcpy(tmp,pc.getString("normRangeName",0,kTRUE),1024) ;
	char* rangeNameToken = strtok(tmp,",") ;
	rangeLim.clear() ;
	while(rangeNameToken) {
	  Double_t rangeLo = frame->getPlotVar()->getMin(rangeNameToken) ;
	  Double_t rangeHi = frame->getPlotVar()->getMax(rangeNameToken) ;
	  rangeLim.push_back(make_pair(rangeLo,rangeHi)) ;
	  rangeNameToken = strtok(0,",") ;
	}
	adjustNorm = kTRUE ;
	hasCustomRange = kTRUE ;	
	coutI(Plotting) << "RooAbsPdf::plotOn(" << GetName() << ") p.d.f. curve is normalized using explicit choice of ranges '" << pc.getString("normRangeName",0,kTRUE) << "'" << endl ;

	nameSuffix.Append(Form("_NormRange[%s]",pc.getString("rangeName"))) ;

      }

      if (hasCustomRange && adjustNorm) {	

	Double_t rangeNevt(0) ;
	list<pair<Double_t,Double_t> >::iterator riter = rangeLim.begin() ;
	for (;riter!=rangeLim.end() ; ++riter) {
	  Double_t nevt= frame->getFitRangeNEvt(riter->first,riter->second) ;
	  rangeNevt += nevt ;
	}
	scaleFactor *= rangeNevt/nExpected ;

      } else {
	scaleFactor *= frame->getFitRangeNEvt()/nExpected ;
      }
    } else if (stype==RelativeExpected) {
      scaleFactor *= nExpected ; 
    } else if (stype==NumEvent) {
      scaleFactor /= nExpected ;
    }
    scaleFactor *= frame->getFitRangeBinW() ;
  } 
  frame->updateNormVars(*frame->getPlotVar()) ;

  // Append overriding scale factor command at end of original command list
  RooCmdArg tmp = RooFit::Normalization(scaleFactor,Raw) ;
  cmdList.Add(&tmp) ;

  // Was a component selected requested
  if (haveCompSel) {
    
    // Get complete set of tree branch nodes
    RooArgSet branchNodeSet ;
    branchNodeServerList(&branchNodeSet) ;
    
    // Discard any non-RooAbsReal nodes
    TIterator* iter = branchNodeSet.createIterator() ;
    RooAbsArg* arg ;
    while((arg=(RooAbsArg*)iter->Next())) {
      if (!dynamic_cast<RooAbsReal*>(arg)) {
	branchNodeSet.remove(*arg) ;
      }
    }
    delete iter ;
    
    // Obtain direct selection
    RooArgSet* dirSelNodes ;
    if (compSet) {
      dirSelNodes = (RooArgSet*) branchNodeSet.selectCommon(*compSet) ;
    } else {
      dirSelNodes = (RooArgSet*) branchNodeSet.selectByName(compSpec) ;
    }
    if (dirSelNodes->getSize()>0) {
      coutI(Plotting) << "RooAbsPdf::plotOn(" << GetName() << ") directly selected PDF components: " << *dirSelNodes << endl ;
      
      // Do indirect selection and activate both
      plotOnCompSelect(dirSelNodes) ;
    } else {
      if (compSet) {
	coutE(Plotting) << "RooAbsPdf::plotOn(" << GetName() << ") ERROR: component selection set " << *compSet << " does not match any components of p.d.f." << endl ;
      } else {
	coutE(Plotting) << "RooAbsPdf::plotOn(" << GetName() << ") ERROR: component selection expression '" << compSpec << "' does not select any components of p.d.f." << endl ;
      }
      return 0 ;
    }

    delete dirSelNodes ;
  }


  RooCmdArg cnsuffix("CurveNameSuffix",0,0,0,0,nameSuffix.Data(),0,0,0) ;
  cmdList.Add(&cnsuffix);

  RooPlot* ret =  RooAbsReal::plotOn(frame,cmdList) ;
  
  // Restore selection status ;
  if (haveCompSel) plotOnCompSelect(0) ;

  if (plotRange) {
    delete plotRange ;
  }
  if (normRange2) {
    delete normRange2 ;
  }  

  return ret ;
}



//_____________________________________________________________________________
void RooAbsPdf::plotOnCompSelect(RooArgSet* selNodes) const
{
  // Helper function for plotting of composite p.d.fs. Given
  // a set of selected components that should be plotted,
  // find all nodes that (in)directly depend on these selected
  // nodes. Mark all directly and indirecty selected nodes
  // as 'selected' using the selectComp() method

  // Get complete set of tree branch nodes
  RooArgSet branchNodeSet ;
  branchNodeServerList(&branchNodeSet) ;

  // Discard any non-PDF nodes
  TIterator* iter = branchNodeSet.createIterator() ;
  RooAbsArg* arg ;
  while((arg=(RooAbsArg*)iter->Next())) {
    if (!dynamic_cast<RooAbsReal*>(arg)) {
      branchNodeSet.remove(*arg) ;
    }
  }

  // If no set is specified, restored all selection bits to kTRUE
  if (!selNodes) {
    // Reset PDF selection bits to kTRUE
    iter->Reset() ;
    while((arg=(RooAbsArg*)iter->Next())) {
      ((RooAbsReal*)arg)->selectComp(kTRUE) ;
    }
    delete iter ;
    return ;
  }


  // Add all nodes below selected nodes
  iter->Reset() ;
  TIterator* sIter = selNodes->createIterator() ;
  RooArgSet tmp ;
  while((arg=(RooAbsArg*)iter->Next())) {
    sIter->Reset() ;
    RooAbsArg* selNode ;
    while((selNode=(RooAbsArg*)sIter->Next())) {
      if (selNode->dependsOn(*arg)) {
	tmp.add(*arg,kTRUE) ;
      }      
    }      
  }
  delete sIter ;

  // Add all nodes that depend on selected nodes
  iter->Reset() ;
  while((arg=(RooAbsArg*)iter->Next())) {
    if (arg->dependsOn(*selNodes)) {
      tmp.add(*arg,kTRUE) ;
    }
  }

  tmp.remove(*selNodes,kTRUE) ;
  tmp.remove(*this) ;
  selNodes->add(tmp) ;
  coutI(Plotting) << "RooAbsPdf::plotOn(" << GetName() << ") indirectly selected PDF components: " << tmp << endl ;

  // Set PDF selection bits according to selNodes
  iter->Reset() ;
  while((arg=(RooAbsArg*)iter->Next())) {
    Bool_t select = selNodes->find(arg->GetName()) ? kTRUE : kFALSE ;
    ((RooAbsReal*)arg)->selectComp(select) ;
  }
  
  delete iter ;
} 




//_____________________________________________________________________________
// coverity[PASS_BY_VALUE]
RooPlot* RooAbsPdf::plotOn(RooPlot *frame, PlotOpt o) const
{
  // Plot oneself on 'frame'. In addition to features detailed in  RooAbsReal::plotOn(),
  // the scale factor for a PDF can be interpreted in three different ways. The interpretation
  // is controlled by ScaleType
  //
  //  Relative  -  Scale factor is applied on top of PDF normalization scale factor 
  //  NumEvent  -  Scale factor is interpreted as a number of events. The surface area
  //               under the PDF curve will match that of a histogram containing the specified
  //               number of event
  //  Raw       -  Scale factor is applied to the raw (projected) probability density.
  //               Not too useful, option provided for completeness.

  // Sanity checks
  if (plotSanityChecks(frame)) return frame ;

  // More sanity checks
  Double_t nExpected(1) ;
  if (o.stype==RelativeExpected) {
    if (!canBeExtended()) {
      coutE(Plotting) << "RooAbsPdf::plotOn(" << GetName() 
		      << "): ERROR the 'Expected' scale option can only be used on extendable PDFs" << endl ;
      return frame ;
    }
    nExpected = expectedEvents(frame->getNormVars()) ;
  }

  // Adjust normalization, if so requested
  if (o.stype != Raw) {    

    if (frame->getFitRangeNEvt() && o.stype==Relative) {
      // If non-default plotting range is specified, adjust number of events in fit range
      o.scaleFactor *= frame->getFitRangeNEvt()/nExpected ;
    } else if (o.stype==RelativeExpected) {
      o.scaleFactor *= nExpected ;
    } else if (o.stype==NumEvent) {
      o.scaleFactor /= nExpected ;
    }
    o.scaleFactor *= frame->getFitRangeBinW() ;
  }
  frame->updateNormVars(*frame->getPlotVar()) ;

  return RooAbsReal::plotOn(frame,o) ;
}




//_____________________________________________________________________________
RooPlot* RooAbsPdf::paramOn(RooPlot* frame, const RooCmdArg& arg1, const RooCmdArg& arg2, 
			    const RooCmdArg& arg3, const RooCmdArg& arg4, const RooCmdArg& arg5, 
			    const RooCmdArg& arg6, const RooCmdArg& arg7, const RooCmdArg& arg8)
{
  // Add a box with parameter values (and errors) to the specified frame
  //
  // The following named arguments are supported
  //
  //   Parameters(const RooArgSet& param) -- Only the specified subset of parameters will be shown. 
  //                                         By default all non-contant parameters are shown
  //   ShowConstants(Bool_t flag)         -- Also display constant parameters
  //   Format(const char* optStr)         -- Classing [arameter formatting options, provided for backward compatibility
  //   Format(const char* what,...)       -- Parameter formatting options, details given below
  //   Label(const chat* label)           -- Add header label to parameter box
  //   Layout(Double_t xmin,              -- Specify relative position of left,right side of box and top of box. Position of 
  //       Double_t xmax, Double_t ymax)     bottom of box is calculated automatically from number lines in box
  //                                 
  //
  // The Format(const char* what,...) has the following structure
  //
  //   const char* what      -- Controls what is shown. "N" adds name, "E" adds error, 
  //                            "A" shows asymmetric error, "U" shows unit, "H" hides the value
  //   FixedPrecision(int n) -- Controls precision, set fixed number of digits
  //   AutoPrecision(int n)  -- Controls precision. Number of shown digits is calculated from error 
  //                            + n specified additional digits (1 is sensible default)
  //
  // Example use: pdf.paramOn(frame, Label("fit result"), Format("NEU",AutoPrecision(1)) ) ;
  //

  // Stuff all arguments in a list
  RooLinkedList cmdList;
  cmdList.Add(const_cast<RooCmdArg*>(&arg1)) ;  cmdList.Add(const_cast<RooCmdArg*>(&arg2)) ;
  cmdList.Add(const_cast<RooCmdArg*>(&arg3)) ;  cmdList.Add(const_cast<RooCmdArg*>(&arg4)) ;
  cmdList.Add(const_cast<RooCmdArg*>(&arg5)) ;  cmdList.Add(const_cast<RooCmdArg*>(&arg6)) ;
  cmdList.Add(const_cast<RooCmdArg*>(&arg7)) ;  cmdList.Add(const_cast<RooCmdArg*>(&arg8)) ;

  // Select the pdf-specific commands 
  RooCmdConfig pc(Form("RooAbsPdf::paramOn(%s)",GetName())) ;
  pc.defineString("label","Label",0,"") ;
  pc.defineDouble("xmin","Layout",0,0.50) ;
  pc.defineDouble("xmax","Layout",1,0.99) ;
  pc.defineInt("ymaxi","Layout",0,Int_t(0.95*10000)) ;
  pc.defineInt("showc","ShowConstants",0,0) ;
  pc.defineObject("params","Parameters",0,0) ;
  pc.defineString("formatStr","Format",0,"NELU") ;
  pc.defineInt("sigDigit","Format",0,2) ;
  pc.defineInt("dummy","FormatArgs",0,0) ;
  pc.defineMutex("Format","FormatArgs") ;

  // Process and check varargs 
  pc.process(cmdList) ;
  if (!pc.ok(kTRUE)) {
    return frame ;
  }

  const char* label = pc.getString("label") ;
  Double_t xmin = pc.getDouble("xmin") ;
  Double_t xmax = pc.getDouble("xmax") ;
  Double_t ymax = pc.getInt("ymaxi") / 10000. ;
  Int_t showc = pc.getInt("showc") ;


  const char* formatStr = pc.getString("formatStr") ;
  Int_t sigDigit = pc.getInt("sigDigit") ;  

  // Decode command line arguments
  RooArgSet* params = static_cast<RooArgSet*>(pc.getObject("params")) ;
  if (!params) {
    params = getParameters(frame->getNormVars()) ;
    if (pc.hasProcessed("FormatArgs")) {
      const RooCmdArg* formatCmd = static_cast<RooCmdArg*>(cmdList.FindObject("FormatArgs")) ;
      paramOn(frame,*params,showc,label,0,0,xmin,xmax,ymax,formatCmd) ;
    } else {
      paramOn(frame,*params,showc,label,sigDigit,formatStr,xmin,xmax,ymax) ;
    }
    delete params ;
  } else {
    RooArgSet* pdfParams = getParameters(frame->getNormVars()) ;    
    RooArgSet* selParams = static_cast<RooArgSet*>(pdfParams->selectCommon(*params)) ;
    if (pc.hasProcessed("FormatArgs")) {
      const RooCmdArg* formatCmd = static_cast<RooCmdArg*>(cmdList.FindObject("FormatArgs")) ;
      paramOn(frame,*selParams,showc,label,0,0,xmin,xmax,ymax,formatCmd) ;
    } else {
      paramOn(frame,*selParams,showc,label,sigDigit,formatStr,xmin,xmax,ymax) ;
    }
    delete selParams ;
    delete pdfParams ;
  }
  
  return frame ;
}




//_____________________________________________________________________________
RooPlot* RooAbsPdf::paramOn(RooPlot* frame, const RooAbsData* data, const char *label,
			    Int_t sigDigits, Option_t *options, Double_t xmin,
			    Double_t xmax ,Double_t ymax) 
{
  // OBSOLETE FUNCTION PROVIDED FOR BACKWARD COMPATIBILITY

  RooArgSet* params = getParameters(data) ;
  TString opts(options) ;  
  paramOn(frame,*params,opts.Contains("c"),label,sigDigits,options,xmin,xmax,ymax) ;
  delete params ;
  return frame ;
}



//_____________________________________________________________________________
RooPlot* RooAbsPdf::paramOn(RooPlot* frame, const RooArgSet& params, Bool_t showConstants, const char *label,
			    Int_t sigDigits, Option_t *options, Double_t xmin,
			    Double_t xmax ,Double_t ymax, const RooCmdArg* formatCmd) 
{
  // Add a text box with the current parameter values and their errors to the frame.
  // Observables of this PDF appearing in the 'data' dataset will be omitted.
  //
  // Optional label will be inserted as first line of the text box. Use 'sigDigits'
  // to modify the default number of significant digits printed. The 'xmin,xmax,ymax'
  // values specify the inital relative position of the text box in the plot frame  


  // parse the options
  TString opts = options;
  opts.ToLower();
  Bool_t showLabel= (label != 0 && strlen(label) > 0);
  
  // calculate the box's size, adjusting for constant parameters
  TIterator* pIter = params.createIterator() ;

  Int_t nPar= params.getSize();
  Double_t ymin(ymax), dy(0.06);
  Int_t index(nPar);
  RooRealVar *var = 0;
  while((var=(RooRealVar*)pIter->Next())) {
    if(showConstants || !var->isConstant()) ymin-= dy;
  }

  if(showLabel) ymin-= dy;

  // create the box and set its options
  TPaveText *box= new TPaveText(xmin,ymax,xmax,ymin,"BRNDC");
  if(!box) return 0;
  box->SetName(Form("%s_paramBox",GetName())) ;
  box->SetFillColor(0);
  box->SetBorderSize(1);
  box->SetTextAlign(12);
  box->SetTextSize(0.04F);
  box->SetFillStyle(1001);
  box->SetFillColor(0);
  //char buffer[512];
  index= nPar;
  pIter->Reset() ;
  while((var=(RooRealVar*)pIter->Next())) {
    if(var->isConstant() && !showConstants) continue;
    
    TString *formatted= options ? var->format(sigDigits, options) : var->format(*formatCmd) ;
    box->AddText(formatted->Data());
    delete formatted;
  }
  // add the optional label if specified
  if(showLabel) box->AddText(label);

  // Add box to frame 
  frame->addObject(box) ;

  delete pIter ;
  return frame ;
}




//_____________________________________________________________________________
Double_t RooAbsPdf::expectedEvents(const RooArgSet*) const 
{ 
  // Return expected number of events from this p.d.f for use in extended
  // likelihood calculations. This default implementation returns zero
  return 0 ; 
} 



//_____________________________________________________________________________
void RooAbsPdf::verboseEval(Int_t stat) 
{ 
  // Change global level of verbosity for p.d.f. evaluations

  _verboseEval = stat ; 
}



//_____________________________________________________________________________
Int_t RooAbsPdf::verboseEval() 
{ 
  // Return global level of verbosity for p.d.f. evaluations

  return _verboseEval ;
}



//_____________________________________________________________________________
void RooAbsPdf::CacheElem::operModeHook(RooAbsArg::OperMode) 
{
  // Dummy implementation
}



//_____________________________________________________________________________
RooAbsPdf::CacheElem::~CacheElem() 
{ 
  // Destructor of normalization cache element. If this element 
  // provides the 'current' normalization stored in RooAbsPdf::_norm
  // zero _norm pointer here before object pointed to is deleted here

  // Zero _norm pointer in RooAbsPdf if it is points to our cache payload
  if (_owner) {
    RooAbsPdf* pdfOwner = static_cast<RooAbsPdf*>(_owner) ;
    if (pdfOwner->_norm == _norm) {
      pdfOwner->_norm = 0 ;
    }
  }

  delete _norm ; 
} 



//_____________________________________________________________________________
RooAbsPdf* RooAbsPdf::createProjection(const RooArgSet& iset) 
{
  // Return a p.d.f that represent a projection of this p.d.f integrated over given observables

  // Construct name for new object
  TString name(GetName()) ;
  name.Append("_Proj[") ;
  if (iset.getSize()>0) {
    TIterator* iter = iset.createIterator() ;
    RooAbsArg* arg ;
    Bool_t first(kTRUE) ;
    while((arg=(RooAbsArg*)iter->Next())) {
      if (first) {
	first=kFALSE ;
      } else {
	name.Append(",") ;
      }
      name.Append(arg->GetName()) ;
    }
    delete iter ;
  }
  name.Append("]") ;
  
  // Return projected p.d.f.
  return new RooProjectedPdf(name.Data(),name.Data(),*this,iset) ;
}



//_____________________________________________________________________________
RooAbsReal* RooAbsPdf::createCdf(const RooArgSet& iset, const RooArgSet& nset) 
{
  // Create a cumulative distribution function of this p.d.f in terms
  // of the observables listed in iset. If no nset argument is given
  // the c.d.f normalization is constructed over the integrated
  // observables, so that its maximum value is precisely 1. It is also
  // possible to choose a different normalization for
  // multi-dimensional p.d.f.s: eg. for a pdf f(x,y,z) one can
  // construct a partial cdf c(x,y) that only when integrated itself
  // over z results in a maximum value of 1. To construct such a cdf pass
  // z as argument to the optional nset argument

  return createCdf(iset,RooFit::SupNormSet(nset)) ;
}



//_____________________________________________________________________________
RooAbsReal* RooAbsPdf::createCdf(const RooArgSet& iset, const RooCmdArg& arg1, const RooCmdArg& arg2,
				 const RooCmdArg& arg3, const RooCmdArg& arg4, const RooCmdArg& arg5, 
				 const RooCmdArg& arg6, const RooCmdArg& arg7, const RooCmdArg& arg8) 
{
  // Create an object that represents the integral of the function over one or more observables listed in iset
  // The actual integration calculation is only performed when the return object is evaluated. The name
  // of the integral object is automatically constructed from the name of the input function, the variables
  // it integrates and the range integrates over
  //
  // The following named arguments are accepted
  //
  // SupNormSet(const RooArgSet&)         -- Observables over which should be normalized _in_addition_ to the
  //                                         integration observables
  // ScanNumCdf()                         -- Apply scanning technique if cdf integral involves numeric integration [ default ] 
  // ScanAllCdf()                         -- Always apply scanning technique 
  // ScanNoCdf()                          -- Never apply scanning technique                  
  // ScanParameters(Int_t nbins,          -- Parameters for scanning technique of making CDF: number
  //                Int_t intOrder)          of sampled bins and order of interpolation applied on numeric cdf

  // Define configuration for this method
  RooCmdConfig pc(Form("RooAbsReal::createCdf(%s)",GetName())) ;
  pc.defineObject("supNormSet","SupNormSet",0,0) ;
  pc.defineInt("numScanBins","ScanParameters",0,1000) ;
  pc.defineInt("intOrder","ScanParameters",1,2) ;
  pc.defineInt("doScanNum","ScanNumCdf",0,1) ;
  pc.defineInt("doScanAll","ScanAllCdf",0,0) ;
  pc.defineInt("doScanNon","ScanNoCdf",0,0) ;
  pc.defineMutex("ScanNumCdf","ScanAllCdf","ScanNoCdf") ;

  // Process & check varargs 
  pc.process(arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8) ;
  if (!pc.ok(kTRUE)) {
    return 0 ;
  }

  // Extract values from named arguments
  const RooArgSet* snset = static_cast<const RooArgSet*>(pc.getObject("supNormSet",0)) ;
  RooArgSet nset ;
  if (snset) {
    nset.add(*snset) ;
  }
  Int_t numScanBins = pc.getInt("numScanBins") ;
  Int_t intOrder = pc.getInt("intOrder") ;
  Int_t doScanNum = pc.getInt("doScanNum") ;
  Int_t doScanAll = pc.getInt("doScanAll") ;
  Int_t doScanNon = pc.getInt("doScanNon") ;

  // If scanning technique is not requested make integral-based cdf and return
  if (doScanNon) {
    return createIntRI(iset,nset) ;
  }
  if (doScanAll) {
    return createScanCdf(iset,nset,numScanBins,intOrder) ;
  }
  if (doScanNum) {
    RooRealIntegral* tmp = (RooRealIntegral*) createIntegral(iset) ;
    Int_t isNum= (tmp->numIntRealVars().getSize()>0) ;
    delete tmp ;

    if (isNum) {
      coutI(NumIntegration) << "RooAbsPdf::createCdf(" << GetName() << ") integration over observable(s) " << iset << " involves numeric integration," << endl 
			    << "      constructing cdf though numeric integration of sampled pdf in " << numScanBins << " bins and applying order " 
			    << intOrder << " interpolation on integrated histogram." << endl 
			    << "      To override this choice of technique use argument ScanNone(), to change scan parameters use ScanParameters(nbins,order) argument" << endl ;
    }
    
    return isNum ? createScanCdf(iset,nset,numScanBins,intOrder) : createIntRI(iset,nset) ;
  }
  return 0 ;
}

RooAbsReal* RooAbsPdf::createScanCdf(const RooArgSet& iset, const RooArgSet& nset, Int_t numScanBins, Int_t intOrder) 
{
  string name = string(GetName()) + "_NUMCDF_" + integralNameSuffix(iset,&nset).Data() ;  
  RooRealVar* ivar = (RooRealVar*) iset.first() ;
  ivar->setBins(numScanBins,"numcdf") ;
  RooNumCdf* ret = new RooNumCdf(name.c_str(),name.c_str(),*this,*ivar,"numcdf") ;
  ret->setInterpolationOrder(intOrder) ;
  return ret ;
}




//_____________________________________________________________________________
RooArgSet* RooAbsPdf::getAllConstraints(const RooArgSet& observables, RooArgSet& constrainedParams, Bool_t stripDisconnected) const 
{
  // This helper function finds and collects all constraints terms of all coponent p.d.f.s
  // and returns a RooArgSet with all those terms

  RooArgSet* ret = new RooArgSet("AllConstraints") ;

  RooArgSet* comps = getComponents() ;
  TIterator* iter = comps->createIterator() ;
  RooAbsArg *arg ;
  while((arg=(RooAbsArg*)iter->Next())) {
    RooAbsPdf* pdf = dynamic_cast<RooAbsPdf*>(arg) ;
    if (pdf && !ret->find(pdf->GetName())) {
      RooArgSet* compRet = pdf->getConstraints(observables,constrainedParams,stripDisconnected) ;
      if (compRet) {
	ret->add(*compRet,kFALSE) ;
	delete compRet ;
      }
    }
  }
  delete iter ;
  delete comps ;

  return ret ;
}


//_____________________________________________________________________________
void RooAbsPdf::clearEvalError() 
{ 
  // Clear the evaluation error flag
  _evalError = kFALSE ; 
}



//_____________________________________________________________________________
Bool_t RooAbsPdf::evalError() 
{ 
  // Return the evaluation error flag
  return _evalError ; 
}



//_____________________________________________________________________________
void RooAbsPdf::raiseEvalError() 
{ 
  // Raise the evaluation error flag
  _evalError = kTRUE ; 
}



//_____________________________________________________________________________
RooNumGenConfig* RooAbsPdf::defaultGeneratorConfig() 
{
  // Returns the default numeric MC generator configuration for all RooAbsReals
  return &RooNumGenConfig::defaultConfig() ;
}


//_____________________________________________________________________________
RooNumGenConfig* RooAbsPdf::specialGeneratorConfig() const 
{
  // Returns the specialized integrator configuration for _this_ RooAbsReal.
  // If this object has no specialized configuration, a null pointer is returned
  return _specGeneratorConfig ;
}



//_____________________________________________________________________________
RooNumGenConfig* RooAbsPdf::specialGeneratorConfig(Bool_t createOnTheFly) 
{
  // Returns the specialized integrator configuration for _this_ RooAbsReal.
  // If this object has no specialized configuration, a null pointer is returned,
  // unless createOnTheFly is kTRUE in which case a clone of the default integrator
  // configuration is created, installed as specialized configuration, and returned

  if (!_specGeneratorConfig && createOnTheFly) {
    _specGeneratorConfig = new RooNumGenConfig(*defaultGeneratorConfig()) ;
  }
  return _specGeneratorConfig ;
}



//_____________________________________________________________________________
const RooNumGenConfig* RooAbsPdf::getGeneratorConfig() const 
{
  // Return the numeric MC generator configuration used for this object. If
  // a specialized configuration was associated with this object, that configuration
  // is returned, otherwise the default configuration for all RooAbsReals is returned

  const RooNumGenConfig* config = specialGeneratorConfig() ;
  if (config) return config ;
  return defaultGeneratorConfig() ;
}



//_____________________________________________________________________________
void RooAbsPdf::setGeneratorConfig(const RooNumGenConfig& config) 
{
  // Set the given configuration as default numeric MC generator
  // configuration for this object
  if (_specGeneratorConfig) {
    delete _specGeneratorConfig ;
  }
  _specGeneratorConfig = new RooNumGenConfig(config) ;  
}



//_____________________________________________________________________________
void RooAbsPdf::setGeneratorConfig() 
{
  // Remove the specialized numeric MC generator configuration associated
  // with this object
  if (_specGeneratorConfig) {
    delete _specGeneratorConfig ;
  }
  _specGeneratorConfig = 0 ;
}



//_____________________________________________________________________________
RooAbsPdf::GenSpec::~GenSpec() 
{
  delete _genContext ;
}


//_____________________________________________________________________________
RooAbsPdf::GenSpec::GenSpec(RooAbsGenContext* context, const RooArgSet& whatVars, RooDataSet* protoData, Int_t nGen, 
			    Bool_t extended, Bool_t randProto, Bool_t resampleProto, TString dsetName) :
  _genContext(context), _whatVars(whatVars), _protoData(protoData), _nGen(nGen), _extended(extended), 
  _randProto(randProto), _resampleProto(resampleProto), _dsetName(dsetName) 
{
}



//_____________________________________________________________________________
void RooAbsPdf::setNormRange(const char* rangeName) 
{ 
  if (rangeName) {
    _normRange = rangeName ; 
  } else {
    _normRange.Clear() ; 
  }

  if (_norm) { 
    _normMgr.sterilize() ;
    _norm = 0 ; 
  }
}


//_____________________________________________________________________________
void RooAbsPdf::setNormRangeOverride(const char* rangeName) 
{
  if (rangeName) {
    _normRangeOverride = rangeName ; 
  } else {
    _normRangeOverride.Clear() ; 
  }

  if (_norm) { 
    _normMgr.sterilize() ;
    _norm = 0 ; 
  }
}
