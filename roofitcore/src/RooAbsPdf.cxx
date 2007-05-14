/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Name:  $:$Id: RooAbsPdf.cxx,v 1.102 2007/05/11 09:11:58 verkerke Exp $
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
// if any of its variables are functions instead of fundamentals. In
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
// [Direct generation of dependents]
//
// Any PDF dependent can be generated with the accept/reject method,
// but for certain PDFs more efficient methods may be implemented. To
// implement direct generation of one or more dependents, two
// functions need to be implemented, similar to those for analytical
// integrals:
//
// Int_t getGenerator(const RooArgSet& generateVars, RooArgSet& directVars) and
// void generateEvent(Int_t code)
//
// The first function advertises dependents that can be generated,
// similar to the way analytical integrals are advertised. The second
// function implements the generator for the advertised dependents
//
// The generated dependent values should be store in the proxy
// objects. For this the assignment operator can be used (i.e. xProxy
// = 3.0 ). Never call assign to any proxy not known to be a dependent
// via the generation code.  Doing so may be ill-defined, e.g. in case
// the proxy holds a function, and will trigger an assert


#include "RooFit.h"

#include "TClass.h"
#include "Riostream.h"
#include "TMath.h"
#include "TObjString.h"
#include "TPaveText.h"
#include "TList.h"
#include "TH1.h"
#include "TH2.h"
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
#include "RooInt.h"

ClassImp(RooAbsPdf) 
;


Int_t RooAbsPdf::_verboseEval = 0;
Bool_t RooAbsPdf::_globalSelectComp = kFALSE ;
Bool_t RooAbsPdf::_evalError = kFALSE ;


RooAbsPdf::RooAbsPdf(const char *name, const char *title) : 
  RooAbsReal(name,title), _norm(0), _normSet(0), _normMgr(10), _selectComp(kTRUE)
{
  // Constructor with name and title only
  resetErrorCounters() ;
  setTraceCounter(0) ;
}


RooAbsPdf::RooAbsPdf(const char *name, const char *title, 
		     Double_t plotMin, Double_t plotMax) :
  RooAbsReal(name,title,plotMin,plotMax), _norm(0), _normSet(0), _normMgr(10), _selectComp(kTRUE)
{
  // Constructor with name, title, and plot range
  resetErrorCounters() ;
  setTraceCounter(0) ;
}



RooAbsPdf::RooAbsPdf(const RooAbsPdf& other, const char* name) : 
  RooAbsReal(other,name), _norm(0), _normSet(0), _normMgr(10), _selectComp(other._selectComp)

{
  // Copy constructor
  resetErrorCounters() ;
  setTraceCounter(other._traceCount) ;
}


RooAbsPdf::~RooAbsPdf()
{
  // Destructor
  //if (_norm) delete _norm ;
}


Double_t RooAbsPdf::getVal(const RooArgSet* nset) const
{
  // Return current value, normalizated by integrating over
  // the dependents in 'nset'. If 'nset' is 0, the unnormalized value. 
  // is returned. All elements of 'nset' must be lvalues


  // Unnormalized values are not cached
  // Doing so would be complicated as _norm->getVal() could
  // spoil the cache and interfere with returning the cached
  // return value. Since unnormalized calls are typically
  // done in integration calls, there is no performance hit.

  if (!nset) {
    Double_t val = evaluate() ;
    Bool_t error = traceEvalPdf(val) ;
    if (_verboseEval>1) cout << IsA()->GetName() << "::getVal(" << GetName() 
			     << "): value = " << val << " (unnormalized)" << endl ;
    if (error) {
      raiseEvalError() ;
      return 0 ;
    }
    return val ;
  }

  // Process change in last data set used
  Bool_t nsetChanged(kFALSE) ;
  if (nset!=_normSet) {
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
    if (normVal==0.) normError=kTRUE ;

    // Raise global error flag if problems occur
    if (normError||error) raiseEvalError() ;

    _value = normError ? 0 : (rawVal / normVal) ;

    if (_verboseEval>1) cout << IsA()->GetName() << "::getVal(" << GetName() << "): value = " 
			     << rawVal << " / " << _norm->getVal() << " = " << _value << endl ;

    clearValueDirty() ; //setValueDirty(kFALSE) ;
    clearShapeDirty() ; //setShapeDirty(kFALSE) ;    
  } 

  if (_traceCount>0) {
    cout << "[" << _traceCount << "] " ;
    Int_t tmp = _traceCount ;
    _traceCount = 0 ;
    Print() ;
    _traceCount = tmp-1  ;
  }

  return _value ;
}


Double_t RooAbsPdf::analyticalIntegralWN(Int_t code, const RooArgSet* normSet, const char* rangeName) const
{
  // Analytical integral with normalization (see RooAbsReal::analyticalIntegralWN() for further information)
  //
  // This function applies the normalization specified by 'normSet' to the integral returned
  // by RooAbsReal::analyticalIntegral(). The passthrough scenario (code=0) is also changed
  // to return a normalized answer

  if (_verboseEval>1) {
    cout << "RooAbsPdf::analyticalIntegralWN(" << GetName() << ") code = " << code << " normset = " ;
    if (normSet) normSet->Print("1") ; else cout << "<none>" << endl ;
  }

  if (code==0) return getVal(normSet) ;
  if (normSet) {
    return analyticalIntegral(code,rangeName) / getNorm(normSet) ;
  } else {
    return analyticalIntegral(code,rangeName) ;
  }
}


Bool_t RooAbsPdf::traceEvalPdf(Double_t value) const
{
  // Check that passed value is positive and not 'not-a-number'.
  // If not, print an error, until the error counter reaches
  // its set maximum.

  // check for a math error or negative value
  Bool_t error= isnan(value) || (value < 0);

  // do nothing if we are no longer tracing evaluations and there was no error
  if(!error) return error ;

  // otherwise, print out this evaluations input values and result
  if(++_errorCount <= 10) {
    cout << "*** Evaluation Error " << _errorCount << " ";
    if(_errorCount == 10) cout << "(no more will be printed) ";
  }
  else {
    return error  ;
  }

  Print() ;
  return error ;
}



Double_t RooAbsPdf::getNorm(const RooArgSet* nset) const
{
  // Return the integral of this PDF over all elements of 'nset'. 

  if (!nset) return 1 ;

  syncNormalization(nset,kTRUE) ;
  if (_verboseEval>1) cout << IsA()->GetName() << "::getNorm(" << GetName() << "): norm(" << _norm << ") = " << _norm->getVal() << endl ;

  Double_t ret = _norm->getVal() ;
  if (ret==0.) {
    if(++_errorCount <= 10) {
      cout << "RooAbsPdf::getNorm(" << GetName() << ":: WARNING normalization is zero, nset = " ;  nset->Print("1") ;
      if(_errorCount == 10) cout << "RooAbsPdf::getNorm(" << GetName() << ") INFO: no more messages will be printed " << endl ;
    }
  }

  return ret ;
}


const RooAbsReal* RooAbsPdf::getNormObj(const RooArgSet* nset, const RooArgSet* iset, const TNamed* rangeName) const 
{
  // Check normalization is already stored
  RooAbsReal* norm = _normMgr.getNormalization(this,nset,iset,rangeName) ;
  if (norm) {
    return norm ;
  }

  // If not create it now
  RooArgSet* depList = getObservables(iset) ;
  norm = createIntegral(*depList,*nset, *getIntegratorConfig(), RooNameReg::str(rangeName)) ;
  delete depList ;

  // Store it in the cache
  _normMgr.setNormalization(this,nset,iset,rangeName,norm) ;

  // And return the newly created integral
  return norm ;
}


Bool_t RooAbsPdf::syncNormalizationPreHook(RooAbsReal*,const RooArgSet*) const 
{ 
  return kFALSE ; 
} 

void RooAbsPdf::syncNormalizationPostHook(RooAbsReal*,const RooArgSet*) const 
{
} 


Bool_t RooAbsPdf::syncNormalization(const RooArgSet* nset, Bool_t adjustProxies) const
{
  // Verify that the normalization integral cached with this PDF
  // is valid for given set of normalization dependents
  //
  // If not, the cached normalization integral (if any) is deleted
  // and a new integral is constructed for use with 'nset'
  // Elements in 'nset' can be discrete and real, but must be lvalues
  //
  // By default, only actual dependents of the PDF listed in 'nset'
  // are integration. This behaviour can be modified in subclasses
  // by overloading the syncNormalizationPreHook() function. 
  // 
  // For functions that declare to be self-normalized by overloading the
  // selfNormalized() function, a unit normalization is always constructed
  
  _normSet = (RooArgSet*) nset ;

  // Check if data sets are identical
  RooAbsReal* norm = _normMgr.getNormalization(this,nset,0) ;
  if (norm) {
    Bool_t nsetChanged = (_norm!=norm) ;
    _norm = norm ;

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
  
  // Allow optional post-processing
  Bool_t fullNorm = syncNormalizationPreHook(_norm,nset) ;


  RooArgSet* depList ;
  if (fullNorm) {
    depList = ((RooArgSet*)nset) ;
  } else {
    depList = getObservables(nset) ;
  }


  if (_verboseEval>0) {
    if (!selfNormalized()) {
      cout << IsA()->GetName() << "::syncNormalization(" << GetName() 
	   << ") recreating normalization integral " << endl ;
      if (depList) depList->printToStream(cout,OneLine) ; else cout << "<none>" << endl ;
    } else {
      cout << IsA()->GetName() << "::syncNormalization(" << GetName() << ") selfNormalized, creating unit norm" << endl;
    }
  }


  // Destroy old normalization & create new
  if (selfNormalized() || !dependsOn(*depList)) {    
    TString ntitle(GetTitle()) ; ntitle.Append(" Unit Normalization") ;
    TString nname(GetName()) ; nname.Append("_UnitNorm") ;
    _norm = new RooRealVar(nname.Data(),ntitle.Data(),1) ;
  } else {

    _norm = createIntegral(*depList,*getIntegratorConfig()) ;
  }

  // Register new normalization with manager (takes ownership)
  _normMgr.setNormalization(this,nset,0,0,_norm) ;

  // Allow optional post-processing
  syncNormalizationPostHook(_norm,nset) ;
 
  if (!fullNorm) delete depList ;
  return kTRUE ;
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
  // Reset error counter to given value, limiting the number
  // of future error messages for this pdf to 'resetValue'

  _errorCount = resetValue ;
  _negCount   = resetValue ;
}



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




void RooAbsPdf::operModeHook() 
{
  // WVE 08/21/01 Probably obsolete now
}



Double_t RooAbsPdf::getLogVal(const RooArgSet* nset) const 
{
  // Return the log of the current value with given normalization
  // An error message is printed if the argument of the log is negative.

  Double_t prob = getVal(nset) ;
  if(prob <= 0) {

    if (_negCount-- > 0) {
      cout << endl 
	   << "RooAbsPdf::getLogVal(" << GetName() << ") WARNING: PDF evaluates to zero or negative value (" << prob << ")" << endl;
      RooArgSet* params = getParameters(nset) ;
      RooArgSet* depends = getObservables(nset) ;	 
      cout << "  Current values of PDF dependents:" ;
      depends->Print("v") ;
      cout << "  Current values of PDF parameters:" ;
      params->Print("v") ;
      delete params ;
      delete depends ;

      if(_negCount == 0) cout << "(no more such warnings will be printed) "<<endl;
    }
    return 0;
  }
  return log(prob);
}



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
    cout << fName << ": this PDF does not support extended maximum likelihood"
         << endl;
    return 0;
  }

  Double_t expected= expectedEvents(nset);
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


RooFitResult* RooAbsPdf::fitTo(RooAbsData& data, RooCmdArg arg1, RooCmdArg arg2, RooCmdArg arg3, RooCmdArg arg4, 
                                                 RooCmdArg arg5, RooCmdArg arg6, RooCmdArg arg7, RooCmdArg arg8) 
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
  // NumCPU(int num)                 -- Parallelize NLL calculation on num CPUs
  // Optimize(Bool_t flag)           -- Activate constant term optimization (on by default)
  // SplitRange(Bool_t flag)         -- Use separate fit ranges in a simultaneous fit. Actual range name for each
  //                                    subsample is assumed to by rangeName_{indexState} where indexState
  //                                    is the state of the master index category of the simultaneous fit
  //
  // Options to control flow of fit procedure
  // ----------------------------------------
  // InitialHesse(Bool_t flag)      -- Flag controls if HESSE before MIGRAD as well, off by default
  // Hesse(Bool_t flag)             -- Flag controls if HESSE is run after MIGRAD, on by default
  // Minos(Bool_t flag)             -- Flag controls if MINOS is run after HESSE, on by default
  // Minos(const RooArgSet& set)    -- Only run MINOS on given subset of arguments
  // Save(Bool_t flag)              -- Flac controls if RooFitResult object is produced and returned, off by default
  // Strategy(Int_t flag)           -- Set Minuit strategy (0 through 2, default is 1)
  // FitOptions(const char* optStr) -- Steer fit with classic options string (for backward compatibility). Use of this option
  //                                   excludes use of any of the new style steering options.
  //
  // Options to control informational output
  // ---------------------------------------
  // Verbose(Bool_t flag)           -- Flag controls if verbose output is printed (NLL, parameter changes during fit
  // Timer(Bool_t flag)             -- Time CPU and wall clock consumption of fit steps, off by default
  // PrintLevel(Int_t level)        -- Set Minuit print level (-1 through 3, default is 1). At -1 all RooFit informational 
  //                                   messages are suppressed as well
  // 
  // 
  
  RooLinkedList l ;
  l.Add((TObject*)&arg1) ;  l.Add((TObject*)&arg2) ;  
  l.Add((TObject*)&arg3) ;  l.Add((TObject*)&arg4) ;
  l.Add((TObject*)&arg5) ;  l.Add((TObject*)&arg6) ;  
  l.Add((TObject*)&arg7) ;  l.Add((TObject*)&arg8) ;
  return fitTo(data,l) ;
}

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

  pc.defineString("fitOpt","FitOptions",0,"") ;
  pc.defineString("rangeName","RangeWithName",0,"",kTRUE) ;
  pc.defineDouble("rangeLo","Range",0,-999.) ;
  pc.defineDouble("rangeHi","Range",1,-999.) ;
  pc.defineInt("splitRange","SplitRange",0,0) ;
  pc.defineInt("optConst","Optimize",0,1) ;
  pc.defineInt("verbose","Verbose",0,0) ;
  pc.defineInt("doSave","Save",0,0) ;
  pc.defineInt("doTimer","Timer",0,0) ;
  pc.defineInt("plevel","PrintLevel",0,1) ;
  pc.defineInt("strat","Strategy",0,1) ;
  pc.defineInt("initHesse","InitialHesse",0,0) ;
  pc.defineInt("hesse","Hesse",0,1) ;
  pc.defineInt("minos","Minos",0,1) ;
  pc.defineInt("ext","Extended",0,0) ;
  pc.defineInt("numcpu","NumCPU",0,1) ;
  pc.defineObject("projDepSet","ProjectedObservables",0,0) ;
  pc.defineObject("minosSet","Minos",0,0) ;
  pc.defineMutex("FitOptions","Verbose") ;
  pc.defineMutex("FitOptions","Save") ;
  pc.defineMutex("FitOptions","Timer") ;
  pc.defineMutex("FitOptions","Strategy") ;
  pc.defineMutex("FitOptions","InitialHesse") ;
  pc.defineMutex("FitOptions","Hesse") ;
  pc.defineMutex("FitOptions","Minos") ;
  pc.defineMutex("Range","RangeWithName") ;

  
  // Process and check varargs 
  pc.process(cmdList) ;
  if (!pc.ok(kTRUE)) {
    return 0 ;
  }

  // Decode command line arguments
  const char* fitOpt = pc.getString("fitOpt",0,kTRUE) ;
  const char* rangeName = pc.getString("rangeName",0,kTRUE) ;
  Int_t optConst = pc.getInt("optConst") ;
  Int_t verbose  = pc.getInt("verbose") ;
  Int_t doSave   = pc.getInt("doSave") ;
  Int_t doTimer  = pc.getInt("doTimer") ;
  Int_t plevel    = pc.getInt("plevel") ;
  Int_t strat    = pc.getInt("strat") ;
  Int_t initHesse= pc.getInt("initHesse") ;
  Int_t hesse    = pc.getInt("hesse") ;
  Int_t minos    = pc.getInt("minos") ;
  Int_t ext      = pc.getInt("ext") ;
  Int_t numcpu   = pc.getInt("numcpu") ;
  Int_t splitr   = pc.getInt("splitRange") ;
  const RooArgSet* minosSet = static_cast<RooArgSet*>(pc.getObject("minosSet")) ;

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
  RooArgSet* tmp = (RooArgSet*) pc.getObject("projDepSet") ;  
  if (tmp) projDeps.add(*tmp) ;
  
  // Construct NLL
  RooAbsReal* nll ;
  if (!rangeName || strchr(rangeName,',')==0) {
    // Simple case: default range, or single restricted range
    nll = new RooNLLVar("nll","-log(likelihood)",*this,data,projDeps,ext,rangeName,numcpu,plevel!=-1,splitr) ;
  } else {
    // Composite case: multiple ranges
    RooArgList nllList ;
    char* buf = new char[strlen(rangeName)+1] ;
    strcpy(buf,rangeName) ;
    char* token = strtok(buf,",") ;
    while(token) {
      RooAbsReal* nllComp = new RooNLLVar(Form("nll_%s",token),"-log(likelihood)",*this,data,projDeps,ext,token,numcpu,plevel!=-1,splitr) ;
      nllList.add(*nllComp) ;
      token = strtok(0,",") ;
    }
    delete[] buf ;
    nll = new RooAddition("nll","-log(likelihood)",nllList,kTRUE) ;
  }
  
  // Instantiate MINUIT
  RooMinuit m(*nll) ;

  if (plevel!=1) {
    m.setPrintLevel(plevel) ;
  }

  if (optConst) {
    // Activate constant term optimization
    m.optimizeConst(1) ;
  }

  RooFitResult *ret = 0 ;

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
      ret = m.save() ;
    } 

  }
  
  // Cleanup
  delete nll ;
  return ret ;
}



RooFitResult* RooAbsPdf::fitTo(RooAbsData& data, Option_t *fitOpt, Option_t *optOpt, const char* fitRange) 
{
  return fitTo(data,RooArgSet(),fitOpt,optOpt,fitRange) ;
}


RooFitResult* RooAbsPdf::fitTo(RooAbsData& data, const RooArgSet& projDeps, Option_t *fitOpt, Option_t *optOpt, const char* fitRange) 
{
  // Fit this PDF to given data set
  //
  // OLD STYLE INTERFACE, PLEASE USE NEW INTERFACE fitTo(RooAbsData& data, RooCmdArg arg1,...,RooCmdArg arg8) 
  //
  // The dataset can be either binned, in which case a binned maximum likelihood fit
  // is performed, or unbinned, in which case an unbinned maximum likelihood fit is performed
  //
  // Available fit options:
  //
  //  "m" = MIGRAD only, i.e. no MINOS 
  //  "s" = estimate step size with HESSE before starting MIGRAD
  //  "h" = run HESSE after MIGRAD
  //  "e" = Perform extended MLL fit
  //  "0" = Run MIGRAD with strategy MINUIT 0 (no correlation matrix calculation at end)
  //        Does not apply to HESSE or MINOS, if run afterwards.
  // 
  //  "q" = Switch off verbose mode
  //  "l" = Save log file with parameter values at each MINUIT step
  //  "v" = Show changed parameters at each MINUIT step
  //  "t" = Time fit 
  //  "r" = Save fit output in RooFitResult object (return value is object RFR pointer)
  //
  // Available optimizer options
  //
  //  "c" = Cache and precalculate components of PDF that exclusively depend on constant parameters
  //  "2" = Do NLL calculation in multi-processor mode on 2 processors
  //  "3" = Do NLL calculation in multi-processor mode on 3 processors
  //  "4" = Do NLL calculation in multi-processor mode on 4 processors
  //
  // The actual fit is performed to a temporary copy of both PDF and data set. Several optimization
  // algorithm are run to increase the efficiency of the likelihood calculation and may increase
  // the speed of complex fits up to an order of magnitude. All optimizations are exact, i.e the fit result
  // of any fit should _exactly_ the same with and without optimization. We strongly encourage
  // to stick to the default optimizer setting (all on). If for any reason you see a difference in the result
  // with and without optimizer, please file a bug report.
  //
  // The function always return null unless the "r" fit option is specified. In that case a pointer to a RooFitResult
  // is returned. The RooFitResult object contains the full fit output, including the correlation matrix.

  // Parse option strings
  TString fopt(fitOpt) ;
  TString oopt(optOpt) ;
  fopt.ToLower() ;
  oopt.ToLower() ;

  Bool_t extended = fopt.Contains("e") ;  
  Bool_t saveRes  = fopt.Contains("r") ;
  Bool_t cOpt     = oopt.Contains("p") || // for backward compatibility
                    oopt.Contains("c") ;
  Bool_t blindfit   = fopt.Contains("b") ;  


  Int_t  ncpu = 1 ;
  if (oopt.Contains("2")) ncpu=2 ;
  if (oopt.Contains("3")) ncpu=3 ;
  if (oopt.Contains("4")) ncpu=4 ;
  if (oopt.Contains("5")) ncpu=5 ;
  if (oopt.Contains("6")) ncpu=6 ;
  if (oopt.Contains("7")) ncpu=7 ;
  if (oopt.Contains("8")) ncpu=8 ;
  if (oopt.Contains("9")) ncpu=9 ;

  // Construct NLL
  RooNLLVar nll("nll","-log(likelihood)",*this,data,projDeps,extended,fitRange,ncpu) ;
  
  // Minimize NLL
  RooMinuit m(nll) ;
  if(blindfit)
    m.setPrintLevel(-1);

  if (cOpt) m.optimizeConst(1) ;
  m.fit(fopt) ;
  
  // Optionally return fit result
  if (saveRes) {
    return m.save() ;
  } else {
    return 0 ;
  }
}



void RooAbsPdf::printToStream(ostream& os, PrintOption opt, TString indent) const
{
  // Print info about this object to the specified stream. In addition to the info
  // from RooAbsArg::printToStream() we add:
  //
  //     Shape : value, units, plot range
  //   Verbose : default binning and print label

  if (opt == OneLine) { 
    RooAbsArg::printToStream(os,opt,indent);
  }

  if (opt == Standard) {
    os << ClassName() << "::" << GetName() << "(" ;
    
    RooArgSet* paramList = getParameters((RooArgSet*)0) ;
    TIterator* pIter = paramList->createIterator() ;

    Bool_t first=kTRUE ;
    RooAbsArg* var ;
    while((var=(RooAbsArg*)pIter->Next())) {
      if (!first) {
	os << "," ;
      } else {
	first=kFALSE ;
      }
      os << var->GetName() ;
    }
    //os << ") = " << getVal(_lastNormSet) << endl ;
    os << ") = " << getVal(0) << endl ;

    delete pIter ;
    delete paramList ;
  }

  if(opt >= Verbose) {
    RooAbsArg::printToStream(os,opt,indent);
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

RooAbsGenContext* RooAbsPdf::genContext(const RooArgSet &vars, const RooDataSet *prototype, 
					const RooArgSet* auxProto, Bool_t verbose) const 
{
  return new RooGenContext(*this,vars,prototype,auxProto,verbose) ;
}


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
  Int_t  nEvents = pc.getInt("nEvents") ;
  Bool_t verbose = pc.getInt("verbose") ;
  Bool_t randProto = pc.getInt("randProto") ;
  Bool_t resampleProto = pc.getInt("resampleProto") ;
  Bool_t extended = pc.getInt("extended") ;

  if (extended) {
    nEvents = RooRandom::randomGenerator()->Poisson(nEvents==0?expectedEvents(&whatVars):nEvents) ;
  }

  if (extended && protoData && !randProto) {
    cout << "RooAbsPdf::generate: WARNING Using generator option Extended() (Poisson distribution of #events) together " << endl
	 << "                     with a prototype dataset implies incomplete sampling or oversampling of proto data." << endl
	 << "                     Set randomize flag in ProtoData() option to randomize prototype dataset order and thus" << endl
         << "                     to randomize the set of over/undersampled prototype events for each generation cycle." << endl ;
  }


  // Forward to appropiate implementation
  if (protoData) {
    return generate(whatVars,*protoData,nEvents,verbose,randProto,resampleProto) ;
  } else {
    return generate(whatVars,nEvents,verbose) ;
  }
  
}



RooDataSet *RooAbsPdf::generate(const RooArgSet &whatVars, Int_t nEvents, Bool_t verbose) const {
  // Generate a new dataset containing the specified variables with
  // events sampled from our distribution. Generate the specified
  // number of events or else try to use expectedEvents() if nEvents <= 0.
  // Any variables of this PDF that are not in whatVars will use their
  // current values and be treated as fixed parameters. Returns zero
  // in case of an error. The caller takes ownership of the returned
  // dataset.

  RooDataSet *generated = 0;
  RooAbsGenContext *context= genContext(whatVars,0,0,verbose);
  if(0 != context && context->isValid()) {
    generated= context->generate(nEvents);
  }
  else {
    cout << ClassName() << "::" << GetName() << ":generate: cannot create a valid context" << endl;
  }
  if(0 != context) delete context;
  return generated;
}

RooDataSet *RooAbsPdf::generate(const RooArgSet &whatVars, const RooDataSet &prototype,
				Int_t nEvents, Bool_t verbose, Bool_t randProtoOrder, Bool_t resampleProto) const {
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

  RooDataSet *generated = 0;
  RooAbsGenContext *context= genContext(whatVars,&prototype,0,verbose);

  // Resampling implies reshuffling in the implementation
  if (resampleProto) {
    randProtoOrder=kTRUE ;
  }

  if (randProtoOrder && prototype.numEntries()!=nEvents) {
    cout << "RooAbsPdf::generate (Re)randomizing event order in prototype dataset (Nevt=" << nEvents << ")" << endl ;
    Int_t* newOrder = randomizeProtoOrder(prototype.numEntries(),nEvents,resampleProto) ;
    context->setProtoDataOrder(newOrder) ;
    delete[] newOrder ;
  }

  if(0 != context && context->isValid()) {
    generated= context->generate(nEvents);
  }
  else {
    cout << ClassName() << "::" << GetName() << ":generate: cannot create a valid context" << endl;
  }
  if(0 != context) delete context;
  return generated;
}



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


Int_t RooAbsPdf::getGenerator(const RooArgSet &/*directVars*/, RooArgSet &/*generatedVars*/, Bool_t /*staticInitOK*/) const {
  // Load generatedVars with the subset of directVars that we can generate events for,
  // and return a code that specifies the generator algorithm we will use. A code of
  // zero indicates that we cannot generate any of the directVars (in this case, nothing
  // should be added to generatedVars). Any non-zero codes will be passed to our generateEvent()
  // implementation, but otherwise its value is arbitrary. The default implemetation of
  // this method returns zero. Subclasses will usually implement this method using the
  // matchArgs() methods to advertise the algorithms they provide.


  return 0 ;
}


void RooAbsPdf::initGenerator(Int_t /*code*/) 
{  
  // One-time initialization to setup the generator for the specified code.
}

void RooAbsPdf::generateEvent(Int_t /*code*/) {
  // Generate an event using the algorithm corresponding to the specified code. The
  // meaning of each code is defined by the getGenerator() implementation. The default
  // implementation does nothing.
}



Bool_t RooAbsPdf::isDirectGenSafe(const RooAbsArg& arg) const 
{
  // Check if PDF depends via more than route on given arg

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


  // Sanity checks
  if (plotSanityChecks(frame)) return frame ;

  // Select the pdf-specific commands 
  RooCmdConfig pc(Form("RooAbsPdf::plotOn(%s)",GetName())) ;
  pc.defineDouble("scaleFactor","Normalization",0,1.0) ;
  pc.defineInt("scaleType","Normalization",0,RooAbsPdf::Relative) ;  
  pc.defineObject("compSet","SelectCompSet",0) ;
  pc.defineString("compSpec","SelectCompSpec",0) ;
  pc.defineObject("asymCat","Asymmetry",0) ;
  pc.defineDouble("rangeLo","Range",0,-999.) ;
  pc.defineDouble("rangeHi","Range",1,-999.) ;
  pc.defineString("rangeName","RangeWithName",0,"") ;
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
  Bool_t haveCompSel = (strlen(compSpec)>0 || compSet) ;

  // Remove PDF-only commands from command list
  pc.stripCmdList(cmdList,"SelectCompSet,SelectCompSpec") ;
  
  // Adjust normalization, if so requested
  if (asymCat) {
    return  RooAbsReal::plotOn(frame,cmdList) ;
  }

  // More sanity checks
  Double_t nExpected(1) ;
  if (stype==RelativeExpected) {
    if (!canBeExtended()) {
      cout << "RooAbsPdf::plotOn(" << GetName() 
	   << "): ERROR the 'Expected' scale option can only be used on extendable PDFs" << endl ;
      return frame ;
    }
    nExpected = expectedEvents(frame->getNormVars()) ;
  }
  
  if (stype != Raw) {    

    if (frame->getFitRangeNEvt() && stype==Relative) {

      Bool_t hasCustomRange(kFALSE), adjustNorm(kFALSE) ;
      Double_t rangeLo(0), rangeHi(0) ;
      // Retrieve plot range to be able to adjust normalization to data
      if (pc.hasProcessed("Range")) {
	rangeLo = pc.getDouble("rangeLo") ;
	rangeHi = pc.getDouble("rangeHi") ;
	adjustNorm = pc.getInt("rangeAdjustNorm") ;
	hasCustomRange = kTRUE ;
      } else if (pc.hasProcessed("RangeWithName")) {    
	rangeLo = frame->getPlotVar()->getMin(pc.getString("rangeName",0,kTRUE)) ;
	rangeHi = frame->getPlotVar()->getMax(pc.getString("rangeName",0,kTRUE)) ;
	adjustNorm = pc.getInt("rangeWNAdjustNorm") ;
	hasCustomRange = kTRUE ;
      } else {
	// Use range of last fit, if it was non-default and no other range was specified
	RooArgSet* plotDep = getObservables(*frame->getPlotVar()) ;
	RooRealVar* plotDepVar = (RooRealVar*) plotDep->find(frame->getPlotVar()->GetName()) ;
	if (plotDepVar->hasBinning("fit")) {
	  rangeLo = plotDepVar->getMin("fit") ;
	  rangeHi = plotDepVar->getMax("fit") ;
	  adjustNorm = kTRUE ;
	  hasCustomRange = kTRUE ;
	  cout << "RooAbsPdf::plotOn(" << GetName() << ") INFO: pdf has been fit over restricted range, plotting only fitted "
	       << "part of PDF normalized data in restricted range" << endl ;
	}
	delete plotDep ;
      }
      if (hasCustomRange && adjustNorm) {	
	scaleFactor *= frame->getFitRangeNEvt(rangeLo,rangeHi)/nExpected ;
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
    
    // Discard any non-PDF nodes
    TIterator* iter = branchNodeSet.createIterator() ;
    RooAbsArg* arg ;
    while((arg=(RooAbsArg*)iter->Next())) {
      if (!dynamic_cast<RooAbsPdf*>(arg)) {
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
    cout << "RooAbsPdf::plotOn(" << GetName() << ") directly selected PDF components: " ;
    dirSelNodes->Print("1") ;
    
    // Do indirect selection and activate both
    plotOnCompSelect(dirSelNodes) ;
  }
  
  RooPlot* ret =  RooAbsReal::plotOn(frame,cmdList) ;
  
  // Restore selection status ;
  if (haveCompSel) plotOnCompSelect(0) ;
  
  return ret ;
}



void RooAbsPdf::plotOnCompSelect(RooArgSet* selNodes) const
{
  // Get complete set of tree branch nodes
  RooArgSet branchNodeSet ;
  branchNodeServerList(&branchNodeSet) ;

  // Discard any non-PDF nodes
  TIterator* iter = branchNodeSet.createIterator() ;
  RooAbsArg* arg ;
  while((arg=(RooAbsArg*)iter->Next())) {
    if (!dynamic_cast<RooAbsPdf*>(arg)) {
      branchNodeSet.remove(*arg) ;
    }
  }

  // If no set is specified, restored all selection bits to kTRUE
  if (!selNodes) {
    // Reset PDF selection bits to kTRUE
    iter->Reset() ;
    while((arg=(RooAbsArg*)iter->Next())) {
      ((RooAbsPdf*)arg)->selectComp(kTRUE) ;
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
  cout << "RooAbsPdf::plotOn(" << GetName() << ") indirectly selected PDF components: " ;
  tmp.Print("1") ;

  // Set PDF selection bits according to selNodes
  iter->Reset() ;
  while((arg=(RooAbsArg*)iter->Next())) {
    Bool_t select = selNodes->find(arg->GetName()) ? kTRUE : kFALSE ;
    ((RooAbsPdf*)arg)->selectComp(select) ;
  }
  
  delete iter ;
} 




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
      cout << "RooAbsPdf::plotOn(" << GetName() 
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



RooPlot* RooAbsPdf::plotCompOn(RooPlot *frame, const RooArgSet& compSet, Option_t* drawOptions,
			       Double_t scaleFactor, ScaleType stype, const RooAbsData* projData, 
			       const RooArgSet* projSet) const 
{
  // THIS FUNCTION IS OBSOLETE AND ONLY RETAINED FOR BACKWARD COMPATIBILITY. 
  // PLEASE USE plotOn(frame,Componenents(...),...)
  //
  // Plot only the PDF components listed in 'compSet' of this PDF on 'frame'. 
  // See RooAbsReal::plotOn() for a description of the remaining arguments and other features

  // Sanity checks
  if (plotSanityChecks(frame)) return frame ;

  // Get complete set of tree branch nodes
  RooArgSet branchNodeSet ;
  branchNodeServerList(&branchNodeSet) ;

  // Discard any non-PDF nodes
  TIterator* iter = branchNodeSet.createIterator() ;
  RooAbsArg* arg ;
  while((arg=(RooAbsArg*)iter->Next())) {
    if (!dynamic_cast<RooAbsPdf*>(arg)) {
      branchNodeSet.remove(*arg) ;
    }
  }
  delete iter ;

  // Get list of directly selected nodes
  RooArgSet* selNodes = (RooArgSet*) branchNodeSet.selectCommon(compSet) ;
  cout << "RooAbsPdf::plotCompOn(" << GetName() << ") directly selected PDF components: " ;
  selNodes->Print("1") ;
  
  return plotCompOnEngine(frame,selNodes,drawOptions,scaleFactor,stype,projData,projSet) ;
}




RooPlot* RooAbsPdf::plotCompOn(RooPlot *frame, const char* compNameList, Option_t* drawOptions,
			       Double_t scaleFactor, ScaleType stype, const RooAbsData* projData, 
			       const RooArgSet* projSet) const 
{
  // THIS FUNCTION IS OBSOLETE AND ONLY RETAINED FOR BACKWARD COMPATIBILITY. 
  // PLEASE USE plotOn(frame,Componenents(...),...)
  //
  // Plot only the PDF components listed in 'compSet' of this PDF on 'frame'. 
  // See RooAbsReal::plotOn() for a description of the remaining arguments and other features

  // Sanity checks
  if (plotSanityChecks(frame)) return frame ;

  // Get complete set of tree branch nodes
  RooArgSet branchNodeSet ;
  branchNodeServerList(&branchNodeSet) ;

  // Discard any non-PDF nodes
  TIterator* iter = branchNodeSet.createIterator() ;
  RooAbsArg* arg ;
  while((arg=(RooAbsArg*)iter->Next())) {
    if (!dynamic_cast<RooAbsPdf*>(arg)) {
      branchNodeSet.remove(*arg) ;
    }
  }
  delete iter ;

  // Get list of directly selected nodes
  RooArgSet* selNodes = (RooArgSet*) branchNodeSet.selectByName(compNameList) ;
  cout << "RooAbsPdf::plotCompOn(" << GetName() << ") directly selected PDF components: " ;
  selNodes->Print("1") ;
  
  return plotCompOnEngine(frame,selNodes,drawOptions,scaleFactor,stype,projData,projSet) ;
}


RooPlot* RooAbsPdf::plotCompOnEngine(RooPlot *frame, RooArgSet* selNodes, Option_t* drawOptions,
			       Double_t scaleFactor, ScaleType stype, const RooAbsData* projData, 
			       const RooArgSet* projSet) const 
{
  // Get complete set of tree branch nodes
  RooArgSet branchNodeSet ;
  branchNodeServerList(&branchNodeSet) ;

  // Discard any non-PDF nodes
  TIterator* iter = branchNodeSet.createIterator() ;
  RooAbsArg* arg ;
  while((arg=(RooAbsArg*)iter->Next())) {
    if (!dynamic_cast<RooAbsPdf*>(arg)) {
      branchNodeSet.remove(*arg) ;
    }
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
//   cout << "RooAbsPdf::plotCompOn(" << GetName() << ") indirectly selected PDF components: " ;
//   tmp.Print("1") ;

  // Set PDF selection bits according to selNodes
  iter->Reset() ;
  while((arg=(RooAbsArg*)iter->Next())) {
    Bool_t select = selNodes->find(arg->GetName()) ? kTRUE : kFALSE ;
    ((RooAbsPdf*)arg)->selectComp(select) ;
  }
 
  // Plot function in selected state
  PlotOpt o ;
  o.drawOptions = drawOptions ;
  o.scaleFactor = scaleFactor ;
  o.stype = stype ;
  o.projData = projData ;
  o.projSet = projSet ;
  frame = plotOn(frame,0) ;

  // Reset PDF selection bits to kTRUE
  iter->Reset() ;
  while((arg=(RooAbsArg*)iter->Next())) {
    ((RooAbsPdf*)arg)->selectComp(kTRUE) ;
  }

  delete selNodes ;
  delete iter ;
  return frame ;
}




RooPlot* RooAbsPdf::plotCompSliceOn(RooPlot *frame, const char* compNameList, const RooArgSet& sliceSet,
				    Option_t* drawOptions, Double_t scaleFactor, ScaleType stype, 
				    const RooAbsData* projData) const 
{
  // THIS FUNCTION IS OBSOLETE AND ONLY RETAINED FOR BACKWARD COMPATIBILITY. 
  // PLEASE USE plotOn(frame,Componenents(...),Slice(...),...)
  //
  // Plot ourselves on given frame, as done in plotOn(), except that the variables 
  // listed in 'sliceSet' are taken out from the default list of projected dimensions created
  // by plotOn().

  RooArgSet projectedVars ;
  makeProjectionSet(frame->getPlotVar(),frame->getNormVars(),projectedVars,kTRUE) ;
  
  // Take out the sliced variables
  TIterator* iter = sliceSet.createIterator() ;
  RooAbsArg* sliceArg ;
  while((sliceArg=(RooAbsArg*)iter->Next())) {
    RooAbsArg* arg = projectedVars.find(sliceArg->GetName()) ;
    if (arg) {
      projectedVars.remove(*arg) ;
    } else {
      cout << "RooAddPdf::plotCompSliceOn(" << GetName() << ") slice variable " 
	   << sliceArg->GetName() << " was not projected anyway" << endl ;
    }
  }
  delete iter ;

  return plotCompOn(frame,compNameList,drawOptions,scaleFactor,stype,projData,&projectedVars) ;
}





RooPlot* RooAbsPdf::plotCompSliceOn(RooPlot *frame, const RooArgSet& compSet, const RooArgSet& sliceSet,
				    Option_t* drawOptions, Double_t scaleFactor, ScaleType stype, 
				    const RooAbsData* projData) const 
{
  // THIS FUNCTION IS OBSOLETE AND ONLY RETAINED FOR BACKWARD COMPATIBILITY. 
  // PLEASE USE plotOn(frame,Componenents(...),Slice(...),...)
  //
  // Plot ourselves on given frame, as done in plotOn(), except that the variables 
  // listed in 'sliceSet' are taken out from the default list of projected dimensions created
  // by plotOn().

  RooArgSet projectedVars ;
  makeProjectionSet(frame->getPlotVar(),frame->getNormVars(),projectedVars,kTRUE) ;
  
  // Take out the sliced variables
  TIterator* iter = sliceSet.createIterator() ;
  RooAbsArg* sliceArg ;
  while((sliceArg=(RooAbsArg*)iter->Next())) {
    RooAbsArg* arg = projectedVars.find(sliceArg->GetName()) ;
    if (arg) {
      projectedVars.remove(*arg) ;
    } else {
      cout << "RooAddPdf::plotCompSliceOn(" << GetName() << ") slice variable " 
	   << sliceArg->GetName() << " was not projected anyway" << endl ;
    }
  }
  delete iter ;

  return plotCompOn(frame,compSet,drawOptions,scaleFactor,stype,projData,&projectedVars) ;
}


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
  //   ShowConstant(Bool_t flag)          -- Also display constant parameters
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
  pc.defineDouble("xmin","Layout",0,0.65) ;
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

RooPlot* RooAbsPdf::paramOn(RooPlot* frame, const RooArgSet& params, Bool_t showConstants, const char *label,
			    Int_t sigDigits, Option_t *options, Double_t xmin,
			    Double_t xmax ,Double_t ymax, const RooCmdArg* formatCmd) 
{
  // Add a text box with the current parameter values and their errors to the frame.
  // Dependents of this PDF appearing in the 'data' dataset will be omitted.
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
  box->SetFillColor(0);
  box->SetBorderSize(1);
  box->SetTextAlign(12);
  box->SetTextSize(0.04F);
  box->SetFillStyle(1001);
  box->SetFillColor(0);
  TText *text = 0;
//char buffer[512];
  index= nPar;
  pIter->Reset() ;
  while((var=(RooRealVar*)pIter->Next())) {
    if(var->isConstant() && !showConstants) continue;
    
    TString *formatted= options ? var->format(sigDigits, options) : var->format(*formatCmd) ;
    text= box->AddText(formatted->Data());
    delete formatted;
  }
  // add the optional label if specified
  if(showLabel) text= box->AddText(label);

  // Add box to frame 
  frame->addObject(box) ;

  delete pIter ;
  return frame ;
}



void RooAbsPdf::fixAddCoefNormalization(const RooArgSet& addNormSet) 
{
  RooArgSet* compSet = getComponents() ;
  TIterator* iter = compSet->createIterator() ;
  RooAbsArg* arg ;
  while((arg=(RooAbsArg*)iter->Next())) {
    RooAbsPdf* pdf = dynamic_cast<RooAbsPdf*>(arg) ;
    if (pdf) {
      if (addNormSet.getSize()>0) {
	pdf->selectNormalization(&addNormSet,kTRUE) ;
      } else {
	pdf->selectNormalization(0,kTRUE) ;
      }
    } 
  }
  delete iter ;
  delete compSet ;  
}

void RooAbsPdf::fixAddCoefRange(const char* rangeName) 
{
  RooArgSet* compSet = getComponents() ;
  TIterator* iter = compSet->createIterator() ;
  RooAbsArg* arg ;
  while((arg=(RooAbsArg*)iter->Next())) {
    RooAbsPdf* pdf = dynamic_cast<RooAbsPdf*>(arg) ;
    if (pdf) {
      pdf->selectNormalizationRange(rangeName,kTRUE) ;
    }
  }
  delete iter ;
  delete compSet ;    
}


RooPlot* RooAbsPdf::plotNLLOn(RooPlot* frame, RooDataSet* data, Bool_t extended, const RooArgSet& /*projDeps*/,
			      Option_t* /*drawOptions*/, Double_t prec, Bool_t fixMinToZero) {
  
  RooNLLVar nll("nll","-log(L)",*this,*data,extended) ;
  if (fixMinToZero) {
    nll.plotOn(frame,RooFit::DrawOption("L"),RooFit::Precision(prec),RooFit::ShiftToZero()) ;
  } else {
    nll.plotOn(frame,RooFit::DrawOption("L"),RooFit::Precision(prec)) ;
  }

  return frame ;
}


Bool_t RooAbsPdf::redirectServersHook(const RooAbsCollection& newServerList, 
				      Bool_t mustReplaceAll, Bool_t nameChange, Bool_t /*isRecursive*/) 
{
  Bool_t ret(kFALSE) ;  

  Int_t i ;
  for (i=0 ; i<_normMgr.cacheSize() ; i++) {
    RooAbsArg* norm = _normMgr.getNormByIndex(i) ;
    ret |= norm->recursiveRedirectServers(newServerList,mustReplaceAll,nameChange) ;
  }
  return ret ;
}


Double_t RooAbsPdf::expectedEvents(const RooArgSet*) const 
{ 
  return 0 ; 
} 
