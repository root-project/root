/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsPdf.cc,v 1.69 2002/06/12 23:53:25 verkerke Exp $
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


#include <iostream.h>
#include <math.h>
#include "TObjString.h"
#include "TPaveText.h"
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
#include "RooFitCore/RooCurve.hh"
#include "RooFitCore/RooNLLBinding.hh"
#include "RooFitCore/RooIntegratorConfig.hh"

ClassImp(RooAbsPdf) 
;


Int_t RooAbsPdf::_verboseEval = 0;
Bool_t RooAbsPdf::_globalSelectComp = kFALSE ;
RooIntegratorConfig* RooAbsPdf::_defaultNormIntConfig(0) ;


RooAbsPdf::RooAbsPdf(const char *name, const char *title) : 
  RooAbsReal(name,title), _norm(0), _lastNormSet(0), _selectComp(kTRUE), _specNormIntConfig(0)
{
  // Constructor with name and title only
  resetErrorCounters() ;
  setTraceCounter(0) ;
}


RooAbsPdf::RooAbsPdf(const char *name, const char *title, 
		     Double_t plotMin, Double_t plotMax) :
  RooAbsReal(name,title,plotMin,plotMax), _norm(0), _lastNormSet(0), _selectComp(kTRUE), _specNormIntConfig(0)
{
  // Constructor with name, title, and plot range
  resetErrorCounters() ;
  setTraceCounter(0) ;
}



RooAbsPdf::RooAbsPdf(const RooAbsPdf& other, const char* name) : 
  RooAbsReal(other,name), _norm(0), _lastNormSet(0), _selectComp(other._selectComp)
{
  // Copy constructor
  resetErrorCounters() ;
  setTraceCounter(other._traceCount) ;

  if (other._specNormIntConfig) {
    _specNormIntConfig = new RooIntegratorConfig(*other._specNormIntConfig) ;
  } else {
    _specNormIntConfig = 0 ;
  }
}


RooAbsPdf::~RooAbsPdf()
{
  // Destructor
  if (_norm) delete _norm ;
  if (_specNormIntConfig) delete _specNormIntConfig ;
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
    if (_verboseEval>1) cout << IsA()->GetName() << "::getVal(" << GetName() << "): value = " << val << " (unnormalized)" << endl ;

    if (error) return 0 ;
    return val ;
  }

  // Process change in last data set used
  Bool_t nsetChanged = (nset != _lastNormSet) ;
  if (nsetChanged) syncNormalization(nset) ;

  // Return value of object. Calculated if dirty, otherwise cached value is returned.
  if ((isValueDirty() || _norm->isValueDirty() || nsetChanged) && operMode()!=AClean) {

    // Evaluate numerator
    Double_t rawVal = evaluate() ;
    Bool_t error = traceEvalPdf(rawVal) ; // Error checking and printing

    // Evaluate denominator
    Double_t normVal(_norm->getVal()) ;
    if (normVal==0.) error=kTRUE ;

    _value = error ? 0 : (rawVal / normVal) ;

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


Double_t RooAbsPdf::analyticalIntegralWN(Int_t code, const RooArgSet* normSet) const
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
    return analyticalIntegral(code) / getNorm(normSet) ;
  } else {
    return analyticalIntegral(code) ;
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

  syncNormalization(nset) ;
  if (_verboseEval>1) cout << IsA()->GetName() << "::getNorm(" << GetName() << "): norm(" << _norm << ") = " << _norm->getVal() << endl ;

  Double_t ret = _norm->getVal() ;
  if (ret==0.) {
    cout << "RooAbsPdf::getNorm(" << GetName() << ":: WARNING normalization is zero, nset = " ; 
    nset->Print("1") ;
    _norm->Print("v") ;
  }

  return ret ;
}




void RooAbsPdf::syncNormalization(const RooArgSet* nset) const
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


  // Check if data sets are identical
  if (nset == _lastNormSet) return ;
  if (_verboseEval>0) cout << GetName() << ":updating lastNormSet from " << _lastNormSet << " to " << nset << endl ;
  RooArgSet* lastNormSet = _lastNormSet;
  _lastNormSet = (RooArgSet*) nset ;
  _lastNameSet.refill(*nset) ;

  // Update dataset pointers of proxies
  ((RooAbsPdf*) this)->setProxyNormSet(nset) ;
  
  // Allow optional post-processing
  Bool_t fullNorm = syncNormalizationPreHook(_norm,nset) ;
  RooArgSet* depList ;
  if (fullNorm) {
    depList = ((RooArgSet*)nset) ;
  } else {
    depList = getDependents(nset) ;

    // Account for LValues in normalization here
    RooArgSet bList ;
    branchNodeServerList(&bList) ;
    TIterator* dIter = nset->createIterator() ;
    RooAbsArg* dep ;
    while (dep=(RooAbsArg*)dIter->Next()) {
      RooAbsArg* tmp = bList.find(dep->GetName()) ;
      if (dynamic_cast<RooAbsLValue*>(tmp)) {
	depList->add(*tmp) ;
      }
    }
    delete dIter ;
    
  }

  if (_verboseEval>0) {
    if (!selfNormalized()) {
      cout << IsA()->GetName() << "::syncNormalization(" << GetName() 
	   << ") recreating normalization integral " 
	   << lastNormSet << " -> " << nset << "=" ;
      if (depList) depList->printToStream(cout,OneLine) ; else cout << "<none>" << endl ;
    } else {
      cout << IsA()->GetName() << "::syncNormalization(" << GetName() << ") selfNormalized, creating unit norm" << endl;
    }
  }


  // Destroy old normalization & create new
  if (_norm) delete _norm ;
    
  TString nname(GetName()) ; nname.Append("Norm") ;
  if (selfNormalized() || !dependsOn(*depList)) {
    TString ntitle(GetTitle()) ; ntitle.Append(" Unit Normalization") ;
    _norm = new RooRealVar(nname.Data(),ntitle.Data(),1) ;
  } else {
    TString ntitle(GetTitle()) ; ntitle.Append(" Integral") ;
    _norm = new RooRealIntegral(nname.Data(),ntitle.Data(),*this,*depList,0,getNormIntConfig()) ;
  }

  // Allow optional post-processing
  syncNormalizationPostHook(_norm,nset) ;
 
  if (!fullNorm) delete depList ;
}



const RooIntegratorConfig* RooAbsPdf::getNormIntConfig() const 
{
  const RooIntegratorConfig* config = getSpecialNormIntConfig() ;
  if (config) return config ;
  return getDefaultNormIntConfig() ;
}


const RooIntegratorConfig* RooAbsPdf::getDefaultNormIntConfig() const 
{
  if (!_defaultNormIntConfig) {
    _defaultNormIntConfig = new RooIntegratorConfig ;
  }
  return _defaultNormIntConfig ;
}


const RooIntegratorConfig* RooAbsPdf::getSpecialNormIntConfig() const 
{
  return _specNormIntConfig ;
}


void RooAbsPdf::setDefaultNormIntConfig(const RooIntegratorConfig& config) 
{
  if (_defaultNormIntConfig) {
    delete _defaultNormIntConfig ;
  }
  _defaultNormIntConfig = new RooIntegratorConfig(config) ;
}


void RooAbsPdf::setNormIntConfig(const RooIntegratorConfig& config) 
{
  if (_specNormIntConfig) {
    delete _specNormIntConfig ;
  }
  _specNormIntConfig = new RooIntegratorConfig(config) ;  
}


void RooAbsPdf::setNormIntConfig() 
{
  if (_specNormIntConfig) {
    delete _specNormIntConfig ;
  }
  _specNormIntConfig = 0 ;
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
    while(arg=(RooAbsArg*)iter->Next()) {
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
      RooArgSet* depends = getDependents(nset) ;	 
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



Double_t RooAbsPdf::extendedTerm(UInt_t observed) const 
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


RooFitResult* RooAbsPdf::fitTo(RooAbsData& data, const RooArgSet& projDeps, Option_t *fitOpt, Option_t *optOpt) 
{
  RooFitContext* cx = fitContext(data,&projDeps) ;
  RooFitResult* result =  cx->fit(fitOpt,optOpt) ;
  delete cx ;
  return result ;
}



RooFitResult* RooAbsPdf::fitTo(RooAbsData& data, Option_t *fitOpt, Option_t *optOpt) 
{
  // Fit this PDF to given data set
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
  //  "p" = Cache and precalculate components of PDF that exclusively depend on constant parameters
  //  "d" = Trim unused elements from data set
  //  "c" = Streamline dirty state propagation
  //  "s" = Calculate and update likelihood components of RooSimultaneous fit independently for each component
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

  RooFitContext* cx = fitContext(data) ;
  RooFitResult* result =  cx->fit(fitOpt,optOpt) ;
  delete cx ;
  return result ;
}



RooFitContext* RooAbsPdf::fitContext(const RooAbsData& dset, const RooArgSet* projDeps) const 
{
  return new RooFitContext(&dset,this,kTRUE,kTRUE,kTRUE,projDeps) ;
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
    while(var=(RooAbsArg*)pIter->Next()) {
      if (!first) {
	os << "," ;
      } else {
	first=kFALSE ;
      }
      os << var->GetName() ;
    }
    os << ") = " << getVal(_lastNormSet) << endl ;

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

RooAbsGenContext* RooAbsPdf::genContext(const RooArgSet &vars, 
					const RooDataSet *prototype, Bool_t verbose) const 
{
  return new RooGenContext(*this,vars,prototype,verbose) ;
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
  RooAbsGenContext *context= genContext(whatVars,0,verbose);
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
				Int_t nEvents, Bool_t verbose) const {
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
  RooAbsGenContext *context= genContext(whatVars,&prototype,verbose);
  if(0 != context && context->isValid()) {
    generated= context->generate(nEvents);
  }
  else {
    cout << ClassName() << "::" << GetName() << ":generate: cannot create a valid context" << endl;
  }
  if(0 != context) delete context;
  return generated;
}

Int_t RooAbsPdf::getGenerator(const RooArgSet &directVars, RooArgSet &generatedVars, Bool_t staticInitOK) const {
  // Load generatedVars with the subset of directVars that we can generate events for,
  // and return a code that specifies the generator algorithm we will use. A code of
  // zero indicates that we cannot generate any of the directVars (in this case, nothing
  // should be added to generatedVars). Any non-zero codes will be passed to our generateEvent()
  // implementation, but otherwise its value is arbitrary. The default implemetation of
  // this method returns zero. Subclasses will usually implement this method using the
  // matchArgs() methods to advertise the algorithms they provide.

  return 0;
}

void RooAbsPdf::initGenerator(Int_t code) 
{  
  // One-time initialization to setup the generator for the specified code.
}

void RooAbsPdf::generateEvent(Int_t code) {
  // Generate an event using the algorithm corresponding to the specified code. The
  // meaning of each code is defined by the getGenerator() implementation. The default
  // implementation does nothing.
}



Bool_t RooAbsPdf::isDirectGenSafe(const RooAbsArg& arg) const 
{
  // Check if PDF depends via more than route on given arg
  TIterator* sIter = serverIterator() ;
  const RooAbsArg *server = 0;
  while(server=(const RooAbsArg*)sIter->Next()) {
    if(server == &arg) continue;
    if(server->dependsOn(arg)) {
      delete sIter ;
      return kFALSE ;
    }
  }
  return kTRUE ;
}



RooPlot* RooAbsPdf::plotOn(RooPlot *frame, Option_t* drawOptions, 
			   Double_t scaleFactor, ScaleType stype, 
			   const RooAbsData* projData, const RooArgSet* projSet) const
{
  // Plot outself on 'frame'. In addition to features detailed in  RooAbsReal::plotOn(),
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
  if (stype==RelativeExpected) {
    if (!canBeExtended()) {
      cout << "RooAbsPdf::plotOn(" << GetName() 
	   << "): ERROR the 'Expected' scale option can only be used on extendable PDFs" << endl ;
      return frame ;
    }
    nExpected = expectedEvents() ;
  }

  // Adjust normalization, if so requested
  if (stype != Raw) {    
    
    if (frame->getFitRangeNEvt() && stype==Relative) {
      scaleFactor *= frame->getFitRangeNEvt()/nExpected ;
    } else if (stype==RelativeExpected) {
      scaleFactor *= nExpected ;
    } else if (stype==NumEvent) {
      scaleFactor /= nExpected ;
    }
    scaleFactor *= frame->getFitRangeBinW() ;
  }
  frame->updateNormVars(*frame->getPlotVar()) ;

  return RooAbsReal::plotOn(frame,drawOptions,scaleFactor,Raw,projData,projSet) ;
}




RooPlot* RooAbsPdf::plotCompOn(RooPlot *frame, const RooArgSet& compSet, Option_t* drawOptions,
			       Double_t scaleFactor, ScaleType stype, const RooAbsData* projData, 
			       const RooArgSet* projSet) const 
{
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
  while(arg=(RooAbsArg*)iter->Next()) {
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
  while(arg=(RooAbsArg*)iter->Next()) {
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
  while(arg=(RooAbsArg*)iter->Next()) {
    if (!dynamic_cast<RooAbsPdf*>(arg)) {
      branchNodeSet.remove(*arg) ;
    }
  }

  // Add all nodes below selected nodes
  iter->Reset() ;
  TIterator* sIter = selNodes->createIterator() ;
  RooArgSet tmp ;
  while(arg=(RooAbsArg*)iter->Next()) {
    sIter->Reset() ;
    RooAbsArg* selNode ;
    while(selNode=(RooAbsArg*)sIter->Next()) {
      if (selNode->dependsOn(*arg)) {
	tmp.add(*arg,kTRUE) ;
      }      
    }      
  }
  delete sIter ;

  // Add all nodes that depend on selected nodes
  iter->Reset() ;
  while(arg=(RooAbsArg*)iter->Next()) {
    if (arg->dependsOn(*selNodes)) {
      tmp.add(*arg,kTRUE) ;
    }
  }

  tmp.remove(*selNodes,kTRUE) ;
  tmp.remove(*this) ;
  selNodes->add(tmp) ;
  cout << "RooAbsPdf::plotCompOn(" << GetName() << ") indirectly selected PDF components: " ;
  tmp.Print("1") ;

  // Set PDF selection bits according to selNodes
  iter->Reset() ;
  while(arg=(RooAbsArg*)iter->Next()) {
    Bool_t select = selNodes->find(arg->GetName()) ? kTRUE : kFALSE ;
    ((RooAbsPdf*)arg)->selectComp(select) ;
  }
 
  // Plot function in selected state
  frame = plotOn(frame,drawOptions,scaleFactor,stype,projData,projSet) ;

  // Reset PDF selection bits to kTRUE
  iter->Reset() ;
  while(arg=(RooAbsArg*)iter->Next()) {
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
  // Plot ourselves on given frame, as done in plotOn(), except that the variables 
  // listed in 'sliceSet' are taken out from the default list of projected dimensions created
  // by plotOn().

  RooArgSet projectedVars ;
  makeProjectionSet(frame->getPlotVar(),frame->getNormVars(),projectedVars,kTRUE) ;
  
  // Take out the sliced variables
  TIterator* iter = sliceSet.createIterator() ;
  RooAbsArg* sliceArg ;
  while(sliceArg=(RooAbsArg*)iter->Next()) {
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
  // Plot ourselves on given frame, as done in plotOn(), except that the variables 
  // listed in 'sliceSet' are taken out from the default list of projected dimensions created
  // by plotOn().

  RooArgSet projectedVars ;
  makeProjectionSet(frame->getPlotVar(),frame->getNormVars(),projectedVars,kTRUE) ;
  
  // Take out the sliced variables
  TIterator* iter = sliceSet.createIterator() ;
  RooAbsArg* sliceArg ;
  while(sliceArg=(RooAbsArg*)iter->Next()) {
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




RooPlot* RooAbsPdf::plotNLLOn(RooPlot* frame, RooDataSet* data, Bool_t extended, const RooArgSet& projDeps,
			      Option_t* drawOptions, Double_t prec, Bool_t fixMinToZero) 
{
  // Plot the negative log likelihood of ourself when applied on the given data set,
  // as function of the plot variable of the frame.

  // Sanity checks on frame
  if (plotSanityChecks(frame)) return frame ;
  RooAbsReal* plotVar = frame->getPlotVar() ;

  // Plot variable may not be a dependent 
  RooArgSet* depSet = getDependents(data) ;
  if (depSet->find(plotVar->GetName())) {
    cout << "RooAbsPdf::plotNLLOn(" << GetName() << ") ERROR: plot variable " 
	 << plotVar->GetName() << " cannot not be a dependent" << endl ;
    delete depSet ;
    return frame ;
  }

  // Clone for plotting
  RooArgSet *cloneList = (RooArgSet*) RooArgSet(*this).snapshot(kTRUE) ;
  if (!cloneList) {
    cout << "RooAbsPdf::plotNLLOn(" << GetName() << ") Couldn't deep-clone self, abort," << endl ;
    return frame ;
  }
  RooAbsPdf* clone     = (RooAbsPdf*) cloneList->find(GetName()) ;
  RooAbsRealLValue* cloneVar = (RooAbsRealLValue*) cloneList->find(plotVar->GetName()) ;

  // Create NLL binding object
  RooNLLBinding nllVar(*clone,*data,*cloneVar,extended,projDeps) ;

  // Construct name and title of curve
  TString name("curve_NLL[") ;
  name.Append(GetName()) ;
  name.Append(",") ;
  name.Append(data->GetName()) ;
  name.Append("]") ;

  TString title("NLL of PDF '") ;
  title.Append(GetTitle()) ;
  title.Append("' with dataset '") ;
  title.Append(data->GetTitle()) ;
  title.Append("'") ;

  // Create curve for NLL binding object
  RooCurve* curve= new RooCurve(name, title, nllVar, 
				frame->GetXaxis()->GetXmin(), frame->GetXaxis()->GetXmax(),frame->GetNbinsX(),
				prec,prec,fixMinToZero) ;

  // Add this new curve to the specified plot frame
  frame->addPlotable(curve, drawOptions);

  delete cloneList ;
  return frame ;
}



RooPlot* RooAbsPdf::paramOn(RooPlot* frame, const RooAbsData* data, const char *label,
			    Int_t sigDigits, Option_t *options, Double_t xmin,
			    Double_t xmax ,Double_t ymax) 
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
  Bool_t showConstants= opts.Contains("c");
  Bool_t showLabel= (label != 0 && strlen(label) > 0);
  
  // calculate the box's size, adjusting for constant parameters
  RooArgSet* params = getParameters(data) ;
  TIterator* pIter = params->createIterator() ;

  Int_t nPar= params->getSize();
  Double_t ymin(ymax), dy(0.06);
  Int_t index(nPar);
  RooRealVar *var = 0;
  while(var=(RooRealVar*)pIter->Next()) {
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
  while(var=(RooRealVar*)pIter->Next()) {
    if(var->isConstant() && !showConstants) continue;
    TString *formatted= var->format(sigDigits, opts.Data());
    text= box->AddText(formatted->Data());
    delete formatted;
  }
  // add the optional label if specified
  if(showLabel) text= box->AddText(label);

  // Add box to frame 
  frame->addObject(box) ;

  delete pIter ;
  delete params ;
  return frame ;
}



TH2F* RooAbsPdf::plotNLLContours(RooAbsData& data, RooRealVar& var1, RooRealVar& var2, Double_t n1, Double_t n2, Double_t n3) 
{
  // Make a one or more 2D contour lines at n1,n2 and n3 sigma from the minimum in the 'var1' vs 'var2' plane.
  //
  // plotNLLContours call MIGRAD first to verify convergence and the existence of a NLL minimum, then calls TMinuit::Contour()
  // for each sigma level. At the end of the calculations the contour lines are plotted on the current canvas pad on
  // top of the returned histogram. The returned TH2F does not contain the contour lines, so please save the canvas to
  // store the results.

  RooFitContext* cx = fitContext(data) ;
  TH2F* ret = cx->plotNLLContours(var1,var2,n1,n2,n3) ;
  delete cx ;
  return ret ;
}


void RooAbsPdf::fixAddCoefNormalization(const RooArgSet& addNormSet) 
{
  RooArgSet* compSet = getComponents() ;
  TIterator* iter = compSet->createIterator() ;
  RooAbsArg* arg ;
  while(arg=(RooAbsArg*)iter->Next()) {
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
