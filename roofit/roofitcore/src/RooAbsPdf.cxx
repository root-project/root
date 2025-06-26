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
/** \class RooAbsPdf
    \ingroup Roofitcore
    \brief Abstract interface for all probability density functions.

## RooAbsPdf, the base class of all PDFs

RooAbsPdf is the base class for all probability density
functions (PDFs). The class provides hybrid analytical/numerical
normalization for its implementations, error tracing, and a Monte Carlo
generator interface.

### A Minimal PDF Implementation

A minimal implementation of a PDF class derived from RooAbsPdf
should override the `evaluate()` function. This function should
return the PDF's value (which does not need to be normalised).


#### Normalization/Integration

Although the normalization of a PDF is an integral part of a
probability density function, normalization is treated separately
in RooAbsPdf. The reason is that a RooAbsPdf object is more than a
PDF: it can be a building block for a more complex composite PDF
if any of its variables are functions instead of variables. In
such cases, the normalization of the composite PDF may not simply be
integral over the dependents of the top-level PDF: these are
functions with potentially non-trivial Jacobian terms themselves.
\note Therefore, no explicit attempt should be made to normalize the
function output in evaluate(). In particular, normalisation constants
can be omitted to speed up the function evaluations, and included later
in the integration of the PDF (see below), which is rarely called in
comparison to the `evaluate()` function.

In addition, RooAbsPdf objects do not have a static concept of what
variables are parameters, and what variables are dependents (which
need to be integrated over for a correct PDF normalization).
Instead, the choice of normalization is always specified each time a
normalized value is requested from the PDF via the getVal()
method.

RooAbsPdf manages the entire normalization logic of each PDF with
the help of a RooRealIntegral object, which coordinates the integration
of a given choice of normalization. By default, RooRealIntegral will
perform an entirely numeric integration of all dependents. However,
PDFs can advertise one or more (partial) analytical integrals of
their function, and these will be used by RooRealIntegral, if it
determines that this is safe (i.e., no hidden Jacobian terms,
multiplication with other PDFs that have one or more dependents in
common, etc).

#### Implementing analytical integrals
To implement analytical integrals, two functions must be implemented. First,

```
Int_t getAnalyticalIntegral(const RooArgSet& integSet, RooArgSet& anaIntSet)
```
should return the analytical integrals that are supported. `integSet`
is the set of dependents for which integration is requested. The
function should copy the subset of dependents it can analytically
integrate to `anaIntSet`, and return a unique identification code for
this integration configuration. If no integration can be
performed, zero should be returned. Second,

```
double analyticalIntegral(Int_t code)
```

implements the actual analytical integral(s) advertised by
`getAnalyticalIntegral()`.  This function will only be called with
codes returned by `getAnalyticalIntegral()`, except code zero.

The integration range for each dependent to be integrated can
be obtained from the dependent's proxy functions `min()` and
`max()`. Never call these proxy functions for any proxy not known to
be a dependent via the integration code. Doing so may be
ill-defined, e.g., in case the proxy holds a function, and will
trigger an assert. Integrated category dependents should always be
summed over all of their states.



### Direct generation of observables

Distributions for any PDF can be generated with the accept/reject method,
but for certain PDFs, more efficient methods may be implemented. To
implement direct generation of one or more observables, two
functions need to be implemented, similar to those for analytical
integrals:

```
Int_t getGenerator(const RooArgSet& generateVars, RooArgSet& directVars)
```
and
```
void generateEvent(Int_t code)
```

The first function advertises observables, for which distributions can be generated,
similar to the way analytical integrals are advertised. The second
function implements the actual generator for the advertised observables.

The generated dependent values should be stored in the proxy
objects. For this, the assignment operator can be used (i.e. `xProxy
= 3.0` ). Never call assign to any proxy not known to be a dependent
via the generation code. Doing so may be ill-defined, e.g. in case
the proxy holds a function, and will trigger an assert.


### Batched function evaluations (Advanced usage)

To speed up computations with large numbers of data events in unbinned fits,
it is beneficial to override `doEval()`. Like this, large spans of
computations can be done, without having to call `evaluate()` for each single data event.
`doEval()` should execute the same computation as `evaluate()`, but it
may choose an implementation that is capable of SIMD computations.
If doEval is not implemented, the classic and slower `evaluate()` will be
called for each data event.
*/

#include "RooAbsPdf.h"

#include "FitHelpers.h"
#include "RooFit/Detail/RooNormalizedPdf.h"
#include "RooMsgService.h"
#include "RooArgSet.h"
#include "RooArgProxy.h"
#include "RooRealProxy.h"
#include "RooRealVar.h"
#include "RooGenContext.h"
#include "RooBinnedGenContext.h"
#include "RooPlot.h"
#include "RooCurve.h"
#include "RooCategory.h"
#include "RooNameReg.h"
#include "RooCmdConfig.h"
#include "RooGlobalFunc.h"
#include "RooRandom.h"
#include "RooNumIntConfig.h"
#include "RooProjectedPdf.h"
#include "RooParamBinning.h"
#include "RooNumCdf.h"
#include "RooFitResult.h"
#include "RooNumGenConfig.h"
#include "RooCachedReal.h"
#include "RooRealIntegral.h"
#include "RooWorkspace.h"
#include "RooNaNPacker.h"
#include "RooFitImplHelpers.h"
#include "RooHelpers.h"
#include "RooFormulaVar.h"
#include "RooDerivative.h"

#include "ROOT/StringUtils.hxx"
#include "TMath.h"
#include "TPaveText.h"
#include "TMatrixD.h"
#include "TMatrixDSym.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <cmath>
#include <stdexcept>

namespace {

inline double getLog(double prob, RooAbsReal const *caller)
{

   if (prob < 0) {
      caller->logEvalError("getLogVal() top-level p.d.f evaluates to a negative number");
      return RooNaNPacker::packFloatIntoNaN(-prob);
   }

   if (std::isinf(prob)) {
      oocoutW(caller, Eval) << "RooAbsPdf::getLogVal(" << caller->GetName()
                            << ") WARNING: top-level pdf has an infinite value" << std::endl;
   }

   if (prob == 0) {
      caller->logEvalError("getLogVal() top-level p.d.f evaluates to zero");

      return -std::numeric_limits<double>::infinity();
   }

   if (TMath::IsNaN(prob)) {
      caller->logEvalError("getLogVal() top-level p.d.f evaluates to NaN");

      return prob;
   }

   return std::log(prob);
}

} // namespace

using std::endl, std::string, std::ostream, std::vector, std::pair, std::make_pair;

using RooHelpers::getColonSeparatedNameString;




Int_t RooAbsPdf::_verboseEval = 0;
TString RooAbsPdf::_normRangeOverride;

////////////////////////////////////////////////////////////////////////////////
/// Default constructor

RooAbsPdf::RooAbsPdf() : _normMgr(this, 10) {}

////////////////////////////////////////////////////////////////////////////////
/// Constructor with name and title only

RooAbsPdf::RooAbsPdf(const char *name, const char *title) :
  RooAbsReal(name,title), _normMgr(this,10), _selectComp(true)
{
  resetErrorCounters() ;
  setTraceCounter(0) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Constructor with name, title, and plot range

RooAbsPdf::RooAbsPdf(const char *name, const char *title,
           double plotMin, double plotMax) :
  RooAbsReal(name,title,plotMin,plotMax), _normMgr(this,10), _selectComp(true)
{
  resetErrorCounters() ;
  setTraceCounter(0) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooAbsPdf::RooAbsPdf(const RooAbsPdf& other, const char* name) :
  RooAbsReal(other,name),
  _normMgr(other._normMgr,this), _selectComp(other._selectComp), _normRange(other._normRange)
{
  resetErrorCounters() ;
  setTraceCounter(other._traceCount) ;

  if (other._specGeneratorConfig) {
    _specGeneratorConfig = std::make_unique<RooNumGenConfig>(*other._specGeneratorConfig);
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooAbsPdf::~RooAbsPdf()
{
}


double RooAbsPdf::normalizeWithNaNPacking(double rawVal, double normVal) const {

    if (normVal < 0. || (normVal == 0. && rawVal != 0)) {
      //Unreasonable normalisations. A zero integral can be tolerated if the function vanishes, though.
      const std::string msg = "p.d.f normalization integral is zero or negative: " + std::to_string(normVal);
      logEvalError(msg.c_str());
      clearValueAndShapeDirty();
      return RooNaNPacker::packFloatIntoNaN(-normVal + (rawVal < 0. ? -rawVal : 0.));
    }

    if (rawVal < 0.) {
      logEvalError(Form("p.d.f value is less than zero (%f), trying to recover", rawVal));
      clearValueAndShapeDirty();
      return RooNaNPacker::packFloatIntoNaN(-rawVal);
    }

    if (TMath::IsNaN(rawVal)) {
      logEvalError("p.d.f value is Not-a-Number");
      clearValueAndShapeDirty();
      return rawVal;
    }

    return (rawVal == 0. && normVal == 0.) ? 0. : rawVal / normVal;
}


////////////////////////////////////////////////////////////////////////////////
/// Return current value, normalized by integrating over
/// the observables in `nset`. If `nset` is 0, the unnormalized value
/// is returned. All elements of `nset` must be lvalues.
///
/// Unnormalized values are not cached.
/// Doing so would be complicated as `_norm->getVal()` could
/// spoil the cache and interfere with returning the cached
/// return value. Since unnormalized calls are typically
/// done in integration calls, there is no performance hit.

double RooAbsPdf::getValV(const RooArgSet* nset) const
{

  // Special handling of case without normalization set (used in numeric integration of pdfs)
  if (!nset) {
    RooArgSet const* tmp = _normSet ;
    _normSet = nullptr ;
    double val = evaluate() ;
    _normSet = tmp ;

    return TMath::IsNaN(val) ? 0. : val;
  }


  // Process change in last data set used
  bool nintChanged(false) ;
  if (!isActiveNormSet(nset) || _norm==nullptr) {
    nintChanged = syncNormalization(nset) ;
  }

  // Return value of object. Calculated if dirty, otherwise cached value is returned.
  if (isValueDirty() || nintChanged || _norm->isValueDirty()) {

    // Evaluate numerator
    const double rawVal = evaluate();

    // Evaluate denominator
    const double normVal = _norm->getVal();

    _value = normalizeWithNaNPacking(rawVal, normVal);

    clearValueAndShapeDirty();
  }

  return _value ;
}


////////////////////////////////////////////////////////////////////////////////
/// Analytical integral with normalization (see RooAbsReal::analyticalIntegralWN() for further information).
///
/// This function applies the normalization specified by `normSet` to the integral returned
/// by RooAbsReal::analyticalIntegral(). The passthrough scenario (code=0) is also changed
/// to return a normalized answer.

double RooAbsPdf::analyticalIntegralWN(Int_t code, const RooArgSet* normSet, const char* rangeName) const
{
  cxcoutD(Eval) << "RooAbsPdf::analyticalIntegralWN(" << GetName() << ") code = " << code << " normset = " << (normSet?*normSet:RooArgSet()) << std::endl ;


  if (code==0) return getVal(normSet) ;
  if (normSet) {
    return analyticalIntegral(code,rangeName) / getNorm(normSet) ;
  } else {
    return analyticalIntegral(code,rangeName) ;
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Check that passed value is positive and not 'not-a-number'.  If
/// not, print an error, until the error counter reaches its set
/// maximum.

bool RooAbsPdf::traceEvalPdf(double value) const
{
  // check for a math error or negative value
  bool error(false) ;
  if (TMath::IsNaN(value)) {
    logEvalError(Form("p.d.f value is Not-a-Number (%f), forcing value to zero",value)) ;
    error=true ;
  }
  if (value<0) {
    logEvalError(Form("p.d.f value is less than zero (%f), forcing value to zero",value)) ;
    error=true ;
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


////////////////////////////////////////////////////////////////////////////////
/// Get normalisation term needed to normalise the raw values returned by
/// getVal(). Note that `getVal(normalisationVariables)` will automatically
/// apply the normalisation term returned here.
/// \param nset Set of variables to normalise over.
double RooAbsPdf::getNorm(const RooArgSet* nset) const
{
  if (!nset) return 1 ;

  syncNormalization(nset,true) ;
  if (_verboseEval>1) cxcoutD(Tracing) << ClassName() << "::getNorm(" << GetName() << "): norm(" << _norm << ") = " << _norm->getVal() << std::endl ;

  double ret = _norm->getVal() ;
  if (ret==0.) {
    if(++_errorCount <= 10) {
      coutW(Eval) << "RooAbsPdf::getNorm(" << GetName() << ":: WARNING normalization is zero, nset = " ;  nset->Print("1") ;
      if(_errorCount == 10) coutW(Eval) << "RooAbsPdf::getNorm(" << GetName() << ") INFO: no more messages will be printed " << std::endl ;
    }
  }

  return ret ;
}



////////////////////////////////////////////////////////////////////////////////
/// Return pointer to RooAbsReal object that implements calculation of integral over observables iset in range
/// rangeName, optionally taking the integrand normalized over observables nset

const RooAbsReal* RooAbsPdf::getNormObj(const RooArgSet* nset, const RooArgSet* iset, const TNamed* rangeName) const
{
  // Check normalization is already stored
  CacheElem* cache = static_cast<CacheElem*>(_normMgr.getObj(nset,iset,nullptr,rangeName)) ;
  if (cache) {
    return cache->_norm.get();
  }

  // If not create it now
  RooArgSet depList;
  getObservables(iset, depList);

  // Normalization is always over all pdf components. Overriding the global
  // component selection temporarily makes all RooRealIntegrals created during
  // that time always include all components.
  GlobalSelectComponentRAII globalSelComp(true);
  RooAbsReal* norm = std::unique_ptr<RooAbsReal>{createIntegral(depList,*nset, *getIntegratorConfig(), RooNameReg::str(rangeName))}.release();

  // Store it in the cache
  _normMgr.setObj(nset,iset,new CacheElem(*norm),rangeName) ;

  // And return the newly created integral
  return norm ;
}



////////////////////////////////////////////////////////////////////////////////
/// Verify that the normalization integral cached with this PDF
/// is valid for given set of normalization observables.
///
/// If not, the cached normalization integral (if any) is deleted
/// and a new integral is constructed for use with 'nset'.
/// Elements in 'nset' can be discrete and real, but must be lvalues.
///
/// For functions that declare to be self-normalized by overloading the
/// selfNormalized() function, a unit normalization is always constructed.

bool RooAbsPdf::syncNormalization(const RooArgSet* nset, bool adjustProxies) const
{
  setActiveNormSet(nset);

  // Check if data sets are identical
  CacheElem* cache = static_cast<CacheElem*>(_normMgr.getObj(nset)) ;
  if (cache) {

    bool nintChanged = (_norm!=cache->_norm.get()) ;
    _norm = cache->_norm.get();

    // In the past, this condition read `if (nintChanged && adjustProxies)`.
    // However, the cache checks if the nset was already cached **by content**,
    // and not by RooArgSet instance! So it can happen that the normalization
    // set object is different, but the integral object is the same, in which
    // case it would be wrong to not adjust the proxies. They always have to be
    // adjusted when the nset changed, which is always the case when
    // `syncNormalization()` is called.
    if (adjustProxies) {
      // Update dataset pointers of proxies
      const_cast<RooAbsPdf*>(this)->setProxyNormSet(nset) ;
    }

    return nintChanged ;
  }

  // Update dataset pointers of proxies
  if (adjustProxies) {
    const_cast<RooAbsPdf*>(this)->setProxyNormSet(nset) ;
  }

  RooArgSet depList;
  getObservables(nset, depList);

  if (_verboseEval>0) {
    if (!selfNormalized()) {
      cxcoutD(Tracing) << ClassName() << "::syncNormalization(" << GetName()
      << ") recreating normalization integral " << std::endl ;
      depList.printStream(ccoutD(Tracing),kName|kValue|kArgs,kSingleLine) ;
    } else {
      cxcoutD(Tracing) << ClassName() << "::syncNormalization(" << GetName() << ") selfNormalized, creating unit norm" << std::endl;
    }
  }

  // Destroy old normalization & create new
  if (selfNormalized() || !dependsOn(depList)) {
    auto ntitle = std::string(GetTitle()) + " Unit Normalization";
    auto nname = std::string(GetName()) + "_UnitNorm";
    _norm = new RooRealVar(nname.c_str(),ntitle.c_str(),1) ;
  } else {
    const char* nr = (_normRangeOverride.Length()>0 ? _normRangeOverride.Data() : (_normRange.Length()>0 ? _normRange.Data() : nullptr)) ;

//     std::cout << "RooAbsPdf::syncNormalization(" << GetName() << ") rangeName for normalization is " << (nr?nr:"<null>") << std::endl ;
    RooAbsReal* normInt;
    {
      // Normalization is always over all pdf components. Overriding the global
      // component selection temporarily makes all RooRealIntegrals created during
      // that time always include all components.
      GlobalSelectComponentRAII selCompRAII(true);
      normInt = std::unique_ptr<RooAbsReal>{createIntegral(depList,*getIntegratorConfig(),nr)}.release();
    }
    static_cast<RooRealIntegral*>(normInt)->setAllowComponentSelection(false);
    normInt->getVal() ;
//     std::cout << "resulting normInt = " << normInt->GetName() << std::endl ;

    const char* cacheParamsStr = getStringAttribute("CACHEPARAMINT") ;
    if (cacheParamsStr && strlen(cacheParamsStr)) {

      std::unique_ptr<RooArgSet> intParams{normInt->getVariables()} ;

      RooArgSet cacheParams = RooHelpers::selectFromArgSet(*intParams, cacheParamsStr);

      if (!cacheParams.empty()) {
   cxcoutD(Caching) << "RooAbsReal::createIntObj(" << GetName() << ") INFO: constructing " << cacheParams.size()
          << "-dim value cache for integral over " << depList << " as a function of " << cacheParams << " in range " << (nr?nr:"<default>") <<  std::endl ;
   string name = Form("%s_CACHE_[%s]",normInt->GetName(),cacheParams.contentsString().c_str()) ;
   RooCachedReal* cachedIntegral = new RooCachedReal(name.c_str(),name.c_str(),*normInt,cacheParams) ;
   cachedIntegral->setInterpolationOrder(2) ;
   cachedIntegral->addOwnedComponents(*normInt) ;
   cachedIntegral->setCacheSource(true) ;
   if (normInt->operMode()==ADirty) {
     cachedIntegral->setOperMode(ADirty) ;
   }
   normInt= cachedIntegral ;
      }

    }
    _norm = normInt ;
  }

  // Register new normalization with manager (takes ownership)
  cache = new CacheElem(*_norm) ;
  _normMgr.setObj(nset,cache) ;

//   std::cout << "making new object " << _norm->GetName() << std::endl ;

  return true ;
}



////////////////////////////////////////////////////////////////////////////////
/// Reset error counter to given value, limiting the number
/// of future error messages for this pdf to 'resetValue'

void RooAbsPdf::resetErrorCounters(Int_t resetValue)
{
  _errorCount = resetValue ;
  _negCount   = resetValue ;
}



////////////////////////////////////////////////////////////////////////////////
/// Reset trace counter to given value, limiting the
/// number of future trace messages for this pdf to 'value'

void RooAbsPdf::setTraceCounter(Int_t value, bool allNodes)
{
  if (!allNodes) {
    _traceCount = value ;
    return ;
  } else {
    RooArgList branchList ;
    branchNodeServerList(&branchList) ;
    for(auto * pdf : dynamic_range_cast<RooAbsPdf*>(branchList)) {
      if (pdf) pdf->setTraceCounter(value,false) ;
    }
  }

}




////////////////////////////////////////////////////////////////////////////////
/// Return the log of the current value with given normalization
/// An error message is printed if the argument of the log is negative.

double RooAbsPdf::getLogVal(const RooArgSet* nset) const
{
  return getLog(getVal(nset), this);
}


////////////////////////////////////////////////////////////////////////////////
/// Check for infinity or NaN.
/// \param[in] inputs Array to check
/// \return True if either infinity or NaN were found.
namespace {
template<class T>
bool checkInfNaNNeg(const T& inputs) {
  // check for a math error or negative value
  bool inf = false;
  bool nan = false;
  bool neg = false;

  for (double val : inputs) { //CHECK_VECTORISE
    inf |= !std::isfinite(val);
    nan |= TMath::IsNaN(val); // Works also during fast math
    neg |= val < 0;
  }

  return inf || nan || neg;
}
}


////////////////////////////////////////////////////////////////////////////////
/// Scan through outputs and fix+log all nans and negative values.
/// \param[in,out] outputs Array to be scanned & fixed.
/// \param[in] begin Begin of event range. Only needed to print the correct event number
/// where the error occurred.
void RooAbsPdf::logBatchComputationErrors(std::span<const double>& outputs, std::size_t begin) const {
  for (unsigned int i=0; i<outputs.size(); ++i) {
    const double value = outputs[i];
    if (TMath::IsNaN(outputs[i])) {
      logEvalError(Form("p.d.f value of (%s) is Not-a-Number (%f) for entry %zu",
          GetName(), value, begin+i));
    } else if (!std::isfinite(outputs[i])){
      logEvalError(Form("p.d.f value of (%s) is (%f) for entry %zu",
          GetName(), value, begin+i));
    } else if (outputs[i] < 0.) {
      logEvalError(Form("p.d.f value of (%s) is less than zero (%f) for entry %zu",
          GetName(), value, begin+i));
    }
  }
}


void RooAbsPdf::getLogProbabilities(std::span<const double> pdfValues, double * output) const {
  for (std::size_t i = 0; i < pdfValues.size(); ++i) {
     output[i] = getLog(pdfValues[i], this);
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Return the extended likelihood term (\f$ N_\mathrm{expect} - N_\mathrm{observed} \cdot \log(N_\mathrm{expect} \f$)
/// of this PDF for the given number of observed events.
///
/// For successful operation, the PDF implementation must indicate that
/// it is extendable by overloading `canBeExtended()`, and must
/// implement the `expectedEvents()` function.
///
/// \param[in] sumEntries The number of observed events.
/// \param[in] nset The normalization set when asking the pdf for the expected
///            number of events.
/// \param[in] sumEntriesW2 The number of observed events when weighting with
///            squared weights. If non-zero, the weight-squared error
///            correction is applied to the extended term.
/// \param[in] doOffset Offset the extended term by a counterterm where the
///            expected number of events equals the observed number of events.
///            This constant shift results in a term closer to zero that is
///            approximately chi-square distributed. It is useful to do this
///            also when summing multiple NLL terms to avoid numeric precision
///            loss that happens if you sum multiple terms of different orders
///            of magnitude.
///
/// The weight-squared error correction works as follows:
/// adjust poisson such that
/// estimate of \f$N_\mathrm{expect}\f$ stays at the same value, but has a different variance, rescale
/// both the observed and expected count of the Poisson with a factor \f$ \sum w_{i} / \sum w_{i}^2 \f$
/// (the effective weight of the Poisson term),
/// i.e., change \f$\mathrm{Poisson}(N_\mathrm{observed} = \sum w_{i} | N_\mathrm{expect} )\f$
/// to \f$ \mathrm{Poisson}(\sum w_{i} \cdot \sum w_{i} / \sum w_{i}^2 | N_\mathrm{expect} \cdot \sum w_{i} / \sum w_{i}^2 ) \f$,
/// weighted by the effective weight \f$ \sum w_{i}^2 / \sum w_{i} \f$ in the likelihood.
/// Since here we compute the likelihood with the weight square, we need to multiply by the
/// square of the effective weight:
///   - \f$ W_\mathrm{expect}   = N_\mathrm{expect} \cdot \sum w_{i} / \sum w_{i}^2 \f$ : effective expected entries
///   - \f$ W_\mathrm{observed} = \sum w_{i} \cdot \sum w_{i} / \sum w_{i}^2 \f$        : effective observed entries
///
/// The extended term for the likelihood weighted by the square of the weight will be then:
///
///  \f$ \left(\sum w_{i}^2 / \sum w_{i}\right)^2 \cdot W_\mathrm{expect} - (\sum w_{i}^2 / \sum w_{i})^2 \cdot W_\mathrm{observed} \cdot \log{W_\mathrm{expect}} \f$
///
///  aund this is using the previous expressions for \f$ W_\mathrm{expect} \f$ and \f$ W_\mathrm{observed} \f$:
///
///  \f$ \sum w_{i}^2 / \sum w_{i} \cdot N_\mathrm{expect} - \sum w_{i}^2 \cdot \log{W_\mathrm{expect}} \f$
///
///  Since the weights are constants in the likelihood we can use \f$\log{N_\mathrm{expect}}\f$ instead of \f$\log{W_\mathrm{expect}}\f$.
///
/// See also RooAbsPdf::extendedTerm(RooAbsData const& data, bool weightSquared, bool doOffset),
/// which takes a dataset to extract \f$N_\mathrm{observed}\f$ and the
/// normalization set.
double RooAbsPdf::extendedTerm(double sumEntries, RooArgSet const* nset, double sumEntriesW2, bool doOffset) const
{
  return extendedTerm(sumEntries, expectedEvents(nset), sumEntriesW2, doOffset);
}

double RooAbsPdf::extendedTerm(double sumEntries, double expected, double sumEntriesW2, bool doOffset) const
{
  // check if this PDF supports extended maximum likelihood fits
  if(!canBeExtended()) {
    coutE(InputArguments) << GetName() << ": this PDF does not support extended maximum likelihood"
         << std::endl;
    return 0.0;
  }

  if(expected < 0.0) {
    coutE(InputArguments) << GetName() << ": calculated negative expected events: " << expected
         << std::endl;
    logEvalError("extendedTerm #expected events is <0 return a  NaN");
    return TMath::QuietNaN();
  }


  // Explicitly handle case Nobs=Nexp=0
  if (std::abs(expected)<1e-10 && std::abs(sumEntries)<1e-10) {
    return 0.0;
  }

  // Check for errors in Nexpected
  if (TMath::IsNaN(expected)) {
    logEvalError("extendedTerm #expected events is a NaN") ;
    return TMath::QuietNaN() ;
  }

  double extra = doOffset
      ? (expected - sumEntries) - sumEntries * (std::log(expected) - std::log(sumEntries))
      : expected - sumEntries * std::log(expected);

  if(sumEntriesW2 != 0.0) {
    extra *= sumEntriesW2 / sumEntries;
  }

  return extra;
}

////////////////////////////////////////////////////////////////////////////////
/// Return the extended likelihood term (\f$ N_\mathrm{expect} - N_\mathrm{observed} \cdot \log(N_\mathrm{expect} \f$)
/// of this PDF for the given number of observed events.
///
/// This function is a wrapper around
/// RooAbsPdf::extendedTerm(double, RooArgSet const *, double, bool) const,
/// where the number of observed events and observables to be used as the
/// normalization set for the pdf is extracted from a RooAbsData.
///
/// For successful operation, the PDF implementation must indicate that
/// it is extendable by overloading `canBeExtended()`, and must
/// implement the `expectedEvents()` function.
///
/// \param[in] data The RooAbsData to retrieve the set of observables and
///            number of expected events.
/// \param[in] weightSquared If set to `true`, the extended term will be scaled by
///            the ratio of squared event weights over event weights:
///            \f$ \sum w_{i}^2 / \sum w_{i} \f$.
///            Intended to be used by fits with the `SumW2Error()` option that
///            can be passed to RooAbsPdf::fitTo()
///            (see the documentation of said function to learn more about the
///            interpretation of fits with squared weights).
/// \param[in] doOffset See RooAbsPdf::extendedTerm(double, RooArgSet const*, double, bool) const.

double RooAbsPdf::extendedTerm(RooAbsData const& data, bool weightSquared, bool doOffset) const {
  double sumW = data.sumEntries();
  double sumW2 = 0.0;
  if (weightSquared) {
    sumW2 = data.sumEntriesW2();
  }
  return extendedTerm(sumW, data.get(), sumW2, doOffset);
}


/** @fn RooAbsPdf::createNLL()
 *
 * @brief Construct representation of -log(L) of PDF with given dataset.
 *
 * If dataset is unbinned, an unbinned likelihood is constructed.
 * If the dataset is binned, a binned likelihood is constructed.
 *
 * @param data Reference to a RooAbsData object representing the dataset.
 * @param cmdArgs Variadic template arguments representing optional command arguments.
 *             You can pass either an arbitrary number of RooCmdArg instances
 *             or a single RooLinkedList that points to the RooCmdArg objects.
 * @return An owning pointer to the created RooAbsReal NLL object.
 *
 * @tparam CmdArgs_t Template types for optional command arguments.
 *                   Can either be an arbitrary number of RooCmdArg or a single RooLinkedList.
 *
 * \note This front-end function should not be re-implemented in derived PDF types.
 *       If you mean to customize the NLL creation routine,
 *       you need to override the virtual RooAbsPdf::createNLLImpl() method.
 *
 * The following named arguments are supported:
 *
 * <table>
 * <tr><th> Type of CmdArg    <th>    Effect on NLL
 * <tr><td> `ConditionalObservables(Args_t &&... argsOrArgSet)`  <td>  Do not normalize PDF over listed observables.
 *                                                 Arguments can either be multiple RooRealVar or a single RooArgSet containing them.
 * <tr><td> `Extended(bool flag)`             <td> Add extended likelihood term, off by default.
 * <tr><td> `Range(const char* name)`         <td>  Fit only data inside range with given name. Multiple comma-separated range names can be specified.
 *                                                  In this case, the unnormalized PDF \f$f(x)\f$ is normalized by the integral over all ranges \f$r_i\f$:
 *                                                  \f[
 *                                                      p(x) = \frac{f(x)}{\sum_i \int_{r_i} f(x) dx}.
 *                                                  \f]
 * <tr><td> `Range(double lo, double hi)` <td>  Fit only data inside given range. A range named "fit" is created on the fly on all observables.
 * <tr><td> `SumCoefRange(const char* name)`  <td> Set the range in which to interpret the coefficients of RooAddPdf components
 * <tr><td> `NumCPU(int num, int istrat)`      <td> Parallelize NLL calculation on num CPUs. (Currently, this setting is ignored with the **cpu** Backend.)
 *   <table>
 *   <tr><th> Strategy   <th> Effect
 *   <tr><td> 0 = RooFit::BulkPartition - *default* <td> Divide events in N equal chunks
 *   <tr><td> 1 = RooFit::Interleave <td> Process event i%N in process N. Recommended for binned data with
 *                     a substantial number of zero-bins, which will be distributed across processes more equitably in this strategy
 *   <tr><td> 2 = RooFit::SimComponents <td> Process each component likelihood of a RooSimultaneous fully in a single process
 *                     and distribute components over processes. This approach can be beneficial if normalization calculation time
 *                     dominates the total computation time of a component (since the normalization calculation must be performed
 *                     in each process in strategies 0 and 1. However beware that if the RooSimultaneous components do not share many
 *                     parameters this strategy is inefficient: as most minuit-induced likelihood calculations involve changing
 *                     a single parameter, only 1 of the N processes will be active most of the time if RooSimultaneous components
 *                     do not share many parameters
 *   <tr><td> 3 = RooFit::Hybrid <td> Follow strategy 0 for all RooSimultaneous components, except those with less than
 *                     30 dataset entries, for which strategy 2 is followed.
 *   </table>
 * <tr><td> `EvalBackend(std::string const&)` <td> Choose a likelihood evaluation backend:
 *   <table>
 *   <tr><th> Backend <th> Description
 *   <tr><td> **cpu** - *default* <td> New vectorized evaluation mode, using faster math functions and auto-vectorisation (currently on a single thread).
 *                         Since ROOT 6.23, this is the default if `EvalBackend()` is not passed, succeeding the **legacy** backend.
 *                         If all RooAbsArg objects in the model support vectorized evaluation,
 *                         likelihood computations are 2 to 10 times faster than with the **legacy** backend (each on a single thread).
 *                         - unless your dataset is so small that the vectorization is not worth it.
 *                         The relative difference of the single log-likelihoods with respect to the legacy mode is usually better than \f$10^{-12}\f$,
 *                         and for fit parameters it's usually better than \f$10^{-6}\f$. In past ROOT releases, this backend could be activated with the now deprecated `BatchMode()` option.
 *   <tr><td> **cuda** <td> Evaluate the likelihood on a GPU that supports CUDA.
 *                          This backend re-uses code from the **cpu** backend, but compiled in CUDA kernels.
 *                          Hence, the results are expected to be identical, modulo some numerical differences that can arise from the different order in which the GPU is summing the log probabilities.
 *                          This backend can drastically speed up the fit if all RooAbsArg object in the model support it.
 *   <tr><td> **legacy** <td> The original likelihood evaluation method.
 *                            Evaluates the PDF for each single data entry at a time before summing the negative log probabilities.
 *                            It supports multi-threading, but you might need more than 20 threads to maybe see about 10% performance gain over the default cpu-backend (that runs currently only on a single thread).
 *   <tr><td> **codegen** <td> **Experimental** - Generates and compiles minimal C++ code for the NLL on-the-fly and wraps it in the returned RooAbsReal.
 *                             Also generates and compiles the code for the gradient using Automatic Differentiation (AD) with [Clad](https://github.com/vgvassilev/clad).
 *                             This analytic gradient is passed to the minimizer, which can result in significant speedups for many-parameter fits,
 *                             even compared to the **cpu** backend. However, if one of the RooAbsArg objects in the model does not support the code generation,
 *                             this backend can't be used.
 *   <tr><td> **codegen_no_grad** <td> **Experimental** - Same as **codegen**, but doesn't generate and compile the gradient code and use the regular numerical differentiation instead.
 *                                     This is expected to be slower, but useful for debugging problems with the analytic gradient.
 *   </table>
 * <tr><td> `Optimize(bool flag)`           <td> Activate constant term optimization (on by default)
 * <tr><td> `SplitRange(bool flag)`         <td> Use separate fit ranges in a simultaneous fit. Actual range name for each subsample is assumed to
 *                                               be `rangeName_indexState`, where `indexState` is the state of the master index category of the simultaneous fit.
 * Using `Range("range"), SplitRange()` as switches, different ranges could be set like this:
 * ```
 * myVariable.setRange("range_pi0", 135, 210);
 * myVariable.setRange("range_gamma", 50, 210);
 * ```
 * <tr><td> `Constrain(const RooArgSet&pars)`          <td> For p.d.f.s that contain internal parameter constraint terms (that is usually product PDFs, where one
 *     term of the product depends on parameters but not on the observable(s),), only apply constraints to the given subset of parameters.
 * <tr><td> `ExternalConstraints(const RooArgSet& )`   <td> Include given external constraints to likelihood by multiplying them with the original likelihood.
 * <tr><td> `GlobalObservables(const RooArgSet&)`      <td> Define the set of normalization observables to be used for the constraint terms.
 *                                                        If none are specified the constrained parameters are used.
 * <tr><td> `GlobalObservablesSource(const char* sourceName)` <td> Which source to prioritize for global observable values.
 *                                                                 Can be either:
 *                                                                 - `data`: to take the values from the dataset,
 *                                                                   falling back to the pdf value if a given global observable is not available.
 *                                                                   If no `GlobalObservables` or `GlobalObservablesTag` command argument is given, the set
 *                                                                   of global observables will be automatically defined to be the set stored in the data.
 *                                                                 - `model`: to take all values from the pdf and completely ignore the set of global observables stored in the data
 *                                                                   (not even using it to automatically define the set of global observables
 *                                                                   if the `GlobalObservables` or `GlobalObservablesTag` command arguments are not given).
 *                                                                 The default option is `data`.
 * <tr><td> `GlobalObservablesTag(const char* tagName)` <td> Define the set of normalization observables to be used for the constraint terms by
 *                                                         a string attribute associated with pdf observables that match the given tagName.
 * <tr><td> `Verbose(bool flag)`           <td> Controls RooFit informational messages in likelihood construction
 * <tr><td> `CloneData(bool flag)`            <td> Use clone of dataset in NLL (default is true).
 *                                                 \warning Deprecated option that is ignored. It is up to the implementation of the NLL creation method if the data is cloned or not.
 * <tr><td> `Offset(std::string const& mode)` <td> Likelihood offsetting mode. Can be either:
 *   <table>
 *   <tr><th> Mode <th> Description
 *   <tr><td> **none** - *default* <td> No offsetting.
 *   <tr><td> **initial** <td> Offset likelihood by initial value (so that starting value of FCN in minuit is zero).
 *                             This can improve numeric stability in simultaneous fits with components with large likelihood values.
 *   <tr><td> **bin** <td> Offset likelihood bin-by-bin with a template histogram model based on the obersved data.
 *                         This results in per-bin values that are all in the same order of magnitude, which reduces precision loss in the sum,
 *                         which can drastically improve numeric stability.
 *                         Furthermore, \f$2\cdot \text{NLL}\f$ defined like this is approximately chi-square distributed, allowing for goodness-of-fit tests.
 *   </table>
 * <tr><td> `IntegrateBins(double precision)` <td> In binned fits, integrate the PDF over the bins instead of using the probability density at the bin centre.
 *                                                 This can reduce the bias observed when fitting functions with high curvature to binned data.
 *                                                 - precision > 0: Activate bin integration everywhere. Use precision between 0.01 and 1.E-6, depending on binning.
 *                                                   Note that a low precision such as 0.01 might yield identical results to 1.E-4, since the integrator might reach 1.E-4 already in its first
 *                                                   integration step. If lower precision is desired (more speed), a RooBinSamplingPdf has to be created manually, and its integrator
 *                                                   has to be manipulated directly.
 *                                                 - precision = 0: Activate bin integration only for continuous PDFs fit to a RooDataHist.
 *                                                 - precision < 0: Deactivate.
 *                                                 \see RooBinSamplingPdf
 * <tr><td> `ModularL(bool flag)`           <td>  Enable or disable modular likelihoods, which will become the default in a future release.
 *                                                This does not change any user-facing code, but only enables a different likelihood class in the back-end. Note that this
 *                                                should be set to true for parallel minimization of likelihoods!
 *                                                Note that it is currently not recommended to use Modular likelihoods without any parallelization enabled in the minimization, since
 *                                                some features such as offsetting might not yet work in this case.
 * </table>
 */


/** @brief Protected implementation of the NLL creation routine.
 *
 * This virtual function can be overridden in case you want to change the NLL creation logic for custom PDFs.
 *
 * \note Never call this function directly. Instead, call RooAbsPdf::createNLL().
 */

std::unique_ptr<RooAbsReal> RooAbsPdf::createNLLImpl(RooAbsData &data, const RooLinkedList &cmdList)
{
   return RooFit::FitHelpers::createNLL(*this, data, cmdList);
}


/** @fn RooAbsPdf::fitTo()
 *
 * @brief Fit PDF to given dataset.
 *
 * If dataset is unbinned, an unbinned maximum likelihood is performed.
 * If the dataset is binned, a binned maximum likelihood is performed.
 * By default the fit is executed through the MINUIT commands MIGRAD, HESSE in succession.
 *
 * @param data Reference to a RooAbsData object representing the dataset.
 * @param cmdArgs Variadic template arguments representing optional command arguments.
 *             You can pass either an arbitrary number of RooCmdArg instances
 *             or a single RooLinkedList that points to the RooCmdArg objects.
 * @return An owning pointer to the created RooAbsReal NLL object.
 * @return RooFitResult with fit status and parameters if option Save() is used, `nullptr` otherwise. The user takes ownership of the fit result.
 *
 * @tparam CmdArgs_t Template types for optional command arguments.
 *              Can either be an arbitrary number of RooCmdArg or a single RooLinkedList.
 *
 * \note This front-end function should not be re-implemented in derived PDF types.
 *       If you mean to customize the likelihood fitting routine,
 *       you need to override the virtual RooAbsPdf::fitToImpl() method.
 *
 * The following named arguments are supported:
 *
 * <table>
 * <tr><th> Type of CmdArg                  <th> Options to control construction of -log(L)
 * <tr><td> <td> All command arguments that can also be passed to the NLL creation method.
 *                \see RooAbsPdf::createNLL()
 *
 * <tr><th><th> Options to control flow of fit procedure
 * <tr><td> `Minimizer("<type>", "<algo>")`   <td>  Choose minimization package and optionally the algorithm to use. Default is MINUIT/MIGRAD through the RooMinimizer interface,
 *                                       but others can be specified (through RooMinimizer interface).
 *   <table>
 *   <tr><th> Type         <th> Algorithm
 *   <tr><td> Minuit       <td>  migrad, simplex, minimize (=migrad+simplex), migradimproved (=migrad+improve)
 *   <tr><td> Minuit2      <td>  migrad, simplex, minimize, scan
 *   <tr><td> GSLMultiMin  <td>  conjugatefr, conjugatepr, bfgs, bfgs2, steepestdescent
 *   <tr><td> GSLSimAn     <td>  -
 *   </table>
 *
 * <tr><td> `InitialHesse(bool flag)`       <td>  Flag controls if HESSE before MIGRAD as well, off by default
 * <tr><td> `Optimize(bool flag)`           <td>  Activate constant term optimization of test statistic during minimization (on by default)
 * <tr><td> `Hesse(bool flag)`              <td>  Flag controls if HESSE is run after MIGRAD, on by default
 * <tr><td> `Minos(bool flag)`              <td>  Flag controls if MINOS is run after HESSE, off by default
 * <tr><td> `Minos(const RooArgSet& set)`     <td>  Only run MINOS on given subset of arguments
 * <tr><td> `Save(bool flag)`               <td>  Flag controls if RooFitResult object is produced and returned, off by default
 * <tr><td> `Strategy(Int_t flag)`            <td>  Set Minuit strategy (0 to 2, default is 1)
 * <tr><td> `MaxCalls(int n)`              <td> Change maximum number of likelihood function calls from MINUIT (if `n <= 0`, the default of 500 * #%parameters is used)
 * <tr><td> `EvalErrorWall(bool flag=true)`    <td>  When parameters are in disallowed regions (e.g. PDF is negative), return very high value to fitter
 *                                                  to force it out of that region. This can, however, mean that the fitter gets lost in this region. If
 *                                                  this happens, try switching it off.
 * <tr><td> `RecoverFromUndefinedRegions(double strength)` <td> When PDF is invalid (e.g. parameter in undefined region), try to direct minimiser away from that region.
 *                                                              `strength` controls the magnitude of the penalty term. Leaving out this argument defaults to 10. Switch off with `strength = 0.`.
 *
 * <tr><td> `SumW2Error(bool flag)`         <td>  Apply correction to errors and covariance matrix.
 *       This uses two covariance matrices, one with the weights, the other with squared weights,
 *       to obtain the correct errors for weighted likelihood fits. If this option is activated, the
 *       corrected covariance matrix is calculated as \f$ V_\mathrm{corr} = V C^{-1} V \f$, where \f$ V \f$ is the original
 *       covariance matrix and \f$ C \f$ is the inverse of the covariance matrix calculated using the
 *       squared weights. This allows to switch between two interpretations of errors:
 *       <table>
 *       <tr><th> SumW2Error <th> Interpretation
 *       <tr><td> true       <td> The errors reflect the uncertainty of the Monte Carlo simulation.
 *                                Use this if you want to know how much accuracy you can get from the available Monte Carlo statistics.
 *
 *                                **Example**: Simulation with 1000 events, the average weight is 0.1.
 *                                The errors are as big as if one fitted to 1000 events.
 *       <tr><td> false      <td> The errors reflect the errors of a dataset, which is as big as the sum of weights.
 *                                Use this if you want to know what statistical errors you would get if you had a dataset with as many
 *                                events as the (weighted) Monte Carlo simulation represents.
 *
 *                                **Example** (Data as above):
 *                                The errors are as big as if one fitted to 100 events.
 *       </table>
 *       \note If the `SumW2Error` correction is enabled, the covariance matrix quality stored in the RooFitResult
 *             object will be the minimum of the original covariance matrix quality and the quality of the covariance
 *             matrix calculated with the squared weights.
 * <tr><td> `AsymptoticError()`               <td> Use the asymptotically correct approach to estimate errors in the presence of weights.
 *                                                 This is slower but more accurate than `SumW2Error`. See also https://arxiv.org/abs/1911.01303).
                                                   This option even correctly implements the case of extended likelihood fits
                                                   (see this [writeup on extended weighted fits](https://root.cern/files/extended_weighted_fits.pdf) that complements the paper linked before).
 * <tr><td> `PrefitDataFraction(double fraction)`
 *                                            <td>  Runs a prefit on a small dataset of size fraction*(actual data size). This can speed up fits
 *                                                  by finding good starting values for the parameters for the actual fit.
 *                                                  \warning Prefitting may give bad results when used in binned analysis.
 *
 * <tr><th><th> Options to control informational output
 * <tr><td> `Verbose(bool flag)`            <td>  Flag controls if verbose output is printed (NLL, parameter changes during fit).
 * <tr><td> `Timer(bool flag)`              <td>  Time CPU and wall clock consumption of fit steps, off by default.
 * <tr><td> `PrintLevel(Int_t level)`         <td>  Set Minuit print level (-1 to 3, default is 1). At -1 all RooFit informational messages are suppressed as well.
 *                                                  See RooMinimizer::PrintLevel for the meaning of the levels.
 * <tr><td> `Warnings(bool flag)`           <td>  Enable or disable MINUIT warnings (enabled by default)
 * <tr><td> `PrintEvalErrors(Int_t numErr)`   <td>  Control number of p.d.f evaluation errors printed per likelihood evaluation.
 *                                                A negative value suppresses output completely, a zero value will only print the error count per p.d.f component,
 *                                                a positive value will print details of each error up to `numErr` messages per p.d.f component.
 * <tr><td> `Parallelize(Int_t nWorkers)`   <td>  Control global parallelization settings. Arguments 1 and above enable the use of RooFit's parallel minimization
 *                                                backend and uses the number given as the number of workers to use in the parallelization. -1 also enables
 *                                                RooFit's parallel minimization backend, and sets the number of workers to the number of available processes.
 *                                                0 disables this feature.
 *                                                In case parallelization is requested, this option implies `ModularL(true)` in the internal call to the NLL creation method.
 * <tr><td> `ParallelGradientOptions(bool enable=true, int orderStrategy=0, int chainFactor=1)`   <td>  **Experimental** - Control gradient parallelization settings. The first argument
 *                                                                                                      only disables or enables gradient parallelization, this is on by default.
 *                                                                                                      The second argument determines the internal partial derivative calculation
 *                                                                                                      ordering strategy. The third argument determines the number of partial
 *                                                                                                      derivatives that are executed per task package on each worker.
 * <tr><td> `ParallelDescentOptions(bool enable=false, int splitStrategy=0, int numSplits=4)`   <td>  **Experimental** - Control settings related to the parallelization of likelihoods
 *                                                                                                      outside of the gradient calculation but in the minimization, most prominently
 *                                                                                                      in the linesearch step. The first argument this disables or enables likelihood
 *                                                                                                      parallelization. The second argument determines whether to split the task batches
 *                                                                                                      per event or per likelihood component. And the third argument how many events or
 *                                                                                                      respectively components to include in each batch.
 * <tr><td> `TimingAnalysis(bool flag)`   <td> **Experimental** - Log timings. This feature logs timings with NewStyle likelihoods on multiple processes simultaneously
 *                                         and outputs the timings at the end of a run to json log files, which can be analyzed with the
 *                                         `RooFit::MultiProcess::HeatmapAnalyzer`. Only works with simultaneous likelihoods.
 * </table>
 */


/** @brief Protected implementation of the likelihood fitting routine.
 *
 * This virtual function can be overridden in case you want to change the likelihood fitting logic for custom PDFs.
 *
 * \note Never call this function directly. Instead, call RooAbsPdf::fitTo().
 */

std::unique_ptr<RooFitResult> RooAbsPdf::fitToImpl(RooAbsData& data, const RooLinkedList& cmdList)
{
   return RooFit::FitHelpers::fitTo(*this, data, cmdList, false);
}


////////////////////////////////////////////////////////////////////////////////
/// Print value of p.d.f, also print normalization integral that was last used, if any

void RooAbsPdf::printValue(ostream& os) const
{
  // silent warning messages coming when evaluating a RooAddPdf without a normalization set
  RooHelpers::LocalChangeMsgLevel locmsg(RooFit::WARNING, 0u, RooFit::Eval, false);

  getVal() ;

  if (_norm) {
    os << getVal() << "/" << _norm->getVal() ;
  } else {
    os << getVal();
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Print multi line detailed information of this RooAbsPdf

void RooAbsPdf::printMultiline(ostream& os, Int_t contents, bool verbose, TString indent) const
{
  RooAbsReal::printMultiline(os,contents,verbose,indent);
  os << indent << "--- RooAbsPdf ---" << std::endl;
  os << indent << "Cached value = " << _value << std::endl ;
  if (_norm) {
    os << indent << " Normalization integral: " << std::endl ;
    auto moreIndent = std::string(indent.Data()) + "   " ;
    _norm->printStream(os,kName|kAddress|kTitle|kValue|kArgs,kSingleLine,moreIndent.c_str()) ;
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Return a binned generator context

RooAbsGenContext* RooAbsPdf::binnedGenContext(const RooArgSet &vars, bool verbose) const
{
  return new RooBinnedGenContext(*this,vars,nullptr,nullptr,verbose) ;
}


////////////////////////////////////////////////////////////////////////////////
/// Interface function to create a generator context from a p.d.f. This default
/// implementation returns a 'standard' context that works for any p.d.f

RooAbsGenContext* RooAbsPdf::genContext(const RooArgSet &vars, const RooDataSet *prototype,
               const RooArgSet* auxProto, bool verbose) const
{
  return new RooGenContext(*this,vars,prototype,auxProto,verbose) ;
}


////////////////////////////////////////////////////////////////////////////////

RooAbsGenContext* RooAbsPdf::autoGenContext(const RooArgSet &vars, const RooDataSet* prototype, const RooArgSet* auxProto,
                   bool verbose, bool autoBinned, const char* binnedTag) const
{
  if (prototype || (auxProto && !auxProto->empty())) {
    return genContext(vars,prototype,auxProto,verbose);
  }

  RooAbsGenContext *context(nullptr) ;
  if ( (autoBinned && isBinnedDistribution(vars)) || ( binnedTag && strlen(binnedTag) && (getAttribute(binnedTag)||string(binnedTag)=="*"))) {
    context = binnedGenContext(vars,verbose) ;
  } else {
    context= genContext(vars,nullptr,nullptr,verbose);
  }
  return context ;
}



////////////////////////////////////////////////////////////////////////////////
/// Generate a new dataset containing the specified variables with events sampled from our distribution.
/// Generate the specified number of events or expectedEvents() if not specified.
/// \param[in] whatVars Choose variables in which to generate events. Variables not listed here will remain
/// constant and not be used for event generation.
/// \param[in] arg1,arg2,arg3,arg4,arg5,arg6 Optional RooCmdArg() to change behaviour of generate().
/// \return RooDataSet *, owned by caller.
///
/// Any variables of this PDF that are not in whatVars will use their
/// current values and be treated as fixed parameters. Returns zero
/// in case of an error.
///
/// <table>
/// <tr><th> Type of CmdArg                    <th> Effect on generate
/// <tr><td> `Name(const char* name)`            <td> Name of the output dataset
/// <tr><td> `Verbose(bool flag)`              <td> Print informational messages during event generation
/// <tr><td> `NumEvents(int nevt)`               <td> Generate specified number of events
/// <tr><td> `Extended()`                        <td> If no number of events to be generated is given,
/// use expected number of events from extended likelihood term.
/// This evidently only works for extended PDFs.
/// <tr><td> `GenBinned(const char* tag)`        <td> Use binned generation for all component pdfs that have 'setAttribute(tag)' set
/// <tr><td> `AutoBinned(bool flag)`           <td> Automatically deploy binned generation for binned distributions (e.g. RooHistPdf, sums and products of
///                                                 RooHistPdfs etc)
/// \note Datasets that are generated in binned mode are returned as weighted unbinned datasets. This means that
/// for each bin, there will be one event in the dataset with a weight corresponding to the (possibly randomised) bin content.
///
///
/// <tr><td> `AllBinned()`                       <td> As above, but for all components.
///       \note The notion of components is only meaningful for simultaneous PDFs
///       as binned generation is always executed at the top-level node for a regular
///       PDF, so for those it only mattes that the top-level node is tagged.
///
/// <tr><td> ProtoData(const RooDataSet& data, bool randOrder)
///          <td> Use specified dataset as prototype dataset. If randOrder in ProtoData() is set to true,
///               the order of the events in the dataset will be read in a random order if the requested
///               number of events to be generated does not match the number of events in the prototype dataset.
///               \note If ProtoData() is used, the specified existing dataset as a prototype: the new dataset will contain
///               the same number of events as the prototype (unless otherwise specified), and any prototype variables not in
///               whatVars will be copied into the new dataset for each generated event and also used to set our PDF parameters.
///               The user can specify a  number of events to generate that will override the default. The result is a
///               copy of the prototype dataset with only variables in whatVars randomized. Variables in whatVars that
///               are not in the prototype will be added as new columns to the generated dataset.
///
/// </table>
///
/// #### Accessing the underlying event generator
/// Depending on the fit model (if it is difficult to sample), it may be necessary to change generator settings.
/// For the default generator (RooFoamGenerator), the number of samples or cells could be increased by e.g. using
///     myPdf->specialGeneratorConfig()->getConfigSection("RooFoamGenerator").setRealValue("nSample",1e4);
///
/// The foam generator e.g. has the following config options:
/// - nCell[123N]D
/// - nSample
/// - chatLevel
/// \see rf902_numgenconfig.C

RooFit::OwningPtr<RooDataSet> RooAbsPdf::generate(const RooArgSet& whatVars, const RooCmdArg& arg1,const RooCmdArg& arg2,
            const RooCmdArg& arg3,const RooCmdArg& arg4, const RooCmdArg& arg5,const RooCmdArg& arg6)
{
  // Select the pdf-specific commands
  RooCmdConfig pc("RooAbsPdf::generate(" + std::string(GetName()) + ")");
  pc.defineObject("proto","PrototypeData",0,nullptr) ;
  pc.defineString("dsetName","Name",0,"") ;
  pc.defineInt("randProto","PrototypeData",0,0) ;
  pc.defineInt("resampleProto","PrototypeData",1,0) ;
  pc.defineInt("verbose","Verbose",0,0) ;
  pc.defineInt("extended","Extended",0,0) ;
  pc.defineInt("nEvents","NumEvents",0,0) ;
  pc.defineInt("autoBinned","AutoBinned",0,1) ;
  pc.defineInt("expectedData","ExpectedData",0,0) ;
  pc.defineDouble("nEventsD","NumEventsD",0,-1.) ;
  pc.defineString("binnedTag","GenBinned",0,"") ;
  pc.defineMutex("GenBinned","ProtoData") ;
  pc.defineMutex("Extended", "NumEvents");

  // Process and check varargs
  pc.process(arg1,arg2,arg3,arg4,arg5,arg6) ;
  if (!pc.ok(true)) {
    return nullptr;
  }

  // Decode command line arguments
  RooDataSet* protoData = static_cast<RooDataSet*>(pc.getObject("proto",nullptr)) ;
  const char* dsetName = pc.getString("dsetName") ;
  bool verbose = pc.getInt("verbose") ;
  bool randProto = pc.getInt("randProto") ;
  bool resampleProto = pc.getInt("resampleProto") ;
  bool extended = pc.getInt("extended") ;
  bool autoBinned = pc.getInt("autoBinned") ;
  const char* binnedTag = pc.getString("binnedTag") ;
  Int_t nEventsI = pc.getInt("nEvents") ;
  double nEventsD = pc.getInt("nEventsD") ;
  //bool verbose = pc.getInt("verbose") ;
  bool expectedData = pc.getInt("expectedData") ;

  double nEvents = (nEventsD>0) ? nEventsD : double(nEventsI);

  // Force binned mode for expected data mode
  if (expectedData) {
    binnedTag="*" ;
  }

  if (extended) {
     if (nEvents == 0) nEvents = expectedEvents(&whatVars);
  } else if (nEvents==0) {
    cxcoutI(Generation) << "No number of events specified , number of events generated is "
           << GetName() << "::expectedEvents() = " << expectedEvents(&whatVars)<< std::endl ;
  }

  if (extended && protoData && !randProto) {
    cxcoutI(Generation) << "WARNING Using generator option Extended() (Poisson distribution of #events) together "
           << "with a prototype dataset implies incomplete sampling or oversampling of proto data. "
           << "Set randomize flag in ProtoData() option to randomize prototype dataset order and thus "
           << "to randomize the set of over/undersampled prototype events for each generation cycle." << std::endl ;
  }


  // Forward to appropriate implementation
  std::unique_ptr<RooDataSet> data;
  if (protoData) {
    data = std::unique_ptr<RooDataSet>{generate(whatVars,*protoData,Int_t(nEvents),verbose,randProto,resampleProto)};
  } else {
     data = std::unique_ptr<RooDataSet>{generate(whatVars,nEvents,verbose,autoBinned,binnedTag,expectedData, extended)};
  }

  // Rename dataset to given name if supplied
  if (dsetName && strlen(dsetName)>0) {
    data->SetName(dsetName) ;
  }

  return RooFit::makeOwningPtr(std::move(data));
}






////////////////////////////////////////////////////////////////////////////////
/// \note This method does not perform any generation. To generate according to generations specification call RooAbsPdf::generate(RooAbsPdf::GenSpec&) const
///
/// Details copied from RooAbsPdf::generate():
/// --------------------------------------------
/// \copydetails RooAbsPdf::generate(const RooArgSet&,const RooCmdArg&,const RooCmdArg&,const RooCmdArg&,const RooCmdArg&,const RooCmdArg&,const RooCmdArg&)

RooAbsPdf::GenSpec* RooAbsPdf::prepareMultiGen(const RooArgSet &whatVars,
                      const RooCmdArg& arg1,const RooCmdArg& arg2,
                      const RooCmdArg& arg3,const RooCmdArg& arg4,
                      const RooCmdArg& arg5,const RooCmdArg& arg6)
{

  // Select the pdf-specific commands
  RooCmdConfig pc("RooAbsPdf::generate(" + std::string(GetName()) + ")");
  pc.defineObject("proto","PrototypeData",0,nullptr) ;
  pc.defineString("dsetName","Name",0,"") ;
  pc.defineInt("randProto","PrototypeData",0,0) ;
  pc.defineInt("resampleProto","PrototypeData",1,0) ;
  pc.defineInt("verbose","Verbose",0,0) ;
  pc.defineInt("extended","Extended",0,0) ;
  pc.defineInt("nEvents","NumEvents",0,0) ;
  pc.defineInt("autoBinned","AutoBinned",0,1) ;
  pc.defineString("binnedTag","GenBinned",0,"") ;
  pc.defineMutex("GenBinned","ProtoData") ;


  // Process and check varargs
  pc.process(arg1,arg2,arg3,arg4,arg5,arg6) ;
  if (!pc.ok(true)) {
    return nullptr ;
  }

  // Decode command line arguments
  RooDataSet* protoData = static_cast<RooDataSet*>(pc.getObject("proto",nullptr)) ;
  const char* dsetName = pc.getString("dsetName") ;
  Int_t nEvents = pc.getInt("nEvents") ;
  bool verbose = pc.getInt("verbose") ;
  bool randProto = pc.getInt("randProto") ;
  bool resampleProto = pc.getInt("resampleProto") ;
  bool extended = pc.getInt("extended") ;
  bool autoBinned = pc.getInt("autoBinned") ;
  const char* binnedTag = pc.getString("binnedTag") ;

  RooAbsGenContext* cx = autoGenContext(whatVars,protoData,nullptr,verbose,autoBinned,binnedTag) ;

  return new GenSpec(cx,whatVars,protoData,nEvents,extended,randProto,resampleProto,dsetName) ;
}


////////////////////////////////////////////////////////////////////////////////
/// If many identical generation requests
/// are needed, e.g. in toy MC studies, it is more efficient to use the prepareMultiGen()/generate()
/// combination than calling the standard generate() multiple times as
/// initialization overhead is only incurred once.

RooFit::OwningPtr<RooDataSet> RooAbsPdf::generate(RooAbsPdf::GenSpec& spec) const
{
  //Int_t nEvt = spec._extended ? RooRandom::randomGenerator()->Poisson(spec._nGen) : spec._nGen ;
  //Int_t nEvt = spec._extended ? RooRandom::randomGenerator()->Poisson(spec._nGen==0?expectedEvents(spec._whatVars):spec._nGen) : spec._nGen ;
  //Int_t nEvt = spec._nGen == 0 ? RooRandom::randomGenerator()->Poisson(expectedEvents(spec._whatVars)) : spec._nGen;

  double nEvt =  spec._nGen == 0 ?  expectedEvents(spec._whatVars) : spec._nGen;

  std::unique_ptr<RooDataSet> ret{generate(*spec._genContext,spec._whatVars,spec._protoData, nEvt,false,spec._randProto,spec._resampleProto,
              spec._init,spec._extended)};
  spec._init = true ;
  return RooFit::makeOwningPtr(std::move(ret));
}





////////////////////////////////////////////////////////////////////////////////
/// Generate a new dataset containing the specified variables with
/// events sampled from our distribution.
///
/// \param[in] whatVars Generate a dataset with the variables (and categories) in this set.
/// Any variables of this PDF that are not in `whatVars` will use their
/// current values and be treated as fixed parameters.
/// \param[in] nEvents Generate the specified number of events or else try to use
/// expectedEvents() if nEvents <= 0 (default).
/// \param[in] verbose Show which generator strategies are being used.
/// \param[in] autoBinned If original distribution is binned, return bin centers and randomise weights
/// instead of generating single events.
/// \param[in] binnedTag
/// \param[in] expectedData Call setExpectedData on the genContext.
/// \param[in] extended Randomise number of events generated according to Poisson(nEvents). Only useful
/// if PDF is extended.
/// \return New dataset. Returns zero in case of an error. The caller takes ownership of the returned
/// dataset.

RooFit::OwningPtr<RooDataSet> RooAbsPdf::generate(const RooArgSet &whatVars, double nEvents, bool verbose, bool autoBinned, const char* binnedTag, bool expectedData, bool extended) const
{
  if (nEvents==0 && extendMode()==CanNotBeExtended) {
    return RooFit::makeOwningPtr(std::make_unique<RooDataSet>("emptyData","emptyData",whatVars));
  }

  // Request for binned generation
  std::unique_ptr<RooAbsGenContext> context{autoGenContext(whatVars,nullptr,nullptr,verbose,autoBinned,binnedTag)};
  if (expectedData) {
    context->setExpectedData(true) ;
  }

  std::unique_ptr<RooDataSet> generated;
  if(nullptr != context && context->isValid()) {
     generated = std::unique_ptr<RooDataSet>{context->generate(nEvents, false, extended)};
  }
  else {
    coutE(Generation)  << "RooAbsPdf::generate(" << GetName() << ") cannot create a valid context" << std::endl;
  }
  return RooFit::makeOwningPtr(std::move(generated));
}




////////////////////////////////////////////////////////////////////////////////
/// Internal method

std::unique_ptr<RooDataSet> RooAbsPdf::generate(RooAbsGenContext& context, const RooArgSet &whatVars, const RooDataSet *prototype,
            double nEvents, bool /*verbose*/, bool randProtoOrder, bool resampleProto,
            bool skipInit, bool extended) const
{
  if (nEvents==0 && (prototype==nullptr || prototype->numEntries()==0)) {
    return std::make_unique<RooDataSet>("emptyData","emptyData",whatVars);
  }

  std::unique_ptr<RooDataSet> generated;

  // Resampling implies reshuffling in the implementation
  if (resampleProto) {
    randProtoOrder=true ;
  }

  if (randProtoOrder && prototype && prototype->numEntries()!=nEvents) {
    coutI(Generation) << "RooAbsPdf::generate (Re)randomizing event order in prototype dataset (Nevt=" << nEvents << ")" << std::endl ;
    Int_t* newOrder = randomizeProtoOrder(prototype->numEntries(),Int_t(nEvents),resampleProto) ;
    context.setProtoDataOrder(newOrder) ;
    delete[] newOrder ;
  }

  if(context.isValid()) {
    generated = std::unique_ptr<RooDataSet>{context.generate(nEvents,skipInit,extended)};
  }
  else {
    coutE(Generation) << "RooAbsPdf::generate(" << GetName() << ") do not have a valid generator context" << std::endl;
  }
  return generated;
}




////////////////////////////////////////////////////////////////////////////////
/// Generate a new dataset using a prototype dataset as a model,
/// with values of the variables in `whatVars` sampled from our distribution.
///
/// \param[in] whatVars Generate for these variables.
/// \param[in] prototype Use this dataset
/// as a prototype: the new dataset will contain the same number of
/// events as the prototype (by default), and any prototype variables not in
/// whatVars will be copied into the new dataset for each generated
/// event and also used to set our PDF parameters. The user can specify a
/// number of events to generate that will override the default. The result is a
/// copy of the prototype dataset with only variables in whatVars
/// randomized. Variables in whatVars that are not in the prototype
/// will be added as new columns to the generated dataset.
/// \param[in] nEvents Number of events to generate. Defaults to 0, which means number
/// of event in prototype dataset.
/// \param[in] verbose Show which generator strategies are being used.
/// \param[in] randProtoOrder Randomise order of retrieval of events from proto dataset.
/// \param[in] resampleProto Resample from the proto dataset.
/// \return The new dataset. Returns zero in case of an error. The caller takes ownership of the
/// returned dataset.

RooFit::OwningPtr<RooDataSet> RooAbsPdf::generate(const RooArgSet &whatVars, const RooDataSet& prototype,
            Int_t nEvents, bool verbose, bool randProtoOrder, bool resampleProto) const
{
  std::unique_ptr<RooAbsGenContext> context{genContext(whatVars,&prototype,nullptr,verbose)};
  if (context) {
    return RooFit::makeOwningPtr(generate(*context,whatVars,&prototype,nEvents,verbose,randProtoOrder,resampleProto));
  }
  coutE(Generation) << "RooAbsPdf::generate(" << GetName() << ") ERROR creating generator context" << std::endl ;
  return nullptr;
}



////////////////////////////////////////////////////////////////////////////////
/// Return lookup table with randomized order for nProto prototype events.

Int_t* RooAbsPdf::randomizeProtoOrder(Int_t nProto, Int_t, bool resampleProto) const
{
  // Make output list
  Int_t* lut = new Int_t[nProto] ;

  // Randomly sample input list into output list
  if (!resampleProto) {
    // In this mode, randomization is a strict reshuffle of the order
    std::iota(lut, lut + nProto, 0); // fill the vector with 0 to nProto - 1
    // Shuffle code taken from https://en.cppreference.com/w/cpp/algorithm/random_shuffle.
    // The std::random_shuffle function was deprecated in C++17. We could have
    // used std::shuffle instead, but this is not straight-forward to use with
    // RooRandom::integer() and we didn't want to change the random number
    // generator. It might cause unwanted effects like reproducibility problems.
    for (int i = nProto-1; i > 0; --i) {
        std::swap(lut[i], lut[RooRandom::integer(i+1)]);
    }
  } else {
    // In this mode, we resample, i.e. events can be used more than once
    std::generate(lut, lut + nProto, [&]{ return RooRandom::integer(nProto); });
  }


  return lut ;
}



////////////////////////////////////////////////////////////////////////////////
/// Load generatedVars with the subset of directVars that we can generate events for,
/// and return a code that specifies the generator algorithm we will use. A code of
/// zero indicates that we cannot generate any of the directVars (in this case, nothing
/// should be added to generatedVars). Any non-zero codes will be passed to our generateEvent()
/// implementation, but otherwise its value is arbitrary. The default implementation of
/// this method returns zero. Subclasses will usually implement this method using the
/// matchArgs() methods to advertise the algorithms they provide.

Int_t RooAbsPdf::getGenerator(const RooArgSet &/*directVars*/, RooArgSet &/*generatedVars*/, bool /*staticInitOK*/) const
{
  return 0 ;
}



////////////////////////////////////////////////////////////////////////////////
/// Interface for one-time initialization to setup the generator for the specified code.

void RooAbsPdf::initGenerator(Int_t /*code*/)
{
}



////////////////////////////////////////////////////////////////////////////////
/// Interface for generation of an event using the algorithm
/// corresponding to the specified code. The meaning of each code is
/// defined by the getGenerator() implementation. The default
/// implementation does nothing.

void RooAbsPdf::generateEvent(Int_t /*code*/)
{
}



////////////////////////////////////////////////////////////////////////////////
/// Check if given observable can be safely generated using the
/// pdfs internal generator mechanism (if that existsP). Observables
/// on which a PDF depends via more than route are not safe
/// for use with internal generators because they introduce
/// correlations not known to the internal generator

bool RooAbsPdf::isDirectGenSafe(const RooAbsArg& arg) const
{
  // Arg must be direct server of self
  if (!findServer(arg.GetName())) return false ;

  // There must be no other dependency routes
  for (const auto server : _serverList) {
    if(server == &arg) continue;
    if(server->dependsOn(arg)) {
      return false ;
    }
  }

  return true ;
}


////////////////////////////////////////////////////////////////////////////////
/// Generate a new dataset containing the specified variables with events sampled from our distribution.
/// \param[in] whatVars Choose variables in which to generate events. Variables not listed here will remain
/// constant and not be used for event generation
/// \param[in] arg1,arg2,arg3,arg4,arg5,arg6 Optional RooCmdArg to change behaviour of generateBinned()
/// \return RooDataHist *, to be managed by caller.
///
/// Generate the specified number of events or expectedEvents() if not specified.
///
/// Any variables of this PDF that are not in whatVars will use their
/// current values and be treated as fixed parameters. Returns zero
/// in case of an error. The caller takes ownership of the returned
/// dataset.
///
/// The following named arguments are supported
/// | Type of CmdArg            | Effect on generation
/// |---------------------------|-----------------------
/// | `Name(const char* name)`  | Name of the output dataset
/// | `Verbose(bool flag)`      | Print informational messages during event generation
/// | `NumEvents(int nevt)`     | Generate specified number of events
/// | `Extended()`              | The actual number of events generated will be sampled from a Poisson distribution with mu=nevt. This can be *much* faster for peaked PDFs, but the number of events is not exactly what was requested.
/// | `ExpectedData()`          | Return a binned dataset _without_ statistical fluctuations (also aliased as Asimov())
///

RooFit::OwningPtr<RooDataHist> RooAbsPdf::generateBinned(const RooArgSet& whatVars, const RooCmdArg& arg1,const RooCmdArg& arg2,
                   const RooCmdArg& arg3,const RooCmdArg& arg4, const RooCmdArg& arg5,const RooCmdArg& arg6) const
{

  // Select the pdf-specific commands
  RooCmdConfig pc("RooAbsPdf::generate(" + std::string(GetName()) + ")");
  pc.defineString("dsetName","Name",0,"") ;
  pc.defineInt("verbose","Verbose",0,0) ;
  pc.defineInt("extended","Extended",0,0) ;
  pc.defineInt("nEvents","NumEvents",0,0) ;
  pc.defineDouble("nEventsD","NumEventsD",0,-1.) ;
  pc.defineInt("expectedData","ExpectedData",0,0) ;

  // Process and check varargs
  pc.process(arg1,arg2,arg3,arg4,arg5,arg6) ;
  if (!pc.ok(true)) {
    return nullptr;
  }

  // Decode command line arguments
  double nEvents = pc.getDouble("nEventsD") ;
  if (nEvents<0) {
    nEvents = pc.getInt("nEvents") ;
  }
  //bool verbose = pc.getInt("verbose") ;
  bool extended = pc.getInt("extended") ;
  bool expectedData = pc.getInt("expectedData") ;
  const char* dsetName = pc.getString("dsetName") ;

  if (extended) {
     //nEvents = (nEvents==0?Int_t(expectedEvents(&whatVars)+0.5):nEvents) ;
    nEvents = (nEvents==0 ? expectedEvents(&whatVars) :nEvents) ;
    cxcoutI(Generation) << " Extended mode active, number of events generated (" << nEvents << ") is Poisson fluctuation on "
         << GetName() << "::expectedEvents() = " << nEvents << std::endl ;
    // If Poisson fluctuation results in zero events, stop here
    if (nEvents==0) {
      return nullptr ;
    }
  } else if (nEvents==0) {
    cxcoutI(Generation) << "No number of events specified , number of events generated is "
         << GetName() << "::expectedEvents() = " << expectedEvents(&whatVars)<< std::endl ;
  }

  // Forward to appropriate implementation
  auto data = generateBinned(whatVars,nEvents,expectedData,extended);

  // Rename dataset to given name if supplied
  if (dsetName && strlen(dsetName)>0) {
    data->SetName(dsetName) ;
  }

  return data;
}




////////////////////////////////////////////////////////////////////////////////
/// Generate a new dataset containing the specified variables with
/// events sampled from our distribution.
///
/// \param[in] whatVars Variables that values should be generated for.
/// \param[in] nEvents  How many events to generate. If `nEvents <=0`, use the value returned by expectedEvents() as target.
/// \param[in] expectedData If set to true (false by default), the returned histogram returns the 'expected'
/// data sample, i.e. no statistical fluctuations are present.
/// \param[in] extended For each bin, generate Poisson(x, mu) events, where `mu` is chosen such that *on average*,
/// one would obtain `nEvents` events. This means that the true number of events will fluctuate around the desired value,
/// but the generation happens a lot faster.
/// Especially if the PDF is sharply peaked, the multinomial event generation necessary to generate *exactly* `nEvents` events can
/// be very slow.
///
/// The binning used for generation of events is the currently set binning for the variables.
/// It can e.g. be changed using
/// ```
/// x.setBins(15);
/// x.setRange(-5., 5.);
/// pdf.generateBinned(RooArgSet(x), 1000);
/// ```
///
/// Any variables of this PDF that are not in `whatVars` will use their
/// current values and be treated as fixed parameters.
/// \return RooDataHist* owned by the caller. Returns `nullptr` in case of an error.
RooFit::OwningPtr<RooDataHist> RooAbsPdf::generateBinned(const RooArgSet &whatVars, double nEvents, bool expectedData, bool extended) const
{
  // Create empty RooDataHist
  auto hist = std::make_unique<RooDataHist>("genData","genData",whatVars);

  // Scale to number of events and introduce Poisson fluctuations
  if (nEvents<=0) {
    if (!canBeExtended()) {
      coutE(InputArguments) << "RooAbsPdf::generateBinned(" << GetName() << ") ERROR: No event count provided and p.d.f does not provide expected number of events" << std::endl ;
      return nullptr;
    } else {

      // Don't round in expectedData or extended mode
      if (expectedData || extended) {
        nEvents = expectedEvents(&whatVars) ;
      } else {
        nEvents = std::round(expectedEvents(&whatVars));
      }
    }
  }

  // Sample p.d.f. distribution
  fillDataHist(hist.get(),&whatVars,1,true) ;

  vector<int> histOut(hist->numEntries()) ;
  double histMax(-1) ;
  Int_t histOutSum(0) ;
  for (int i=0 ; i<hist->numEntries() ; i++) {
    hist->get(i) ;
    if (expectedData) {

      // Expected data, multiply p.d.f by nEvents
      double w=hist->weight()*nEvents ;
      hist->set(i, w, sqrt(w));

    } else if (extended) {

      // Extended mode, set contents to Poisson(pdf*nEvents)
      double w = RooRandom::randomGenerator()->Poisson(hist->weight()*nEvents) ;
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
    Int_t nEvtExtra = std::abs(Int_t(nEvents)-histOutSum) ;
    Int_t wgt = (histOutSum>nEvents) ? -1 : 1 ;

    // Perform simple binned accept/reject procedure to get to exact event count
    std::size_t counter = 0;
    bool havePrintedInfo = false;
    while(nEvtExtra>0) {

      Int_t ibinRand = RooRandom::randomGenerator()->Integer(hist->numEntries()) ;
      hist->get(ibinRand) ;
      double ranY = RooRandom::randomGenerator()->Uniform(histMax) ;

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

      if ((counter++ > 10*nEvents || nEvents > 1.E7) && !havePrintedInfo) {
        havePrintedInfo = true;
        coutP(Generation) << "RooAbsPdf::generateBinned(" << GetName() << ") Performing costly accept/reject sampling. If this takes too long, use "
            << "extended mode to speed up the process." << std::endl;
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
    double corr = nEvents/hist->sumEntries() ;
    for (int i=0 ; i<hist->numEntries() ; i++) {
      hist->get(i) ;
      hist->set(hist->weight()*corr,sqrt(hist->weight()*corr)) ;
    }

  }

  return RooFit::makeOwningPtr(std::move(hist));
}



////////////////////////////////////////////////////////////////////////////////
/// Special generator interface for generation of 'global observables' -- for RooStats tools

RooFit::OwningPtr<RooDataSet> RooAbsPdf::generateSimGlobal(const RooArgSet& whatVars, Int_t nEvents)
{
  return generate(whatVars,nEvents) ;
}

namespace {
void removeRangeOverlap(std::vector<std::pair<double, double>>& ranges) {
  //Sort from left to right
  std::sort(ranges.begin(), ranges.end());

  for (auto it = ranges.begin(); it != ranges.end(); ++it) {
    double& startL = it->first;
    double& endL   = it->second;

    for (auto innerIt = it+1; innerIt != ranges.end(); ++innerIt) {
      const double startR = innerIt->first;
      const double endR   = innerIt->second;

      if (startL <= startR && startR <= endL) {
        //Overlapping ranges, extend left one
        endL = std::max(endL, endR);
        *innerIt = make_pair(0., 0.);
      }
    }
  }

  auto newEnd = std::remove_if(ranges.begin(), ranges.end(),
      [](const std::pair<double,double>& input){
          return input.first == input.second;
      });
  ranges.erase(newEnd, ranges.end());
}
}


////////////////////////////////////////////////////////////////////////////////
/// Plot (project) PDF on specified frame.
/// - If a PDF is plotted in an empty frame, it
/// will show a unit-normalized curve in the frame variable. When projecting a multi-
/// dimensional PDF onto the frame axis, hidden parameters are taken are taken at
/// their current value.
/// - If a PDF is plotted in a frame in which a dataset has already been plotted, it will
/// show a projection integrated over all variables that were present in the shown
/// dataset (except for the one on the x-axis). The normalization of the curve will
/// be adjusted to the event count of the plotted dataset. An informational message
/// will be printed for each projection step that is performed.
/// - If a PDF is plotted in a frame showing a dataset *after* a fit, the above happens,
/// but the PDF will be drawn and normalised only in the fit range. If this is not desired,
/// plotting and normalisation range can be overridden using Range() and NormRange() as
/// documented in the table below.
///
/// This function takes the following named arguments (for more arguments, see also
/// RooAbsReal::plotOn(RooPlot*,const RooCmdArg&,const RooCmdArg&,const RooCmdArg&,const RooCmdArg&,
/// const RooCmdArg&,const RooCmdArg&,const RooCmdArg&,const RooCmdArg&,const RooCmdArg&,
/// const RooCmdArg&) const )
///
///
/// <table>
/// <tr><th> Type of argument <th> Controlling normalisation
/// <tr><td> `NormRange(const char* name)`      <td>  Calculate curve normalization w.r.t. specified range[s].
///               See the tutorial rf212_plottingInRanges_blinding.C
///               \note Setting a Range() by default also sets a NormRange() on the same range, meaning that the
///               PDF is plotted and normalised in the same range. Overriding this can be useful if the PDF was fit
///               in limited range[s] such as side bands, `NormRange("sidebandLeft,sidebandRight")`, but the PDF
///               should be drawn in the full range, `Range("")`.
///
/// <tr><td> `Normalization(double scale, ScaleType code)`   <td>  Adjust normalization by given scale factor.
///               Interpretation of number depends on code:
///                 `RooAbsReal::Relative`: relative adjustment factor
///                 `RooAbsReal::NumEvent`: scale to match given number of events.
///
/// <tr><th> Type of argument <th> Misc control
/// <tr><td> `Name(const chat* name)`           <td>  Give curve specified name in frame. Useful if curve is to be referenced later
/// <tr><td> `Asymmetry(const RooCategory& c)`  <td>  Show the asymmetry of the PDF in given two-state category
///               \f$ \frac{F(+)-F(-)}{F(+)+F(-)} \f$ rather than the PDF projection. Category must have two
///               states with indices -1 and +1 or three states with indices -1,0 and +1.
/// <tr><td> `ShiftToZero(bool flag)`         <td>  Shift entire curve such that lowest visible point is at exactly zero.
///               Mostly useful when plotting -log(L) or \f$ \chi^2 \f$ distributions
/// <tr><td> `AddTo(const char* name, double_t wgtSelf, double_t wgtOther)`  <td>  Create a projection of this PDF onto the x-axis, but
///               instead of plotting it directly, add it to an existing curve with given name (and relative weight factors).
/// <tr><td> `Components(const char* names)`  <td>  When plotting sums of PDFs, plot only the named components (*e.g.* only
///                                                 the signal of a signal+background model).
/// <tr><td> `Components(const RooArgSet& compSet)` <td> As above, but pass a RooArgSet of the components themselves.
///
/// <tr><th> Type of argument                 <th> Projection control
/// <tr><td> `Slice(const RooArgSet& set)`     <td> Override default projection behaviour by omitting observables listed
///                                    in set from the projection, i.e. by not integrating over these.
///                                    Slicing is usually only sensible in discrete observables, by e.g. creating a slice
///                                    of the PDF at the current value of the category observable.
/// <tr><td> `Slice(RooCategory& cat, const char* label)`        <td> Override default projection behaviour by omitting the specified category
///                                    observable from the projection, i.e., by not integrating over all states of this category.
///                                    The slice is positioned at the given label value. Multiple Slice() commands can be given to specify slices
///                                    in multiple observables, e.g.
/// ```{.cpp}
///   pdf.plotOn(frame, Slice(tagCategory, "2tag"), Slice(jetCategory, "3jet"));
/// ```
/// <tr><td> `Project(const RooArgSet& set)`    <td>  Override default projection behaviour by projecting
///               over observables given in set, completely ignoring the default projection behavior. Advanced use only.
/// <tr><td> `ProjWData(const RooAbsData& d)`   <td>  Override default projection _technique_ (integration). For observables
///               present in given dataset projection of PDF is achieved by constructing an average over all observable
///               values in given set. Consult RooFit plotting tutorial for further explanation of meaning & use of this technique
/// <tr><td> `ProjWData(const RooArgSet& s, const RooAbsData& d)`   <td>  As above but only consider subset 's' of
///               observables in dataset 'd' for projection through data averaging
/// <tr><td> `ProjectionRange(const char* rn)`  <td>  When projecting the PDF onto the plot axis, it is usually integrated
///               over the full range of the invisible variables. The ProjectionRange overrides this.
///               This is useful if the PDF was fitted in a limited range in y, but it is now projected onto x. If
///               `ProjectionRange("<name of fit range>")` is passed, the projection is normalised correctly.
///
/// <tr><th> Type of argument <th> Plotting control
/// <tr><td> `LineStyle(Int_t style)`           <td>  Select line style by ROOT line style code, default is solid
/// <tr><td> `LineColor(Int_t color)`           <td>  Select line color by ROOT color code, default is blue
/// <tr><td> `LineWidth(Int_t width)`           <td>  Select line with in pixels, default is 3
/// <tr><td> `FillStyle(Int_t style)`           <td>  Select fill style, default is not filled. If a filled style is selected,
///                                                 also use VLines() to add vertical downward lines at end of curve to ensure proper closure
/// <tr><td> `FillColor(Int_t color)`           <td>  Select fill color by ROOT color code
/// <tr><td> `Range(const char* name)`          <td>  Only draw curve in range defined by given name. Multiple comma-separated ranges can be given.
///                                                   An empty string "" or `nullptr` means to use the default range of the variable.
/// <tr><td> `Range(double lo, double hi)`      <td>  Only draw curve in specified range
/// <tr><td> `VLines()`                         <td>  Add vertical lines to y=0 at end points of curve
/// <tr><td> `Precision(double eps)`          <td>  Control precision of drawn curve w.r.t to scale of plot, default is 1e-3. A higher precision will
///    result in more and more densely spaced curve points. A negative precision value will disable
///    adaptive point spacing and restrict sampling to the grid point of points defined by the binning
///    of the plotted observable (recommended for expensive functions such as profile likelihoods)
/// <tr><td> `Invisible(bool flag)`           <td>  Add curve to frame, but do not display. Useful in combination AddTo()
/// <tr><td> `VisualizeError(const RooFitResult& fitres, double Z=1, bool linearMethod=true)`
///                                  <td> Visualize the uncertainty on the parameters, as given in fitres, at 'Z' sigma.
///                                       The linear method is fast but may not be accurate in the presence of strong correlations (~>0.9) and at Z>2 due to linear and Gaussian approximations made.
///                                       Intervals from the sampling method can be asymmetric, and may perform better in the presence of strong correlations, but may take (much) longer to calculate
///                                  \note To include the uncertainty from the expected number of events,
///                                        the Normalization() argument with `ScaleType` `RooAbsReal::RelativeExpected` has to be passed, e.g.
/// ```{.cpp}
///   pdf.plotOn(frame, VisualizeError(fitResult), Normalization(1.0, RooAbsReal::RelativeExpected));
/// ```
///
/// <tr><td> `VisualizeError(const RooFitResult& fitres, const RooArgSet& param, double Z=1, bool linearMethod=true)`
///                                  <td> Visualize the uncertainty on the subset of parameters 'param', as given in fitres, at 'Z' sigma
/// </table>

RooPlot* RooAbsPdf::plotOn(RooPlot* frame, RooLinkedList& cmdList) const
{

  // Pre-processing if p.d.f. contains a fit range and there is no command specifying one,
  // add a fit range as default range
  std::unique_ptr<RooCmdArg> plotRange;
  std::unique_ptr<RooCmdArg> normRange2;
  if (getStringAttribute("fitrange") && !cmdList.FindObject("Range") &&
      !cmdList.FindObject("RangeWithName")) {
    plotRange.reset(static_cast<RooCmdArg*>(RooFit::Range(getStringAttribute("fitrange")).Clone()));
    cmdList.Add(plotRange.get());
  }

  if (getStringAttribute("fitrange") && !cmdList.FindObject("NormRange")) {
    normRange2.reset(static_cast<RooCmdArg*>(RooFit::NormRange(getStringAttribute("fitrange")).Clone()));
    cmdList.Add(normRange2.get());
  }

  if (plotRange || normRange2) {
    coutI(Plotting) << "RooAbsPdf::plotOn(" << GetName() << ") p.d.f was fitted in a subrange and no explicit "
          << (plotRange?"Range()":"") << ((plotRange&&normRange2)?" and ":"")
          << (normRange2?"NormRange()":"") << " was specified. Plotting / normalising in fit range. To override, do one of the following"
          << "\n\t- Clear the automatic fit range attribute: <pdf>.removeStringAttribute(\"fitrange\");"
          << "\n\t- Explicitly specify the plotting range: Range(\"<rangeName>\")."
          << "\n\t- Explicitly specify where to compute the normalisation: NormRange(\"<rangeName>\")."
          << "\n\tThe default (full) range can be denoted with Range(\"\") / NormRange(\"\")."<< std::endl ;
  }

  // Sanity checks
  if (plotSanityChecks(frame)) return frame ;

  // Select the pdf-specific commands
  RooCmdConfig pc("RooAbsPdf::plotOn(" + std::string(GetName()) + ")");
  pc.defineDouble("scaleFactor","Normalization",0,1.0) ;
  pc.defineInt("scaleType","Normalization",0,Relative) ;
  pc.defineSet("compSet","SelectCompSet",0) ;
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
  if (!pc.ok(true)) {
    return frame ;
  }

  // Decode command line arguments
  ScaleType stype = (ScaleType) pc.getInt("scaleType") ;
  double scaleFactor = pc.getDouble("scaleFactor") ;
  const RooAbsCategoryLValue* asymCat = static_cast<const RooAbsCategoryLValue*>(pc.getObject("asymCat")) ;
  const char* compSpec = pc.getString("compSpec") ;
  const RooArgSet* compSet = pc.getSet("compSet");
  bool haveCompSel = ( (compSpec && strlen(compSpec)>0) || compSet) ;

  // Suffix for curve name
  std::string nameSuffix ;
  if (compSpec && strlen(compSpec)>0) {
    nameSuffix.append("_Comp[") ;
    nameSuffix.append(compSpec) ;
    nameSuffix.append("]") ;
  } else if (compSet) {
    nameSuffix += "_Comp[" + compSet->contentsString() + "]";
  }

  // Remove PDF-only commands from command list
  RooCmdConfig::stripCmdList(cmdList,"SelectCompSet,SelectCompSpec") ;

  // Adjust normalization, if so requested
  if (asymCat) {
    RooCmdArg cnsuffix("CurveNameSuffix",0,0,0,0,nameSuffix.c_str(),nullptr,nullptr,nullptr) ;
    cmdList.Add(&cnsuffix);
    return  RooAbsReal::plotOn(frame,cmdList) ;
  }

  // More sanity checks
  double nExpected(1) ;
  if (stype==RelativeExpected) {
    if (!canBeExtended()) {
      coutE(Plotting) << "RooAbsPdf::plotOn(" << GetName()
            << "): ERROR the 'Expected' scale option can only be used on extendable PDFs" << std::endl ;
      return frame ;
    }
    frame->updateNormVars(*frame->getPlotVar()) ;
    nExpected = expectedEvents(frame->getNormVars()) ;
  }

  if (stype != Raw) {

    if (frame->getFitRangeNEvt() && stype==Relative) {

      bool hasCustomRange(false);
      bool adjustNorm(false);

      std::vector<pair<double,double> > rangeLim;

      // Retrieve plot range to be able to adjust normalization to data
      if (pc.hasProcessed("Range")) {

        double rangeLo = pc.getDouble("rangeLo") ;
        double rangeHi = pc.getDouble("rangeHi") ;
        rangeLim.push_back(make_pair(rangeLo,rangeHi)) ;
        adjustNorm = pc.getInt("rangeAdjustNorm") ;
        hasCustomRange = true ;

        coutI(Plotting) << "RooAbsPdf::plotOn(" << GetName() << ") only plotting range ["
            << rangeLo << "," << rangeHi << "]" ;
        if (!pc.hasProcessed("NormRange")) {
          ccoutI(Plotting) << ", curve is normalized to data in " << (adjustNorm?"given":"full") << " range" << std::endl ;
        } else {
          ccoutI(Plotting) << std::endl ;
        }

        nameSuffix.append(Form("_Range[%f_%f]",rangeLo,rangeHi)) ;

      } else if (pc.hasProcessed("RangeWithName")) {

        for (const std::string& rangeNameToken : ROOT::Split(pc.getString("rangeName", "", false), ",")) {
          const char* thisRangeName = rangeNameToken.empty() ? nullptr : rangeNameToken.c_str();
          if (thisRangeName && !frame->getPlotVar()->hasRange(thisRangeName)) {
            coutE(Plotting) << "Range '" << rangeNameToken << "' not defined for variable '"
                << frame->getPlotVar()->GetName() << "'. Ignoring ..." << std::endl;
            continue;
          }
          rangeLim.push_back(frame->getPlotVar()->getRange(thisRangeName));
        }
        adjustNorm = pc.getInt("rangeWNAdjustNorm") ;
        hasCustomRange = true ;

        coutI(Plotting) << "RooAbsPdf::plotOn(" << GetName() << ") only plotting range '" << pc.getString("rangeName", "", false) << "'" ;
        if (!pc.hasProcessed("NormRange")) {
          ccoutI(Plotting) << ", curve is normalized to data in " << (adjustNorm?"given":"full") << " range" << std::endl ;
        } else {
          ccoutI(Plotting) << std::endl ;
        }

        nameSuffix.append("_Range[" + std::string(pc.getString("rangeName")) + "]");
      }
      // Specification of a normalization range override those in a regular range
      if (pc.hasProcessed("NormRange")) {
        rangeLim.clear();
        for (const auto& rangeNameToken : ROOT::Split(pc.getString("normRangeName", "", false), ",")) {
          const char* thisRangeName = rangeNameToken.empty() ? nullptr : rangeNameToken.c_str();
          if (thisRangeName && !frame->getPlotVar()->hasRange(thisRangeName)) {
            coutE(Plotting) << "Range '" << rangeNameToken << "' not defined for variable '"
                << frame->getPlotVar()->GetName() << "'. Ignoring ..." << std::endl;
            continue;
          }
          rangeLim.push_back(frame->getPlotVar()->getRange(thisRangeName));
        }
        adjustNorm = true ;
        hasCustomRange = true ;
        coutI(Plotting) << "RooAbsPdf::plotOn(" << GetName() << ") p.d.f. curve is normalized using explicit choice of ranges '" << pc.getString("normRangeName", "", false) << "'" << std::endl ;

        nameSuffix.append("_NormRange[" + std::string(pc.getString("rangeName")) + "]");

      }

      if (hasCustomRange && adjustNorm) {
        // If overlapping ranges were given, remove them now
        const std::size_t oldSize = rangeLim.size();
        removeRangeOverlap(rangeLim);

        if (oldSize != rangeLim.size() && !pc.hasProcessed("NormRange")) {
          // User gave overlapping ranges. This leads to double-counting events and integrals, and must
          // therefore be avoided. If a NormRange has been given, the overlap is already gone.
          // It's safe to plot even with overlap now.
          coutE(Plotting) << "Requested plot/integration ranges overlap. For correct plotting, new ranges "
              "will be defined." << std::endl;
          auto plotVar = dynamic_cast<RooRealVar*>(frame->getPlotVar());
          assert(plotVar);
          std::string rangesNoOverlap;
          for (auto it = rangeLim.begin(); it != rangeLim.end(); ++it) {
            std::stringstream rangeName;
            rangeName << "Remove_overlap_range_" << it - rangeLim.begin();
            plotVar->setRange(rangeName.str().c_str(), it->first, it->second);
            if (!rangesNoOverlap.empty())
              rangesNoOverlap += ",";
            rangesNoOverlap += rangeName.str();
          }

          auto rangeArg = static_cast<RooCmdArg*>(cmdList.FindObject("RangeWithName"));
          if (rangeArg) {
            rangeArg->setString(0, rangesNoOverlap.c_str());
          } else {
            plotRange = std::make_unique<RooCmdArg>(RooFit::Range(rangesNoOverlap.c_str()));
            cmdList.Add(plotRange.get());
          }
        }

        double rangeNevt(0) ;
        for (const auto& riter : rangeLim) {
          double nevt= frame->getFitRangeNEvt(riter.first, riter.second);
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
  tmp.setInt(1,1) ; // Flag this normalization command as created for internal use (so that VisualizeError can strip it)
  cmdList.Add(&tmp) ;

  // Was a component selected requested
  if (haveCompSel) {

    // Get complete set of tree branch nodes
    RooArgSet branchNodeSet ;
    branchNodeServerList(&branchNodeSet) ;

    // Discard any non-RooAbsReal nodes
    for (const auto arg : branchNodeSet) {
      if (!dynamic_cast<RooAbsReal*>(arg)) {
        branchNodeSet.remove(*arg) ;
      }
    }

    // Obtain direct selection
    std::unique_ptr<RooArgSet> dirSelNodes;
    if (compSet) {
      dirSelNodes = std::unique_ptr<RooArgSet>{branchNodeSet.selectCommon(*compSet)};
    } else {
      dirSelNodes = std::unique_ptr<RooArgSet>{branchNodeSet.selectByName(compSpec)};
    }
    if (!dirSelNodes->empty()) {
      coutI(Plotting) << "RooAbsPdf::plotOn(" << GetName() << ") directly selected PDF components: " << *dirSelNodes << std::endl ;

      // Do indirect selection and activate both
      plotOnCompSelect(dirSelNodes.get());
    } else {
      if (compSet) {
   coutE(Plotting) << "RooAbsPdf::plotOn(" << GetName() << ") ERROR: component selection set " << *compSet << " does not match any components of p.d.f." << std::endl ;
      } else {
   coutE(Plotting) << "RooAbsPdf::plotOn(" << GetName() << ") ERROR: component selection expression '" << compSpec << "' does not select any components of p.d.f." << std::endl ;
      }
      return nullptr ;
    }
  }


  RooCmdArg cnsuffix("CurveNameSuffix",0,0,0,0,nameSuffix.c_str(),nullptr,nullptr,nullptr) ;
  cmdList.Add(&cnsuffix);

  RooPlot* ret =  RooAbsReal::plotOn(frame,cmdList) ;

  // Restore selection status ;
  if (haveCompSel) plotOnCompSelect(nullptr) ;

  return ret ;
}


//_____________________________________________________________________________
/// Plot oneself on 'frame'. In addition to features detailed in  RooAbsReal::plotOn(),
/// the scale factor for a PDF can be interpreted in three different ways. The interpretation
/// is controlled by ScaleType
/// ```
///  Relative  -  Scale factor is applied on top of PDF normalization scale factor
///  NumEvent  -  Scale factor is interpreted as a number of events. The surface area
///               under the PDF curve will match that of a histogram containing the specified
///               number of event
///  Raw       -  Scale factor is applied to the raw (projected) probability density.
///               Not too useful, option provided for completeness.
/// ```
// coverity[PASS_BY_VALUE]
RooPlot* RooAbsPdf::plotOn(RooPlot *frame, PlotOpt o) const
{

  // Sanity checks
  if (plotSanityChecks(frame)) return frame ;

  // More sanity checks
  double nExpected(1) ;
  if (o.stype==RelativeExpected) {
    if (!canBeExtended()) {
      coutE(Plotting) << "RooAbsPdf::plotOn(" << GetName()
            << "): ERROR the 'Expected' scale option can only be used on extendable PDFs" << std::endl ;
      return frame ;
    }
    frame->updateNormVars(*frame->getPlotVar()) ;
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




////////////////////////////////////////////////////////////////////////////////
/// The following named arguments are supported
/// <table>
/// <tr><th> Type of CmdArg                     <th> Effect on parameter box
/// <tr><td> `Parameters(const RooArgSet& param)` <td>  Only the specified subset of parameters will be shown. By default all non-constant parameters are shown.
/// <tr><td> `ShowConstants(bool flag)`         <td>  Also display constant parameters
/// <tr><td> `Format(const char* what,...)`       <td>  Parameter formatting options.
///   | Parameter              | Format
///   | ---------------------- | --------------------------
///   | `const char* what`     |  Controls what is shown. "N" adds name (alternatively, "T" adds the title), "E" adds error, "A" shows asymmetric error, "U" shows unit, "H" hides the value
///   | `FixedPrecision(int n)`|  Controls precision, set fixed number of digits
///   | `AutoPrecision(int n)` |  Controls precision. Number of shown digits is calculated from error + n specified additional digits (1 is sensible default)
/// <tr><td> `Label(const chat* label)`           <td>  Add label to parameter box. Use `\n` for multi-line labels.
/// <tr><td> `Layout(double xmin, double xmax, double ymax)` <td>  Specify relative position of left/right side of box and top of box.
///                                                                      Coordinates are given as position on the pad between 0 and 1.
///                                                                      The lower end of the box is calculated automatically from the number of lines in the box.
/// </table>
///
///
/// Example use:
/// ```
/// pdf.paramOn(frame, Label("fit result"), Format("NEU",AutoPrecision(1)) ) ;
/// ```
///

RooPlot* RooAbsPdf::paramOn(RooPlot* frame, const RooCmdArg& arg1, const RooCmdArg& arg2,
             const RooCmdArg& arg3, const RooCmdArg& arg4, const RooCmdArg& arg5,
             const RooCmdArg& arg6, const RooCmdArg& arg7, const RooCmdArg& arg8)
{
  // Stuff all arguments in a list
  RooLinkedList cmdList;
  cmdList.Add(const_cast<RooCmdArg*>(&arg1)) ;  cmdList.Add(const_cast<RooCmdArg*>(&arg2)) ;
  cmdList.Add(const_cast<RooCmdArg*>(&arg3)) ;  cmdList.Add(const_cast<RooCmdArg*>(&arg4)) ;
  cmdList.Add(const_cast<RooCmdArg*>(&arg5)) ;  cmdList.Add(const_cast<RooCmdArg*>(&arg6)) ;
  cmdList.Add(const_cast<RooCmdArg*>(&arg7)) ;  cmdList.Add(const_cast<RooCmdArg*>(&arg8)) ;

  // Select the pdf-specific commands
  RooCmdConfig pc("RooAbsPdf::paramOn(" + std::string(GetName()) + ")");
  pc.defineString("label","Label",0,"") ;
  pc.defineDouble("xmin","Layout",0,0.65) ;
  pc.defineDouble("xmax","Layout",1,0.9) ;
  pc.defineInt("ymaxi","Layout",0,Int_t(0.9*10000)) ;
  pc.defineInt("showc","ShowConstants",0,0) ;
  pc.defineSet("params","Parameters",0,nullptr) ;
  pc.defineInt("dummy","FormatArgs",0,0) ;

  // Process and check varargs
  pc.process(cmdList) ;
  if (!pc.ok(true)) {
    return frame ;
  }

  auto formatCmd = static_cast<RooCmdArg const*>(cmdList.FindObject("FormatArgs")) ;

  const char* label = pc.getString("label") ;
  double xmin = pc.getDouble("xmin") ;
  double xmax = pc.getDouble("xmax") ;
  double ymax = pc.getInt("ymaxi") / 10000. ;
  int showc = pc.getInt("showc") ;

  // Decode command line arguments
  std::unique_ptr<RooArgSet> params{getParameters(frame->getNormVars())} ;
  if(RooArgSet* requestedParams = pc.getSet("params")) {
    params = std::unique_ptr<RooArgSet>{params->selectCommon(*requestedParams)};
  }
  paramOn(frame,*params,showc,label,xmin,xmax,ymax,formatCmd);

  return frame ;
}


////////////////////////////////////////////////////////////////////////////////
/// Add a text box with the current parameter values and their errors to the frame.
/// Observables of this PDF appearing in the 'data' dataset will be omitted.
///
/// An optional label will be inserted if passed. Multi-line labels can be generated
/// by adding `\n` to the label string. Use 'sigDigits'
/// to modify the default number of significant digits printed. The 'xmin,xmax,ymax'
/// values specify the initial relative position of the text box in the plot frame.

RooPlot* RooAbsPdf::paramOn(RooPlot* frame, const RooArgSet& params, bool showConstants, const char *label,
             double xmin, double xmax ,double ymax, const RooCmdArg* formatCmd)
{

  // parse the options
  bool showLabel= (label != nullptr && strlen(label) > 0);

  // calculate the box's size, adjusting for constant parameters

  double ymin(ymax);
  double dy(0.06);
  for (const auto param : params) {
    auto var = static_cast<RooRealVar*>(param);
    if(showConstants || !var->isConstant()) ymin-= dy;
  }

  std::string labelString = label;
  unsigned int numLines = std::count(labelString.begin(), labelString.end(), '\n') + 1;
  if (showLabel) ymin -= numLines * dy;

  // create the box and set its options
  TPaveText *box= new TPaveText(xmin,ymax,xmax,ymin,"BRNDC");
  if(!box) return nullptr;
  box->SetName((std::string(GetName()) + "_paramBox").c_str());
  box->SetFillColor(0);
  box->SetBorderSize(0);
  box->SetTextAlign(12);
  box->SetTextSize(0.04F);
  box->SetFillStyle(0);

  for (const auto param : params) {
    auto var = static_cast<const RooRealVar*>(param);
    if(var->isConstant() && !showConstants) continue;

    box->AddText((formatCmd ? var->format(*formatCmd) : var->format(2, "NELU")).c_str());
  }

  // add the optional label if specified
  if (showLabel) {
    for (const auto& line : ROOT::Split(label, "\n")) {
      box->AddText(line.c_str());
    }
  }

  // Add box to frame
  frame->addObject(box) ;

  return frame ;
}




////////////////////////////////////////////////////////////////////////////////
/// Return expected number of events from this p.d.f for use in extended
/// likelihood calculations. This default implementation returns zero

double RooAbsPdf::expectedEvents(const RooArgSet*) const
{
  return 0 ;
}



////////////////////////////////////////////////////////////////////////////////
/// Change global level of verbosity for p.d.f. evaluations

void RooAbsPdf::verboseEval(Int_t stat)
{
  _verboseEval = stat ;
}



////////////////////////////////////////////////////////////////////////////////
/// Return global level of verbosity for p.d.f. evaluations

Int_t RooAbsPdf::verboseEval()
{
  return _verboseEval ;
}



////////////////////////////////////////////////////////////////////////////////
/// Destructor of normalization cache element. If this element
/// provides the 'current' normalization stored in RooAbsPdf::_norm
/// zero _norm pointer here before object pointed to is deleted here

RooAbsPdf::CacheElem::~CacheElem()
{
  // Zero _norm pointer in RooAbsPdf if it is points to our cache payload
  if (_owner) {
    RooAbsPdf* pdfOwner = static_cast<RooAbsPdf*>(_owner) ;
    if (pdfOwner->_norm == _norm.get()) {
      pdfOwner->_norm = nullptr ;
    }
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Return a p.d.f that represent a projection of this p.d.f integrated over given observables

RooAbsPdf* RooAbsPdf::createProjection(const RooArgSet& iset)
{
  // Construct name for new object
  std::string name = std::string{GetName()} + "_Proj[" + RooHelpers::getColonSeparatedNameString(iset, ',') + "]";

  // Return projected p.d.f.
  return new RooProjectedPdf(name.c_str(),name.c_str(),*this,iset) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Create a cumulative distribution function of this p.d.f in terms
/// of the observables listed in iset. If no nset argument is given
/// the c.d.f normalization is constructed over the integrated
/// observables, so that its maximum value is precisely 1. It is also
/// possible to choose a different normalization for
/// multi-dimensional p.d.f.s: eg. for a pdf f(x,y,z) one can
/// construct a partial cdf c(x,y) that only when integrated itself
/// over z results in a maximum value of 1. To construct such a cdf pass
/// z as argument to the optional nset argument

RooFit::OwningPtr<RooAbsReal> RooAbsPdf::createCdf(const RooArgSet& iset, const RooArgSet& nset)
{
  return createCdf(iset,RooFit::SupNormSet(nset)) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Create an object that represents the integral of the function over one or more observables listed in `iset`.
/// The actual integration calculation is only performed when the return object is evaluated. The name
/// of the integral object is automatically constructed from the name of the input function, the variables
/// it integrates and the range integrates over
///
/// The following named arguments are accepted
/// | Type of CmdArg    |    Effect on CDF
/// | ---------------------|-------------------
/// | SupNormSet(const RooArgSet&)         | Observables over which should be normalized _in addition_ to the integration observables
/// | ScanNumCdf()                         | Apply scanning technique if cdf integral involves numeric integration [ default ]
/// | ScanAllCdf()                         | Always apply scanning technique
/// | ScanNoCdf()                          | Never apply scanning technique
/// | ScanParameters(Int_t nbins, Int_t intOrder) | Parameters for scanning technique of making CDF: number of sampled bins and order of interpolation applied on numeric cdf

RooFit::OwningPtr<RooAbsReal> RooAbsPdf::createCdf(const RooArgSet& iset, const RooCmdArg& arg1, const RooCmdArg& arg2,
             const RooCmdArg& arg3, const RooCmdArg& arg4, const RooCmdArg& arg5,
             const RooCmdArg& arg6, const RooCmdArg& arg7, const RooCmdArg& arg8)
{
  // Define configuration for this method
  RooCmdConfig pc("RooAbsReal::createCdf(" + std::string(GetName()) + ")");
  pc.defineSet("supNormSet","SupNormSet",0,nullptr) ;
  pc.defineInt("numScanBins","ScanParameters",0,1000) ;
  pc.defineInt("intOrder","ScanParameters",1,2) ;
  pc.defineInt("doScanNum","ScanNumCdf",0,1) ;
  pc.defineInt("doScanAll","ScanAllCdf",0,0) ;
  pc.defineInt("doScanNon","ScanNoCdf",0,0) ;
  pc.defineMutex("ScanNumCdf","ScanAllCdf","ScanNoCdf") ;

  // Process & check varargs
  pc.process(arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8) ;
  if (!pc.ok(true)) {
    return nullptr ;
  }

  // Extract values from named arguments
  const RooArgSet* snset = pc.getSet("supNormSet",nullptr);
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
    std::unique_ptr<RooAbsReal> tmp{createIntegral(iset)} ;
    Int_t isNum= !static_cast<RooRealIntegral&>(*tmp).numIntRealVars().empty();

    if (isNum) {
      coutI(NumIntegration) << "RooAbsPdf::createCdf(" << GetName() << ") integration over observable(s) " << iset << " involves numeric integration," << std::endl
             << "      constructing cdf though numeric integration of sampled pdf in " << numScanBins << " bins and applying order "
             << intOrder << " interpolation on integrated histogram." << std::endl
             << "      To override this choice of technique use argument ScanNone(), to change scan parameters use ScanParameters(nbins,order) argument" << std::endl ;
    }

    return isNum ? createScanCdf(iset,nset,numScanBins,intOrder) : createIntRI(iset,nset) ;
  }
  return nullptr ;
}

RooFit::OwningPtr<RooAbsReal> RooAbsPdf::createScanCdf(const RooArgSet& iset, const RooArgSet& nset, Int_t numScanBins, Int_t intOrder)
{
  string name = string(GetName()) + "_NUMCDF_" + integralNameSuffix(iset,&nset).Data() ;
  RooRealVar* ivar = static_cast<RooRealVar*>(iset.first()) ;
  ivar->setBins(numScanBins,"numcdf") ;
  auto ret = std::make_unique<RooNumCdf>(name.c_str(),name.c_str(),*this,*ivar,"numcdf");
  ret->setInterpolationOrder(intOrder) ;
  return RooFit::makeOwningPtr<RooAbsReal>(std::move(ret));
}




////////////////////////////////////////////////////////////////////////////////
/// This helper function finds and collects all constraints terms of all component p.d.f.s
/// and returns a RooArgSet with all those terms.

RooArgSet* RooAbsPdf::getAllConstraints(const RooArgSet& observables, RooArgSet& constrainedParams,
                                        bool stripDisconnected) const
{
  RooArgSet constraints;
  RooArgSet pdfParams;

  std::unique_ptr<RooArgSet> comps(getComponents());
  for (const auto arg : *comps) {
    auto pdf = dynamic_cast<const RooAbsPdf*>(arg) ;
    if (pdf && !constraints.find(pdf->GetName())) {
      std::unique_ptr<RooArgSet> compRet(
              pdf->getConstraints(observables,constrainedParams, pdfParams));
      if (compRet) {
        constraints.add(*compRet,false) ;
      }
    }
  }

  RooArgSet conParams;

  // Strip any constraints that are completely decoupled from the other product terms
  RooArgSet* finalConstraints = new RooArgSet("AllConstraints") ;
  for(auto * pdf : static_range_cast<RooAbsPdf*>(constraints)) {

    RooArgSet tmp;
    pdf->getParameters(nullptr, tmp);
    conParams.add(tmp,true) ;

    if (pdf->dependsOnValue(pdfParams) || !stripDisconnected) {
      finalConstraints->add(*pdf) ;
    } else {
      coutI(Minimization) << "RooAbsPdf::getAllConstraints(" << GetName() << ") omitting term " << pdf->GetName()
           << " as constraint term as it does not share any parameters with the other pdfs in product. "
           << "To force inclusion in likelihood, add an explicit Constrain() argument for the target parameter" << std::endl ;
    }
  }

  // Now remove from constrainedParams all parameters that occur exclusively in constraint term and not in regular pdf term

  RooArgSet cexl;
  conParams.selectCommon(constrainedParams, cexl);
  cexl.remove(pdfParams,true,true) ;
  constrainedParams.remove(cexl,true,true) ;

  return finalConstraints ;
}


////////////////////////////////////////////////////////////////////////////////
/// Returns the default numeric MC generator configuration for all RooAbsReals

RooNumGenConfig* RooAbsPdf::defaultGeneratorConfig()
{
  return &RooNumGenConfig::defaultConfig() ;
}


////////////////////////////////////////////////////////////////////////////////
/// Returns the specialized integrator configuration for _this_ RooAbsReal.
/// If this object has no specialized configuration, a null pointer is returned

RooNumGenConfig* RooAbsPdf::specialGeneratorConfig() const
{
  return _specGeneratorConfig.get();
}



////////////////////////////////////////////////////////////////////////////////
/// Returns the specialized integrator configuration for _this_ RooAbsReal.
/// If this object has no specialized configuration, a null pointer is returned,
/// unless createOnTheFly is true in which case a clone of the default integrator
/// configuration is created, installed as specialized configuration, and returned

RooNumGenConfig* RooAbsPdf::specialGeneratorConfig(bool createOnTheFly)
{
  if (!_specGeneratorConfig && createOnTheFly) {
    _specGeneratorConfig = std::make_unique<RooNumGenConfig>(*defaultGeneratorConfig()) ;
  }
  return _specGeneratorConfig.get();
}



////////////////////////////////////////////////////////////////////////////////
/// Return the numeric MC generator configuration used for this object. If
/// a specialized configuration was associated with this object, that configuration
/// is returned, otherwise the default configuration for all RooAbsReals is returned

const RooNumGenConfig* RooAbsPdf::getGeneratorConfig() const
{
  const RooNumGenConfig* config = specialGeneratorConfig() ;
  if (config) return config ;
  return defaultGeneratorConfig() ;
}



////////////////////////////////////////////////////////////////////////////////
/// Set the given configuration as default numeric MC generator
/// configuration for this object

void RooAbsPdf::setGeneratorConfig(const RooNumGenConfig& config)
{
  _specGeneratorConfig = std::make_unique<RooNumGenConfig>(config);
}



////////////////////////////////////////////////////////////////////////////////
/// Remove the specialized numeric MC generator configuration associated
/// with this object

void RooAbsPdf::setGeneratorConfig()
{
  _specGeneratorConfig.reset();
}

RooAbsPdf::GenSpec::~GenSpec() = default;


////////////////////////////////////////////////////////////////////////////////

RooAbsPdf::GenSpec::GenSpec(RooAbsGenContext* context, const RooArgSet& whatVars, RooDataSet* protoData, Int_t nGen,
             bool extended, bool randProto, bool resampleProto, TString dsetName, bool init) :
  _genContext(context), _whatVars(whatVars), _protoData(protoData), _nGen(nGen), _extended(extended),
  _randProto(randProto), _resampleProto(resampleProto), _dsetName(dsetName), _init(init)
{
}


namespace {

void sterilizeClientCaches(RooAbsArg & arg) {
  auto const& clients = arg.clients();
  for(std::size_t iClient = 0; iClient < clients.size(); ++iClient) {

    const std::size_t oldClientsSize = clients.size();
    RooAbsArg* client = clients[iClient];

    for(int iCache = 0; iCache < client->numCaches(); ++iCache) {
      if(auto cacheMgr = dynamic_cast<RooObjCacheManager*>(client->getCache(iCache))) {
        cacheMgr->sterilize();
      }
    }

    // It can happen that the objects cached by the client are also clients of
    // the arg itself! In that case, the position of the client in the client
    // list might have changed, and we need to find the new index.
    if(clients.size() != oldClientsSize) {
      auto clientIter = std::find(clients.begin(), clients.end(), client);
      if(clientIter == clients.end()) {
        throw std::runtime_error("After a clients caches were cleared, the client was gone! This should not happen.");
      }
      iClient = std::distance(clients.begin(), clientIter);
    }
  }
}

} // namespace


////////////////////////////////////////////////////////////////////////////////

void RooAbsPdf::setNormRange(const char* rangeName)
{
  _normRange = rangeName ? rangeName : "";

  // the stuff that the clients have cached may depend on the normalization range
  sterilizeClientCaches(*this);

  if (_norm) {
    _normMgr.sterilize() ;
    _norm = nullptr ;
  }
}


////////////////////////////////////////////////////////////////////////////////

void RooAbsPdf::setNormRangeOverride(const char* rangeName)
{
  _normRangeOverride = rangeName ? rangeName : "";

  // the stuff that the clients have cached may depend on the normalization range
  sterilizeClientCaches(*this);

  if (_norm) {
    _normMgr.sterilize() ;
    _norm = nullptr ;
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Hook function intercepting redirectServer calls. Discard current
/// normalization object if any server is redirected

bool RooAbsPdf::redirectServersHook(const RooAbsCollection & newServerList, bool mustReplaceAll,
                                    bool nameChange, bool isRecursiveStep)
{
  // If servers are redirected, the cached normalization integrals and
  // normalization sets are most likely invalid.
  _normMgr.sterilize();

  // Object is own by _normCacheManager that will delete object as soon as cache
  // is sterilized by server redirect
  _norm = nullptr ;

  // Similar to the situation with the normalization integral above: if a
  // server is redirected, the cached normalization set might not point to
  // the right observables anymore. We need to reset it.
  setActiveNormSet(nullptr);
  return RooAbsReal::redirectServersHook(newServerList, mustReplaceAll, nameChange, isRecursiveStep);
}


std::unique_ptr<RooAbsArg>
RooAbsPdf::compileForNormSet(RooArgSet const &normSet, RooFit::Detail::CompileContext &ctx) const
{
   if (normSet.empty() || selfNormalized()) {
      return RooAbsReal::compileForNormSet(normSet, ctx);
   }
   std::unique_ptr<RooAbsPdf> pdfClone(static_cast<RooAbsPdf *>(this->Clone()));
   ctx.compileServers(*pdfClone, normSet);

   auto newArg = std::make_unique<RooFit::Detail::RooNormalizedPdf>(*pdfClone, normSet);

   // The direct servers are this pdf and the normalization integral, which
   // don't need to be compiled further.
   for (RooAbsArg *server : newArg->servers()) {
      ctx.markAsCompiled(*server);
   }
   ctx.markAsCompiled(*newArg);
   newArg->addOwnedComponents(std::move(pdfClone));
   return newArg;
}

/// Returns an object that represents the expected number of events for a given
/// normalization set, similar to how createIntegral() returns an object that
/// returns the integral. This is used to build the computation graph for the
/// final likelihood.
std::unique_ptr<RooAbsReal> RooAbsPdf::createExpectedEventsFunc(const RooArgSet * /*nset*/) const
{
   std::stringstream errMsg;
   errMsg << "The pdf \"" << GetName() << "\" of type " << ClassName()
          << " did not overload RooAbsPdf::createExpectedEventsFunc()!";
   coutE(InputArguments) << errMsg.str() << std::endl;
   return nullptr;
}
