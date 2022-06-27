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
/** \class RooRealSumPdf
    \ingroup Roofitcore


The class RooRealSumPdf implements a PDF constructed from a sum of functions:
\f[
  \mathrm{PDF}(x) = \frac{ \sum_{i=1}^{n-1} \mathrm{coef}_i * \mathrm{func}_i(x) + \left[ 1 - \sum_{i=1}^{n-1} \mathrm{coef}_i \right] * \mathrm{func}_n(x) }
            {\sum_{i=1}^{n-1} \mathrm{coef}_i * \int \mathrm{func}_i(x)dx  + \left[ 1 - \sum_{i=1}^{n-1} \mathrm{coef}_i \right] * \int \mathrm{func}_n(x) dx }
\f]

where \f$\mathrm{coef}_i\f$ and \f$\mathrm{func}_i\f$ are RooAbsReal objects, and \f$ x \f$ is the collection of dependents.
In the present version \f$\mathrm{coef}_i\f$ may not depend on \f$ x \f$, but this limitation could be removed should the need arise.

If the number of coefficients is one less than the number of functions, the PDF is assumed to be normalised. Due to this additional constraint,
\f$\mathrm{coef}_n\f$ is computed from the other coefficients.

### Extending the PDF
If an \f$ n^\mathrm{th} \f$ coefficient is provided, the PDF **can** be used as an extended PDF, *i.e.* the total number of events will be measured in addition
to the fractions of the various functions. **This requires setting the last argument of the constructor to `true`.**
\note For the RooAddPdf, the extension happens automatically.

### Difference to RooAddPdf / RooRealSumFunc
- RooAddPdf is a PDF of PDFs, *i.e.* its components need to be normalised and non-negative.
- RooRealSumPdf is a PDF of functions, *i.e.*, its components can be negative, but their sum cannot be. The normalisation
  is computed automatically, unless the PDF is extended (see above).
- RooRealSumFunc is a sum of functions. It is neither normalised, nor need it be positive.

*/

#include "RooRealSumPdf.h"

#include "RooRealIntegral.h"
#include "RooRealProxy.h"
#include "RooRealVar.h"
#include "RooMsgService.h"
#include "RooNaNPacker.h"
#include "RunContext.h"

#include <TError.h>

#include <algorithm>
#include <memory>
#include <stdexcept>

using namespace std;

ClassImp(RooRealSumPdf);

bool RooRealSumPdf::_doFloorGlobal = false ;

////////////////////////////////////////////////////////////////////////////////
/// Default constructor
/// coverity[UNINIT_CTOR]

RooRealSumPdf::RooRealSumPdf() : _normIntMgr(this,10)
{

}



////////////////////////////////////////////////////////////////////////////////
/// Constructor with name and title

RooRealSumPdf::RooRealSumPdf(const char *name, const char *title) :
  RooAbsPdf(name,title),
  _normIntMgr(this,10),
  _funcList("!funcList","List of functions",this),
  _coefList("!coefList","List of coefficients",this),
  _extended(false),
  _doFloor(false)
{

}



////////////////////////////////////////////////////////////////////////////////
/// Construct p.d.f consisting of \f$ \mathrm{coef}_1 * \mathrm{func}_1 + (1-\mathrm{coef}_1) * \mathrm{func}_2 \f$.
/// The input coefficients and functions are allowed to be negative
/// but the resulting sum is not, which is enforced at runtime.

RooRealSumPdf::RooRealSumPdf(const char *name, const char *title,
           RooAbsReal& func1, RooAbsReal& func2, RooAbsReal& coef1) :
  RooRealSumPdf(name, title)
{
  // Special constructor with two functions and one coefficient

  _funcList.add(func1) ;
  _funcList.add(func2) ;
  _coefList.add(coef1) ;

}


////////////////////////////////////////////////////////////////////////////////
/// Constructor for a PDF from a list of functions and coefficients.
/// It implements
/// \f[
///   \sum_i \mathrm{coef}_i \cdot \mathrm{func}_i,
/// \f]
/// if \f$ N_\mathrm{coef} = N_\mathrm{func} \f$. With `extended=true`, the coefficients can take any values. With `extended=false`,
/// there is the danger of getting a degenerate minimisation problem because a PDF has to be normalised, which needs one degree
/// of freedom less.
///
/// A plain (normalised) PDF can therefore be implemented with one less coefficient. RooFit then computes
/// \f[
///   \sum_i^{N-1} \mathrm{coef}_i \cdot \mathrm{func}_i + (1 - \sum_i \mathrm{coef}_i ) \cdot \mathrm{func}_N,
/// \f]
/// if \f$ N_\mathrm{coef} = N_\mathrm{func} - 1 \f$.
///
/// All coefficients and functions are allowed to be negative
/// but the sum (*i.e.* the PDF) is not, which is enforced at runtime.
///
/// \param name Name of the PDF
/// \param title Title (for plotting)
/// \param inFuncList List of functions to sum
/// \param inCoefList List of coefficients
/// \param extended   Interpret as extended PDF (requires equal number of functions and coefficients)

RooRealSumPdf::RooRealSumPdf(const char *name, const char *title,
    const RooArgList& inFuncList, const RooArgList& inCoefList, bool extended) :
  RooRealSumPdf(name, title)
{
  _extended = extended;
  RooRealSumPdf::initializeFuncsAndCoefs(*this, inFuncList, inCoefList, _funcList, _coefList);
}


void RooRealSumPdf::initializeFuncsAndCoefs(RooAbsReal const& caller,
                                            const RooArgList& inFuncList, const RooArgList& inCoefList,
                                            RooArgList& funcList, RooArgList& coefList)
{
  const std::string className = caller.ClassName();
  const std::string constructorName = className + "::" + className;

  if (!(inFuncList.size()==inCoefList.size()+1 || inFuncList.size()==inCoefList.size())) {
    oocoutE(&caller, InputArguments) << constructorName << "(" << caller.GetName()
           << ") number of pdfs and coefficients inconsistent, must have Nfunc=Ncoef or Nfunc=Ncoef+1" << std::endl;
    throw std::invalid_argument(className + ": Number of PDFs and coefficients is inconsistent.");
  }

  // Constructor with N functions and N or N-1 coefs
  for (unsigned int i = 0; i < inCoefList.size(); ++i) {
    const auto& func = inFuncList[i];
    const auto& coef = inCoefList[i];

    if (!dynamic_cast<const RooAbsReal*>(&coef)) {
      oocoutW(&caller, InputArguments) << constructorName << "(" << caller.GetName() << ") coefficient " << coef.GetName() << " is not of type RooAbsReal, ignored" << std::endl ;
      continue ;
    }
    if (!dynamic_cast<const RooAbsReal*>(&func)) {
      oocoutW(&caller, InputArguments) << constructorName << "(" << caller.GetName() << ") func " << func.GetName() << " is not of type RooAbsReal, ignored" << std::endl ;
      continue ;
    }
    funcList.add(func) ;
    coefList.add(coef) ;
  }

  if (inFuncList.size() == inCoefList.size() + 1) {
    const auto& func = inFuncList[inFuncList.size()-1];
    if (!dynamic_cast<const RooAbsReal*>(&func)) {
      oocoutE(&caller, InputArguments) << constructorName << "(" << caller.GetName() << ") last func " << func.GetName() << " is not of type RooAbsReal, fatal error" << std::endl ;
      throw std::invalid_argument(className + ": Function passed as is not of type RooAbsReal.");
    }
    funcList.add(func);
  }

}




////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooRealSumPdf::RooRealSumPdf(const RooRealSumPdf& other, const char* name) :
  RooAbsPdf(other,name),
  _normIntMgr(other._normIntMgr,this),
  _funcList("!funcList",this,other._funcList),
  _coefList("!coefList",this,other._coefList),
  _extended(other._extended),
  _doFloor(other._doFloor)
{

}



////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooRealSumPdf::~RooRealSumPdf()
{

}





////////////////////////////////////////////////////////////////////////////////

RooAbsPdf::ExtendMode RooRealSumPdf::extendMode() const
{
  return (_extended && (_funcList.getSize()==_coefList.getSize())) ? CanBeExtended : CanNotBeExtended ;
}


double RooRealSumPdf::evaluate(RooAbsReal const& caller,
                               RooArgList const& funcList,
                               RooArgList const& coefList,
                               bool doFloor,
                               bool & hasWarnedBefore)
{
  // Do running sum of coef/func pairs, calculate lastCoef.
  double value = 0;
  double sumCoeff = 0.;
  for (unsigned int i = 0; i < funcList.size(); ++i) {
    const auto func = static_cast<RooAbsReal*>(&funcList[i]);
    const auto coef = static_cast<RooAbsReal*>(i < coefList.size() ? &coefList[i] : nullptr);
    const double coefVal = coef != nullptr ? coef->getVal() : (1. - sumCoeff);

    // Warn about degeneration of last coefficient
    if (coef == nullptr && (coefVal < 0 || coefVal > 1.)) {
      if (!hasWarnedBefore) {
        oocoutW(&caller, Eval) << caller.ClassName() << "::evaluate(" << caller.GetName()
            << ") WARNING: sum of FUNC coefficients not in range [0-1], value="
            << sumCoeff << ". This means that the PDF is not properly normalised."
            << " If the PDF was meant to be extended, provide as many coefficients as functions." << std::endl;
        hasWarnedBefore = true;
      }
      // Signal that we are in an undefined region:
      value = RooNaNPacker::packFloatIntoNaN(100.f * (coefVal < 0. ? -coefVal : coefVal - 1.));
    }

    if (func->isSelectedComp()) {
      value += func->getVal() * coefVal;
    }

    sumCoeff += coefVal;
  }

  // Introduce floor if so requested
  return value < 0 && doFloor ? 0.0 : value;
}


////////////////////////////////////////////////////////////////////////////////
/// Calculate the current value

double RooRealSumPdf::evaluate() const
{
  return evaluate(*this, _funcList, _coefList, _doFloor || _doFloorGlobal, _haveWarned);
}


void RooRealSumPdf::computeBatch(cudaStream_t* /*stream*/, double* output, size_t nEvents, RooFit::Detail::DataMap const& dataMap) const {
  // Do running sum of coef/func pairs, calculate lastCoef.
  for (unsigned int j = 0; j < nEvents; ++j) {
    output[j] = 0.0;
  }

  double sumCoeff = 0.;
  for (unsigned int i = 0; i < _funcList.size(); ++i) {
    const auto func = static_cast<RooAbsReal*>(&_funcList[i]);
    const auto coef = static_cast<RooAbsReal*>(i < _coefList.size() ? &_coefList[i] : nullptr);
    const double coefVal = coef != nullptr ? dataMap.at(coef)[0] : (1. - sumCoeff);

    if (func->isSelectedComp()) {
      auto funcValues = dataMap.at(func);
      if(funcValues.size() == 1) {
        for (unsigned int j = 0; j < nEvents; ++j) {
          output[j] += funcValues[0] * coefVal;
        }
      } else {
        for (unsigned int j = 0; j < nEvents; ++j) {
          output[j] += funcValues[j] * coefVal;
        }
      }
    }

    // Warn about degeneration of last coefficient
    if (coef == nullptr && (coefVal < 0 || coefVal > 1.)) {
      if (!_haveWarned) {
        coutW(Eval) << "RooRealSumPdf::evaluateSpan(" << GetName()
            << ") WARNING: sum of FUNC coefficients not in range [0-1], value="
            << sumCoeff << ". This means that the PDF is not properly normalised. If the PDF was meant to be extended, provide as many coefficients as functions." << endl ;
        _haveWarned = true;
      }
      // Signal that we are in an undefined region by handing back one NaN.
      output[0] = RooNaNPacker::packFloatIntoNaN(100.f * (coefVal < 0. ? -coefVal : coefVal - 1.));
    }

    sumCoeff += coefVal;
  }

  // Introduce floor if so requested
  if (_doFloor || _doFloorGlobal) {
    for (unsigned int j = 0; j < nEvents; ++j) {
      output[j] += std::max(0., output[j]);
    }
  }
}


bool RooRealSumPdf::checkObservables(RooAbsReal const& caller, RooArgSet const* nset,
                                     RooArgList const& funcList, RooArgList const& coefList)
{
  bool ret(false) ;

  for (unsigned int i=0; i < coefList.size(); ++i) {
    const auto& coef = coefList[i];
    const auto& func = funcList[i];

    if (func.observableOverlaps(nset, coef)) {
      oocoutE(&caller, InputArguments) << caller.ClassName() << "::checkObservables(" << caller.GetName()
             << "): ERROR: coefficient " << coef.GetName()
             << " and FUNC " << func.GetName() << " have one or more observables in common" << std::endl;
      ret = true ;
    }
    if (coef.dependsOn(*nset)) {
      oocoutE(&caller, InputArguments) << caller.ClassName() << "::checkObservables(" << caller.GetName()
             << "): ERROR coefficient " << coef.GetName()
             << " depends on one or more of the following observables" ; nset->Print("1") ;
      ret = true ;
    }
  }

  return ret ;
}


////////////////////////////////////////////////////////////////////////////////
/// Check if FUNC is valid for given normalization set.
/// Coefficient and FUNC must be non-overlapping, but func-coefficient
/// pairs may overlap each other.
///
/// In the present implementation, coefficients may not be observables or derive
/// from observables.

bool RooRealSumPdf::checkObservables(const RooArgSet* nset) const
{
  return checkObservables(*this, nset, _funcList, _coefList);
}


////////////////////////////////////////////////////////////////////////////////
/// Advertise that all integrals can be handled internally.

Int_t RooRealSumPdf::getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& analVars,
                    const RooArgSet* normSet2, const char* rangeName) const
{
  return getAnalyticalIntegralWN(*this, _normIntMgr, _funcList, _coefList, allVars, analVars, normSet2, rangeName);
}


Int_t RooRealSumPdf::getAnalyticalIntegralWN(RooAbsReal const& caller, RooObjCacheManager & normIntMgr,
                                             RooArgList const& funcList, RooArgList const& /*coefList*/,
                                             RooArgSet& allVars, RooArgSet& analVars,
                                             const RooArgSet* normSet2, const char* rangeName)
{
  // Handle trivial no-integration scenario
  if (allVars.empty()) return 0 ;
  if (caller.getForceNumInt()) return 0 ;

  // Select subset of allVars that are actual dependents
  analVars.add(allVars) ;
  RooArgSet* normSet = normSet2 ? caller.getObservables(normSet2) : 0 ;


  // Check if this configuration was created before
  Int_t sterileIdx(-1) ;
  auto* cache = static_cast<CacheElem*>(normIntMgr.getObj(normSet,&analVars,&sterileIdx,RooNameReg::ptr(rangeName)));
  if (cache) {
    //cout << "RooRealSumPdf("<<this<<")::getAnalyticalIntegralWN:"<<GetName()<<"("<<allVars<<","<<analVars<<","<<(normSet2?*normSet2:RooArgSet())<<","<<(rangeName?rangeName:"<none>") << " -> " << _normIntMgr.lastIndex()+1 << " (cached)" << endl;
    return normIntMgr.lastIndex()+1 ;
  }

  // Create new cache element
  cache = new CacheElem ;

  // Make list of function projection and normalization integrals
  for (const auto elm : funcList) {
    const auto func = static_cast<RooAbsReal*>(elm);

    RooAbsReal* funcInt = func->createIntegral(analVars,rangeName) ;
    if(funcInt->InheritsFrom(RooRealIntegral::Class())) ((RooRealIntegral*)funcInt)->setAllowComponentSelection(true);
    cache->_funcIntList.addOwned(*funcInt) ;
    if (normSet && normSet->getSize()>0) {
      RooAbsReal* funcNorm = func->createIntegral(*normSet) ;
      cache->_funcNormList.addOwned(*funcNorm) ;
    }
  }

  // Store cache element
  Int_t code = normIntMgr.setObj(normSet,&analVars,(RooAbsCacheElement*)cache,RooNameReg::ptr(rangeName)) ;

  if (normSet) {
    delete normSet ;
  }

  //cout << "RooRealSumPdf("<<this<<")::getAnalyticalIntegralWN:"<<GetName()<<"("<<allVars<<","<<analVars<<","<<(normSet2?*normSet2:RooArgSet())<<","<<(rangeName?rangeName:"<none>") << " -> " << code+1 << endl;
  return code+1 ;
}




////////////////////////////////////////////////////////////////////////////////
/// Implement analytical integrations by deferring integration of component
/// functions to integrators of components.

double RooRealSumPdf::analyticalIntegralWN(Int_t code, const RooArgSet* normSet2, const char* rangeName) const
{
  return analyticalIntegralWN(*this, _normIntMgr, _funcList, _coefList, code, normSet2, rangeName, _haveWarned);
}

double RooRealSumPdf::analyticalIntegralWN(RooAbsReal const& caller, RooObjCacheManager & normIntMgr,
                                           RooArgList const& funcList, RooArgList const& coefList,
                                           Int_t code, const RooArgSet* normSet2, const char* rangeName,
                                           bool hasWarnedBefore)
{
  // Handle trivial passthrough scenario
  if (code==0) return caller.getVal(normSet2) ;


  // WVE needs adaptation for rangeName feature
  auto* cache = static_cast<CacheElem*>(normIntMgr.getObjByIndex(code-1));
  if (cache==0) { // revive the (sterilized) cache
    //cout << "RooRealSumPdf("<<this<<")::analyticalIntegralWN:"<<GetName()<<"("<<code<<","<<(normSet2?*normSet2:RooArgSet())<<","<<(rangeName?rangeName:"<none>") << ": reviving cache "<< endl;
    RooArgSet vars;
    caller.getParameters(nullptr, vars);
    RooArgSet iset = normIntMgr.selectFromSet2(vars, code-1);
    RooArgSet nset = normIntMgr.selectFromSet1(vars, code-1);
    RooArgSet dummy;
    Int_t code2 = caller.getAnalyticalIntegralWN(iset,dummy,&nset,rangeName);
    R__ASSERT(code==code2); // must have revived the right (sterilized) slot...
    cache = (CacheElem*) normIntMgr.getObjByIndex(code-1) ;
    R__ASSERT(cache!=0);
  }

  double value(0) ;

  // N funcs, N-1 coefficients
  double lastCoef(1) ;
  auto funcIt = funcList.begin();
  auto funcIntIt = cache->_funcIntList.begin();
  for (const auto coefArg : coefList) {
    assert(funcIt != funcList.end());
    const auto coef = static_cast<const RooAbsReal*>(coefArg);
    const auto func = static_cast<const RooAbsReal*>(*funcIt++);
    const auto funcInt = static_cast<RooAbsReal*>(*funcIntIt++);

    double coefVal = coef->getVal(normSet2) ;
    if (coefVal) {
      assert(func);
      if (normSet2 ==0 || func->isSelectedComp()) {
        assert(funcInt);
        value += funcInt->getVal()*coefVal ;
      }
      lastCoef -= coef->getVal(normSet2) ;
    }
  }

  const bool haveLastCoef = funcList.size() == coefList.size();

  if (!haveLastCoef) {
    // Add last func with correct coefficient
    const auto func = static_cast<const RooAbsReal*>(*funcIt);
    const auto funcInt = static_cast<RooAbsReal*>(*funcIntIt);
    assert(func);

    if (normSet2 ==0 || func->isSelectedComp()) {
      assert(funcInt);
      value += funcInt->getVal()*lastCoef ;
    }

    // Warn about coefficient degeneration
    if (!hasWarnedBefore && (lastCoef<0 || lastCoef>1)) {
      oocoutW(&caller, Eval) << caller.ClassName() << "::evaluate(" << caller.GetName()
            << " WARNING: sum of FUNC coefficients not in range [0-1], value="
            << 1-lastCoef << endl ;
    }
  }

  double normVal(1) ;
  if (normSet2 && normSet2->getSize()>0) {
    normVal = 0 ;

    // N funcs, N-1 coefficients
    auto funcNormIter = cache->_funcNormList.begin();
    for (const auto coefAsArg : coefList) {
      auto coef = static_cast<RooAbsReal*>(coefAsArg);
      auto funcNorm = static_cast<RooAbsReal*>(*funcNormIter++);

      double coefVal = coef->getVal(normSet2);
      if (coefVal) {
        assert(funcNorm);
        normVal += funcNorm->getVal()*coefVal ;
      }
    }

    // Add last func with correct coefficient
    if (!haveLastCoef) {
      auto funcNorm = static_cast<RooAbsReal*>(*funcNormIter);
      assert(funcNorm);

      normVal += funcNorm->getVal()*lastCoef;
    }
  }

  return value / normVal;
}


////////////////////////////////////////////////////////////////////////////////

double RooRealSumPdf::expectedEvents(const RooArgSet* nset) const
{
  double n = getNorm(nset) ;
  if (n<0) {
    logEvalError("Expected number of events is negative") ;
  }
  return n ;
}


////////////////////////////////////////////////////////////////////////////////

std::list<double>* RooRealSumPdf::binBoundaries(RooAbsRealLValue& obs, double xlo, double xhi) const
{
  return binBoundaries(_funcList, obs, xlo, xhi);
}


std::list<double>* RooRealSumPdf::binBoundaries(RooArgList const& funcList, RooAbsRealLValue& obs, double xlo, double xhi)
{
  std::list<double>* sumBinB = nullptr;
  bool needClean(false) ;

  // Loop over components pdf
  for (auto * func : static_range_cast<RooAbsReal*>(funcList)) {

    list<double>* funcBinB = func->binBoundaries(obs,xlo,xhi) ;

    // Process hint
    if (funcBinB) {
      if (!sumBinB) {
   // If this is the first hint, then just save it
   sumBinB = funcBinB ;
      } else {

   std::list<double>* newSumBinB = new list<double>(sumBinB->size()+funcBinB->size()) ;

   // Merge hints into temporary array
   merge(funcBinB->begin(),funcBinB->end(),sumBinB->begin(),sumBinB->end(),newSumBinB->begin()) ;

   // Copy merged array without duplicates to new sumBinBArrau
   delete sumBinB ;
   delete funcBinB ;
   sumBinB = newSumBinB ;
   needClean = true ;
      }
    }
  }

  // Remove consecutive duplicates
  if (needClean) {
    list<double>::iterator new_end = unique(sumBinB->begin(),sumBinB->end()) ;
    sumBinB->erase(new_end,sumBinB->end()) ;
  }

  return sumBinB ;
}



/// Check if all components that depend on `obs` are binned.
bool RooRealSumPdf::isBinnedDistribution(const RooArgSet& obs) const
{
  return isBinnedDistribution(_funcList, obs);
}


bool RooRealSumPdf::isBinnedDistribution(RooArgList const& funcList, const RooArgSet& obs)
{
  for (const auto elm : funcList) {
    auto func = static_cast<RooAbsReal*>(elm);

    if (func->dependsOn(obs) && !func->isBinnedDistribution(obs)) {
      return false ;
    }
  }

  return true  ;
}



////////////////////////////////////////////////////////////////////////////////

std::list<double>* RooRealSumPdf::plotSamplingHint(RooAbsRealLValue& obs, double xlo, double xhi) const
{
  return plotSamplingHint(_funcList, obs, xlo, xhi);
}

std::list<double>* RooRealSumPdf::plotSamplingHint(RooArgList const& funcList, RooAbsRealLValue& obs, double xlo, double xhi)
{
  std::list<double>* sumHint = 0 ;
  bool needClean(false) ;

  // Loop over components pdf
  for (const auto elm : funcList) {
    auto func = static_cast<RooAbsReal*>(elm);

    list<double>* funcHint = func->plotSamplingHint(obs,xlo,xhi) ;

    // Process hint
    if (funcHint) {
      if (!sumHint) {

   // If this is the first hint, then just save it
   sumHint = funcHint ;

      } else {

   list<double>* newSumHint = new list<double>(sumHint->size()+funcHint->size()) ;

   // Merge hints into temporary array
   merge(funcHint->begin(),funcHint->end(),sumHint->begin(),sumHint->end(),newSumHint->begin()) ;

   // Copy merged array without duplicates to new sumHintArrau
   delete sumHint ;
   sumHint = newSumHint ;
   needClean = true ;
      }
    }
  }

  // Remove consecutive duplicates
  if (needClean) {
    sumHint->erase(std::unique(sumHint->begin(),sumHint->end()), sumHint->end()) ;
  }

  return sumHint ;
}




////////////////////////////////////////////////////////////////////////////////
/// Label OK'ed components of a RooRealSumPdf with cache-and-track

void RooRealSumPdf::setCacheAndTrackHints(RooArgSet& trackNodes)
{
  setCacheAndTrackHints(_funcList, trackNodes);
}


void RooRealSumPdf::setCacheAndTrackHints(RooArgList const& funcList, RooArgSet& trackNodes)
{
  for (const auto sarg : funcList) {
    if (sarg->canNodeBeCached()==Always) {
      trackNodes.add(*sarg) ;
      //cout << "tracking node RealSumPdf component " << sarg->ClassName() << "::" << sarg->GetName() << endl ;
    }
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Customized printing of arguments of a RooRealSumPdf to more intuitively reflect the contents of the
/// product operator construction

void RooRealSumPdf::printMetaArgs(ostream& os) const
{
  printMetaArgs(_funcList, _coefList, os);
}


void RooRealSumPdf::printMetaArgs(RooArgList const& funcList, RooArgList const& coefList, ostream& os)
{

  bool first(true) ;

  if (!coefList.empty()) {
    auto funcIter = funcList.begin();

    for (const auto coef : coefList) {
      if (!first) {
        os << " + " ;
      } else {
        first = false ;
      }
      const auto func = *(funcIter++);
      os << coef->GetName() << " * " << func->GetName();
    }

    if (funcIter != funcList.end()) {
      os << " + [%] * " << (*funcIter)->GetName() ;
    }
  } else {

    for (const auto func : funcList) {
      if (!first) {
        os << " + " ;
      } else {
        first = false ;
      }
      os << func->GetName() ;
    }
  }

  os << " " ;
}
