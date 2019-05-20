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
to the fractions of the various functions. This requires setting the last argument of the constructor


*/

#include "RooFit.h"
#include "Riostream.h"

#include "TError.h"
#include "TIterator.h"
#include "TList.h"
#include "RooRealSumPdf.h"
#include "RooRealProxy.h"
#include "RooPlot.h"
#include "RooRealVar.h"
#include "RooAddGenContext.h"
#include "RooRealConstant.h"
#include "RooRealIntegral.h"
#include "RooMsgService.h"
#include "RooNameReg.h"

#include <algorithm>
#include <memory>

using namespace std;

ClassImp(RooRealSumPdf);
;

Bool_t RooRealSumPdf::_doFloorGlobal = kFALSE ; 

////////////////////////////////////////////////////////////////////////////////
/// Default constructor
/// coverity[UNINIT_CTOR]

RooRealSumPdf::RooRealSumPdf() 
{
  _funcIter  = _funcList.createIterator() ;
  _coefIter  = _coefList.createIterator() ;
  _extended = kFALSE ;
  _doFloor = kFALSE ;
}



////////////////////////////////////////////////////////////////////////////////
/// Constructor with name and title

RooRealSumPdf::RooRealSumPdf(const char *name, const char *title) :
  RooAbsPdf(name,title), 
  _normIntMgr(this,10),
  _haveLastCoef(kFALSE),
  _funcList("!funcList","List of functions",this),
  _coefList("!coefList","List of coefficients",this),
  _extended(kFALSE),
  _doFloor(kFALSE)
{
  _funcIter   = _funcList.createIterator() ;
  _coefIter  = _coefList.createIterator() ;
}



////////////////////////////////////////////////////////////////////////////////
/// Construct p.d.f consisting of \f$ \mathrm{coef}_1 * \mathrm{func}_1 + (1-\mathrm{coef}_1) * \mathrm{func}_2 \f$.
/// The input coefficients and functions are allowed to be negative
/// but the resulting sum is not, which is enforced at runtime.

RooRealSumPdf::RooRealSumPdf(const char *name, const char *title,
		     RooAbsReal& func1, RooAbsReal& func2, RooAbsReal& coef1) : 
  RooAbsPdf(name,title),
  _normIntMgr(this,10),
  _haveLastCoef(kFALSE),
  _funcList("!funcList","List of functions",this),
  _coefList("!coefList","List of coefficients",this),
  _extended(kFALSE),
  _doFloor(kFALSE)
{
  // Special constructor with two functions and one coefficient
  _funcIter  = _funcList.createIterator() ;
  _coefIter = _coefList.createIterator() ;

  _funcList.add(func1) ;  
  _funcList.add(func2) ;
  _coefList.add(coef1) ;

}


////////////////////////////////////////////////////////////////////////////////
/// Constructor for a PDF implementing
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

RooRealSumPdf::RooRealSumPdf(const char *name, const char *title, const RooArgList& inFuncList, const RooArgList& inCoefList, Bool_t extended) :
  RooAbsPdf(name,title),
  _normIntMgr(this,10),
  _haveLastCoef(kFALSE),
  _funcList("!funcList","List of functions",this),
  _coefList("!coefList","List of coefficients",this),
  _extended(extended),
  _doFloor(kFALSE)
{ 
  if (!(inFuncList.getSize()==inCoefList.getSize()+1 || inFuncList.getSize()==inCoefList.getSize())) {
    coutE(InputArguments) << "RooRealSumPdf::RooRealSumPdf(" << GetName() 
			  << ") number of pdfs and coefficients inconsistent, must have Nfunc=Ncoef or Nfunc=Ncoef+1" << endl ;
    assert(0) ;
  }

  _funcIter  = _funcList.createIterator() ;
  _coefIter = _coefList.createIterator() ;
 
  // Constructor with N functions and N or N-1 coefs
  TIterator* funcIter = inFuncList.createIterator() ;
  TIterator* coefIter = inCoefList.createIterator() ;
  RooAbsArg* func ;
  RooAbsArg* coef ;

  while((coef = (RooAbsArg*)coefIter->Next())) {
    func = (RooAbsArg*) funcIter->Next() ;

    if (!dynamic_cast<RooAbsReal*>(coef)) {
      coutW(InputArguments) << "RooRealSumPdf::RooRealSumPdf(" << GetName() << ") coefficient " << coef->GetName() << " is not of type RooAbsReal, ignored" << endl ;
      continue ;
    }
    if (!dynamic_cast<RooAbsReal*>(func)) {
      coutW(InputArguments) << "RooRealSumPdf::RooRealSumPdf(" << GetName() << ") func " << func->GetName() << " is not of type RooAbsReal, ignored" << endl ;
      continue ;
    }
    _funcList.add(*func) ;
    _coefList.add(*coef) ;    
  }

  func = (RooAbsReal*) funcIter->Next() ;
  if (func) {
    if (!dynamic_cast<RooAbsReal*>(func)) {
      coutE(InputArguments) << "RooRealSumPdf::RooRealSumPdf(" << GetName() << ") last func " << coef->GetName() << " is not of type RooAbsReal, fatal error" << endl ;
      assert(0) ;
    }
    _funcList.add(*func) ;  
  } else {
    _haveLastCoef = kTRUE ;
  }
  
  delete funcIter ;
  delete coefIter  ;
}




////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooRealSumPdf::RooRealSumPdf(const RooRealSumPdf& other, const char* name) :
  RooAbsPdf(other,name),
  _normIntMgr(other._normIntMgr,this),
  _haveLastCoef(other._haveLastCoef),
  _funcList("!funcList",this,other._funcList),
  _coefList("!coefList",this,other._coefList),
  _extended(other._extended),
  _doFloor(other._doFloor)
{
  _funcIter  = _funcList.createIterator() ;
  _coefIter = _coefList.createIterator() ;
}



////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooRealSumPdf::~RooRealSumPdf()
{
  delete _funcIter ;
  delete _coefIter ;
}





////////////////////////////////////////////////////////////////////////////////

RooAbsPdf::ExtendMode RooRealSumPdf::extendMode() const 
{
  return (_extended && (_funcList.getSize()==_coefList.getSize())) ? CanBeExtended : CanNotBeExtended ;
}




////////////////////////////////////////////////////////////////////////////////
/// Calculate the current value

Double_t RooRealSumPdf::evaluate() const 
{
  Double_t value(0) ;

  // Do running sum of coef/func pairs, calculate lastCoef.
      
  // N funcs, N-1 coefficients 
  Double_t lastCoef(1) ;
  auto funcIt = _funcList.begin();
  for (const auto coefArg : _coefList) {
    assert(funcIt != _funcList.end());
    auto func = static_cast<const RooAbsReal*>(*funcIt++);
    auto coef = static_cast<const RooAbsReal*>(coefArg);

    Double_t coefVal = coef->getVal() ;
    if (coefVal) {
      cxcoutD(Eval) << "RooRealSumPdf::eval(" << GetName() << ") coefVal = " << coefVal << " funcVal = " << func->IsA()->GetName() << "::" << func->GetName() << " = " << func->getVal() << endl ;
      if (func->isSelectedComp()) {
        value += func->getVal()*coefVal ;
      }
      lastCoef -= coef->getVal() ;
    }
  }
  
  if (!_haveLastCoef) {
    assert(funcIt != _funcList.end());
    // Add last func with correct coefficient
    auto func = static_cast<const RooAbsReal*>(*funcIt);
    if (func->isSelectedComp()) {
      value += func->getVal()*lastCoef ;
    }

    cxcoutD(Eval) << "RooRealSumPdf::eval(" << GetName() << ") lastCoef = " << lastCoef << " funcVal = " << func->getVal() << endl ;
    
    // Warn about coefficient degeneration
    if (lastCoef<0 || lastCoef>1) {
      coutW(Eval) << "RooRealSumPdf::evaluate(" << GetName() 
		  << ") WARNING: sum of FUNC coefficients not in range [0-1], value=" 
		  << 1-lastCoef << ". This means that the PDF is not properly normalised. If the PDF was meant to be extended, provide as many coefficients as functions." << endl ;
    } 
  }

  // Introduce floor if so requested
  if (value<0 && (_doFloor || _doFloorGlobal)) {
    value = 0 ;
  }
  
  return value ;
}




////////////////////////////////////////////////////////////////////////////////
/// Check if FUNC is valid for given normalization set.
/// Coeffient and FUNC must be non-overlapping, but func-coefficient 
/// pairs may overlap each other
///
/// In the present implementation, coefficients may not be observables or derive
/// from observables

Bool_t RooRealSumPdf::checkObservables(const RooArgSet* nset) const 
{
  Bool_t ret(kFALSE) ;

  _funcIter->Reset() ;
  _coefIter->Reset() ;
  RooAbsReal* coef ;
  RooAbsReal* func ;
  while((coef=(RooAbsReal*)_coefIter->Next())) {
    func = (RooAbsReal*)_funcIter->Next() ;
    if (func->observableOverlaps(nset,*coef)) {
      coutE(InputArguments) << "RooRealSumPdf::checkObservables(" << GetName() << "): ERROR: coefficient " << coef->GetName() 
			    << " and FUNC " << func->GetName() << " have one or more observables in common" << endl ;
      ret = kTRUE ;
    }
    if (coef->dependsOn(*nset)) {
      coutE(InputArguments) << "RooRealPdf::checkObservables(" << GetName() << "): ERROR coefficient " << coef->GetName() 
			    << " depends on one or more of the following observables" ; nset->Print("1") ;
      ret = kTRUE ;
    }
  }
  
  return ret ;
}




////////////////////////////////////////////////////////////////////////////////
/// Advertise that all integrals can be handled internally.

Int_t RooRealSumPdf::getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& analVars, 
					     const RooArgSet* normSet2, const char* rangeName) const 
{
  // Handle trivial no-integration scenario
  if (allVars.getSize()==0) return 0 ;
  if (_forceNumInt) return 0 ;

  // Select subset of allVars that are actual dependents
  analVars.add(allVars) ;
  RooArgSet* normSet = normSet2 ? getObservables(normSet2) : 0 ;


  // Check if this configuration was created before
  Int_t sterileIdx(-1) ;
  CacheElem* cache = (CacheElem*) _normIntMgr.getObj(normSet,&analVars,&sterileIdx,RooNameReg::ptr(rangeName)) ;
  if (cache) {
    //cout << "RooRealSumPdf("<<this<<")::getAnalyticalIntegralWN:"<<GetName()<<"("<<allVars<<","<<analVars<<","<<(normSet2?*normSet2:RooArgSet())<<","<<(rangeName?rangeName:"<none>") << " -> " << _normIntMgr.lastIndex()+1 << " (cached)" << endl;
    return _normIntMgr.lastIndex()+1 ;
  }
  
  // Create new cache element
  cache = new CacheElem ;

  // Make list of function projection and normalization integrals 
  _funcIter->Reset() ;
  RooAbsReal *func ;
  while((func=(RooAbsReal*)_funcIter->Next())) {
    RooAbsReal* funcInt = func->createIntegral(analVars,rangeName) ;
    if(funcInt->InheritsFrom(RooRealIntegral::Class())) ((RooRealIntegral*)funcInt)->setAllowComponentSelection(true);
    cache->_funcIntList.addOwned(*funcInt) ;
    if (normSet && normSet->getSize()>0) {
      RooAbsReal* funcNorm = func->createIntegral(*normSet) ;
      cache->_funcNormList.addOwned(*funcNorm) ;
    }
  }

  // Store cache element
  Int_t code = _normIntMgr.setObj(normSet,&analVars,(RooAbsCacheElement*)cache,RooNameReg::ptr(rangeName)) ;

  if (normSet) {
    delete normSet ;
  }

  //cout << "RooRealSumPdf("<<this<<")::getAnalyticalIntegralWN:"<<GetName()<<"("<<allVars<<","<<analVars<<","<<(normSet2?*normSet2:RooArgSet())<<","<<(rangeName?rangeName:"<none>") << " -> " << code+1 << endl;
  return code+1 ; 
}




////////////////////////////////////////////////////////////////////////////////
/// Implement analytical integrations by deferring integration of component
/// functions to integrators of components.

Double_t RooRealSumPdf::analyticalIntegralWN(Int_t code, const RooArgSet* normSet2, const char* rangeName) const 
{
  // Handle trivial passthrough scenario
  if (code==0) return getVal(normSet2) ;


  // WVE needs adaptation for rangeName feature
  CacheElem* cache = (CacheElem*) _normIntMgr.getObjByIndex(code-1) ;
  if (cache==0) { // revive the (sterilized) cache
     //cout << "RooRealSumPdf("<<this<<")::analyticalIntegralWN:"<<GetName()<<"("<<code<<","<<(normSet2?*normSet2:RooArgSet())<<","<<(rangeName?rangeName:"<none>") << ": reviving cache "<< endl;
     std::unique_ptr<RooArgSet> vars( getParameters(RooArgSet()) );
     std::unique_ptr<RooArgSet> iset(  _normIntMgr.nameSet2ByIndex(code-1)->select(*vars) );
     std::unique_ptr<RooArgSet> nset(  _normIntMgr.nameSet1ByIndex(code-1)->select(*vars) );
     RooArgSet dummy;
     Int_t code2 = getAnalyticalIntegralWN(*iset,dummy,nset.get(),rangeName);
     R__ASSERT(code==code2); // must have revived the right (sterilized) slot...
     cache = (CacheElem*) _normIntMgr.getObjByIndex(code-1) ;
     R__ASSERT(cache!=0);
  }

  RooFIter funcIntIter = cache->_funcIntList.fwdIterator() ;
  RooFIter coefIter = _coefList.fwdIterator() ;
  RooFIter funcIter = _funcList.fwdIterator() ;
  RooAbsReal *coef(0), *funcInt(0), *func(0) ;
  Double_t value(0) ;

  // N funcs, N-1 coefficients 
  Double_t lastCoef(1) ;
  while((coef=(RooAbsReal*)coefIter.next())) {
    funcInt = (RooAbsReal*)funcIntIter.next() ;
    func    = (RooAbsReal*)funcIter.next() ;
    Double_t coefVal = coef->getVal(normSet2) ;
    if (coefVal) {
      assert(func);
      if (normSet2 ==0 || func->isSelectedComp()) {
	assert(funcInt);
	value += funcInt->getVal()*coefVal ;
      }
      lastCoef -= coef->getVal(normSet2) ;
    }
  }
  
  if (!_haveLastCoef) {
    // Add last func with correct coefficient
    funcInt = (RooAbsReal*) funcIntIter.next() ;
    if (normSet2 ==0 || func->isSelectedComp()) {
      assert(funcInt);
      value += funcInt->getVal()*lastCoef ;
    }
    
    // Warn about coefficient degeneration
    if (lastCoef<0 || lastCoef>1) {
      coutW(Eval) << "RooRealSumPdf::evaluate(" << GetName() 
		  << " WARNING: sum of FUNC coefficients not in range [0-1], value=" 
		  << 1-lastCoef << endl ;
    } 
  }
  
  Double_t normVal(1) ;
  if (normSet2 && normSet2->getSize()>0) {
    normVal = 0 ;

    // N funcs, N-1 coefficients 
    RooAbsReal* funcNorm ;
    RooFIter funcNormIter = cache->_funcNormList.fwdIterator() ;
    RooFIter coefIter2 = _coefList.fwdIterator() ;
    while((coef=(RooAbsReal*)coefIter2.next())) {
      funcNorm = (RooAbsReal*)funcNormIter.next() ;
      Double_t coefVal = coef->getVal(normSet2) ;
      if (coefVal) {
	assert(funcNorm);
	normVal += funcNorm->getVal()*coefVal ;
      }
    }
    
    // Add last func with correct coefficient
    if (!_haveLastCoef) {
      funcNorm = (RooAbsReal*) funcNormIter.next() ;
      assert(funcNorm);
      normVal += funcNorm->getVal()*lastCoef ;
    }      
  }

  return value / normVal;
}


////////////////////////////////////////////////////////////////////////////////

Double_t RooRealSumPdf::expectedEvents(const RooArgSet* nset) const
{
  Double_t n = getNorm(nset) ;  
  if (n<0) {
    logEvalError("Expected number of events is negative") ;
  }
  return n ;
}


////////////////////////////////////////////////////////////////////////////////

std::list<Double_t>* RooRealSumPdf::binBoundaries(RooAbsRealLValue& obs, Double_t xlo, Double_t xhi) const
{
  list<Double_t>* sumBinB = 0 ;
  Bool_t needClean(kFALSE) ;
  
  RooFIter iter = _funcList.fwdIterator() ;
  RooAbsReal* func ;
  // Loop over components pdf
  while((func=(RooAbsReal*)iter.next())) {

    list<Double_t>* funcBinB = func->binBoundaries(obs,xlo,xhi) ;
    
    // Process hint
    if (funcBinB) {
      if (!sumBinB) {
	// If this is the first hint, then just save it
	sumBinB = funcBinB ;
      } else {
	
	list<Double_t>* newSumBinB = new list<Double_t>(sumBinB->size()+funcBinB->size()) ;

	// Merge hints into temporary array
	merge(funcBinB->begin(),funcBinB->end(),sumBinB->begin(),sumBinB->end(),newSumBinB->begin()) ;
	
	// Copy merged array without duplicates to new sumBinBArrau
	delete sumBinB ;
	delete funcBinB ;
	sumBinB = newSumBinB ;
	needClean = kTRUE ;	
      }
    }
  }

  // Remove consecutive duplicates
  if (needClean) {
    list<Double_t>::iterator new_end = unique(sumBinB->begin(),sumBinB->end()) ;
    sumBinB->erase(new_end,sumBinB->end()) ;
  }

  return sumBinB ;
}



//_____________________________________________________________________________B
Bool_t RooRealSumPdf::isBinnedDistribution(const RooArgSet& obs) const 
{
  // If all components that depend on obs are binned that so is the product
  
  RooFIter iter = _funcList.fwdIterator() ;
  RooAbsReal* func ;
  while((func=(RooAbsReal*)iter.next())) {
    if (func->dependsOn(obs) && !func->isBinnedDistribution(obs)) {
      return kFALSE ;
    }
  }
  
  return kTRUE  ;  
}





////////////////////////////////////////////////////////////////////////////////

std::list<Double_t>* RooRealSumPdf::plotSamplingHint(RooAbsRealLValue& obs, Double_t xlo, Double_t xhi) const
{
  list<Double_t>* sumHint = 0 ;
  Bool_t needClean(kFALSE) ;
  
  RooFIter iter = _funcList.fwdIterator() ;
  RooAbsReal* func ;
  // Loop over components pdf
  while((func=(RooAbsReal*)iter.next())) {

    list<Double_t>* funcHint = func->plotSamplingHint(obs,xlo,xhi) ;
    
    // Process hint
    if (funcHint) {
      if (!sumHint) {

	// If this is the first hint, then just save it
	sumHint = funcHint ;

      } else {
	
	list<Double_t>* newSumHint = new list<Double_t>(sumHint->size()+funcHint->size()) ;
	
	// Merge hints into temporary array
	merge(funcHint->begin(),funcHint->end(),sumHint->begin(),sumHint->end(),newSumHint->begin()) ;

	// Copy merged array without duplicates to new sumHintArrau
	delete sumHint ;
	sumHint = newSumHint ;
	needClean = kTRUE ;	
      }
    }
  }

  // Remove consecutive duplicates
  if (needClean) {
    list<Double_t>::iterator new_end = unique(sumHint->begin(),sumHint->end()) ;
    sumHint->erase(new_end,sumHint->end()) ;
  }

  return sumHint ;
}




////////////////////////////////////////////////////////////////////////////////
/// Label OK'ed components of a RooRealSumPdf with cache-and-track

void RooRealSumPdf::setCacheAndTrackHints(RooArgSet& trackNodes) 
{
  RooFIter siter = funcList().fwdIterator() ;
  RooAbsArg* sarg ;
  while ((sarg=siter.next())) {
    if (sarg->canNodeBeCached()==Always) {
      trackNodes.add(*sarg) ;
      //cout << "tracking node RealSumPdf component " << sarg->IsA()->GetName() << "::" << sarg->GetName() << endl ;
    }
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Customized printing of arguments of a RooRealSumPdf to more intuitively reflect the contents of the
/// product operator construction

void RooRealSumPdf::printMetaArgs(ostream& os) const 
{
  _funcIter->Reset() ;
  _coefIter->Reset() ;

  Bool_t first(kTRUE) ;
    
  RooAbsArg* coef, *func ;
  if (_coefList.getSize()!=0) { 
    while((coef=(RooAbsArg*)_coefIter->Next())) {
      if (!first) {
	os << " + " ;
      } else {
	first = kFALSE ;
      }
      func=(RooAbsArg*)_funcIter->Next() ;
      os << coef->GetName() << " * " << func->GetName() ;
    }
    func = (RooAbsArg*) _funcIter->Next() ;
    if (func) {
      os << " + [%] * " << func->GetName() ;
    }
  } else {
    
    while((func=(RooAbsArg*)_funcIter->Next())) {
      if (!first) {
	os << " + " ;
      } else {
	first = kFALSE ;
      }
      os << func->GetName() ; 
    }  
  }

  os << " " ;    
}
