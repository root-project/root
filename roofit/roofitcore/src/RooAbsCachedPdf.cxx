 /*****************************************************************************
  * Project: RooFit                                                           *
  *                                                                           *
  * Copyright (c) 2000-2005, Regents of the University of California          *
  *                          and Stanford University. All rights reserved.    *
  *                                                                           *
  * Redistribution and use in source and binary forms,                        *
  * with or without modification, are permitted according to the terms        *
  * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
  *****************************************************************************/

/**
\file RooAbsCachedPdf.cxx
\class RooAbsCachedPdf
\ingroup Roofitcore

RooAbsCachedPdf is the abstract base class for p.d.f.s that need or
want to cache their evaluate() output in a RooHistPdf defined in
terms of the used observables. This base class manages the creation
and storage of all RooHistPdf cache p.d.fs and the RooDataHists
that define their shape. Implementations of RooAbsCachedPdf must
define member function fillCacheObject() which serves to fill an
already created RooDataHist with the p.d.fs function values. In
addition the member functions actualObservables() and
actualParameters() must be define which report what the actual
observables to be cached are for a given set of observables passed
by the user to getVal() and on which parameters need to be tracked
for changes to trigger a refilling of the cache histogram.
**/

#include "Riostream.h"
using namespace std ;

#include "RooFit.h"
#include "TString.h"
#include "RooAbsCachedPdf.h"
#include "RooAbsReal.h"
#include "RooMsgService.h"
#include "RooDataHist.h"
#include "RooHistPdf.h"
#include "RooGlobalFunc.h"
#include "RooRealVar.h"
#include "RooChangeTracker.h"
#include "RooExpensiveObjectCache.h"

ClassImp(RooAbsCachedPdf);



////////////////////////////////////////////////////////////////////////////////
/// Constructor

RooAbsCachedPdf::RooAbsCachedPdf(const char *name, const char *title, Int_t ipOrder) :
  RooAbsPdf(name,title),
  _cacheMgr(this,10),
  _ipOrder(ipOrder),
  _disableCache(kFALSE)
 {
 }



////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooAbsCachedPdf::RooAbsCachedPdf(const RooAbsCachedPdf& other, const char* name) :
   RooAbsPdf(other,name),
   _cacheMgr(other._cacheMgr,this),
   _ipOrder(other._ipOrder),
   _disableCache(other._disableCache)
 {
 }



////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooAbsCachedPdf::~RooAbsCachedPdf()
{
}



////////////////////////////////////////////////////////////////////////////////
/// Implementation of getVal() overriding default implementation
/// of RooAbsPdf. Return normalized value stored in cache p.d.f
/// rather than return value of evaluate() which is undefined
/// for RooAbsCachedPdf

Double_t RooAbsCachedPdf::getValV(const RooArgSet* nset) const
{
  if (_disableCache) {
    return RooAbsPdf::getValV(nset) ;
  }

  // Calculate current unnormalized value of object
  PdfCacheElem* cache = getCache(nset) ;

  Double_t value = cache->pdf()->getVal(nset) ;

  _value = value ;
  return _value ;
}



////////////////////////////////////////////////////////////////////////////////
/// Return pointer to RooHistPdf cache pdf for given choice of observables

RooAbsPdf* RooAbsCachedPdf::getCachePdf(const RooArgSet* nset) const
{
  PdfCacheElem* cache = getCache(nset) ;

  if (cache) {
    return cache->pdf() ;
  } else {
    return 0 ;
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Return pointer to RooDataHist cache histogram for given choice of observables

RooDataHist* RooAbsCachedPdf::getCacheHist(const RooArgSet* nset) const
{
  PdfCacheElem* cache = getCache(nset) ;

  if (cache) {
    return cache->hist() ;
  } else {
    return 0 ;
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Mark all bins of given cache as unitialized (value -1)

void RooAbsCachedPdf::clearCacheObject(PdfCacheElem& cache) const
{
  cache.hist()->setAllWeights(-1) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Retrieve cache object associated with given choice of observables. If cache object
/// does not exist, create and fill and register it on the fly. If recalculate=false
/// recalculation of cache contents of existing caches that are marked dirty due to
/// dependent parameter changes is suppressed.

RooAbsCachedPdf::PdfCacheElem* RooAbsCachedPdf::getCache(const RooArgSet* nset, Bool_t recalculate) const
{
  // Check if this configuration was created becfore
  Int_t sterileIdx(-1) ;
  PdfCacheElem* cache = (PdfCacheElem*) _cacheMgr.getObj(nset,0,&sterileIdx) ;

  // Check if we have a cache histogram in the global expensive object cache
  if (cache) {
    if (cache->paramTracker()->hasChanged(kTRUE) && (recalculate || !cache->pdf()->haveUnitNorm()) ) {
      cxcoutD(Eval) << "RooAbsCachedPdf::getCache(" << GetName() << ") cache " << cache << " pdf "
		    << cache->pdf()->GetName() << " requires recalculation as parameters changed" << endl ;
      fillCacheObject(*cache) ;
      cache->pdf()->setValueDirty() ;
    }
    return cache ;
  }

  // Create and fill cache
  std::unique_ptr<PdfCacheElem> cacheUniqePtr{createCache(nset)} ;
  cache = cacheUniqePtr.get(); // non-owning pointer

  // Check if we have contents registered already in global expensive object cache
  RooDataHist* htmp = (RooDataHist*) expensiveObjectCache().retrieveObject(cache->hist()->GetName(),RooDataHist::Class(),cache->paramTracker()->parameters()) ;

  if (htmp) {

    cache->hist()->reset() ;
    cache->hist()->add(*htmp) ;

  } else {

    fillCacheObject(*cache) ;

    RooDataHist* eoclone = new RooDataHist(*cache->hist()) ;
    eoclone->removeSelfFromDir() ;
    expensiveObjectCache().registerObject(GetName(),cache->hist()->GetName(),*eoclone,cache->paramTracker()->parameters()) ;

  }


  // Store this cache configuration
  Int_t code = _cacheMgr.setObj(nset,0,std::move(cacheUniqePtr),0) ;

  coutI(Caching) << "RooAbsCachedPdf::getCache(" << GetName() << ") creating new cache " << cache << " with pdf "
		 << cache->pdf()->GetName() << " for nset " << (nset?*nset:RooArgSet()) << " with code " << code ;
  if (htmp) {
    ccoutI(Caching) << " from preexisting content." ;
  }
  ccoutI(Caching) << endl ;

  return cache ;
}




////////////////////////////////////////////////////////////////////////////////
/// Constructor of cache object which owns RooDataHist cache histogram,
/// RooHistPdf pdf that represents is shape and RooChangeTracker meta
/// object that tracks changes in listed dependent parameter of cache.

RooAbsCachedPdf::PdfCacheElem::PdfCacheElem(const RooAbsCachedPdf& self, const RooArgSet* nsetIn) :
  _pdf(0), _paramTracker(0), _hist(0), _norm(0)
{
  // Create cache object itself -- Default implementation is a RooHistPdf
  RooArgSet* nset2 = self.actualObservables(nsetIn?*nsetIn:RooArgSet()) ;

  RooArgSet orderedObs ;
  if (nset2) {
    self.preferredObservableScanOrder(*nset2,orderedObs) ;
  }

  // Create RooDataHist
  TString hname = self.GetName() ;
  hname.Append("_") ;
  hname.Append(self.inputBaseName()) ;
  hname.Append("_CACHEHIST") ;
  hname.Append(self.cacheNameSuffix(orderedObs)) ;
  hname.Append(self.histNameSuffix()) ;
  _hist = new RooDataHist(hname,hname,orderedObs,self.binningName()) ;
  _hist->removeSelfFromDir() ;

  //RooArgSet* observables= self.getObservables(orderedObs) ;
  // cout << "orderedObs = " << orderedObs << " observables = " << *observables << endl ;

  // Get set of p.d.f. observable corresponding to set of histogram observables
  RooArgSet pdfObs ;
  RooArgSet pdfFinalObs ;
  TIterator* iter = orderedObs.createIterator() ;
  RooAbsArg* harg ;
  while((harg=(RooAbsArg*)iter->Next())) {
    RooAbsArg& po = self.pdfObservable(*harg) ;
    pdfObs.add(po) ;
    if (po.isFundamental()) {
      pdfFinalObs.add(po) ;
    } else {
      RooArgSet* tmp = po.getVariables() ;
      pdfFinalObs.add(*tmp) ;
      delete tmp ;
    }
  }
  delete iter ;

  // Create RooHistPdf
  TString pdfname = self.inputBaseName() ;
  pdfname.Append("_CACHE") ;
  pdfname.Append(self.cacheNameSuffix(pdfFinalObs)) ;
  // add a different name when cache is built in case nsetIn is not an empty list
  if (nsetIn && nsetIn->getSize() > 0) {
     pdfname.Append("_NORM");
     for (auto *arg : *nsetIn)
        pdfname.Append(TString::Format("_%s", arg->GetName()));
  }
  _pdf = new RooHistPdf(pdfname,pdfname,pdfObs,orderedObs,*_hist,self.getInterpolationOrder()) ;
  if (nsetIn) {
    _nset.addClone(*nsetIn) ;
  }

  // Create pseudo-object that tracks changes in parameter values

  RooArgSet* params = self.actualParameters(pdfFinalObs) ;
  params->remove(pdfFinalObs,kTRUE,kTRUE) ;

  string name= Form("%s_CACHEPARAMS",_pdf->GetName()) ;
  _paramTracker = new RooChangeTracker(name.c_str(),name.c_str(),*params,kTRUE) ;
  _paramTracker->hasChanged(kTRUE) ; // clear dirty flag as cache is up-to-date upon creation

  // Introduce formal dependency of RooHistPdf on parameters so that const optimization code
  // makes the correct decisions
  _pdf->addServerList(*params) ;

  // Set initial state of cache to dirty
  _pdf->setValueDirty() ;

  //delete observables ;
  delete params ;
  delete nset2 ;

}



////////////////////////////////////////////////////////////////////////////////
/// Construct string with unique suffix for cache objects based on
/// observable names that define cache configuration

TString RooAbsCachedPdf::cacheNameSuffix(const RooArgSet& nset) const
{
  TString name ;
  name.Append("_Obs[") ;
  if (nset.getSize()>0) {
    TIterator* iter = nset.createIterator() ;
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
  const char* payloadUS = payloadUniqueSuffix() ;
  if (payloadUS) {
    name.Append(payloadUS) ;
  }
  return name ;
}



////////////////////////////////////////////////////////////////////////////////
/// Change the interpolation order that is used in RooHistPdf cache
/// representation smoothing the RooDataHist shapes.

void RooAbsCachedPdf::setInterpolationOrder(Int_t order)
{
  _ipOrder = order ;

  Int_t i ;
  for (i=0 ; i<_cacheMgr.cacheSize() ; i++) {
    PdfCacheElem* cache = (PdfCacheElem*) _cacheMgr.getObjByIndex(i) ;
    if (cache) {
      cache->pdf()->setInterpolationOrder(order) ;
    }
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Returns all RooAbsArg objects contained in the cache element

RooArgList RooAbsCachedPdf::PdfCacheElem::containedArgs(Action)
{
  RooArgList ret(*_pdf) ;
  ret.add(*_paramTracker) ;
  if (_norm) ret.add(*_norm) ;
  return ret ;
}



////////////////////////////////////////////////////////////////////////////////
/// Cache element destructor

RooAbsCachedPdf::PdfCacheElem::~PdfCacheElem()
{
  if (_norm) {
    delete _norm ;
  }
  if (_pdf) {
    delete _pdf ;
  }
  if (_paramTracker) {
    delete _paramTracker ;
  }
  if (_hist) {
    delete _hist ;
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Print contents of cache when printing self as part of object tree

void RooAbsCachedPdf::PdfCacheElem::printCompactTreeHook(ostream& os, const char* indent, Int_t curElem, Int_t maxElem)
{
  if (curElem==0) {
    os << indent << "--- RooAbsCachedPdf begin cache ---" << endl ;
  }

  TString indent2(indent) ;
  os << Form("[%d] Configuration for observables ",curElem) << _nset << endl ;
  indent2 += Form("[%d] ",curElem) ;
  _pdf->printCompactTree(os,indent2) ;
  if (_norm) {
    os << Form("[%d] Norm ",curElem) ;
    _norm->printStream(os,kName|kArgs,kSingleLine) ;
  }

  if (curElem==maxElem) {
    os << indent << "--- RooAbsCachedPdf end cache --- " << endl ;
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Force RooRealIntegral to offer all our actual observable for internal
/// integration

Bool_t RooAbsCachedPdf::forceAnalyticalInt(const RooAbsArg& dep) const
{
  RooArgSet* actObs = actualObservables(dep) ;
  Bool_t ret = (actObs->getSize()>0) ;
  delete actObs ;
  return ret ;
}



////////////////////////////////////////////////////////////////////////////////
/// Advertises internal (analytical) integration capabilities. Call
/// is forwarded to RooHistPdf cache p.d.f of cache that is used for
/// given choice of observables

Int_t RooAbsCachedPdf::getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& analVars, const RooArgSet* normSet, const char* rangeName) const
{
  if (allVars.getSize()==0) {
    return 0 ;
  }

  PdfCacheElem* cache = getCache(normSet?normSet:&allVars) ;
  Int_t code = cache->pdf()->getAnalyticalIntegralWN(allVars,analVars,normSet,rangeName) ;

  if (code==0) {
    return 0 ;
  }

  RooArgSet* all = new RooArgSet ;
  RooArgSet* ana = new RooArgSet ;
  RooArgSet* nrm = new RooArgSet ;
  all->addClone(allVars) ;
  ana->addClone(analVars) ;
  if (normSet) {
    nrm->addClone(*normSet) ;
  }
  std::vector<Int_t> codeList(2);
  codeList[0] = code ;
  codeList[1] = cache->pdf()->haveUnitNorm() ? 1 : 0 ;
  Int_t masterCode = _anaReg.store(codeList,all,ana,nrm)+1 ; // takes ownership of all sets


  // Mark all observables as internally integrated
  if (cache->pdf()->haveUnitNorm()) {
    analVars.add(allVars,kTRUE) ;
  }

  return masterCode ;
}



////////////////////////////////////////////////////////////////////////////////
/// Implements internal (analytical) integration capabilities. Call
/// is forwarded to RooHistPdf cache p.d.f of cache that is used for
/// given choice of observables

Double_t RooAbsCachedPdf::analyticalIntegralWN(Int_t code, const RooArgSet* normSet, const char* rangeName) const
{
  if (code==0) {
    return getVal(normSet) ;
  }

  RooArgSet *allVars(0),*anaVars(0),*normSet2(0),*dummy(0) ;
  const std::vector<Int_t> codeList = _anaReg.retrieve(code-1,allVars,anaVars,normSet2,dummy) ;

  PdfCacheElem* cache = getCache(normSet2?normSet2:anaVars,kFALSE) ;
  Double_t ret = cache->pdf()->analyticalIntegralWN(codeList[0],normSet,rangeName) ;

  if (codeList[1]>0) {
    RooArgSet factObs(*allVars) ;
    factObs.remove(*anaVars,kTRUE,kTRUE) ;
    TIterator* iter = factObs.createIterator() ;
    RooAbsLValue* arg ;
    while((arg=dynamic_cast<RooAbsLValue*>(iter->Next()))) {
      ret *= arg->volume(rangeName) ;
    }
    delete iter ;
  }

  return ret ;
}
