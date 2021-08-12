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

#include "RooAbsCachedPdf.h"
#include "RooMsgService.h"
#include "RooDataHist.h"
#include "RooHistPdf.h"
#include "RooExpensiveObjectCache.h"

ClassImp(RooAbsCachedPdf);



////////////////////////////////////////////////////////////////////////////////
/// Constructor

RooAbsCachedPdf::RooAbsCachedPdf(const char *name, const char *title, int ipOrder) :
  RooAbsPdf(name,title),
  _cacheMgr(this,10),
  _ipOrder(ipOrder)
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
/// Implementation of getVal() overriding default implementation
/// of RooAbsPdf. Return normalized value stored in cache p.d.f
/// rather than return value of evaluate() which is undefined
/// for RooAbsCachedPdf

double RooAbsCachedPdf::getValV(const RooArgSet* nset) const
{
  if (_disableCache) {
    return RooAbsPdf::getValV(nset) ;
  }

  // Calculate current unnormalized value of object
  return _value = getCache(nset)->pdf()->getVal(nset) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Return pointer to RooHistPdf cache pdf for given choice of observables

RooAbsPdf* RooAbsCachedPdf::getCachePdf(const RooArgSet* nset) const
{
  PdfCacheElem* cache = getCache(nset) ;
  return cache ? cache->pdf() : nullptr;
}


////////////////////////////////////////////////////////////////////////////////
/// Return pointer to RooDataHist cache histogram for given choice of observables

RooDataHist* RooAbsCachedPdf::getCacheHist(const RooArgSet* nset) const
{
  PdfCacheElem* cache = getCache(nset) ;
  return cache ? cache->hist() : nullptr;
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

RooAbsCachedPdf::PdfCacheElem* RooAbsCachedPdf::getCache(const RooArgSet* nset, bool recalculate) const
{
  // Check if this configuration was created becfore
  int sterileIdx = -1 ;
  auto cache = static_cast<PdfCacheElem*>(_cacheMgr.getObj(nset,0,&sterileIdx));

  // Check if we have a cache histogram in the global expensive object cache
  if (cache) {
    if (cache->paramTracker()->hasChanged(true) && (recalculate || !cache->pdf()->haveUnitNorm()) ) {
      cxcoutD(Eval) << "RooAbsCachedPdf::getCache(" << GetName() << ") cache " << cache << " pdf "
		    << cache->pdf()->GetName() << " requires recalculation as parameters changed" << std::endl ;
      fillCacheObject(*cache) ;
      cache->pdf()->setValueDirty() ;
    }
    return cache ;
  }

  // Create and fill cache
  cache = createCache(nset) ;

  // Check if we have contents registered already in global expensive object cache
  auto htmp = static_cast<RooDataHist const*>(expensiveObjectCache().retrieveObject(cache->hist()->GetName(),RooDataHist::Class(),cache->paramTracker()->parameters()));

  if (htmp) {

    cache->hist()->reset() ;
    cache->hist()->add(*htmp) ;

  } else {

    fillCacheObject(*cache) ;

    auto eoclone = new RooDataHist(*cache->hist()) ;
    eoclone->removeSelfFromDir() ;
    expensiveObjectCache().registerObject(GetName(),cache->hist()->GetName(),*eoclone,cache->paramTracker()->parameters()) ;

  }


  // Store this cache configuration
  int code = _cacheMgr.setObj(nset,0,(static_cast<RooAbsCacheElement*>(cache)),0) ;

  coutI(Caching) << "RooAbsCachedPdf::getCache(" << GetName() << ") creating new cache " << cache << " with pdf "
		 << cache->pdf()->GetName() << " for nset " << (nset?*nset:RooArgSet()) << " with code " << code ;
  if (htmp) {
    ccoutI(Caching) << " from preexisting content." ;
  }
  ccoutI(Caching) << std::endl ;

  return cache ;
}




////////////////////////////////////////////////////////////////////////////////
/// Constructor of cache object which owns RooDataHist cache histogram,
/// RooHistPdf pdf that represents is shape and RooChangeTracker meta
/// object that tracks changes in listed dependent parameter of cache.

RooAbsCachedPdf::PdfCacheElem::PdfCacheElem(const RooAbsCachedPdf& self, const RooArgSet* nsetIn)
{
  // Create cache object itself -- Default implementation is a RooHistPdf
  std::unique_ptr<RooArgSet> nset2{self.actualObservables(nsetIn?*nsetIn:RooArgSet())};

  RooArgSet orderedObs ;
  if (nset2) {
    self.preferredObservableScanOrder(*nset2,orderedObs) ;
  }

  // Create RooDataHist
  auto hname = std::string(self.GetName()) + "_" + self.inputBaseName() + "_CACHEHIST"
               + self.cacheNameSuffix(orderedObs).c_str() + self.histNameSuffix().Data();
  _hist = std::make_unique<RooDataHist>(hname,hname,orderedObs,self.binningName()) ;
  _hist->removeSelfFromDir() ;

  //RooArgSet* observables= self.getObservables(orderedObs) ;
  // cout << "orderedObs = " << orderedObs << " observables = " << *observables << std::endl ;

  // Get set of p.d.f. observable corresponding to set of histogram observables
  RooArgSet pdfObs ;
  RooArgSet pdfFinalObs ;
  for(auto const& harg : orderedObs) {
    RooAbsArg& po = self.pdfObservable(*harg) ;
    pdfObs.add(po) ;
    if (po.isFundamental()) {
      pdfFinalObs.add(po) ;
    } else {
      pdfFinalObs.add(*std::unique_ptr<RooArgSet>(po.getVariables()));
    }
  }

  // Create RooHistPdf
  auto pdfname = std::string(self.inputBaseName()) + "_CACHE" + self.cacheNameSuffix(pdfFinalObs);
  // add a different name when cache is built in case nsetIn is not an empty list
  if (nsetIn && !nsetIn->empty()) {
     pdfname += "_NORM";
     for (auto *arg : *nsetIn)
        pdfname += std::string("_") + arg->GetName();
  }
  _pdf = std::make_unique<RooHistPdf>(pdfname.c_str(),pdfname.c_str(),pdfObs,orderedObs,*_hist,self.getInterpolationOrder()) ;
  if (nsetIn) {
    _nset.addClone(*nsetIn) ;
  }

  // Create pseudo-object that tracks changes in parameter values

  std::unique_ptr<RooArgSet> params{self.actualParameters(pdfFinalObs)};
  params->remove(pdfFinalObs,true,true) ;

  auto name = std::string(_pdf->GetName()) + "_CACHEPARAMS";
  _paramTracker = std::make_unique<RooChangeTracker>(name.c_str(),name.c_str(),*params,true) ;
  _paramTracker->hasChanged(true) ; // clear dirty flag as cache is up-to-date upon creation

  // Introduce formal dependency of RooHistPdf on parameters so that const optimization code
  // makes the correct decisions
  _pdf->addServerList(*params) ;

  // Set initial state of cache to dirty
  _pdf->setValueDirty() ;
}



////////////////////////////////////////////////////////////////////////////////
/// Construct string with unique suffix for cache objects based on
/// observable names that define cache configuration

std::string RooAbsCachedPdf::cacheNameSuffix(const RooArgSet& nset) const
{
  std::string name = "_Obs[";
  if (!nset.empty()) {
    bool first(true) ;
    for(auto const& arg : nset) {
      if (first) {
        first=false ;
      } else {
        name += ",";
      }
      name += arg->GetName();
    }
  }

  name += "]";
  if (const char* payloadUS = payloadUniqueSuffix()) {
    name += payloadUS;
  }
  return name ;
}



////////////////////////////////////////////////////////////////////////////////
/// Change the interpolation order that is used in RooHistPdf cache
/// representation smoothing the RooDataHist shapes.

void RooAbsCachedPdf::setInterpolationOrder(int order)
{
  _ipOrder = order ;

  for (int i=0 ; i<_cacheMgr.cacheSize() ; i++) {
    if (auto cache = static_cast<PdfCacheElem*>(_cacheMgr.getObjByIndex(i))) {
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
/// Print contents of cache when printing self as part of object tree

void RooAbsCachedPdf::PdfCacheElem::printCompactTreeHook(std::ostream& os, const char* indent, int curElem, int maxElem)
{
  if (curElem==0) {
    os << indent << "--- RooAbsCachedPdf begin cache ---" << std::endl ;
  }

  os << "[" << curElem << "]" << " Configuration for observables " << _nset << std::endl;
  auto indent2 = std::string(indent) + "[" + std::to_string(curElem) + "]";
  _pdf->printCompactTree(os,indent2.c_str()) ;
  if (_norm) {
    os << "[" << curElem << "] Norm ";
    _norm->printStream(os,kName|kArgs,kSingleLine) ;
  }

  if (curElem==maxElem) {
    os << indent << "--- RooAbsCachedPdf end cache --- " << std::endl ;
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Force RooRealIntegral to offer all our actual observable for internal
/// integration

bool RooAbsCachedPdf::forceAnalyticalInt(const RooAbsArg& dep) const
{
  return !std::unique_ptr<RooArgSet>{actualObservables(dep)}->empty();
}



////////////////////////////////////////////////////////////////////////////////
/// Advertises internal (analytical) integration capabilities. Call
/// is forwarded to RooHistPdf cache p.d.f of cache that is used for
/// given choice of observables

int RooAbsCachedPdf::getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& analVars, const RooArgSet* normSet, const char* rangeName) const
{
  if (allVars.empty()) {
    return 0 ;
  }

  PdfCacheElem* cache = getCache(normSet?normSet:&allVars) ;
  int code = cache->pdf()->getAnalyticalIntegralWN(allVars,analVars,normSet,rangeName) ;

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
  std::vector<int> codeList(2);
  codeList[0] = code ;
  codeList[1] = cache->pdf()->haveUnitNorm() ? 1 : 0 ;
  int masterCode = _anaReg.store(codeList,all,ana,nrm)+1 ; // takes ownership of all sets


  // Mark all observables as internally integrated
  if (cache->pdf()->haveUnitNorm()) {
    analVars.add(allVars,true) ;
  }

  return masterCode ;
}



////////////////////////////////////////////////////////////////////////////////
/// Implements internal (analytical) integration capabilities. Call
/// is forwarded to RooHistPdf cache p.d.f of cache that is used for
/// given choice of observables

double RooAbsCachedPdf::analyticalIntegralWN(int code, const RooArgSet* normSet, const char* rangeName) const
{
  if (code==0) {
    return getVal(normSet) ;
  }

  RooArgSet *allVars(0),*anaVars(0),*normSet2(0),*dummy(0) ;
  const std::vector<int> codeList = _anaReg.retrieve(code-1,allVars,anaVars,normSet2,dummy) ;

  PdfCacheElem* cache = getCache(normSet2?normSet2:anaVars,false) ;
  double ret = cache->pdf()->analyticalIntegralWN(codeList[0],normSet,rangeName) ;

  if (codeList[1]>0) {
    RooArgSet factObs(*allVars) ;
    factObs.remove(*anaVars,true,true) ;
    for(auto * arg : dynamic_range_cast<RooAbsLValue*>(factObs)) {
      ret *= arg->volume(rangeName) ;
    }
  }

  return ret ;
}
