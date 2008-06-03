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

 // -- CLASS DESCRIPTION [PDF] -- 
 // Your description goes here... 

#include "Riostream.h" 
using namespace std ;

#include "RooFit.h"
#include "TString.h"
#include "RooAbsCachedReal.h" 
#include "RooAbsReal.h" 
#include "RooMsgService.h"
#include "RooDataHist.h"
#include "RooHistFunc.h"
#include "RooChangeTracker.h"

ClassImp(RooAbsCachedReal) 



RooAbsCachedReal::RooAbsCachedReal(const char *name, const char *title, Int_t ipOrder) :
  RooAbsReal(name,title), 
  _cacheMgr(this,10),
  _ipOrder(ipOrder),
  _disableCache(kFALSE)
 { 
 } 


RooAbsCachedReal::RooAbsCachedReal(const RooAbsCachedReal& other, const char* name) :  
   RooAbsReal(other,name), 
   _cacheMgr(other._cacheMgr,this),
   _ipOrder(other._ipOrder),
   _disableCache(other._disableCache)
 { 
 } 


RooAbsCachedReal::~RooAbsCachedReal() 
{
}


Double_t RooAbsCachedReal::getVal(const RooArgSet* nset) const 
{
  if (_disableCache) {
    return RooAbsReal::getVal(nset) ;
  }

  // Cannot call cached p.d.f w.o nset
  // if (!nset) return evaluate() ;

  // Calculate current unnormalized value of object
  FuncCacheElem* cache = getCache(nset) ;
 
  _value = cache->func()->getVal() ;

  return _value ;
}



void RooAbsCachedReal::clearCacheObject(FuncCacheElem& cache) const 
{
  // Mark all bins as unitialized (value -1)
  cache.hist()->setAllWeights(-1) ;  
}


RooAbsCachedReal::FuncCacheElem* RooAbsCachedReal::createCache(const RooArgSet* nset) const
{
  return new FuncCacheElem(const_cast<RooAbsCachedReal&>(*this),nset) ;
}


RooAbsCachedReal::FuncCacheElem* RooAbsCachedReal::getCache(const RooArgSet* nset) const
{
  // Retrieve object representing projection integral of input p.d.f over observables iset, while normalizing
  // over observables nset. The code argument returned by reference is the unique code defining this particular
  // projection configuration
 
  // Check if this configuration was created becfore
  Int_t sterileIdx(-1) ;
  FuncCacheElem* cache = (FuncCacheElem*) _cacheMgr.getObj(nset,0,&sterileIdx,0) ;
  if (cache) {
    if (cache->paramTracker()->hasChanged(kTRUE)) {
      coutI(Eval) << "RooAbsCachedPdf::getCache(" << GetName() << ") cache " << cache << " function " 
		  << cache->func()->GetName() << " requires recalculation as parameters changed" << endl ;
      fillCacheObject(*cache) ;  
      cache->func()->setValueDirty() ;
    }
    return cache ;
  }

  cache = createCache(nset) ;
  fillCacheObject(*cache) ;  

  // Store this cache configuration
  Int_t code = _cacheMgr.setObj(nset,0,((RooAbsCacheElement*)cache),0) ;
  coutI(Caching) << "RooAbsCachedReal::getCache(" << GetName() << ") creating new cache " << cache->func()->GetName() << " for nset " << (nset?*nset:RooArgSet()) << " with code " << code << endl ;
  
  return cache ;
}


RooAbsCachedReal::FuncCacheElem::FuncCacheElem(const RooAbsCachedReal& self, const RooArgSet* nset) 
{
  // Create cache object itself -- Default implementation is a RooHistFunc
  RooArgSet* nset2 = self.actualObservables(nset?*nset:RooArgSet()) ;

  RooArgSet orderedObs ;
  if (nset2) {
    self.preferredObservableScanOrder(*nset2,orderedObs) ;
  }

  // Create RooDataHist
  TString hname = self.inputBaseName() ;
  hname.Append("_CACHEHIST_") ;
  hname.Append(self.cacheNameSuffix(*nset2)) ;

  _hist = new RooDataHist(hname,hname,*nset2,self.binningName()) ;

  RooArgSet* observables= self.actualObservables(*nset2) ;

  // Create RooHistFunc
  TString funcname = self.inputBaseName() ;
  funcname.Append("_CACHE") ;
  funcname.Append(self.cacheNameSuffix(*nset2)) ;
  _func = new RooHistFunc(funcname,funcname,*observables,*_hist,self.getInterpolationOrder()) ;

  // Set initial state of cache to dirty
  _func->setValueDirty() ;

  // Create pseudo-object that tracks changes in parameter values
  // Create pseudo-object that tracks changes in parameter values
  RooArgSet* params = self.actualParameters(orderedObs) ;
  string name= Form("%s_CACHEPARAMS",_func->GetName()) ;
  _paramTracker = new RooChangeTracker(name.c_str(),name.c_str(),*params,kTRUE) ;
  _paramTracker->hasChanged(kTRUE) ; // clear dirty flag as cache is up-to-date upon creation

  // Introduce formal dependency of RooHistFunc on parameters so that const optimization code
  // makes the correct decisions
  _func->addServerList(*params) ;

  delete observables ;
  delete params ;
  delete nset2 ;
  
}



TString RooAbsCachedReal::cacheNameSuffix(const RooArgSet& nset) const 
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
  return name ;
}



void RooAbsCachedReal::setInterpolationOrder(Int_t order) 
{
  _ipOrder = order ;

  Int_t i ;
  for (i=0 ; i<_cacheMgr.cacheSize() ; i++) {
    FuncCacheElem* cache = (FuncCacheElem*) _cacheMgr.getObjByIndex(i) ;
    if (cache) {
      cache->func()->setInterpolationOrder(order) ;
    }
  }
}


RooArgList RooAbsCachedReal::FuncCacheElem::containedArgs(Action) 
{
  RooArgList ret(*func()) ;

  ret.add(*_paramTracker) ;
  return ret ;
}



void RooAbsCachedReal::FuncCacheElem::printCompactTreeHook(ostream& os, const char* indent, Int_t curElem, Int_t maxElem) 
{
  // Print contents of cache when printing self as part of object tree
  if (curElem==0) {
    os << indent << "--- RooAbsCachedReal begin cache ---" << endl ;
  }

  TString indent2(indent) ;
  indent2 += Form("[%d] ",curElem) ;
  func()->printCompactTree(os,indent2) ;

  if (curElem==maxElem) {
    os << indent << "--- RooAbsCachedReal end cache --- " << endl ;
  }
}



Int_t RooAbsCachedReal::getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& analVars, const RooArgSet* normSet, const char* rangeName) const 
{
  FuncCacheElem* cache = getCache(normSet?normSet:&allVars) ;
  Int_t code = cache->func()->getAnalyticalIntegralWN(allVars,analVars,normSet,rangeName) ;
  _anaIntMap[code].first = &allVars ;
  _anaIntMap[code].second = normSet ;
  return code ;
}


Double_t RooAbsCachedReal::analyticalIntegralWN(Int_t code, const RooArgSet* normSet, const char* rangeName) const
{  
  if (code==0) {
    return getVal(normSet) ; 
  }  

  const RooArgSet* anaVars = _anaIntMap[code].first ;
  const RooArgSet* normSet2 = _anaIntMap[code].second ;

  FuncCacheElem* cache = getCache(normSet2?normSet2:anaVars) ;
  return cache->func()->analyticalIntegralWN(code,normSet,rangeName) ;
  
}





