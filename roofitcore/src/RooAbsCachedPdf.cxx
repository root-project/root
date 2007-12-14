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
#include "RooAbsCachedPdf.h" 
#include "RooAbsReal.h" 
#include "RooMsgService.h"
#include "RooDataHist.h"
#include "RooHistPdf.h"

ClassImp(RooAbsCachedPdf) 



RooAbsCachedPdf::RooAbsCachedPdf(const char *name, const char *title, Int_t ipOrder) :
  RooAbsPdf(name,title), 
  _cacheMgr(this,10),
  _ipOrder(ipOrder),
  _disableCache(kFALSE)
 { 
 } 


RooAbsCachedPdf::RooAbsCachedPdf(const RooAbsCachedPdf& other, const char* name) :  
   RooAbsPdf(other,name), 
   _cacheMgr(other._cacheMgr,this),
   _ipOrder(other._ipOrder),
   _disableCache(other._disableCache)
 { 
 } 


RooAbsCachedPdf::~RooAbsCachedPdf() 
{
}


Double_t RooAbsCachedPdf::getVal(const RooArgSet* nset) const 
{
  if (_disableCache) {
    return RooAbsPdf::getVal(nset) ;
  }

  // Cannot call cached p.d.f w.o nset
  if (!nset) return evaluate() ;

  // Calculate current unnormalized value of object
  const CacheElem* cache = getCache(nset) ;
 
  _value = cache->_pdf->getVal(nset) ;
  
  return _value ;
}



void RooAbsCachedPdf::clearCacheObject(CacheElem& cache) const 
{
  // Mark all bins as unitialized (value -1)
  cache._hist->setAllWeights(-1) ;  
}



const RooAbsCachedPdf::CacheElem* RooAbsCachedPdf::getCache(const RooArgSet* nset) const
{
  // Retrieve object representing projection integral of input p.d.f over observables iset, while normalizing
  // over observables nset. The code argument returned by reference is the unique code defining this particular
  // projection configuration
 
  // Check if this configuration was created becfore
  Int_t sterileIdx(-1) ;
  CacheElem* cache = (CacheElem*) _cacheMgr.getObj(nset,0,&sterileIdx,0) ;
  if (cache) {
    if (cache->_params->isValueDirty()) {
      coutI(Caching) << "RooAbsCachedPdf::getCache(" << GetName() << ") cache " << cache->_pdf->GetName() << " requires recalculation as parameters changed" << endl ;
      fillCacheObject(*cache) ;  
      cache->_pdf->setValueDirty() ;
      cache->_params->getVal() ;
    }
    return cache ;
  }

  cache = new CacheElem ;

  // Create cache object itself -- Default implementation is a RooHistPdf
  RooArgSet* nset2 = actualObservables(*nset) ;

  // Create RooDataHist
  TString hname = inputBaseName() ;
  hname.Append("_CACHEHIST_") ;
  hname.Append(cacheNameSuffix(*nset2)) ;

  cache->_hist = new RooDataHist(hname,hname,*nset2) ;

  RooArgSet* observables= actualObservables(*nset2) ;

  // Create RooHistPdf
  TString pdfname = inputBaseName() ;
  pdfname.Append("_CACHE") ;
  pdfname.Append(cacheNameSuffix(*nset2)) ;
  cache->_pdf = new RooHistPdf(pdfname,pdfname,*observables,*cache->_hist,_ipOrder) ;

  fillCacheObject(*cache) ;
  
  delete observables ;

  // Set initial state of cache to dirty
  cache->_pdf->setValueDirty() ;

  // Create pseudo-object that tracks changes in parameter values
  RooArgSet* params = actualParameters(*nset2) ;
  cache->_params = new RooFormulaVar(Form("%s_CACHEPARAMS",cache->_pdf->GetName()),"1",*params) ;
  cache->_params->getVal() ; // clear dirty flag as cache is up-to-date upon creation

  // Introduce formal dependency of RooHistPdf on parameters so that const optimization code
  // makes the correct decisions
  cache->_pdf->addServerList(*params) ;

  delete params ;
  delete nset2 ;

  // Store this cache configuration
  Int_t code = _cacheMgr.setObj(nset,0,((RooAbsCacheElement*)cache),0) ;
  coutI(Caching) << "RooAbsCachedPdf::getCache(" << GetName() << ") creating new cache " << cache->_pdf->GetName() << " for nset " << (nset?*nset:RooArgSet()) << " with code " << code << endl ;
  
  return cache ;
}




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
  return name ;
}



void RooAbsCachedPdf::setInterpolationOrder(Int_t order) 
{
  _ipOrder = order ;

  Int_t i ;
  for (i=0 ; i<_cacheMgr.cacheSize() ; i++) {
    CacheElem* cache = (CacheElem*) _cacheMgr.getObjByIndex(i) ;
    if (cache) {
      cache->_pdf->setInterpolationOrder(order) ;
    }
  }
}


RooArgList RooAbsCachedPdf::CacheElem::containedArgs(Action) 
{
  RooArgList ret(*_pdf) ;

  ret.add(*_params) ;
  return ret ;
}



void RooAbsCachedPdf::CacheElem::printCompactTreeHook(ostream& os, const char* indent, Int_t curElem, Int_t maxElem) 
{
  // Print contents of cache when printing self as part of object tree
  if (curElem==0) {
    os << indent << "--- RooAbsCachedPdf begin cache ---" << endl ;
  }

  TString indent2(indent) ;
  indent2 += Form("[%d] ",curElem) ;
  _pdf->printCompactTree(os,indent2) ;

  if (curElem==maxElem) {
    os << indent << "--- RooAbsCachedPdf end cache --- " << endl ;
  }
}



Int_t RooAbsCachedPdf::getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& analVars, const RooArgSet* normSet, const char* rangeName) const 
{
  const CacheElem* cache = getCache(normSet?normSet:&allVars) ;
  Int_t code = cache->_pdf->getAnalyticalIntegralWN(allVars,analVars,normSet,rangeName) ;
  _anaIntMap[code].first = &allVars ;
  _anaIntMap[code].second = normSet ;
  return code ;
}


Double_t RooAbsCachedPdf::analyticalIntegralWN(Int_t code, const RooArgSet* normSet, const char* rangeName) const
{  
  if (code==0) {
    return getVal(normSet) ; 
  }  

  const RooArgSet* anaVars = _anaIntMap[code].first ;
  const RooArgSet* normSet2 = _anaIntMap[code].second ;

  const CacheElem* cache = getCache(normSet2?normSet2:anaVars) ;
  return cache->_pdf->analyticalIntegralWN(code,normSet,rangeName) ;
  
}





