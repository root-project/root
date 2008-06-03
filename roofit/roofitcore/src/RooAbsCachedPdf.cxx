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
#include "RooGlobalFunc.h"
#include "RooRealVar.h"
#include "RooChangeTracker.h"

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

  // Calculate current unnormalized value of object
  PdfCacheElem* cache = getCache(nset) ;
  
  Double_t value = cache->pdf()->getVal(nset) ;  

  _value = value ;    
  return _value ;
}


RooAbsPdf* RooAbsCachedPdf::getCachePdf(const RooArgSet* nset) const 
{
  PdfCacheElem* cache = getCache(nset) ;

  if (cache) {
    return cache->pdf() ;
  } else {
    return 0 ;
  }
}

RooDataHist* RooAbsCachedPdf::getCacheHist(const RooArgSet* nset) const 
{
  PdfCacheElem* cache = getCache(nset) ;

  if (cache) {
    return cache->hist() ;
  } else {
    return 0 ;
  }
}


void RooAbsCachedPdf::clearCacheObject(PdfCacheElem& cache) const 
{
  // Mark all bins as unitialized (value -1)
  cache.hist()->setAllWeights(-1) ;  
}



RooAbsCachedPdf::PdfCacheElem* RooAbsCachedPdf::getCache(const RooArgSet* nset, Bool_t recalculate) const
{
  // Retrieve object representing projection integral of input p.d.f over observables iset, while normalizing
  // over observables nset. The code argument returned by reference is the unique code defining this particular
  // projection configuration
 
  // Check if this configuration was created becfore
  Int_t sterileIdx(-1) ;
  PdfCacheElem* cache = (PdfCacheElem*) _cacheMgr.getObj(nset,0,&sterileIdx,0) ;
  if (cache) {
    if (cache->paramTracker()->hasChanged(kTRUE) && (recalculate || !cache->pdf()->haveUnitNorm()) ) {
      coutI(Eval) << "RooAbsCachedPdf::getCache(" << GetName() << ") cache " << cache << " pdf " 
		  << cache->pdf()->GetName() << " requires recalculation as parameters changed" << endl ;
      fillCacheObject(*cache) ;  
      cache->pdf()->setValueDirty() ;
    }
    return cache ;
  }


  cache = createCache(nset) ; 
  fillCacheObject(*cache) ;  
  

  // Store this cache configuration
  Int_t code = _cacheMgr.setObj(nset,0,((RooAbsCacheElement*)cache),0) ;

  coutI(Caching) << "RooAbsCachedPdf::getCache(" << GetName() << ") creating new cache " << cache << " with pdf "
		 << cache->pdf()->GetName() << " for nset " << (nset?*nset:RooArgSet()) << " with code " << code << endl ;  
  
  return cache ;
}




RooAbsCachedPdf::PdfCacheElem::PdfCacheElem(const RooAbsCachedPdf& self, const RooArgSet* nset) : 
  _pdf(0), _paramTracker(0), _hist(0), _norm(0) 
{
  // Create cache object itself -- Default implementation is a RooHistPdf
  RooArgSet* nset2 = self.actualObservables(nset?*nset:RooArgSet()) ;

  RooArgSet orderedObs ;
  if (nset2) {
    self.preferredObservableScanOrder(*nset2,orderedObs) ;
  }

  // Create RooDataHist
  TString hname = self.inputBaseName() ;
  hname.Append("_CACHEHIST_") ;
  hname.Append(self.cacheNameSuffix(orderedObs)) ;

  _hist = new RooDataHist(hname,hname,orderedObs,self.binningName()) ;

  RooArgSet* observables= self.actualObservables(orderedObs) ;

  // Create RooHistPdf
  TString pdfname = self.inputBaseName() ;
  pdfname.Append("_CACHE") ;
  pdfname.Append(self.cacheNameSuffix(orderedObs)) ;
  _pdf = new RooHistPdf(pdfname,pdfname,*observables,*_hist,self.getInterpolationOrder()) ;
  if (nset) {
    _nset.addClone(*nset) ;
  }

  // Create pseudo-object that tracks changes in parameter values
  RooArgSet* params = self.actualParameters(orderedObs) ;
  string name= Form("%s_CACHEPARAMS",_pdf->GetName()) ;
  _paramTracker = new RooChangeTracker(name.c_str(),name.c_str(),*params,kTRUE) ;
  _paramTracker->hasChanged(kTRUE) ; // clear dirty flag as cache is up-to-date upon creation

  // Introduce formal dependency of RooHistPdf on parameters so that const optimization code
  // makes the correct decisions
  _pdf->addServerList(*params) ;

  // Set initial state of cache to dirty
  _pdf->setValueDirty() ;

  delete observables ;
  delete params ;
  delete nset2 ;

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
    PdfCacheElem* cache = (PdfCacheElem*) _cacheMgr.getObjByIndex(i) ;
    if (cache) {
      cache->pdf()->setInterpolationOrder(order) ;
    }
  }
}


RooArgList RooAbsCachedPdf::PdfCacheElem::containedArgs(Action) 
{
  RooArgList ret(*_pdf) ;
  ret.add(*_paramTracker) ;
  if (_norm) ret.add(*_norm) ;
  return ret ;
}


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


void RooAbsCachedPdf::PdfCacheElem::printCompactTreeHook(ostream& os, const char* indent, Int_t curElem, Int_t maxElem) 
{
  // Print contents of cache when printing self as part of object tree
  if (curElem==0) {
    os << indent << "--- RooAbsCachedPdf begin cache ---" << endl ;
  }

  TString indent2(indent) ;
  os << Form("[%d] Configuration for observables ",curElem) << _nset << endl ;
  indent2 += Form("[%d] ",curElem) ;
  _pdf->printCompactTree(os,indent2) ;
  os << Form("[%d] Norm ",curElem) ;
  _norm->printStream(os,kName|kArgs,kSingleLine) ;
  
  if (curElem==maxElem) {
    os << indent << "--- RooAbsCachedPdf end cache --- " << endl ;
  }
}


Bool_t RooAbsCachedPdf::forceAnalyticalInt(const RooAbsArg& dep) const 
{
  RooArgSet* actObs = actualObservables(dep) ;
  Bool_t ret = (actObs->getSize()>0) ;
  delete actObs ;
  return ret ;
}


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
  Int_t codeList[2] ;
  codeList[0] = code ;
  codeList[1] = cache->pdf()->haveUnitNorm() ? 1 : 0 ;
  Int_t masterCode = _anaReg.store(codeList,2,all,ana,nrm)+1 ; // takes ownership of all sets

  
  // Mark all observables as internally integrated 
  if (cache->pdf()->haveUnitNorm()) {
    analVars.add(allVars,kTRUE) ;
  }

  return masterCode ;
}


Double_t RooAbsCachedPdf::analyticalIntegralWN(Int_t code, const RooArgSet* normSet, const char* rangeName) const
{  
  if (code==0) {
    return getVal(normSet) ; 
  }  

  RooArgSet *allVars(0),*anaVars(0),*normSet2(0),*dummy(0) ;
  const Int_t* codeList = _anaReg.retrieve(code-1,allVars,anaVars,normSet2,dummy) ;


  
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





