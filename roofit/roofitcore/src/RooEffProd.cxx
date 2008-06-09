/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   GR, Gerhard Raven, NIKHEF/VU                                            *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/


// -- CLASS DESCRIPTION [PDF] --
// The class RooEffProd implements the product of a PDF with an efficiency function.
// The normalization integral of the product is calculated numerically, but the
// event generation is handled by a specialized generator context that implements
// the event generation in a more efficient for cases where the PDF has an internal
// generator that is smarter than accept reject. 

#include "RooFit.h"
#include "RooEffProd.h"
#include "RooEffGenContext.h"
#include "RooNameReg.h"
#include "RooRealVar.h"

ClassImp(RooEffProd)
  ;

RooEffProd::RooEffProd(const char *name, const char *title, 
                             RooAbsPdf& inPdf, RooAbsReal& inEff) :
  RooAbsPdf(name,title),
  _cacheMgr(this,10),
  _pdf("pdf","pre-efficiency pdf", this,inPdf),
  _eff("eff","efficiency function",this,inEff),
  _nset(0),
  _fixedNset(0)
{  

}


RooEffProd::RooEffProd(const RooEffProd& other, const char* name) : 
  RooAbsPdf(other, name),
  _cacheMgr(other._cacheMgr,this),
  _pdf("pdf",this,other._pdf),
  _eff("acc",this,other._eff),
  _nset(0),
  _fixedNset(0) 
{
}


RooEffProd::~RooEffProd() 
{
}

Double_t RooEffProd::getVal(const RooArgSet* set) const 
{  
  _nset = _fixedNset ? _fixedNset : set ;
  return RooAbsPdf::getVal(set) ;
}

Double_t RooEffProd::evaluate() const
{
    return eff()->getVal() * pdf()->getVal(_nset);
}

RooAbsGenContext* RooEffProd::genContext(const RooArgSet &vars, const RooDataSet *prototype,
                                            const RooArgSet* auxProto, Bool_t verbose) const
{
  assert(pdf()!=0);
  assert(eff()!=0);
  return new RooEffGenContext(*this,*pdf(),*eff(),vars,prototype,auxProto,verbose) ;
}


Int_t RooEffProd::getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& analVars, 
					  const RooArgSet* normSet, const char* rangeName) const 
{
  // No special handling required if a normalization set is given
  if (normSet && normSet->getSize()>0) {    
    return 0 ;
  }
  // No special handling required if running with a fixed normalization set
  if (_fixedNset) {
    return 0 ;
  }

  // Special handling of integrals w/o normalization set. We need to pass _a_ normalization set
  // to pdf().getVal(nset) in evaluate() because for certain p.d.fs the shape depends on the
  // chosen normalization set. This functions correctly automatically for plain getVal() calls,
  // however when the (numeric) normalization is calculated, getVal() is called without any normalization
  // set causing the normalization to be calculated for a possibly different shape. To fix that
  // integrals over a RooEffProd without normalization setup are explicitly handled here. These integrals
  // are calculated using a clone of the RooEffProd that has a fixed normalization set passed to the
  // underlying p.d.f regardless of the normalization set passed to getVal(). Here the set of observables
  // over which is integrated is passed.

  // Declare that we can analytically integrate all requested observables
  analVars.add(allVars) ;

  // Check if cache was previously created
  Int_t sterileIndex(-1) ;
  CacheElem* cache = (CacheElem*) _cacheMgr.getObj(&allVars,&allVars,&sterileIndex,RooNameReg::ptr(rangeName)) ;
  if (cache) {
    return _cacheMgr.lastIndex()+1;
  }


  // Construct cache with clone of p.d.f that has fixed normalization set that is passed to input pdf
  cache = new CacheElem ;
  cache->_intObs.addClone(allVars) ;
  cache->_clone = (RooEffProd*) clone(Form("%s_clone",GetName())) ;
  cache->_clone->_fixedNset = &cache->_intObs ;
  cache->_int = cache->_clone->createIntegral(cache->_intObs) ;

  // Store cache and return index as code
  Int_t code = _cacheMgr.setObj(&allVars,&allVars,(RooAbsCacheElement*)cache,RooNameReg::ptr(rangeName)) ; 

  return code+1 ;
}




Double_t RooEffProd::analyticalIntegralWN(Int_t code, const RooArgSet* normSet, const char* /*rangeName*/) const 
{
  // Return analytical integral defined by given scenario code

  // No integration scenario
  if (code==0) {
    return getVal(normSet) ;
  }

  // Partial integration scenarios
  CacheElem* cache = (CacheElem*) _cacheMgr.getObjByIndex(code-1) ;
  return cache->_int->getVal() ;
}


RooArgList RooEffProd::CacheElem::containedArgs(Action) 
{
  RooArgList ret(_intObs) ;
  ret.add(*_int) ;
  ret.add(*_clone) ;
  return ret ;
}
