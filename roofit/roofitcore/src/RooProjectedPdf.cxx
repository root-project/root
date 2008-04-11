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
 // Class RooProjectedPdf represent a projection of a given input p.d.f
 // The actual projection integral for it value and normalization are
 // calculated on the fly

#include "Riostream.h" 

#include "RooFit.h"
#include "RooProjectedPdf.h" 
#include "RooMsgService.h"
#include "RooAbsReal.h" 



 ClassImp(RooProjectedPdf) 
   ;

RooProjectedPdf::RooProjectedPdf() : _curNormSet(0)
{
}


 RooProjectedPdf::RooProjectedPdf(const char *name, const char *title, RooAbsReal& _intpdf, const RooArgSet& intObs) :
   RooAbsPdf(name,title), 
   intpdf("IntegratedPdf","intpdf",this,_intpdf,kFALSE,kFALSE),
   intobs("IntegrationObservables","intobs",this,kFALSE,kFALSE),
   deps("!Dependents","deps",this,kTRUE,kTRUE),
   _cacheMgr(this,10)
 { 
   // Constructor of projected of input p.d.f _intpdf over observables intObs

   intobs.add(intObs) ;

   // Add all other dependens of projected p.d.f. directly
   RooArgSet* tmpdeps = _intpdf.getParameters(intObs) ;
   deps.add(*tmpdeps) ;
   delete tmpdeps ;
 } 


 RooProjectedPdf::RooProjectedPdf(const RooProjectedPdf& other, const char* name) :  
   RooAbsPdf(other,name), 
   intpdf("IntegratedPdf",this,other.intpdf),
   intobs("IntegrarionObservable",this,other.intobs),
   deps("!Dependents",this,other.deps),
   _cacheMgr(other._cacheMgr,this)
{ 
   // Copy constructor
 } 


Double_t RooProjectedPdf::getVal(const RooArgSet* set) const 
{
  // Special version of getVal() overrides RooAbsReal::getVal() to save value of current normalization set
  _curNormSet = (RooArgSet*)set ;
  return RooAbsPdf::getVal(set) ;
}


Double_t RooProjectedPdf::evaluate() const 
{
  // Evaluate projected p.d.f

  // Calculate current unnormalized value of object
  int code ;
  const RooAbsReal* proj = getProjection(&intobs,_curNormSet,code) ;
  return proj->getVal() ;
}


const RooAbsReal* RooProjectedPdf::getProjection(const RooArgSet* iset, const RooArgSet* nset, int& code) const
{
  // Retrieve object representing projection integral of input p.d.f over observables iset, while normalizing
  // over observables nset. The code argument returned by reference is the unique code defining this particular
  // projection configuration
 
  // Check if this configuration was created before
  Int_t sterileIdx(-1) ;
  CacheElem* cache = (CacheElem*) _cacheMgr.getObj(nset,&intobs,&sterileIdx,0) ;
  if (cache) {
    return static_cast<const RooAbsReal*>(cache->_projection);
  }

  RooArgSet* nset2 =  intpdf.arg().getObservables(*nset) ;
  RooAbsReal* proj = intpdf.arg().createIntegral(*iset,nset2) ;
  delete nset2 ;

  cache = new CacheElem ;
  cache->_projection = proj ;

  code = _cacheMgr.setObj(nset,iset,(RooAbsCacheElement*)cache,0) ;
  coutI(Integration) << "RooProjectedPdf::getProjection(" << GetName() << ") creating new projection " << proj->GetName() << " with code " << code << endl ;

  return proj ;
}


RooAbsPdf* RooProjectedPdf::createProjection(const RooArgSet& iset) 
{
  // Special version of RooAbsReal::createProjection that deals with projections of projections. Instead of integrating
  // twice, a new RooProjectedPdf is returned that is configured to perform the complete integration in one step
  RooArgSet combiset(iset) ;
  combiset.add(intobs) ;
  return static_cast<RooAbsPdf&>( const_cast<RooAbsReal&>(intpdf.arg()) ).createProjection(combiset) ;
}


Bool_t RooProjectedPdf::forceAnalyticalInt(const RooAbsArg& /*dep*/) const 
{
  // Force RooRealIntegral to relegate integration of all observables to internal logic
  return kTRUE ;
}


Int_t RooProjectedPdf::getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& analVars, const RooArgSet* normSet, const char* /*rangeName*/) const 
{ 
  // Mark all requested variables as analytically integrated
  analVars.add(allVars) ;
  
  // Create the appropriate integral
  int code ;
  RooArgSet allVars2(allVars) ;
  allVars2.add(intobs) ;
  getProjection(&allVars2,normSet,code) ;
  
  return code+1 ; 
} 



Double_t RooProjectedPdf::analyticalIntegralWN(Int_t code, const RooArgSet* /*normSet*/, const char* /*rangeName*/) const 
 { 
   // Retrieve appropriate projection by code
   CacheElem *cache = (CacheElem*) _cacheMgr.getObjByIndex(code-1) ;
   
   return cache->_projection->getVal() ;
 } 



Int_t RooProjectedPdf::getGenerator(const RooArgSet& /*directVars*/, RooArgSet& /*generateVars*/, Bool_t /*staticInitOK*/) const 
 { 
   // No internal generator is implemented
   return 0 ; 
 } 



void RooProjectedPdf::generateEvent(Int_t /*code*/) 
 { 
   return; 
 } 



Bool_t RooProjectedPdf::redirectServersHook(const RooAbsCollection& newServerList, Bool_t /*mustReplaceAll*/, 
				       Bool_t /*nameChange*/, Bool_t /*isRecursive*/) 
{
  // Intercept a server redirection all and update list of dependents if necessary 

  // Redetermine explicit list of dependents if intPdf is being replaced
  RooAbsArg* newPdf = newServerList.find(intpdf.arg().GetName()) ;
  if (newPdf) {

    // Determine if set of dependens of new p.d.f is different from old p.d.f.
    RooArgSet olddeps(deps) ;
    RooArgSet* newdeps = newPdf->getParameters(intobs) ;
    RooArgSet* common = (RooArgSet*) newdeps->selectCommon(deps) ;    
    newdeps->remove(*common,kTRUE,kTRUE) ;
    olddeps.remove(*common,kTRUE,kTRUE) ;

    // If so, adjust composition of deps Listproxy
    if (newdeps->getSize()>0) {
      deps.add(*newdeps) ;
    }
    if (olddeps.getSize()>0) {
      deps.remove(olddeps,kTRUE,kTRUE) ;
    }

    delete common ;
    delete newdeps ;
  }

  return kFALSE ;
}



RooArgList RooProjectedPdf::CacheElem::containedArgs(Action)
{
  RooArgList ret(*_projection) ;  
  return ret ;
}



void RooProjectedPdf::CacheElem::printCompactTreeHook(ostream& os, const char* indent, Int_t curElem, Int_t maxElem) 
{
  // Print contents of cache when printing self as part of object tree
  if (curElem==0) {
    os << indent << "RooProjectedPdf begin projection cache" << endl ;
  }

  TString indent2(indent) ;
  indent2 += Form("[%d] ",curElem) ;

  _projection->printCompactTree(os,indent2) ;

  if(curElem==maxElem) {
    os << indent << "RooProjectedPdf end projection cache" << endl ;
  }
}


