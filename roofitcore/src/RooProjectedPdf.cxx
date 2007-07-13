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

#include <iostream> 

#include "RooProjectedPdf.h" 
#include "RooMsgService.h"
#include "RooAbsReal.h" 

 ClassImp(RooProjectedPdf) 

 RooProjectedPdf::RooProjectedPdf(const char *name, const char *title, RooAbsReal& _intpdf, const RooArgSet& intObs) :
   RooAbsPdf(name,title), 
   intpdf("IntegratedPdf","intpdf",this,_intpdf,kFALSE,kFALSE),
   intobs("IntegrationObservables","intobs",this,kFALSE,kFALSE),
   deps("!Dependents","deps",this,kTRUE,kTRUE)
 { 
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
   deps("!Dependents",this,other.deps)
 { 
 } 


Double_t RooProjectedPdf::getVal(const RooArgSet* set) const 
{
  _curNormSet = (RooArgSet*)set ;
  return RooAbsPdf::getVal(set) ;
}


Double_t RooProjectedPdf::evaluate() const 
{
  // Calculate current unnormalized value of object
  int code ;
  const RooAbsReal* proj = getProjection(&intobs,_curNormSet,code) ;
  return proj->getVal() ;
}


const RooAbsReal* RooProjectedPdf::getProjection(const RooArgSet* iset, const RooArgSet* nset, int& code) const
{
   // Check if this configuration was created before
  Int_t sterileIdx(-1) ;
  RooArgList* projList = _projListMgr.getNormList(this,nset,&intobs,&sterileIdx,0) ;
  if (projList) {
    return static_cast<const RooAbsReal*>(projList->at(0));
  }

  RooAbsReal* proj = intpdf.arg().createIntegral(*iset,nset) ;
  projList = new RooArgList(*proj) ;

  code = _projListMgr.setNormList(this,nset,iset,projList,0) ;
  coutI("Integration") << "RooProjectedPdf::getProjection(" << GetName() << ") creating new projection " << proj->GetName() << " with code " << code << endl ;
  
  return proj ;
}


RooAbsPdf* RooProjectedPdf::createProjection(const RooArgSet& iset) 
{
  RooArgSet combiset(iset) ;
  combiset.add(intobs) ;
  return static_cast<RooAbsPdf&>( const_cast<RooAbsReal&>(intpdf.arg()) ).createProjection(combiset) ;
}


Bool_t RooProjectedPdf::forceAnalyticalInt(const RooAbsArg& /*dep*/) const 
{
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
   RooArgList *projList = _projListMgr.getNormListByIndex(code-1) ;
   
   return ((RooAbsReal*)projList->at(0))->getVal() ;
 } 



Int_t RooProjectedPdf::getGenerator(const RooArgSet& /*directVars*/, RooArgSet& /*generateVars*/, Bool_t /*staticInitOK*/) const 
 { 
   return 0 ; 
 } 



void RooProjectedPdf::generateEvent(Int_t /*code*/) 
 { 
   return; 
 } 



void RooProjectedPdf::clearCache() 
{
  _projListMgr.sterilize()  ;
}



Bool_t RooProjectedPdf::redirectServersHook(const RooAbsCollection& newServerList, Bool_t /*mustReplaceAll*/, 
				       Bool_t /*nameChange*/, Bool_t /*isRecursive*/) 
{
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

  // Throw away cache, as figuring out redirections on the cache is an unsolvable problem. 
  clearCache() ;

  return kFALSE ;
}


void RooProjectedPdf::operModeHook() 
{
  Int_t i ;
  for (i=0 ; i<_projListMgr.cacheSize() ; i++) {
    RooArgList* plist = _projListMgr.getNormListByIndex(i) ;
    if (plist) {
      TIterator* iter = plist->createIterator() ;
      RooAbsArg* arg ;
      while((arg=(RooAbsArg*)iter->Next())) {
	arg->setOperMode(_operMode) ;
      }
      delete iter ;
    }
  }
  return ;
}

void RooProjectedPdf::printCompactTreeHook(ostream& os, const char* indent) 
{
  Int_t i ;
  os << indent << "RooProjectedPdf begin projection cache" << endl ;

  for (i=0 ; i<_projListMgr.cacheSize() ; i++) {
    RooArgList* plist = _projListMgr.getNormListByIndex(i) ;    
    if (plist) {
      TIterator* iter = plist->createIterator() ;
      RooAbsArg* arg ;
      TString indent2(indent) ;
      indent2 += Form("[%d] ",i) ;
      while((arg=(RooAbsArg*)iter->Next())) {      
	arg->printCompactTree(os,indent2) ;
      }
      delete iter ;
    }
  }

  os << indent << "RooProjectedPdf end projection cache" << endl ;
}


