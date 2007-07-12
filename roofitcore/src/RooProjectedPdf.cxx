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

 #include <iostream> 

 #include "RooProjectedPdf.h" 
 #include "RooAbsReal.h" 

 ClassImp(RooProjectedPdf) 

 RooProjectedPdf::RooProjectedPdf(const char *name, const char *title, RooAbsReal& _intpdf, const RooArgSet& intObs) :
   RooAbsPdf(name,title), 
   intpdf("intpdf","intpdf",this,_intpdf),
   intobs("intobs","intobs",this,kFALSE,kFALSE)
 { 
   intobs.add(intObs) ;
 } 


 RooProjectedPdf::RooProjectedPdf(const RooProjectedPdf& other, const char* name) :  
   RooAbsPdf(other,name), 
   intpdf("intpdf",this,other.intpdf),
   intobs("intobs",this,other.intobs)
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
  cout << "RooProjectedPdf::getProjection(" << GetName() << ") creating new projection " << proj->GetName() << " with code " << code << endl ;

  
  return proj ;
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



Bool_t RooProjectedPdf::redirectServersHook(const RooAbsCollection& /*newServerList*/, Bool_t /*mustReplaceAll*/, 
				       Bool_t /*nameChange*/, Bool_t /*isRecursive*/) 
{
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


