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

//////////////////////////////////////////////////////////////////////////////
//
// BEGIN_HTML
// Class RooProjectedPdf is a RooAbsPdf implementation that represent a projection 
// of a given input p.d.f and the object returned by RooAbsPdf::createProjection.
// <p>
// The actual projection integral for it value and normalization are
// calculated on the fly in getVal() once the normalization observables are known.
// Class RooProjectedPdf can cache projected p.d.f.s for multiple normalization
// observables simultaneously.
// <p>
// The createProjection() method of RooProjectedPdf is overloaded and will
// return a new RooProjectedPdf that performs the projection of itself
// and the requested additional projections in one integration step
// The performance of <pre>f->createProjection(x)->createProjection(y)</pre>
// is therefore identical to that of <pre>f->createProjection(RooArgSet(x,y))</pre>
// END_HTML
//

#include "Riostream.h" 

#include "RooFit.h"
#include "RooProjectedPdf.h" 
#include "RooMsgService.h"
#include "RooAbsReal.h" 
#include "RooRealVar.h"
#include "RooNameReg.h"


 ClassImp(RooProjectedPdf) 
   ;


//_____________________________________________________________________________
RooProjectedPdf::RooProjectedPdf() : _curNormSet(0)
{
  // Default constructor
}



//_____________________________________________________________________________
 RooProjectedPdf::RooProjectedPdf(const char *name, const char *title, RooAbsReal& _intpdf, const RooArgSet& intObs) :
   RooAbsPdf(name,title), 
   intpdf("!IntegratedPdf","intpdf",this,_intpdf,kFALSE,kFALSE),
   intobs("!IntegrationObservables","intobs",this,kFALSE,kFALSE),
   deps("!Dependents","deps",this,kTRUE,kTRUE),
   _cacheMgr(this,10)
 { 
   // Construct projection of input pdf '_intpdf' over observables 'intObs'

   intobs.add(intObs) ;

   // Add all other dependens of projected p.d.f. directly
   RooArgSet* tmpdeps = _intpdf.getParameters(intObs) ;
   deps.add(*tmpdeps) ;
   delete tmpdeps ;
 } 



//_____________________________________________________________________________
 RooProjectedPdf::RooProjectedPdf(const RooProjectedPdf& other, const char* name) :  
   RooAbsPdf(other,name), 
   intpdf("!IntegratedPdf",this,other.intpdf),
   intobs("!IntegrationObservable",this,other.intobs),
   deps("!Dependents",this,other.deps),
   _cacheMgr(other._cacheMgr,this)
{ 
   // Copy constructor
 } 



//_____________________________________________________________________________
Double_t RooProjectedPdf::getVal(const RooArgSet* set) const 
{
  // Special version of getVal() overrides RooAbsReal::getVal() to save value of current normalization set

  _curNormSet = (RooArgSet*)set ;

  return RooAbsPdf::getVal(set) ;
}



//_____________________________________________________________________________
Double_t RooProjectedPdf::evaluate() const 
{
  // Evaluate projected p.d.f

  // Calculate current unnormalized value of object
  int code ;
  const RooAbsReal* proj = getProjection(&intobs,_curNormSet,0,code) ;
  
  return proj->getVal() ;
}



//_____________________________________________________________________________
const RooAbsReal* RooProjectedPdf::getProjection(const RooArgSet* iset, const RooArgSet* nset, const char* rangeName, int& code) const
{
  // Retrieve object representing projection integral of input p.d.f
  // over observables iset, while normalizing over observables
  // nset. The code argument returned by reference is the unique code
  // defining this particular projection configuration


  // Check if this configuration was created before
  Int_t sterileIdx(-1) ;
  CacheElem* cache = (CacheElem*) _cacheMgr.getObj(iset,nset,&sterileIdx,RooNameReg::ptr(rangeName)) ;
  if (cache) {
    code = _cacheMgr.lastIndex() ;
    return static_cast<const RooAbsReal*>(cache->_projection);
  }

  RooArgSet* nset2 =  intpdf.arg().getObservables(*nset) ;

  if (iset) {
    nset2->add(*iset) ;
  }
  RooAbsReal* proj = intpdf.arg().createIntegral(iset?*iset:RooArgSet(),nset2,0,rangeName) ;
  delete nset2 ;

  cache = new CacheElem ;
  cache->_projection = proj ;

  code = _cacheMgr.setObj(iset,nset,(RooAbsCacheElement*)cache,RooNameReg::ptr(rangeName)) ;

  coutI(Integration) << "RooProjectedPdf::getProjection(" << GetName() << ") creating new projection " << proj->GetName() << " with code " << code << endl ;

  return proj ;
}



//_____________________________________________________________________________
RooAbsPdf* RooProjectedPdf::createProjection(const RooArgSet& iset) 
{
  // Special version of RooAbsReal::createProjection that deals with
  // projections of projections. Instead of integrating twice, a new
  // RooProjectedPdf is returned that is configured to perform the
  // complete integration in one step

  RooArgSet combiset(iset) ;
  combiset.add(intobs) ;
  return static_cast<RooAbsPdf&>( const_cast<RooAbsReal&>(intpdf.arg()) ).createProjection(combiset) ;
}



//_____________________________________________________________________________
Bool_t RooProjectedPdf::forceAnalyticalInt(const RooAbsArg& /*dep*/) const 
{
  // Force RooRealIntegral to relegate integration of all observables to internal logic

  return kTRUE ;
}



//_____________________________________________________________________________
Int_t RooProjectedPdf::getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& analVars, const RooArgSet* normSet, const char* rangeName) const 
{ 
  // Mark all requested variables as internally integrated

  analVars.add(allVars) ;
  
  // Create the appropriate integral
  int code ;
  RooArgSet allVars2(allVars) ;
  allVars2.add(intobs) ;
  getProjection(&allVars2,normSet,rangeName,code) ;
  
  return code+1 ; 
} 



//_____________________________________________________________________________
Double_t RooProjectedPdf::analyticalIntegralWN(Int_t code, const RooArgSet* /*normSet*/, const char* rangeName) const 
{ 
  // Return analytical integral represent by appropriate element of projection cache
  
  CacheElem *cache = (CacheElem*) _cacheMgr.getObjByIndex(code-1) ;
  
  if (cache) {
    Double_t ret= cache->_projection->getVal() ;
    return ret ;
  } else {
    
    RooArgSet* vars = getParameters(RooArgSet()) ;
    vars->add(intobs) ;
    RooArgSet* iset = _cacheMgr.nameSet1ByIndex(code-1)->select(*vars) ;
    RooArgSet* nset = _cacheMgr.nameSet2ByIndex(code-1)->select(*vars) ;
    
    Int_t code2(-1) ;
    const RooAbsReal* proj = getProjection(iset,nset,rangeName,code2) ;
    
    delete vars ;
    delete nset ;
    delete iset ;
    
    Double_t ret =  proj->getVal() ;
    return ret ;
  } 
  
} 



//_____________________________________________________________________________
Int_t RooProjectedPdf::getGenerator(const RooArgSet& /*directVars*/, RooArgSet& /*generateVars*/, Bool_t /*staticInitOK*/) const 
 { 
   // No internal generator is implemented
   return 0 ; 
 } 



//_____________________________________________________________________________
void RooProjectedPdf::generateEvent(Int_t /*code*/) 
 { 
   // No internal generator is implemented
   return; 
 } 



//_____________________________________________________________________________
Bool_t RooProjectedPdf::redirectServersHook(const RooAbsCollection& newServerList, Bool_t /*mustReplaceAll*/, 
				       Bool_t /*nameChange*/, Bool_t /*isRecursive*/) 
{
  // Intercept a server redirection all and update list of dependents if necessary 
  // Specifically update the set proxy 'deps' which introduces the dependency
  // on server value dirty flags of ourselves

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



//_____________________________________________________________________________
RooArgList RooProjectedPdf::CacheElem::containedArgs(Action)
{
  // Return RooAbsArg elements contained in projection cache element.
  RooArgList ret(*_projection) ;  
  return ret ;
}



//_____________________________________________________________________________
void RooProjectedPdf::printMetaArgs(ostream& os) const
{
  // Customized printing of arguments of a RooRealIntegral to more intuitively reflect the contents of the
  // integration operation

  os << "Int " << intpdf.arg().GetName() ;
 
  os << " d" ;
  os << intobs ;
  os << " " ;
  
}




//_____________________________________________________________________________
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


