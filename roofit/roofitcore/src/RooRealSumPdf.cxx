/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
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
// Class RooRealSumPdf implements a PDF constructed from a sum of
// functions:
//
//                 Sum(i=1,n-1) coef_i * func_i(x) + [ 1 - (Sum(i=1,n-1) coef_i ] * func_n(x)
//   pdf(x) =    ------------------------------------------------------------------------------
//             Sum(i=1,n-1) coef_i * Int(func_i)dx + [ 1 - (Sum(i=1,n-1) coef_i ] * Int(func_n)dx
//
//
// where coef_i and func_i are RooAbsReal objects, and x is the collection of dependents. 
// In the present version coef_i may not depend on x, but this limitation may be removed in the future
//

#include "RooFit.h"
#include "Riostream.h"

#include "TIterator.h"
#include "TList.h"
#include "RooRealSumPdf.h"
#include "RooRealProxy.h"
#include "RooPlot.h"
#include "RooRealVar.h"
#include "RooAddGenContext.h"
#include "RooRealConstant.h"
#include "RooRealIntegral.h"
#include "RooMsgService.h"
#include "RooNameReg.h"
#include <memory>


ClassImp(RooRealSumPdf)
;


//_____________________________________________________________________________
RooRealSumPdf::RooRealSumPdf() 
{
  // Default constructor
  // coverity[UNINIT_CTOR]
  _funcIter  = _funcList.createIterator() ;
  _coefIter  = _coefList.createIterator() ;
  _extended = kFALSE ;
}



//_____________________________________________________________________________
RooRealSumPdf::RooRealSumPdf(const char *name, const char *title) :
  RooAbsPdf(name,title), 
  _normIntMgr(this,10),
  _haveLastCoef(kFALSE),
  _funcList("!funcList","List of functions",this),
  _coefList("!coefList","List of coefficients",this),
  _extended(kFALSE)
{
  // Constructor with name and title
  _funcIter   = _funcList.createIterator() ;
  _coefIter  = _coefList.createIterator() ;
}



//_____________________________________________________________________________
RooRealSumPdf::RooRealSumPdf(const char *name, const char *title,
		     RooAbsReal& func1, RooAbsReal& func2, RooAbsReal& coef1) : 
  RooAbsPdf(name,title),
  _normIntMgr(this,10),
  _haveLastCoef(kFALSE),
  _funcList("!funcList","List of functions",this),
  _coefList("!coefList","List of coefficients",this),
  _extended(kFALSE)
{
  // Construct p.d.f consisting of coef1*func1 + (1-coef1)*func2
  // The input coefficients and functions are allowed to be negative
  // but the resulting sum is not, which is enforced at runtime

  // Special constructor with two functions and one coefficient
  _funcIter  = _funcList.createIterator() ;
  _coefIter = _coefList.createIterator() ;

  _funcList.add(func1) ;  
  _funcList.add(func2) ;
  _coefList.add(coef1) ;

}


//_____________________________________________________________________________
RooRealSumPdf::RooRealSumPdf(const char *name, const char *title, const RooArgList& inFuncList, const RooArgList& inCoefList, Bool_t extended) :
  RooAbsPdf(name,title),
  _normIntMgr(this,10),
  _haveLastCoef(kFALSE),
  _funcList("!funcList","List of functions",this),
  _coefList("!coefList","List of coefficients",this),
  _extended(extended)
{ 
  // Constructor p.d.f implementing sum_i [ coef_i * func_i ], if N_coef==N_func
  // or sum_i [ coef_i * func_i ] + (1 - sum_i [ coef_i ] )* func_N if Ncoef==N_func-1
  // 
  // All coefficients and functions are allowed to be negative
  // but the sum is not, which is enforced at runtime.

  if (!(inFuncList.getSize()==inCoefList.getSize()+1 || inFuncList.getSize()==inCoefList.getSize())) {
    coutE(InputArguments) << "RooRealSumPdf::RooRealSumPdf(" << GetName() 
			  << ") number of pdfs and coefficients inconsistent, must have Nfunc=Ncoef or Nfunc=Ncoef+1" << endl ;
    assert(0) ;
  }

  _funcIter  = _funcList.createIterator() ;
  _coefIter = _coefList.createIterator() ;
 
  // Constructor with N functions and N or N-1 coefs
  TIterator* funcIter = inFuncList.createIterator() ;
  TIterator* coefIter = inCoefList.createIterator() ;
  RooAbsArg* func ;
  RooAbsArg* coef ;

  while((coef = (RooAbsArg*)coefIter->Next())) {
    func = (RooAbsArg*) funcIter->Next() ;

    if (!dynamic_cast<RooAbsReal*>(coef)) {
      coutW(InputArguments) << "RooRealSumPdf::RooRealSumPdf(" << GetName() << ") coefficient " << coef->GetName() << " is not of type RooAbsReal, ignored" << endl ;
      continue ;
    }
    if (!dynamic_cast<RooAbsReal*>(func)) {
      coutW(InputArguments) << "RooRealSumPdf::RooRealSumPdf(" << GetName() << ") func " << func->GetName() << " is not of type RooAbsReal, ignored" << endl ;
      continue ;
    }
    _funcList.add(*func) ;
    _coefList.add(*coef) ;    
  }

  func = (RooAbsReal*) funcIter->Next() ;
  if (func) {
    if (!dynamic_cast<RooAbsReal*>(func)) {
      coutE(InputArguments) << "RooRealSumPdf::RooRealSumPdf(" << GetName() << ") last func " << coef->GetName() << " is not of type RooAbsReal, fatal error" << endl ;
      assert(0) ;
    }
    _funcList.add(*func) ;  
  } else {
    _haveLastCoef = kTRUE ;
  }
  
  delete funcIter ;
  delete coefIter  ;
}




//_____________________________________________________________________________
RooRealSumPdf::RooRealSumPdf(const RooRealSumPdf& other, const char* name) :
  RooAbsPdf(other,name),
  _normIntMgr(other._normIntMgr,this),
  _haveLastCoef(other._haveLastCoef),
  _funcList("!funcList",this,other._funcList),
  _coefList("!coefList",this,other._coefList),
  _extended(other._extended)
{
  // Copy constructor

  _funcIter  = _funcList.createIterator() ;
  _coefIter = _coefList.createIterator() ;
}



//_____________________________________________________________________________
RooRealSumPdf::~RooRealSumPdf()
{
  // Destructor
  delete _funcIter ;
  delete _coefIter ;
}





//_____________________________________________________________________________
RooAbsPdf::ExtendMode RooRealSumPdf::extendMode() const 
{
  return (_extended && (_funcList.getSize()==_coefList.getSize())) ? CanBeExtended : CanNotBeExtended ;
}




//_____________________________________________________________________________
Double_t RooRealSumPdf::evaluate() const 
{
  // Calculate the current value

  Double_t value(0) ;

  // Do running sum of coef/func pairs, calculate lastCoef.
  _funcIter->Reset() ;
  _coefIter->Reset() ;
  RooAbsReal* coef ;
  RooAbsReal* func ;
      
  // N funcs, N-1 coefficients 
  Double_t lastCoef(1) ;
  while((coef=(RooAbsReal*)_coefIter->Next())) {
    func = (RooAbsReal*)_funcIter->Next() ;
    Double_t coefVal = coef->getVal() ;
    if (coefVal) {
      cxcoutD(Eval) << "RooRealSumPdf::eval(" << GetName() << ") coefVal = " << coefVal << " funcVal = " << func->getVal() << endl ;
      if (func->isSelectedComp()) {
	value += func->getVal()*coefVal ;
      }
      lastCoef -= coef->getVal() ;
    }
  }
  
  if (!_haveLastCoef) {
    // Add last func with correct coefficient
    func = (RooAbsReal*) _funcIter->Next() ;
    if (func->isSelectedComp()) {
      value += func->getVal()*lastCoef ;
    }

    cxcoutD(Eval) << "RooRealSumPdf::eval(" << GetName() << ") lastCoef = " << lastCoef << " funcVal = " << func->getVal() << endl ;
    
    // Warn about coefficient degeneration
    if (lastCoef<0 || lastCoef>1) {
      coutW(Eval) << "RooRealSumPdf::evaluate(" << GetName() 
		  << " WARNING: sum of FUNC coefficients not in range [0-1], value=" 
		  << 1-lastCoef << endl ;
    } 
  }

  return value ;
}




//_____________________________________________________________________________
Bool_t RooRealSumPdf::checkObservables(const RooArgSet* nset) const 
{
  // Check if FUNC is valid for given normalization set.
  // Coeffient and FUNC must be non-overlapping, but func-coefficient 
  // pairs may overlap each other
  //
  // In the present implementation, coefficients may not be observables or derive
  // from observables

  Bool_t ret(kFALSE) ;

  _funcIter->Reset() ;
  _coefIter->Reset() ;
  RooAbsReal* coef ;
  RooAbsReal* func ;
  while((coef=(RooAbsReal*)_coefIter->Next())) {
    func = (RooAbsReal*)_funcIter->Next() ;
    if (func->observableOverlaps(nset,*coef)) {
      coutE(InputArguments) << "RooRealSumPdf::checkObservables(" << GetName() << "): ERROR: coefficient " << coef->GetName() 
			    << " and FUNC " << func->GetName() << " have one or more observables in common" << endl ;
      ret = kTRUE ;
    }
    if (coef->dependsOn(*nset)) {
      coutE(InputArguments) << "RooRealPdf::checkObservables(" << GetName() << "): ERROR coefficient " << coef->GetName() 
			    << " depends on one or more of the following observables" ; nset->Print("1") ;
      ret = kTRUE ;
    }
  }
  
  return ret ;
}




//_____________________________________________________________________________
Int_t RooRealSumPdf::getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& analVars, 
					     const RooArgSet* normSet2, const char* rangeName) const 
{
  //cout << "RooRealSumPdf::getAnalyticalIntegralWN:"<<GetName()<<"("<<allVars<<",analVars,"<<(normSet2?*normSet2:RooArgSet())<<","<<(rangeName?rangeName:"<none>") << endl;
  // Advertise that all integrals can be handled internally.

  // Handle trivial no-integration scenario
  if (allVars.getSize()==0) return 0 ;
  if (_forceNumInt) return 0 ;

  // Select subset of allVars that are actual dependents
  analVars.add(allVars) ;
  RooArgSet* normSet = normSet2 ? getObservables(normSet2) : 0 ;


  // Check if this configuration was created before
  Int_t sterileIdx(-1) ;
  CacheElem* cache = (CacheElem*) _normIntMgr.getObj(normSet,&analVars,&sterileIdx,RooNameReg::ptr(rangeName)) ;
  if (cache) {
    //cout << "RooRealSumPdf("<<this<<")::getAnalyticalIntegralWN:"<<GetName()<<"("<<allVars<<","<<analVars<<","<<(normSet2?*normSet2:RooArgSet())<<","<<(rangeName?rangeName:"<none>") << " -> " << _normIntMgr.lastIndex()+1 << " (cached)" << endl;
    return _normIntMgr.lastIndex()+1 ;
  }
  
  // Create new cache element
  cache = new CacheElem ;

  // Make list of function projection and normalization integrals 
  _funcIter->Reset() ;
  RooAbsReal *func ;
  while((func=(RooAbsReal*)_funcIter->Next())) {
    RooAbsReal* funcInt = func->createIntegral(analVars,rangeName) ;
    cache->_funcIntList.addOwned(*funcInt) ;
    if (normSet && normSet->getSize()>0) {
      RooAbsReal* funcNorm = func->createIntegral(*normSet) ;
      cache->_funcNormList.addOwned(*funcNorm) ;
    }
  }

  // Store cache element
  Int_t code = _normIntMgr.setObj(normSet,&analVars,(RooAbsCacheElement*)cache,RooNameReg::ptr(rangeName)) ;

  if (normSet) {
    delete normSet ;
  }

  //cout << "RooRealSumPdf("<<this<<")::getAnalyticalIntegralWN:"<<GetName()<<"("<<allVars<<","<<analVars<<","<<(normSet2?*normSet2:RooArgSet())<<","<<(rangeName?rangeName:"<none>") << " -> " << code+1 << endl;
  return code+1 ; 
}




//_____________________________________________________________________________
Double_t RooRealSumPdf::analyticalIntegralWN(Int_t code, const RooArgSet* normSet2, const char* rangeName) const 
{
  //cout << "RooRealSumPdf::analyticalIntegralWN:"<<GetName()<<"("<<code<<","<<(normSet2?*normSet2:RooArgSet())<<","<<(rangeName?rangeName:"<none>") << endl;
  // Implement analytical integrations by deferring integration of component
  // functions to integrators of components

  // Handle trivial passthrough scenario
  if (code==0) return getVal(normSet2) ;


  // WVE needs adaptation for rangeName feature
  CacheElem* cache = (CacheElem*) _normIntMgr.getObjByIndex(code-1) ;
  if (cache==0) { // revive the (sterilized) cache
     //cout << "RooRealSumPdf("<<this<<")::analyticalIntegralWN:"<<GetName()<<"("<<code<<","<<(normSet2?*normSet2:RooArgSet())<<","<<(rangeName?rangeName:"<none>") << ": reviving cache "<< endl;
     std::auto_ptr<RooArgSet> vars( getParameters(RooArgSet()) );
     std::auto_ptr<RooArgSet> iset(  _normIntMgr.nameSet2ByIndex(code-1)->select(*vars) );
     std::auto_ptr<RooArgSet> nset(  _normIntMgr.nameSet1ByIndex(code-1)->select(*vars) );
     RooArgSet dummy;
     Int_t code2 = getAnalyticalIntegralWN(*iset,dummy,nset.get(),rangeName);
     assert(code==code2); // must have revived the right (sterilized) slot...
     cache = (CacheElem*) _normIntMgr.getObjByIndex(code-1) ;
     assert(cache!=0);
  }

  TIterator* funcIntIter = cache->_funcIntList.createIterator() ;
  _coefIter->Reset() ;
  _funcIter->Reset() ;
  RooAbsReal *coef(0), *funcInt(0), *func(0) ;
  Double_t value(0) ;

  // N funcs, N-1 coefficients 
  Double_t lastCoef(1) ;
  while((coef=(RooAbsReal*)_coefIter->Next())) {
    funcInt = (RooAbsReal*)funcIntIter->Next() ;
    func    = (RooAbsReal*)_funcIter->Next() ;
    Double_t coefVal = coef->getVal(normSet2) ;
    if (coefVal) {
      assert(func);
      if (func->isSelectedComp()) {
    assert(funcInt);
	value += funcInt->getVal()*coefVal ;
      }
      lastCoef -= coef->getVal(normSet2) ;
    }
  }
  
  if (!_haveLastCoef) {
    // Add last func with correct coefficient
    funcInt = (RooAbsReal*) funcIntIter->Next() ;
    if (func->isSelectedComp()) {
      assert(funcInt);
      value += funcInt->getVal()*lastCoef ;
    }
    
    // Warn about coefficient degeneration
    if (lastCoef<0 || lastCoef>1) {
      coutW(Eval) << "RooRealSumPdf::evaluate(" << GetName() 
		  << " WARNING: sum of FUNC coefficients not in range [0-1], value=" 
		  << 1-lastCoef << endl ;
    } 
  }

  delete funcIntIter ;
  
  Double_t normVal(1) ;
  if (normSet2 && normSet2->getSize()>0) {
    normVal = 0 ;

    // N funcs, N-1 coefficients 
    RooAbsReal* funcNorm ;
    TIterator* funcNormIter = cache->_funcNormList.createIterator() ;
    _coefIter->Reset() ;
    while((coef=(RooAbsReal*)_coefIter->Next())) {
      funcNorm = (RooAbsReal*)funcNormIter->Next() ;
      Double_t coefVal = coef->getVal(normSet2) ;
      if (coefVal) {
    assert(funcNorm);
	normVal += funcNorm->getVal()*coefVal ;
      }
    }
    
    // Add last func with correct coefficient
    if (!_haveLastCoef) {
      funcNorm = (RooAbsReal*) funcNormIter->Next() ;
      assert(funcNorm);
      normVal += funcNorm->getVal()*lastCoef ;
    }
      
    delete funcNormIter ;      
  }

  return value / normVal;
}


//_____________________________________________________________________________
Double_t RooRealSumPdf::expectedEvents(const RooArgSet* nset) const
{
  //  return getNorm(nset) ;
  Double_t n = getNorm(nset) ;  
  if (n<0) {
    logEvalError("Expected number of events is negative") ;
  }
  return n ;
}


//_____________________________________________________________________________
void RooRealSumPdf::printMetaArgs(ostream& os) const 
{
  // Customized printing of arguments of a RooRealSumPdf to more intuitively reflect the contents of the
  // product operator construction

  _funcIter->Reset() ;
  _coefIter->Reset() ;

  Bool_t first(kTRUE) ;
    
  RooAbsArg* coef, *func ;
  if (_coefList.getSize()!=0) { 
    while((coef=(RooAbsArg*)_coefIter->Next())) {
      if (!first) {
	os << " + " ;
      } else {
	first = kFALSE ;
      }
      func=(RooAbsArg*)_funcIter->Next() ;
      os << coef->GetName() << " * " << func->GetName() ;
    }
    func = (RooAbsArg*) _funcIter->Next() ;
    if (func) {
      os << " + [%] * " << func->GetName() ;
    }
  } else {
    
    while((func=(RooAbsArg*)_funcIter->Next())) {
      if (!first) {
	os << " + " ;
      } else {
	first = kFALSE ;
      }
      os << func->GetName() ; 
    }  
  }

  os << " " ;    
}
