/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   17-Jan-2002 WV Created initial version
 *
 * Copyright (C) 2002 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [PDF] --
//
// Class RooRealSumPdf implements a PDF constructed from a sum of
// real valued objects, i.e.
//
//                 Sum(i=1,n-1) coef_i * func_i(x) + [ 1 - (Sum(i=1,n-1) coef_i ] * func_n(x)
//   pdf(x) =    ------------------------------------------------------------------------------
//             Sum(i=1,n-1) coef_i * Int(func_i)dx + [ 1 - (Sum(i=1,n-1) coef_i ] * Int(func_n)dx
//
//
// where coef_i and func_i are RooAbsReal objects, and x is the collection of dependents. 
// In the present version coef_i may not depend on x, but this limitation will be removed in the future
//

#include "TIterator.h"
#include "TList.h"
#include "RooFitCore/RooRealSumPdf.hh"
#include "RooFitCore/RooRealProxy.hh"
#include "RooFitCore/RooPlot.hh"
#include "RooFitCore/RooRealVar.hh"
#include "RooFitCore/RooAddGenContext.hh"
#include "RooFitCore/RooRealConstant.hh"
#include "RooFitCore/RooRealIntegral.hh"

ClassImp(RooRealSumPdf)
;


RooRealSumPdf::RooRealSumPdf(const char *name, const char *title) :
  RooAbsPdf(name,title), 
  _coefList("coefList","List of coefficients",this),
  _funcList("funcList","List of functions",this),
  _lastFuncIntSet(0),
  _lastFuncNormSet(0),
  _funcIntList(0),
  _funcNormList(0),
  _codeReg(10)  
{
  // Dummy constructor 
  _funcIter   = _funcList.createIterator() ;
  _coefIter  = _coefList.createIterator() ;
}


RooRealSumPdf::RooRealSumPdf(const char *name, const char *title,
		     RooAbsReal& func1, RooAbsReal& func2, RooAbsReal& coef1) : 
  RooAbsPdf(name,title),
  _coefList("coefList","List of coefficients",this),
  _funcList("funcProxyList","List of functions",this),
  _lastFuncIntSet(0),
  _lastFuncNormSet(0),
  _funcIntList(0),
  _funcNormList(0),
  _codeReg(10)
{
  // Special constructor with two functions and one coefficient

  _funcIter  = _funcList.createIterator() ;
  _coefIter = _coefList.createIterator() ;

  _funcList.add(func1) ;  
  _funcList.add(func2) ;
  _coefList.add(coef1) ;

}

RooRealSumPdf::RooRealSumPdf(const char *name, const char *title, const RooArgList& funcList, const RooArgList& coefList) :
  RooAbsPdf(name,title),
  _coefList("coefList","List of coefficients",this),
  _funcList("funcProxyList","List of functions",this),
  _lastFuncIntSet(0),
  _lastFuncNormSet(0),
  _funcIntList(0),
  _funcNormList(0),
  _codeReg(10)
{ 
  // Constructor from list of functions and list of coefficients.
  // Each func list element (i) is paired with coefficient list element (i).
  // The number of coefficients must be one less than to the number of functions,
  //
  // All functions and coefficients must inherit from RooAbsReal. 

  if (funcList.getSize()!=coefList.getSize()+1) {
    cout << "RooRealSumPdf::RooRealSumPdf(" << GetName() 
	 << ") number of pdfs and coefficients inconsistent, must have Nfunc=Ncoef+1" << endl ;
    assert(0) ;
  }

  _funcIter  = _funcList.createIterator() ;
  _coefIter = _coefList.createIterator() ;
 
  // Constructor with N functions and N or N-1 coefs
  TIterator* funcIter = funcList.createIterator() ;
  TIterator* coefIter = coefList.createIterator() ;
  RooAbsReal* func ;
  RooAbsReal* coef ;

  while(coef = (RooAbsReal*)coefIter->Next()) {
    func = (RooAbsReal*) funcIter->Next() ;

    if (!dynamic_cast<RooAbsReal*>(coef)) {
      cout << "RooRealSumPdf::RooRealSumPdf(" << GetName() << ") coefficient " << coef->GetName() << " is not of type RooAbsReal, ignored" << endl ;
      continue ;
    }
    if (!dynamic_cast<RooAbsReal*>(func)) {
      cout << "RooRealSumPdf::RooRealSumPdf(" << GetName() << ") func " << func->GetName() << " is not of type RooAbsReal, ignored" << endl ;
      continue ;
    }
    _funcList.add(*func) ;
    _coefList.add(*coef) ;    
  }

  func = (RooAbsReal*) funcIter->Next() ;
  if (func) {
    if (!dynamic_cast<RooAbsReal*>(func)) {
      cout << "RooRealSumPdf::RooRealSumPdf(" << GetName() << ") last func " << coef->GetName() << " is not of type RooAbsReal, fatal error" << endl ;
      assert(0) ;
    }
    _funcList.add(*func) ;  
  } 
  
  delete funcIter ;
  delete coefIter  ;
}




RooRealSumPdf::RooRealSumPdf(const RooRealSumPdf& other, const char* name) :
  RooAbsPdf(other,name),
  _coefList("coefList",this,other._coefList),
  _funcList("funcProxyList",this,other._funcList),
  _lastFuncIntSet(0),
  _lastFuncNormSet(0),
  _funcIntList(0),
  _funcNormList(0),
  _codeReg(other._codeReg)
{
  // Copy constructor

  _funcIter  = _funcList.createIterator() ;
  _coefIter = _coefList.createIterator() ;
}


RooRealSumPdf::~RooRealSumPdf()
{
  // Destructor
  delete _funcIter ;
  delete _coefIter ;

  if (_funcIntList) delete _funcIntList ;
  if (_funcNormList) delete _funcNormList ;		     
}






Double_t RooRealSumPdf::evaluate() const 
{
  // Calculate the current value

  const RooArgSet* nset = _funcList.nset() ;

  Double_t value(0) ;

  // Do running sum of coef/func pairs, calculate lastCoef.
  _funcIter->Reset() ;
  _coefIter->Reset() ;
  RooAbsReal* coef ;
  RooAbsReal* func ;
      
  // N funcs, N-1 coefficients 
  Double_t lastCoef(1) ;
  while(coef=(RooAbsReal*)_coefIter->Next()) {
    func = (RooAbsReal*)_funcIter->Next() ;
    Double_t coefVal = coef->getVal(nset) ;
    if (coefVal) {
      value += func->getVal(nset)*coefVal ;
      lastCoef -= coef->getVal(nset) ;
    }
  }
  
  // Add last func with correct coefficient
  func = (RooAbsReal*) _funcIter->Next() ;
  value += func->getVal(nset)*lastCoef ;
  
  // Warn about coefficient degeneration
  if (lastCoef<0 || lastCoef>1) {
    cout << "RooRealSumPdf::evaluate(" << GetName() 
	 << " WARNING: sum of FUNC coefficients not in range [0-1], value=" 
	 << 1-lastCoef << endl ;
  } 
  
  return value ;
}




Bool_t RooRealSumPdf::checkDependents(const RooArgSet* nset) const 
{
  // Check if FUNC is valid for given normalization set.
  // Coeffient and FUNC must be non-overlapping, but func-coefficient 
  // pairs may overlap each other
  //
  // For the moment, coefficients may not be dependents or derive
  // from dependents

  Bool_t ret(kFALSE) ;

  _funcIter->Reset() ;
  _coefIter->Reset() ;
  RooAbsReal* coef ;
  RooAbsReal* func ;
  while(coef=(RooAbsReal*)_coefIter->Next()) {
    func = (RooAbsReal*)_funcIter->Next() ;
    if (func->dependentOverlaps(nset,*coef)) {
      cout << "RooRealSumPdf::checkDependents(" << GetName() << "): ERROR: coefficient " << coef->GetName() 
	   << " and FUNC " << func->GetName() << " have one or more dependents in common" << endl ;
      ret = kTRUE ;
    }
    if (coef->dependsOn(*nset)) {
      cout << "RooRealPdf::checkDependents(" << GetName() << "): ERROR coefficient " << coef->GetName() 
	   << " depends on one or more of the following dependents" ; nset->Print("1") ;
      ret = kTRUE ;
    }
  }
  
  return ret ;
}




Int_t RooRealSumPdf::getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& analVars, const RooArgSet* normSet2) const 
{
  // Handle trivial no-integration scenario
  if (allVars.getSize()==0) return 0 ;
  if (_forceNumInt) return 0 ;

  // Select subset of allVars that are actual dependents
  RooArgSet* allDeps = getDependents(allVars) ;
  RooArgSet* normSet = normSet2 ? getDependents(normSet2) : 0 ;

  _funcIter->Reset() ;
  _coefIter->Reset() ;

//   cout << "allDeps = " << allDeps << " " ; allDeps->Print("1") ;
//   cout << "normSet = " ; if (normSet) normSet->Print("1") ; else cout << "<none>" << endl ;

  analVars.add(*allDeps) ;

  Int_t tmp(0) ;
  Int_t masterCode = _codeReg.store(&tmp,1,allDeps,normSet) + 1 ;
//   cout << "RooRealSumPdf::getAI masterCode = " << masterCode << endl ;

  return masterCode ;
}




Double_t RooRealSumPdf::analyticalIntegralWN(Int_t code, const RooArgSet* normSet2) const 
{
  // Handle trivial passthrough scenario
  if (code==0) return getVal(normSet2) ;

//   cout << "RooRealSumPdf::aiWN code = " << code << endl ;

  RooArgSet *allDeps, *normSet ;
  const Int_t* tmp = _codeReg.retrieve(code-1,allDeps,normSet) ;
  syncFuncIntList(allDeps) ;
  if (normSet) syncFuncNormList(normSet) ;

  // Do running sum of coef/func pairs, calculate lastCoef.
//     cout << "_funcIntList = " << _funcIntList << endl ;
  TIterator* funcIntIter = _funcIntList->createIterator() ;
  _coefIter->Reset() ;
  RooAbsReal* coef ;
  RooAbsReal* funcInt ;

  Double_t value(0) ;

  // N funcs, N-1 coefficients 
  Double_t lastCoef(1) ;
  while(coef=(RooAbsReal*)_coefIter->Next()) {
    funcInt = (RooAbsReal*)funcIntIter->Next() ;
    Double_t coefVal = coef->getVal(normSet) ;
    if (coefVal) {
      value += funcInt->getVal()*coefVal ;
//       cout << "aiWN value += " << funcInt->getVal() << " * " << coefVal << endl ;
      lastCoef -= coef->getVal(normSet) ;
    }
  }
  
  // Add last func with correct coefficient
  funcInt = (RooAbsReal*) funcIntIter->Next() ;
  value += funcInt->getVal()*lastCoef ;
//   cout << "aiWN value += " << funcInt->getVal() << " * " << lastCoef << endl ;
  
  // Warn about coefficient degeneration
  if (lastCoef<0 || lastCoef>1) {
    cout << "RooRealSumPdf::evaluate(" << GetName() 
	 << " WARNING: sum of FUNC coefficients not in range [0-1], value=" 
	 << 1-lastCoef << endl ;
  } 
  delete funcIntIter ;
  
  Double_t normVal(1) ;
  if (normSet) {
    normVal = 0 ;

    // N funcs, N-1 coefficients 
    RooAbsReal* funcNorm ;
//     cout << "_funcNormList = " << _funcNormList << endl ;
    TIterator* funcNormIter = _funcNormList->createIterator() ;
    _coefIter->Reset() ;
    while(coef=(RooAbsReal*)_coefIter->Next()) {
      funcNorm = (RooAbsReal*)funcNormIter->Next() ;
      Double_t coefVal = coef->getVal(normSet) ;
      if (coefVal) {
	normVal += funcNorm->getVal()*coefVal ;
// 	cout << "aiWN norm += " << funcNorm->getVal() << " * " << coefVal << endl ;
      }
    }
    
    // Add last func with correct coefficient
    funcNorm = (RooAbsReal*) funcNormIter->Next() ;
    normVal += funcNorm->getVal()*lastCoef ;
//     cout << "aiWN norm += " << funcNorm->getVal() << " * " << lastCoef << endl ;
    
    delete funcNormIter ;      
  }


//   cout << "RGS:aiWN value = " << value << " / " << normVal << endl ;
  return value / normVal;
}



void RooRealSumPdf::syncFuncIntList(const RooArgSet* intSet) const
{
  if (intSet==_lastFuncIntSet) return ;
  _lastFuncIntSet = (RooArgSet*) intSet ;

//   cout << "RooRealSumPdf::syncFuncIntList: remaking integrals" << endl ;

  if (_funcIntList) delete _funcIntList ;


  // Make list of function integrals
  _funcIntList = new RooArgList ;  
  _funcIter->Reset() ;
  RooAbsReal *func, *coef ;
  while(func=(RooAbsReal*)_funcIter->Next()) {
    RooAbsReal* funcInt = func->createIntegral(*intSet) ;
    _funcIntList->addOwned(*funcInt) ;
  }
}



void RooRealSumPdf::syncFuncNormList(const RooArgSet* normSet) const 
{
  if (normSet==_lastFuncNormSet) return ;
  _lastFuncNormSet = (RooArgSet*) normSet ;

  if (_funcNormList) delete _funcNormList ;

//   cout << "RooRealSumPdf::syncFuncNormList: remaking integrals" << endl ;
  
  // Make list of function normalization integrals
  _funcNormList = new RooArgList ;  
  RooAbsReal *func, *coef ;
  _funcIter->Reset() ;
  while(func=(RooAbsReal*)_funcIter->Next()) {
    RooAbsReal* funcInt = func->createIntegral(*normSet) ;
    _funcNormList->addOwned(*funcInt) ;
  }
}

