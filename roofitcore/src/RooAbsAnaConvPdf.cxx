/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAbsAnaConvPdf.cc,v 1.11 2005/12/01 16:10:17 wverkerke Exp $
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

// -- CLASS DESCRIPTION [PDF] --
// 
//  RooAbsAnaConvPdf is the base class of for PDFs that represents a
//  physics model that can be analytically convolved with a resolution model
//  
//  To achieve factorization between the physics model and the resolution
//  model, each physics model must be able to be written in the form
//           _ _                 _              _ 
//    Phys(x,a,b) = Sum_k coef_k(a) * basis_k(x,b)
//  
//  where basis_k are a limited number of functions in terms of the variable
//  to be convoluted and coef_k are coefficients independent of the convolution
//  variable.
//  
//  Classes derived from RooResolutionModel implement 
//         _ _                        _                  _
//   R_k(x,b,c) = Int(dx') basis_k(x',b) * resModel(x-x',c)
// 
//  which RooAbsAnaConvPdf uses to construct the pdf for [ Phys (x) R ] :
//          _ _ _                 _          _ _
//    PDF(x,a,b,c) = Sum_k coef_k(a) * R_k(x,b,c)
//
//  A minimal implementation of a RooAbsAnaConvPdf physics model consists of
//  
//  - A constructor that declares the required basis functions using the declareBasis() method.
//    The declareBasis() function assigns a unique identifier code to each declare basis
//
//  - An implementation of coefficient(Int_t code) returning the coefficient value for each
//    declared basis function
//
//  Optionally, analytical integrals can be provided for the coefficient functions. The
//  interface for this is quite similar to that for integrals of regular PDFs. Two functions,
//
//   Int_t getCoefAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars) 
//   Double_t coefAnalyticalIntegral(Int_t coef, Int_t code),
//
//  advertise the coefficient integration capabilities and implement them respectively.
//  Please see RooAbsPdf for additional details. Advertised analytical integrals must be
//  valid for all coefficients.


#include "RooFit.h"

#include "Riostream.h"
#include "Riostream.h"
#include "RooAbsAnaConvPdf.h"
#include "RooResolutionModel.h"
#include "RooRealVar.h"
#include "RooFormulaVar.h"
#include "RooConvGenContext.h"
#include "RooGenContext.h"
#include "RooTruthModel.h"
#include "RooConvCoefVar.h"
#include "RooNameReg.h"

ClassImp(RooAbsAnaConvPdf) 
;


RooAbsAnaConvPdf::RooAbsAnaConvPdf(const char *name, const char *title, 
				   const RooResolutionModel& model, RooRealVar& convVar) :
  RooAbsPdf(name,title), _isCopy(kFALSE),
  _model((RooResolutionModel*)&model), _convVar((RooRealVar*)&convVar),
  _convSet("convSet","Set of resModel X basisFunc convolutions",this),
  _convNormSet(0), _convSetIter(_convSet.createIterator()),
  _coefNormMgr(10),
  _codeReg(10)
{
  // Constructor. The supplied resolution model must be constructed with the same
  // convoluted variable as this physics model ('convVar')
  _convNormSet = new RooArgSet(convVar,"convNormSet") ;
}


RooAbsAnaConvPdf::RooAbsAnaConvPdf(const RooAbsAnaConvPdf& other, const char* name) : 
  RooAbsPdf(other,name), _isCopy(kTRUE),
  _model(other._model), _convVar(other._convVar), 
  _convSet("convSet",this,other._convSet),
  _basisList(other._basisList),
  _convNormSet(new RooArgSet(*other._convNormSet)),
  _convSetIter(_convSet.createIterator()),
  _coefNormMgr(other._coefNormMgr),
  _codeReg(other._codeReg)
{
  // Copy constructor

//   TIterator* iter = other._coefVarList.createIterator() ;
//   RooAbsArg* arg ;
//   while(((arg=(RooAbsArg*)iter->Next()))) {
//     _coefVarList.addOwned(*(RooAbsArg*)arg->Clone()) ;
//   }
//   delete iter ;
}



RooAbsAnaConvPdf::~RooAbsAnaConvPdf()
{
  // Destructor
  if (_convNormSet) {
    delete _convNormSet ;
  }
    
  delete _convSetIter ;

  if (!_isCopy) {
    TIterator* iter = _convSet.createIterator() ;
    RooAbsArg* arg ;
    while (((arg = (RooAbsArg*)iter->Next()))) {
      _convSet.remove(*arg) ;
      delete arg ;
    }
    delete iter ;
  }

}


Int_t RooAbsAnaConvPdf::declareBasis(const char* expression, const RooArgList& params) 
{
  // Declare a basis function for use in this physics model. The string expression 
  // must be a valid RooFormulVar expression representing the basis function, referring
  // to the convolution variable as '@0', and any additional parameters (supplied in
  // 'params' as '@1','@2' etc.
  //
  // The return value is a unique identifier code, that will be passed to coefficient()
  // to identify the basis function for which the coefficient is requested. If the
  // resolution model used does not support the declared basis function, code -1 is
  // returned. 
  //

  // Sanity check
  if (_isCopy) {
    cout << "RooAbsAnaConvPdf::declareBasis(" << GetName() << "): ERROR attempt to "
	 << " declare basis functions in a copied RooAbsAnaConvPdf" << endl ;
    return -1 ;
  }

  // Resolution model must support declared basis
  if (!_model->isBasisSupported(expression)) {
    cout << "RooAbsAnaConvPdf::declareBasis(" << GetName() << "): resolution model " 
	 << _model->GetName() 
	 << " doesn't support basis function " << expression << endl ;
    return -1 ;
  }

  // Instantiate basis function
  RooArgList basisArgs(*_convVar) ;
  basisArgs.add(params) ;

  TString basisName(expression) ;
  TIterator* iter = basisArgs.createIterator() ;
  RooAbsArg* arg  ;
  while(((arg=(RooAbsArg*)iter->Next()))) {
    basisName.Append("_") ;
    basisName.Append(arg->GetName()) ;
  }
  delete iter ;  

  RooFormulaVar* basisFunc = new RooFormulaVar(basisName,expression,basisArgs) ;
  basisFunc->setOperMode(operMode()) ;
  _basisList.addOwned(*basisFunc) ;

  // Instantiate resModel x basisFunc convolution
  RooAbsReal* conv = _model->convolution(basisFunc,this) ;
  if (!conv) {
    cout << "RooAbsAnaConvPdf::declareBasis(" << GetName() << "): unable to construct convolution with basis function '" 
	 << expression << "'" << endl ;
    return -1 ;
  }
  _convSet.add(*conv) ;

  return _convSet.index(conv) ;
}


Bool_t RooAbsAnaConvPdf::changeModel(const RooResolutionModel& newModel) 
{
  // Change the resolution model to given model
  TIterator* cIter = _convSet.createIterator() ;
  RooResolutionModel* conv ;
  RooArgList newConvSet ;
  Bool_t allOK(kTRUE) ;
  while(((conv=(RooResolutionModel*)cIter->Next()))) {

    // Build new resolution model
    RooResolutionModel* newConv = newModel.convolution((RooFormulaVar*)&conv->basis(),this) ;
    if (!newConvSet.add(*newConv)) {
      allOK = kFALSE ;
      break ;
    }
  }
  delete cIter ;

  // Check if all convolutions were succesfully built
  if (!allOK) {
    // Delete new basis functions created sofar
    TIterator* iter = newConvSet.createIterator() ;
    while(((conv=(RooResolutionModel*)iter->Next()))) delete conv ;
    delete iter ;

    return kTRUE ;
  }
  
  // Replace old convolutions with new set
  _convSet.removeAll() ;
  _convSet.addOwned(newConvSet) ;

  _model = (RooResolutionModel*) &newModel ;
  return kFALSE ;
}




RooAbsGenContext* RooAbsAnaConvPdf::genContext(const RooArgSet &vars, const RooDataSet *prototype, 
					       const RooArgSet* auxProto, Bool_t verbose) const 
{
  RooArgSet* modelDep = _model->getObservables(&vars) ;
  modelDep->remove(*convVar(),kTRUE,kTRUE) ;
  Int_t numAddDep = modelDep->getSize() ;
  delete modelDep ;

  if (dynamic_cast<RooTruthModel*>(_model)) {
    // Truth resolution model: use generic context explicitly allowing generation of convolution variable
    RooArgSet forceDirect(*convVar()) ;
    return new RooGenContext(*this,vars,prototype,auxProto,verbose,&forceDirect) ;
  } 

  // Check if physics PDF and resolution model can both directly generate the convolution variable
  RooArgSet dummy ;
  Bool_t pdfCanDir = (getGenerator(*convVar(),dummy) != 0) ;
  RooResolutionModel* conv = (RooResolutionModel*) _convSet.at(0) ;
  Bool_t resCanDir = conv && (conv->getGenerator(*convVar(),dummy)!=0) && conv->isDirectGenSafe(*convVar()) ;

  if (numAddDep>0 || !pdfCanDir || !resCanDir) {
    // Any resolution model with more dependents than the convolution variable
    // or pdf or resmodel do not support direct generation
    return new RooGenContext(*this,vars,prototype,auxProto,verbose) ;
  } 
  
  // Any other resolution model: use specialized generator context
  return new RooConvGenContext(*this,vars,prototype,auxProto,verbose) ;
}


Bool_t RooAbsAnaConvPdf::isDirectGenSafe(const RooAbsArg& arg) const 
{
  // All direct generation of convolution arg if model is truth model
  if (!TString(_convVar->GetName()).CompareTo(arg.GetName()) && 
      dynamic_cast<RooTruthModel*>(_model)) {
    return kTRUE ;
  }

  return RooAbsPdf::isDirectGenSafe(arg) ;
}



const RooRealVar* RooAbsAnaConvPdf::convVar() const
{
  // Return a pointer to the convolution variable instance used in the resolution model
  RooResolutionModel* conv = (RooResolutionModel*) _convSet.at(0) ;
  if (!conv) return 0 ;  
  return &conv->convVar() ;
}



Double_t RooAbsAnaConvPdf::evaluate() const
{
  // Calculate the current unnormalized value of the PDF
  //
  // PDF = sum_k coef_k * [ basis_k (x) ResModel ]
  //
  Double_t result(0) ;

  _convSetIter->Reset() ;
  RooAbsPdf* conv ;
  Int_t index(0) ;
  while(((conv=(RooAbsPdf*)_convSetIter->Next()))) {
    Double_t coef = coefficient(index++) ;
    if (coef!=0.) {
      result += conv->getVal(0)*coef ;
   }
  }
  
  return result ;
}


Int_t RooAbsAnaConvPdf::getAnalyticalIntegralWN(RooArgSet& allVars, 
	  				        RooArgSet& analVars, const RooArgSet* normSet2, const char* /*rangeName*/) const 
{
  // Handle trivial no-integration scenario
  if (allVars.getSize()==0) return 0 ;

  // Select subset of allVars that are actual dependents
  RooArgSet* allDeps = getObservables(allVars) ;
  RooArgSet* normSet = normSet2 ? getObservables(normSet2) : 0 ;

  RooAbsArg *arg ;
  RooResolutionModel *conv ;

  RooArgSet* intSetAll = new RooArgSet(*allDeps,"intSetAll") ;

  // Split intSetAll in coef/conv parts
  RooArgSet* intCoefSet = new RooArgSet("intCoefSet") ; 
  RooArgSet* intConvSet = new RooArgSet("intConvSet") ;
  TIterator* varIter  = intSetAll->createIterator() ;
  TIterator* convIter = _convSet.createIterator() ;

  while(((arg=(RooAbsArg*) varIter->Next()))) {
    Bool_t ok(kTRUE) ;
    convIter->Reset() ;
    while(((conv=(RooResolutionModel*) convIter->Next()))) {
      if (conv->dependsOn(*arg)) ok=kFALSE ;
    }
    
    if (ok) {
      intCoefSet->add(*arg) ;
    } else {
      intConvSet->add(*arg) ;
    }
    
  }
  delete varIter ;


  // Split normSetAll in coef/conv parts
  RooArgSet* normCoefSet = new RooArgSet("normCoefSet") ;  
  RooArgSet* normConvSet = new RooArgSet("normConvSet") ;
  RooArgSet* normSetAll = normSet ? (new RooArgSet(*normSet,"normSetAll")) : 0 ;
  if (normSetAll) {
    varIter  =  normSetAll->createIterator() ;
    while(((arg=(RooAbsArg*) varIter->Next()))) {
      Bool_t ok(kTRUE) ;
      convIter->Reset() ;
      while(((conv=(RooResolutionModel*) convIter->Next()))) {
	if (conv->dependsOn(*arg)) ok=kFALSE ;
      }
      
      if (ok) {
	normCoefSet->add(*arg) ;
      } else {
	normConvSet->add(*arg) ;
      }
      
    }
    delete varIter ;
  }
  delete convIter ;

//    cout << "allVars     = " ; allVars.Print("1") ;
//    cout << "allDeps     = " ; allDeps->Print("1") ;
//    cout << "normSet     = " ; if (normSet) normSet->Print("1") ; else cout << "<none>" << endl ;
//    cout << "intCoefSet  = " << intCoefSet << " " ; intCoefSet->Print("1") ;
//    cout << "intConvSet  = " << intConvSet << " " ; intConvSet->Print("1") ;
//    cout << "normCoefSet = " << normCoefSet << " " ; normCoefSet->Print("1") ;
//    cout << "normConvSet = " << normConvSet << " " ; normConvSet->Print("1") ;

  if (intCoefSet->getSize()==0) {
    delete intCoefSet ; intCoefSet=0 ;
  }
  if (intConvSet->getSize()==0) {
    delete intConvSet ; intConvSet=0 ;
  }
  if (normCoefSet->getSize()==0) {
    delete normCoefSet ; normCoefSet=0 ;
  }
  if (normConvSet->getSize()==0) {
    delete normConvSet ; normConvSet=0 ;
  }


  // Store integration configuration in registry
  Int_t masterCode(0) ;
  Int_t tmp(0) ;

  masterCode = _codeReg.store(&tmp,1,intCoefSet,intConvSet,normCoefSet,normConvSet)+1 ; // takes ownership of all sets

  analVars.add(*allDeps) ;
  delete allDeps ;
  if (normSet) delete normSet ;
  if (normSetAll) delete normSetAll ;
  delete intSetAll ;

//   cout << this << "---> masterCode = " << masterCode << endl ;
  
  return masterCode  ;
}



Double_t RooAbsAnaConvPdf::analyticalIntegralWN(Int_t code, const RooArgSet* normSet, const char* rangeName) const 
{
  // Return analytical integral defined by given scenario code.
  //
  // For unnormalized integrals this is
  //                    _                _     
  //   PDF = sum_k Int(dx) coef_k * Int(dy) [ basis_k (x) ResModel ].
  //       _
  // where x is the set of coefficient dependents to be integrated
  // and y the set of basis function dependents to be integrated. 
  //
  // For normalized integrals this becomes
  //
  //         sum_k Int(dx) coef_k * Int(dy) [ basis_k (x) ResModel ].
  //  PDF =  --------------------------------------------------------
  //         sum_k Int(dv) coef_k * Int(dw) [ basis_k (x) ResModel ].
  //
  // where x is the set of coefficient dependents to be integrated,
  // y the set of basis function dependents to be integrated,
  // v is the set of coefficient dependents over which is normalized and
  // w is the set of basis function dependents over which is normalized.
  //
  // Set x must be contained in v and set y must be contained in w.
  //

  // WVE needs adaptation to handle new rangeName feature

  // Handle trivial passthrough scenario
  if (code==0) return getVal(normSet) ;

//   cout << "RooConvPdf::aiWN code = " << code << endl ;

  // Unpack master code
  RooArgSet *intCoefSet, *intConvSet, *normCoefSet, *normConvSet ;
  _codeReg.retrieve(code-1,intCoefSet,intConvSet,normCoefSet,normConvSet) ;
//   cout << "ai: mastercode = " << code << endl ;
//   cout << "intCoefSet: " << intCoefSet << " " ; if (intCoefSet) intCoefSet->Print("1") ; else cout << "<none>" << endl ;
//   cout << "intConvSet: " << intConvSet << " "  ; if (intConvSet) intConvSet->Print("1") ; else cout << "<none>" << endl ;
//   cout << "normCoefSet: " << normCoefSet << " "  ; if (normCoefSet) normCoefSet->Print("1") ; else cout << "<none>" << endl ;
//   cout << "normConvSet: " << normConvSet << " "  ; if (normConvSet) normConvSet->Print("1") ; else cout << "<none>" << endl ;

  //const_cast<RooAbsAnaConvPdf*>(this)->printCompactTree() ;

  RooResolutionModel* conv ;
  Int_t index(0) ;
  Double_t answer(0) ;
  _convSetIter->Reset() ;

  if (normCoefSet==0&&normConvSet==0) {

    // Integral over unnormalized function
    Double_t integral(0) ;
    while(((conv=(RooResolutionModel*)_convSetIter->Next()))) {
      Double_t coef = getCoefNorm(index++,intCoefSet,rangeName) ; 
      //cout << "coefInt[" << index << "] = " << coef << " " ; intCoefSet->Print("1") ; 
      if (coef!=0) integral += coef*(rangeName ? conv->getNormObj(0,intConvSet,RooNameReg::ptr(rangeName))->getVal() :  conv->getNorm(intConvSet) ) ; 
    }
    answer = integral ;
    
  } else {

    // Integral over normalized function
    Double_t integral(0) ;
    Double_t norm(0) ;
    while(((conv=(RooResolutionModel*)_convSetIter->Next()))) {

      Double_t coefInt = getCoefNorm(index,intCoefSet,rangeName) ;
      Double_t term = (rangeName ? conv->getNormObj(0,intConvSet,RooNameReg::ptr(rangeName))->getVal() : conv->getNorm(intConvSet) ) ;
      //cout << "coefInt[" << index << "] = " << coefInt << "*" << term << " " << (intCoefSet?*intCoefSet:RooArgSet()) << endl ;
      if (coefInt!=0) integral += coefInt*term ;

      Double_t coefNorm = getCoefNorm(index,normCoefSet) ;
      term = conv->getNorm(normConvSet) ;
      //cout << "coefNorm[" << index << "] = " << coefNorm << "*" << term << " " << (normCoefSet?*normCoefSet:RooArgSet()) << endl ;
      if (coefNorm!=0) norm += coefNorm*term ;

      index++ ;
    }
    answer = integral/norm ;    
  }

  return answer ;
}



Int_t RooAbsAnaConvPdf::getCoefAnalyticalIntegral(RooArgSet& /*allVars*/, RooArgSet& /*analVars*/, const char* /*rangeName*/) const 
{
  // Default implementation of function advertising integration capabilities: no integrals
  // are advertised.

  return 0 ;
}



Double_t RooAbsAnaConvPdf::coefAnalyticalIntegral(Int_t coef, Int_t code, const char* /*rangeName*/) const 
{
  // Default implementation of function implementing advertised integrals. Only
  // the pass-through scenario (no integration) is implemented.

  if (code==0) return coefficient(coef) ;
  cout << "RooAbsAnaConvPdf::coefAnalyticalIntegral(" << GetName() << ") ERROR: unrecognized integration code: " << code << endl ;
  assert(0) ;
  return 1 ;
}



Bool_t RooAbsAnaConvPdf::forceAnalyticalInt(const RooAbsArg& /*dep*/) const
{
  // This function forces RooRealIntegral to offer all integration dependents
  // to RooAbsAnaConvPdf::getAnalyticalIntegralWN() for consideration for
  // analytical integration, if RRI considers this to be unsafe (e.g. due
  // to hidden Jacobian terms). 
  //
  // RooAbsAnaConvPdf will not attempt to actually integrate all these dependents
  // but feed them to the resolution models integration interface, which will
  // make the final determination on how to integrate these dependents.

  return kTRUE ;
}                                                                                                                         
               


Double_t RooAbsAnaConvPdf::getCoefNorm(Int_t coefIdx, const RooArgSet* nset, const char* rangeName) const 
{
  if (nset==0) return coefficient(coefIdx) ;

  RooArgList* normList = _coefNormMgr.getNormList(this,nset,0,0,RooNameReg::ptr(rangeName)) ;
  if (!normList) {

    // Make list of coefficient normalizations
    Int_t i ;
    normList = new RooArgList("coefNormList") ;
    if (_coefVarList.getSize()==0) makeCoefVarList() ;  

    for (i=0 ; i<_coefVarList.getSize() ; i++) {
      RooAbsReal* coefInt = static_cast<RooAbsReal&>(*_coefVarList.at(i)).createIntegral(*nset,rangeName) ;
      normList->addOwned(*coefInt) ;      
    }  

    _coefNormMgr.setNormList(this,nset,0,normList,RooNameReg::ptr(rangeName)) ;
  }

  //cout << "RooAbsAnaConvPdf::getCoefNorm(" << GetName() << ") coefIdx = " << coefIdx << " nset = " << (nset?*nset:RooArgSet()) << " value = " <<  ((RooAbsReal*)normList->at(coefIdx))->getVal() << endl ;
  return ((RooAbsReal*)normList->at(coefIdx))->getVal() ;
}



void RooAbsAnaConvPdf::makeCoefVarList() const
{
  // Build complete list of coefficient variables 
  RooArgSet* coefVars = getParameters((RooArgSet*)0) ;
  TIterator* iter = coefVars->createIterator() ;
  RooAbsArg* arg ;
  Int_t i ;
  while(((arg=(RooAbsArg*)iter->Next()))) {
    for (i=0 ; i<_convSet.getSize() ; i++) {
      if (_convSet.at(i)->dependsOn(*arg)) {
	coefVars->remove(*arg,kTRUE) ;
      }
    }
  }
  delete iter ;
  
  // Instantate a coefficient variables
  for (i=0 ; i<_convSet.getSize() ; i++) {
    RooAbsReal* coefVar = new RooConvCoefVar("coefVar","coefVar",*this,i,coefVars) ;
    _coefVarList.addOwned(*coefVar) ;
  }
  
  delete coefVars ;
}


Bool_t RooAbsAnaConvPdf::redirectServersHook(const RooAbsCollection& /*newServerList*/, Bool_t /*mustReplaceAll*/, 
				       Bool_t /*nameChange*/, Bool_t /*isRecursive*/) 
{
  // Throw away cache of coefCoefVars and normalization integrals constructed in terms of those coefficients
  _coefVarList.removeAll() ;
  _coefNormMgr.reset() ;
  return kFALSE ;
}





Bool_t RooAbsAnaConvPdf::syncNormalizationPreHook(RooAbsReal* /*norm*/,const RooArgSet* nset) const 
{
  // Overload of hook function in RooAbsPdf::syncNormalization(). This functions serves
  // two purposes: 
  //
  //   - Modify default normalization behaviour of RooAbsPdf: integration requests over
  //     unrelated variables are properly executed (introducing a trivial multiplication
  //     for each unrelated dependent). This is necessary if composite resolution models
  //     are used in which the components do not necessarily all have the same set
  //     of dependents.
  //
  //   - Built the sub set of normalization dependents that is contained the basis function/
  //     resolution model convolution (to be used in syncNormalizationPostHook().

  delete _convNormSet ;
  RooArgSet convNormArgs("convNormArgs") ;

  // Make iterator over data set arguments
  TIterator* dsIter = nset->createIterator() ;
  RooAbsArg* dsArg ;

  // Make iterator over convolution integrals
  TIterator* cvIter = _convSet.createIterator() ;
  RooResolutionModel* conv ;

  // Build integration list for convolutions
  while (((dsArg = (RooAbsArg*) dsIter->Next()))) {
    cvIter->Reset() ;
    while(((conv = (RooResolutionModel*) cvIter->Next()))) {
      if (conv->dependsOn(*dsArg)) {
	// Add any data set variable that occurs in any convolution integral
	convNormArgs.add(*dsArg) ;
      }
    }
  }
  delete dsIter ;
  delete cvIter ;
  _convNormSet = new RooArgSet(convNormArgs,"convNormSet") ;
  
  return kFALSE ;
}




void RooAbsAnaConvPdf::syncNormalizationPostHook(RooAbsReal* /*norm*/,const RooArgSet* /*nset*/) const 
{
  // Overload of hook function in RooAbsPdf::syncNormalization(). This function propagates
  // the syncNormalization() call to all basis-function/resolution-model convolution component
  // objects and fixes the physics models client-server links by adding each variable that
  // serves any of the convolution objects normalizations. PDFs by default have all client-server
  // links that control the unnormalized value (as returned by evaluate()), but convoluted PDFs
  // have a non-trivial normalization term that may introduce dependencies on additional server
  // that exclusively appear in the normalization.

  TIterator* cvIter = _convSet.createIterator() ;
  RooResolutionModel* conv ;

  // Make convolution normalizations servers of the convoluted pdf normalization
  while(((conv=(RooResolutionModel*)cvIter->Next()))) {
    conv->syncNormalization(_convNormSet) ;

    // Add leaf node servers of convolution normalization integrals to our normalization
    // integral, except for the integrated variables

    RooArgSet leafList("leafNodeServerList") ;
    conv->normLeafServerList(leafList) ;
    TIterator* sIter = leafList.createIterator() ;

    RooAbsArg* server ;
    while(((server=(RooAbsArg*)sIter->Next()))) {
      if (!_norm->findServer(*server)) {
	_norm->addServer(*server,kTRUE,kFALSE) ;
      }
    }
    delete sIter ;

  }  
  delete cvIter ;

  return ;
}


void RooAbsAnaConvPdf::printToStream(ostream& os, PrintOption opt, TString indent) const {
  // Print info about this object to the specified stream. In addition to the info
  // from RooAbsPdf::printToStream() we add:
  //
  //   Verbose : detailed information on convolution integrals

  RooAbsPdf::printToStream(os,opt,indent);
  if(opt >= Verbose) {
    os << indent << "--- RooAbsAnaConvPdf ---" << endl;
    TIterator* iter = _convSet.createIterator() ;
    RooResolutionModel* conv ;
    while (((conv=(RooResolutionModel*)iter->Next()))) {
      conv->printToStream(os,Verbose,"    ") ;
    }
  }
}


