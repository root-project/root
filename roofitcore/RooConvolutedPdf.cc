/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooConvolutedPdf.cc,v 1.26 2001/11/14 18:42:37 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   05-Jun-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [PDF] --
// 
//  RooConvolutedPdf is the base class of for PDFs that represents a
//  physics model that can be convoluted with a resolution model
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
//  which RooConvolutedPdf uses to construct the pdf for [ Phys (x) R ] :
//          _ _ _                 _          _ _
//    PDF(x,a,b,c) = Sum_k coef_k(a) * R_k(x,b,c)
//
//  A minimal implementation of a RooConvolutedPdf physics model consists of
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


#include <iostream.h>
#include "RooFitCore/RooConvolutedPdf.hh"
#include "RooFitCore/RooResolutionModel.hh"
#include "RooFitCore/RooRealVar.hh"
#include "RooFitCore/RooFormulaVar.hh"
#include "RooFitCore/RooConvGenContext.hh"
#include "RooFitCore/RooGenContext.hh"
#include "RooFitCore/RooTruthModel.hh"
#include "RooFitCore/RooConvCoefVar.hh"

ClassImp(RooConvolutedPdf) 
;


RooConvolutedPdf::RooConvolutedPdf(const char *name, const char *title, 
				   const RooResolutionModel& model, RooRealVar& convVar) :
  RooAbsPdf(name,title), _isCopy(kFALSE),
  _model((RooResolutionModel*)&model), _convVar((RooRealVar*)&convVar),
  _convSet("convSet","Set of resModel X basisFunc convolutions",this),
  _convNormSet(0), _convSetIter(_convSet.createIterator()),
  _codeReg(10), _lastCoefNormSet(0)
{
  // Constructor. The supplied resolution model must be constructed with the same
  // convoluted variable as this physics model ('convVar')
  _convNormSet = new RooArgSet(convVar,"convNormSet") ;
}


RooConvolutedPdf::RooConvolutedPdf(const RooConvolutedPdf& other, const char* name) : 
  RooAbsPdf(other,name), _convVar(0), _isCopy(kTRUE),
  _convSet("convSet",this,other._convSet),
  _convNormSet(new RooArgSet(*other._convNormSet)),
  _convSetIter(_convSet.createIterator()),
  _codeReg(other._codeReg),
  _model(other._model),
  _basisList(other._basisList),
  _lastCoefNormSet(0)
{
  // Copy constructor

  TIterator* iter = other._coefVarList.createIterator() ;
  RooAbsArg* arg ;
  while(arg=(RooAbsArg*)iter->Next()) {
    _coefVarList.addOwned(*(RooAbsArg*)arg->Clone()) ;
  }
  delete iter ;
}



RooConvolutedPdf::~RooConvolutedPdf()
{
  // Destructor
  if (_convNormSet) {
    delete _convNormSet ;
  }
    
  delete _convSetIter ;

  if (!_isCopy) {
    TIterator* iter = _convSet.createIterator() ;
    RooAbsArg* arg ;
    while (arg = (RooAbsArg*)iter->Next()) {
      _convSet.remove(*arg) ;
      delete arg ;
    }
    delete iter ;
  }

}


Int_t RooConvolutedPdf::declareBasis(const char* expression, const RooArgList& params) 
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
  if (!_model || !_convVar) {
    cout << "RooConvolutedPdf::declareBasis(" << GetName() << "): ERROR attempt to "
	 << " declare basis functions in a copied RooConvolutedPdf" << endl ;
    return -1 ;
  }

  // Resolution model must support declared basis
  if (!_model->isBasisSupported(expression)) {
    cout << "RooConvolutedPdf::declareBasis(" << GetName() << "): resolution model " 
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
  while(arg=(RooAbsArg*)iter->Next()) {
    basisName.Append("_") ;
    basisName.Append(arg->GetName()) ;
  }
  delete iter ;  

  RooFormulaVar* basisFunc = new RooFormulaVar(basisName,expression,basisArgs) ;
  _basisList.addOwned(*basisFunc) ;

  // Instantiate resModel x basisFunc convolution
  RooAbsReal* conv = _model->convolution(basisFunc,this) ;
  if (!conv) {
    cout << "RooConvolutedPdf::declareBasis(" << GetName() << "): unable to construct convolution with basis function '" 
	 << expression << "'" << endl ;
    return -1 ;
  }
  _convSet.add(*conv) ;

  return _convSet.index(conv) ;
}


Bool_t RooConvolutedPdf::changeModel(const RooResolutionModel& newModel) 
{
  // Change the resolution model to given model
  TIterator* cIter = _convSet.createIterator() ;
  RooResolutionModel* conv ;
  RooArgList newConvSet ;
  Bool_t allOK(kTRUE) ;
  while(conv=(RooResolutionModel*)cIter->Next()) {

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
    while(conv=(RooResolutionModel*)iter->Next()) delete conv ;
    delete iter ;

    return kTRUE ;
  }
  
  // Replace old convolutions with new set
  _convSet.removeAll() ;
  _convSet.addOwned(newConvSet) ;

  _model = (RooResolutionModel*) &newModel ;
  return kFALSE ;
}




RooAbsGenContext* RooConvolutedPdf::genContext(const RooArgSet &vars, 
					       const RooDataSet *prototype, Bool_t verbose) const 
{
  RooArgSet* modelDep = _model->getDependents(&vars) ;
  modelDep->remove(*convVar(),kTRUE,kTRUE) ;
  Int_t numAddDep = modelDep->getSize() ;
  delete modelDep ;

  if (dynamic_cast<RooTruthModel*>(_model)) {
    // Truth resolution model: use generic context explicitly allowing generation of convolution variable
    RooArgSet forceDirect(*convVar()) ;
    return new RooGenContext(*this,vars,prototype,verbose,&forceDirect) ;
  } 

  // Check if physics PDF and resolution model can both directly generate the convolution variable
  RooArgSet dummy ;
  Bool_t pdfCanDir = (getGenerator(*convVar(),dummy) != 0) ;
  RooResolutionModel* conv = (RooResolutionModel*) _convSet.at(0) ;
  Bool_t resCanDir = conv && (conv->getGenerator(*convVar(),dummy)!=0) ;

  if (numAddDep>0 || !pdfCanDir || !resCanDir) {
    // Any resolution model with more dependents than the convolution variable
    // or pdf or resmodel do not support direct generation
    return new RooGenContext(*this,vars,prototype,verbose) ;
  } 
  
  // Any other resolution model: use specialized generator context
  return new RooConvGenContext(*this,vars,prototype,verbose) ;
}



const RooRealVar* RooConvolutedPdf::convVar() const
{
  // Return a pointer to the convolution variable instance used in the resolution model
  RooResolutionModel* conv = (RooResolutionModel*) _convSet.at(0) ;
  if (!conv) return 0 ;  
  return &conv->convVar() ;
}



Double_t RooConvolutedPdf::evaluate() const
{
  // Calculate the current unnormalized value of the PDF
  //
  // PDF = sum_k coef_k * [ basis_k (x) ResModel ]
  //
  Double_t result(0) ;

  _convSetIter->Reset() ;
  RooAbsPdf* conv ;
  Int_t index(0) ;
  while(conv=(RooAbsPdf*)_convSetIter->Next()) {
    Double_t coef = coefficient(index++) ;
    if (coef!=0.) {
      result += conv->getVal(0)*coef ;
   }
  }
  
  return result ;
}


Int_t RooConvolutedPdf::getAnalyticalIntegralWN(RooArgSet& allVars, 
	  				        RooArgSet& analVars, const RooArgSet* normSet) const 
{

  // Handle trivial no-integration scenario
  if (allVars.getSize()==0) return 0 ;

  RooAbsArg *arg ;
  RooResolutionModel *conv ;

  RooArgSet* intSetAll = new RooArgSet(allVars) ;

  // Split intSetAll in coef/conv parts
  RooArgSet* intCoefSet = new RooArgSet ; 
  RooArgSet* intConvSet = new RooArgSet ;
  TIterator* varIter  = intSetAll->createIterator() ;
  TIterator* convIter = _convSet.createIterator() ;

  while(arg=(RooAbsArg*) varIter->Next()) {
    Bool_t ok(kTRUE) ;
    convIter->Reset() ;
    while(conv=(RooResolutionModel*) convIter->Next()) {
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
  RooArgSet* normCoefSet = new RooArgSet ;  
  RooArgSet* normConvSet = new RooArgSet ;
  RooArgSet* normSetAll = normSet ? (new RooArgSet(*normSet)) : 0 ;
  if (normSetAll) {
    varIter  =  normSetAll->createIterator() ;
    while(arg=(RooAbsArg*) varIter->Next()) {
      Bool_t ok(kTRUE) ;
      convIter->Reset() ;
      while(conv=(RooResolutionModel*) convIter->Next()) {
	if (conv->dependsOn(*arg)) ok=kFALSE ;
      }
      
      if (ok) {
	normCoefSet->add(*arg) ;
      } else {
	normConvSet->add(*arg) ;
      }
      
    }
    delete varIter ;
    delete convIter ;
  }

//   cout << "allVars     = " ; allVars.Print("1") ;
//   cout << "normSet     = " ; if (normSet) normSet->Print("1") ; else cout << "<none>" << endl ;
//   cout << "intCoefSet  = " << intCoefSet << " " ; intCoefSet->Print("1") ;
//   cout << "intConvSet  = " << intConvSet << " " ; intConvSet->Print("1") ;
//   cout << "normCoefSet = " << normCoefSet << " " ; normCoefSet->Print("1") ;
//   cout << "normConvSet = " << normConvSet << " " ; normConvSet->Print("1") ;
  
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
  analVars.add(allVars) ;

//   cout << this << "---> masterCode = " << masterCode << endl ;
  
  return masterCode  ;
}



Double_t RooConvolutedPdf::analyticalIntegralWN(Int_t code, const RooArgSet* normSet) const 
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

  // Handle trivial passthrough scenario
  if (code==0) return getVal(normSet) ;

//   cout << "RooConvPdf::aiWN code = " << code << endl ;

  // Unpack master code
  RooArgSet *intCoefSet, *intConvSet, *normCoefSet, *normConvSet ;
  const Int_t* tmp = _codeReg.retrieve(code-1,intCoefSet,intConvSet,normCoefSet,normConvSet) ;
//   cout << "ai: mastercode = " << code << endl ;
//   cout << "intCoefSet: " << intCoefSet << " " ; if (intCoefSet) intCoefSet->Print("1") ; else cout << "<none>" << endl ;
//   cout << "intConvSet: " << intConvSet << " "  ; if (intConvSet) intConvSet->Print("1") ; else cout << "<none>" << endl ;
//   cout << "normCoefSet: " << normCoefSet << " "  ; if (normCoefSet) normCoefSet->Print("1") ; else cout << "<none>" << endl ;
//   cout << "normConvSet: " << normConvSet << " "  ; if (normConvSet) normConvSet->Print("1") ; else cout << "<none>" << endl ;

  RooResolutionModel* conv ;
  Int_t index(0) ;
  Double_t answer(0) ;
  _convSetIter->Reset() ;

  if (normCoefSet==0&&normConvSet==0) {

    // Integral over unnormalized function
    Double_t integral(0) ;
    while(conv=(RooResolutionModel*)_convSetIter->Next()) {
      Double_t coef = getCoefNorm(index++,intCoefSet) ; 
//       cout << "coefInt[" << index << "] = " << coef << " " ; intCoefSet->Print("1") ; 
      if (coef!=0) integral += coef*conv->getNorm(intConvSet) ;
    }
    answer = integral ;
    
  } else {

    // Integral over normalized function
    Double_t integral(0) ;
    Double_t norm(0) ;
    while(conv=(RooResolutionModel*)_convSetIter->Next()) {

      Double_t coefInt = getCoefNorm2(index,intCoefSet) ;
//       cout << "coefInt[" << index << "] = " << coefInt << " " ; intCoefSet->Print("1") ;
      if (coefInt!=0) integral += coefInt*conv->getNormSpecial(intConvSet) ;

      Double_t coefNorm = getCoefNorm(index,normCoefSet) ;
//       cout << "coefNorm[" << index << "] = " << coefNorm << " " ; normCoefSet->Print("1") ;
      if (coefNorm!=0) norm += coefNorm*conv->getNorm(normConvSet) ;

      index++ ;
    }
    answer = integral/norm ;    
  }

  return answer ;
}



Int_t RooConvolutedPdf::getCoefAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars) const 
{
  // Default implementation of function advertising integration capabilities: no integrals
  // are advertised.

  return 0 ;
}



Double_t RooConvolutedPdf::coefAnalyticalIntegral(Int_t coef, Int_t code) const 
{
  // Default implementation of function implementing advertised integrals. Only
  // the pass-through scenario (no integration) is implemented.

  if (code==0) return coefficient(coef) ;
  cout << "RooConvolutedPdf::coefAnalyticalIntegral(" << GetName() << ") ERROR: unrecognized integration code: " << code << endl ;
  assert(0) ;
  return 1 ;
}



Bool_t RooConvolutedPdf::forceAnalyticalInt(const RooAbsArg& dep) const
{
  // This function forces RooRealIntegral to offer all integration dependents
  // to RooConvolutedPdf::getAnalyticalIntegralWN() for consideration for
  // analytical integration, if RRI considers this to be unsafe (e.g. due
  // to hidden Jacobian terms). 
  //
  // RooConvolutedPdf will not attempt to actually integrate all these dependents
  // but feed them to the resolution models integration interface, which will
  // make the final determination on how to integrate these dependents.

  return kTRUE ;
}                                                                                                                         
               


Double_t RooConvolutedPdf::getCoefNorm(Int_t coefIdx, const RooArgSet* nset=0) const 
{
  if (nset==0) return coefficient(coefIdx) ;

  if (nset != _lastCoefNormSet) {
    syncCoefNormalizations(_coefNormList,nset) ;
    _lastCoefNormSet = (RooArgSet*) nset ;
  }

  return ((RooAbsReal*)_coefNormList.at(coefIdx))->getVal() ; ;
}




Double_t RooConvolutedPdf::getCoefNorm2(Int_t coefIdx, const RooArgSet* nset=0) const 
{
  if (nset==0) return coefficient(coefIdx) ;

  if (nset != _lastCoefNormSet2) {
    syncCoefNormalizations(_coefNormList2,nset) ;
    _lastCoefNormSet2 = (RooArgSet*) nset ;
  }

  return ((RooAbsReal*)_coefNormList2.at(coefIdx))->getVal() ; ;
}




void RooConvolutedPdf::syncCoefNormalizations(RooArgList& coefNormList, const RooArgSet* nset) const
{
  if (!nset) return ;

  // Remove all old normalizations
  coefNormList.removeAll() ;

  if (_coefVarList.getSize()==0) {
    
    // Build complete list of coefficient variables 
    RooArgSet* coefVars = getParameters((RooArgSet*)0) ;
    TIterator* iter = coefVars->createIterator() ;
    RooAbsArg* arg ;
    Int_t i ;
    while(arg=(RooAbsArg*)iter->Next()) {
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

  Int_t i ;
  for (i=0 ; i<_coefVarList.getSize() ; i++) {
    RooRealIntegral* coefInt = new RooRealIntegral("coefInt","coefInt",(RooAbsReal&)(*_coefVarList.at(i)),*nset) ;
    coefNormList.addOwned(*coefInt) ;
  }  
}



Bool_t RooConvolutedPdf::syncNormalizationPreHook(RooAbsReal* norm,const RooArgSet* nset) const 
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
  while (dsArg = (RooAbsArg*) dsIter->Next()) {
    cvIter->Reset() ;
    while(conv = (RooResolutionModel*) cvIter->Next()) {
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




void RooConvolutedPdf::syncNormalizationPostHook(RooAbsReal* norm,const RooArgSet* nset) const 
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
  while(conv=(RooResolutionModel*)cvIter->Next()) {
    conv->syncNormalization(_convNormSet) ;

    // Add leaf node servers of convolution normalization integrals to our normalization
    // integral, except for the integrated variables

    RooArgSet leafList("leafNodeServerList") ;
    conv->normLeafServerList(leafList) ;
    TIterator* sIter = leafList.createIterator() ;

    RooAbsArg* server ;
    while(server=(RooAbsArg*)sIter->Next()) {
      if (!_norm->findServer(*server)) {
	_norm->addServer(*server,kTRUE,kFALSE) ;
      }
    }
    delete sIter ;

  }  
  delete cvIter ;

  return ;
}



void RooConvolutedPdf::printToStream(ostream& os, PrintOption opt, TString indent) const {
  // Print info about this object to the specified stream. In addition to the info
  // from RooAbsPdf::printToStream() we add:
  //
  //   Verbose : detailed information on convolution integrals

  RooAbsPdf::printToStream(os,opt,indent);
  if(opt >= Verbose) {
    os << indent << "--- RooConvolutedPdf ---" << endl;
    TIterator* iter = _convSet.createIterator() ;
    RooResolutionModel* conv ;
    while (conv=(RooResolutionModel*)iter->Next()) {
      conv->printToStream(os,Verbose,"    ") ;
    }
  }
}




