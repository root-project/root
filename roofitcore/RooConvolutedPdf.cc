/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooConvolutedPdf.cc,v 1.16 2001/09/20 01:40:10 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   05-Jun-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION --
// 

#include <iostream.h>
#include "RooFitCore/RooConvolutedPdf.hh"
#include "RooFitCore/RooResolutionModel.hh"
#include "RooFitCore/RooRealVar.hh"
#include "RooFitCore/RooFormulaVar.hh"

ClassImp(RooConvolutedPdf) 
;


RooConvolutedPdf::RooConvolutedPdf(const char *name, const char *title, 
				   const RooResolutionModel& model, RooRealVar& convVar) :
  RooAbsPdf(name,title), _isCopy(kFALSE),
  _model((RooResolutionModel*)&model), _convVar((RooRealVar*)&convVar),
  _convSet("convSet","Set of resModel X basisFunc convolutions",this),
  _convNormSet(0), _convSetIter(_convSet.createIterator()),
  _codeReg(10)
{
  // Constructor
  _convNormSet = new RooArgSet(convVar,"convNormSet") ;
}


RooConvolutedPdf::RooConvolutedPdf(const RooConvolutedPdf& other, const char* name) : 
  RooAbsPdf(other,name), _model(0), _convVar(0), _isCopy(kTRUE),
  _convSet("convSet",this,other._convSet),
  _convNormSet(new RooArgSet(*other._convNormSet)),
  _convSetIter(_convSet.createIterator()),
  _codeReg(other._codeReg)
{
  // Copy constructor
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

  // Delete all basis functions we created 
  _basisList.Delete() ;
}


Int_t RooConvolutedPdf::declareBasis(const char* expression, const RooArgSet& params) 
{
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
  RooArgSet basisArgs(*_convVar) ;
  basisArgs.add(params) ;
  RooFormulaVar* basisFunc = new RooFormulaVar(expression,expression,basisArgs) ;
  _basisList.Add(basisFunc) ;

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



const RooRealVar* RooConvolutedPdf::convVar() const
{
  RooResolutionModel* conv = (RooResolutionModel*) _convSet.at(0) ;
  if (!conv) return 0 ;  
  return &conv->convVar() ;
}



Double_t RooConvolutedPdf::evaluate() const
{

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


RooArgSet* RooConvolutedPdf::parseIntegrationRequest(const RooArgSet& intSet, Int_t& coefCode, RooArgSet* analVars) const
{
  // Split allVars in a list for coefficients and a list for convolutions
  RooArgSet allVarsCoef("allVarsCoef") ;
  RooArgSet allVarsConv("allVarsConvInt") ;

  RooAbsArg* arg ;
  RooAbsArg* conv ;

  TIterator* varIter  = intSet.createIterator() ;
  TIterator* convIter = _convSet.createIterator() ;
  while(arg=(RooAbsArg*) varIter->Next()) {
    Bool_t ok(kTRUE) ;
    convIter->Reset() ;
    while(conv=(RooAbsArg*) convIter->Next()) {
      
      if (conv->dependsOn(*arg)) ok=kFALSE ;
    }

    if (ok) {
      allVarsCoef.add(*arg) ;
    } else {
      allVarsConv.add(*arg) ;
    }

  }
  delete varIter ;
  delete convIter ;

  // Get analytical integration code for coefficients
  if (analVars) {
    coefCode = getCoefAnalyticalIntegral(allVarsCoef,*analVars) ;
  } else {
    RooArgSet tmp ;
    coefCode = getCoefAnalyticalIntegral(allVarsCoef,tmp) ;
  }

  // If convolution integration set is empty, return null ptr, otherwise return heap clone of set
  return allVarsConv.getSize()? new RooArgSet(allVarsConv) : 0 ;
}



Int_t RooConvolutedPdf::getAnalyticalIntegral(RooArgSet& allVars, 
					      RooArgSet& analVars, const RooArgSet* normSet) const 
{
  // Handle trivial no-integration scenario
  if (allVars.getSize()==0) return 0 ;

  // Process integration request
  Int_t intCoefCode(0) ;
  RooArgSet *intConvSet = parseIntegrationRequest(allVars,intCoefCode,&analVars) ;

  // Process normalization if integration request
  Int_t normCoefCode(0) ;    
  RooArgSet *normConvSet(0) ;
  if (normSet) {
    normConvSet = parseIntegrationRequest(*normSet,normCoefCode) ;
  }
  
  // Optional messaging
  if (_verboseEval>0) {
    cout << "RooConvolutedPdf::getAI(" << GetName() << ") coefficients integrate analytically " ; analVars.Print("1") ;
    cout << "RooConvolutedPdf::getAI(" << GetName() << ") intCoefCode  = " << intCoefCode << endl ;
    cout << "RooConvolutedPdf::getAI(" << GetName() << ") normCoefCode = " << normCoefCode << endl ;
    cout << "RooConvolutedPdf::getAI(" << GetName() << ") intConvSet  = " ; 
    if (intConvSet) intConvSet->Print("1") ; else cout << "<none>" << endl ;
    cout << "RooConvolutedPdf::getAI(" << GetName() << ") normConvSet  = " ; 
    if (normConvSet) normConvSet->Print("1") ; else cout << "<none>" << endl ;
  }

  // Register convolution dependents integrated as analytical
  analVars.add(*intConvSet) ;

  // Store integration configuration in registry
  Int_t masterCode(0) ;
  Int_t tmp[2] ;
  tmp[0] = intCoefCode ;
  tmp[1] = normCoefCode ;
  masterCode = _codeReg.store(tmp,2,intConvSet,normConvSet)+1 ; // takes ownership of intConvSet,normConvSet
  
  if (_verboseEval>0) {
    cout << "RooConvolutedPdf::getAI(" << GetName() << ") masterCode " << masterCode 
	 << " will integrate analytically " ; analVars.Print("1") ;  
  }
  return masterCode  ;
}



Double_t RooConvolutedPdf::analyticalIntegral(Int_t code, const RooArgSet* normSet) const 
{
  // Handle trivial passthrough scenario
  if (code==0) return getVal(normSet) ;

  // Unpack master code
  RooArgSet *intConvSet, *normConvSet ;
  const Int_t* tmp = _codeReg.retrieve(code-1,intConvSet,normConvSet) ;
  Int_t intCoefCode = tmp[0] ;
  Int_t normCoefCode = tmp[1] ;
  
  RooResolutionModel* conv ;
  Int_t index(0) ;
  Double_t answer(0) ;
  _convSetIter->Reset() ;

//   cout << "RooConvolutedPdf::aI(" << GetName() << "): intCoefCode = " << intCoefCode << ", normCoefCode = " << normCoefCode << endl ;
//   cout << "          intConvSet = " ; if (intConvSet) intConvSet->Print("1") ; else cout << "<none>" << endl ;
//   cout << "         normConvSet = " ; if (normConvSet) normConvSet->Print("1") ; else cout << "<none>" << endl ;

  if (normSet==0) {

    // Integral over unnormalized function
    Double_t integral(0) ;
    while(conv=(RooResolutionModel*)_convSetIter->Next()) {
      Double_t coef = coefAnalyticalIntegral(index++,intCoefCode) ;
      if (coef!=0) {
	integral += coef*conv->getNorm(intConvSet) ;
      }
    }
    answer = integral ;
    
  } else {

    // Integral over normalized function
    Double_t integral(0) ;
    Double_t norm(0) ;
    while(conv=(RooResolutionModel*)_convSetIter->Next()) {

      Double_t coefInt = coefAnalyticalIntegral(index,intCoefCode) ;
      if (coefInt!=0) {
	integral += coefInt*conv->getNormSpecial(intConvSet) ;
      }
      Double_t coefNorm = (intCoefCode==normCoefCode)?coefInt:coefAnalyticalIntegral(index,normCoefCode) ;
      if (coefNorm!=0) {
	norm += coefNorm*conv->getNorm(normConvSet) ; 	
      }
      index++ ;
    }
    answer = integral/norm ;    
  }

  return answer ;
}



Int_t RooConvolutedPdf::getCoefAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars) const 
{
  return 0 ;
}



Double_t RooConvolutedPdf::coefAnalyticalIntegral(Int_t coef, Int_t code) const 
{
  if (code==0) return coefficient(coef) ;
  cout << "RooConvolutedPdf::coefAnalyticalIntegral(" << GetName() << ") ERROR: unrecognized integration code: " << code << endl ;
  assert(0) ;
  return 1 ;
}



Bool_t RooConvolutedPdf::forceAnalyticalInt(const RooAbsArg& dep) const
{
  // Force 'analytical' integration of whatever is delegated to the convolution integrals
  return kTRUE ;
}                                                                                                                         
               



Bool_t RooConvolutedPdf::syncNormalizationPreHook(RooAbsReal* norm,const RooArgSet* nset) const 
{
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




