/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooConvolutedPdf.cc,v 1.7 2001/08/02 21:39:09 verkerke Exp $
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
  RooAbsPdf(name,title), 
  _model((RooResolutionModel*)&model), _convVar((RooRealVar*)&convVar),
  _convSet("convSet","Set of resModel X basisFunc convolutions",this),
  _convNormSet(0), _convSetIter(_convSet.MakeIterator())
{
  // Constructor
  _convNormSet = new RooArgSet(convVar,"convNormSet") ;
}


RooConvolutedPdf::RooConvolutedPdf(const RooConvolutedPdf& other, const char* name) : 
  RooAbsPdf(other,name), _model(0), _convVar(0),
  _convSet("convSet",this,other._convSet),
  _convNormSet(new RooArgSet(*other._convNormSet)),
  _convSetIter(_convSet.MakeIterator())
{
  // Copy constructor
}



RooConvolutedPdf::~RooConvolutedPdf()
{
  // Destructor

  if (_convNormSet) delete _convNormSet ;
  delete _convSetIter ;
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

  // Instantiate resModel x basisFunc convolution
  RooAbsReal* conv = _model->convolution(basisFunc,this) ;
  _convSet.add(*conv) ;

  // WVE must store or delete basisFunc 
  return _convSet.IndexOf(conv) ;
}



const RooRealVar* RooConvolutedPdf::convVar() const
{
  RooResolutionModel* conv = (RooResolutionModel*) _convSet.At(0) ;
  if (!conv) return 0 ;  
  return &conv->convVar() ;
}



Double_t RooConvolutedPdf::evaluate(const RooArgSet* nset) const
{

  Double_t result(0) ;

  _convSetIter->Reset() ;
  RooAbsPdf* conv ;
  Int_t index(0) ;
  while(conv=(RooAbsPdf*)_convSetIter->Next()) {
    Double_t coef = coefficient(index++) ;
    if (coef!=0.) {
//       cout << "RooConvPdf::evaluate(" << GetName() << "): conv x coef = " 
// 	      << conv->getVal(0) << " x " << coef << endl ;
      result += conv->getVal(0)*coef ;
   }
  }
  
  return result ;
}



Int_t RooConvolutedPdf::getAnalyticalIntegral(RooArgSet& allVars, 
					      RooArgSet& analVars) const 
{
  // Make subset of allVars that excludes dependents of any convolution integral
  TIterator* varIter  = allVars.MakeIterator() ;
  TIterator* convIter = _convSet.MakeIterator() ;
  
  RooArgSet allVarsCoef("allVarsCoef") ;
  RooAbsArg* arg ;
  RooAbsArg* conv ;
  while(arg=(RooAbsArg*) varIter->Next()) {
    Bool_t ok(kTRUE) ;
    convIter->Reset() ;
    while(conv=(RooAbsArg*) convIter->Next()) {
      if (conv->dependsOn(*arg)) ok=kFALSE ;
    }
    if (ok) allVarsCoef.add(*arg) ;
  }
  delete varIter ;
  delete convIter ;
  
  // Determine integration capability of coefficients
  Int_t coefCode = getCoefAnalyticalIntegral(allVarsCoef,analVars) ;
  
  // Add integrations capability over convolution variable
  if (matchArgs(allVars,analVars,*_convNormSet)) return 1000+coefCode ;
  
  return coefCode ;
}



Double_t RooConvolutedPdf::analyticalIntegral(Int_t code) const 
{
  if (code==0) {

    return getVal(0) ;

  } else if (code>=1000) {
    
    // Integration list includes convolution variable
    Double_t norm(0) ;
    TIterator* iter = _convSet.MakeIterator() ;
    RooAbsPdf* conv ;
    Int_t index(0) ;
    while(conv=(RooAbsPdf*)iter->Next()) {
      Double_t coef = coefAnalyticalIntegral(index++,code-1000) ;
      if (coef!=0) {
	Double_t tmp = conv->getNorm(_convNormSet) ;
	if (_verboseEval>1) cout << "RooConvolutedPdf::analyticalIntegral(" << GetName() 
				 << "): norm of '" << conv->GetName() << "' = " << tmp
				 << " * (coef = " << coef << ") = " << tmp*coef << endl ;
	norm   += tmp*coef ;
      }
    }
    
    delete iter ;
    return norm ;       

  } else {

    // Integration list doesn't include convolution variable
    Double_t norm(0) ;
    TIterator* iter = _convSet.MakeIterator() ;
    RooAbsPdf* conv ;
    Int_t index(0) ;
    while(conv=(RooAbsPdf*)iter->Next()) {
      Double_t coef = coefAnalyticalIntegral(index++,code) ;
      if (coef!=0) {
	Double_t tmp = conv->getVal(0) ;
	norm   += tmp*coef ;
      }
    }
    
    delete iter ;
    return norm ;       
  }
}



Int_t RooConvolutedPdf::getCoefAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars) const 
{
  return 0 ;
}



Double_t RooConvolutedPdf::coefAnalyticalIntegral(Int_t coef, Int_t code) const 
{
  if (code==0) return coefficient(coef) ;
  return 1 ;
}



Bool_t RooConvolutedPdf::forceAnalyticalInt(const RooAbsArg& dep) const
{
  // Force 'analytical' integration of whatever is delegated to the convolution integrals
//   cout << "forceInt(" << dep.GetName() << ") = " 
//        << (_convNormSet->get()->FindObject(dep.GetName())?"kTRUE":"kFALSE") << endl ;
  return _convNormSet->FindObject(dep.GetName())?kTRUE:kFALSE ;
}                                                                                                                         
               


void RooConvolutedPdf::dump(const RooArgSet* nset) const
{
  TIterator* iter = _convSet.MakeIterator() ;
  RooAbsPdf* conv ;
  Int_t index(0) ;
  while(conv=(RooAbsPdf*)iter->Next()) {
    Double_t coef = coefficient(index++) ;
    if (coef!=0) {
//        cout << "convolution: value = " << conv->getVal(0)*coef 
//  	   << " norm = " << conv->getNorm(nset)*coef  << endl ;
    }
  }
  
  delete iter ;
}



Bool_t RooConvolutedPdf::syncNormalizationPreHook(RooAbsReal* norm,const RooArgSet* nset) const 
{
  delete _convNormSet ;
  RooArgSet convNormArgs("convNormArgs") ;

  // Make iterator over data set arguments
  TIterator* dsIter = nset->MakeIterator() ;
  RooAbsArg* dsArg ;

  // Make iterator over convolution integrals
  TIterator* cvIter = _convSet.MakeIterator() ;
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
  TIterator* cvIter = _convSet.MakeIterator() ;
  RooResolutionModel* conv ;

  // Make convolution normalizations servers of the convoluted pdf normalization
  while(conv=(RooResolutionModel*)cvIter->Next()) {
    conv->syncNormalization(_convNormSet) ;

    // Add leaf node servers of convolution normalization integrals to our normalization
    // integral, except for the integrated variables

    RooArgSet leafList("leafNodeServerList") ;
    conv->normLeafServerList(leafList) ;
    TIterator* sIter = leafList.MakeIterator() ;

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
    TIterator* iter = _convSet.MakeIterator() ;
    RooResolutionModel* conv ;
    while (conv=(RooResolutionModel*)iter->Next()) {
      conv->printToStream(os,Verbose,"    ") ;
    }
  }
}




