/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooConvolutedPdf.cc,v 1.2 2001/06/09 05:08:47 verkerke Exp $
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
  _convDummyDataSet("convDummyDataSet","Dummy data set for convolution integrals",convVar)
{
  // Constructor
}


RooConvolutedPdf::RooConvolutedPdf(const RooConvolutedPdf& other, const char* name) : 
  RooAbsPdf(other,name), _model(0), _convVar(0),
  _convSet("convSet",this,other._convSet),
  _convDummyDataSet(other._convDummyDataSet) 
{
  // Copy constructor
}



RooConvolutedPdf::~RooConvolutedPdf()
{
  // Destructor
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
  RooAbsReal* conv = _model->convolution(basisFunc) ;
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



Double_t RooConvolutedPdf::evaluate(const RooDataSet* dset) const
{
  Double_t result(0) ;

  TIterator* iter = _convSet.MakeIterator() ;
  RooAbsPdf* conv ;
  Int_t index(0) ;
  while(conv=(RooAbsPdf*)iter->Next()) {
    Double_t coef = coefficient(index++) ;
    if (coef!=0) {
      result += conv->getVal(0)*coef ;
    }
  }
  
  delete iter ;
  return result ;
}



Int_t RooConvolutedPdf::getAnalyticalIntegral(RooArgSet& allVars, 
					      RooArgSet& analVars) const 
{
  // Make subset of allVars that excludes dependents of any convolution integral
  TIterator* varIter  = allVars.MakeIterator() ;
  TIterator* convIter = _convSet.MakeIterator() ;
  
  RooArgSet allVarsCoef ;
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
  if (matchArgs(allVars,analVars,*convVar())) return 1000+coefCode ;
  
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
	Double_t tmp = conv->getNorm(&_convDummyDataSet) ;
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
  return (&dep == ((RooAbsArg*)convVar())) ;
}                                                                                                                                        


void RooConvolutedPdf::dump(const RooDataSet* dset) const
{
  TIterator* iter = _convSet.MakeIterator() ;
  RooAbsPdf* conv ;
  Int_t index(0) ;
  while(conv=(RooAbsPdf*)iter->Next()) {
    Double_t coef = coefficient(index++) ;
    if (coef!=0) {
      cout << "convolution: value = " << conv->getVal(0)*coef 
	   << " norm = " << conv->getNorm(dset)*coef  << endl ;
    }
  }
  
  delete iter ;
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




