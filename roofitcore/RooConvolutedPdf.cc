/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooConvolutedPdf.cc,v 1.1 2001/06/08 05:51:05 verkerke Exp $
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
  _convSet("convSet","Set of resModel X basisFunc convolutions",this)
{
  // Constructor
}


RooConvolutedPdf::RooConvolutedPdf(const RooConvolutedPdf& other, const char* name) : 
  RooAbsPdf(other,name), _model(0), _convVar(0),
  _convSet("convSet",this,other._convSet)
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



Double_t RooConvolutedPdf::evaluate(const RooDataSet* dset) const
{
  Double_t result(0), norm(0) ;

  TIterator* iter = _convSet.MakeIterator() ;
  RooAbsPdf* conv ;
  Int_t index(0) ;
  while(conv=(RooAbsPdf*)iter->Next()) {
    Double_t coef = coefficient(index++) ;
    if (coef!=0) {
      result += conv->getVal(0)*coef ;
      norm   += conv->getNorm(dset)*coef ;
    }
  }
  
  delete iter ;
  return result/norm ;
}



Bool_t RooConvolutedPdf::selfNormalized(const RooAbsArg& dep) const 
{
  RooResolutionModel* conv = (RooResolutionModel*) _convSet.At(0) ;
  if (!conv) return kFALSE ;
  
  if (&dep == &conv->convVar()) return kTRUE ;
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




