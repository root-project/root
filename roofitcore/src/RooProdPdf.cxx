/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitTools
 *    File: $Id: RooProdPdf.cc,v 1.10 2001/09/20 01:40:11 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   06-Jan-2000 DK Created initial version
 *   19-Apr-2000 DK Add the printEventStats() method
 *   26-Jun-2000 DK Add support for extended likelihood fits
 *   02-Jul-2000 DK Add support for multiple terms (instead of only 2)
 *   05-Jul-2000 DK Add support for extended maximum likelihood and a
 *                  new method for this: setNPar()
 *   03-May02001 WV Port to RooFitCore/RooFitModels
 *
 * Copyright (C) 2000 Stanford University
 *****************************************************************************/

// -- CLASS DESCRIPTION --
// RooProdPdf is an efficient implementation of a product of PDFs of the form 
//
//  PDF_1 * PDF_2 * ... * PDF_N
//
// RooProdPdf relies on each component PDF to be normalized and will perform no 
// explicit normalization itself. 
// A condition for this to work is that each pdf in the product may not share
// any servers with any other PDF. 

#include "TIterator.h"
#include "RooFitCore/RooProdPdf.hh"
#include "RooFitCore/RooRealProxy.hh"

ClassImp(RooProdPdf)
;


RooProdPdf::RooProdPdf(const char *name, const char *title, Double_t cutOff) :
  RooAbsPdf(name,title), 
  _pdfList("_pdfList","List of PDFs",this),
  _pdfIter(_pdfList.createIterator()), 
  _cutOff(cutOff),
  _codeReg(10)
{
  // Dummy constructor
}


RooProdPdf::RooProdPdf(const char *name, const char *title,
		       RooAbsPdf& pdf1, RooAbsPdf& pdf2, Double_t cutOff) : 
  RooAbsPdf(name,title), 
  _pdfList("_pdfList","List of PDFs",this),
  _pdfIter(_pdfList.createIterator()), 
  _cutOff(cutOff),
  _codeReg(10)
{
  // Constructor with 2 PDFs
  addPdf(pdf1) ;
  addPdf(pdf2) ;    
}



RooProdPdf::RooProdPdf(const char* name, const char* title, RooArgList& pdfList, Double_t cutOff) :
  RooAbsPdf(name,title), 
  _pdfList("_pdfList","List of PDFs",this),
  _pdfIter(_pdfList.createIterator()), 
  _cutOff(cutOff),
  _codeReg(10)
{
  // Constructor with 2 PDFs
  TIterator* iter = pdfList.createIterator() ;
  RooAbsPdf* pdf ;
  while(pdf=(RooAbsPdf*)iter->Next()) {
    if (!dynamic_cast<RooAbsPdf*>(pdf)) {
      cout << "RooProdPdf::RooProdPdf(" << GetName() << ") list arg " 
	   << pdf->GetName() << " is not a PDF, ignored" << endl ;
      continue ;
    }
    addPdf(*pdf) ;
  }
  delete iter ;
}


RooProdPdf::RooProdPdf(const RooProdPdf& other, const char* name) :
  RooAbsPdf(other,name), 
  _pdfList("_pdfList",this,other._pdfList),
  _pdfIter(_pdfList.createIterator()), 
  _cutOff(other._cutOff),
  _codeReg(other._codeReg)
{
  // Copy constructor
}


RooProdPdf::~RooProdPdf()
{
  // Destructor

  delete _pdfIter ;
}



void RooProdPdf::addPdf(RooAbsPdf& pdf) 
{    
  // Add PDF to product of PDFs
  _pdfList.add(pdf) ;
}


Double_t RooProdPdf::evaluate() const 
{
  // Calculate current value of object

  Double_t value(1) ;
    
  // Calculate running product of pdfs
  RooAbsReal* pdf ;
  _pdfIter->Reset() ;
  const RooArgSet* nset(_pdfList.nset()) ;
  while(pdf=(RooAbsReal*)_pdfIter->Next()) {    
    value *= pdf->getVal(nset) ;
    if (value<_cutOff) break ;
  }

  return value ;
}


Int_t RooProdPdf::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const RooArgSet* normSet) const 
{
  // Determine which part (if any) of given integral can be performed analytically.
  // If any analytical integration is possible, return integration scenario code

  // Determine if we can express the integral as a partial product of pdfs:
  // If integration is requested over all of a component pdfs' dependents
  // we know a priori, because the pdf is by construction normalized, that
  // it will evaluate to 1 and can thus be dropped from the product.
  _pdfIter->Reset() ;
  RooAbsPdf* pdf ;
  Int_t code(0), n(0) ;
  Bool_t allFact(kTRUE) ;
  Int_t* subCode = new Int_t[_pdfList.getSize()] ;
  while(pdf=(RooAbsPdf*)_pdfIter->Next()) {
    Bool_t fact(kTRUE) ;
    RooArgSet *pdfDepList = pdf->getDependents(normSet) ;
    TIterator* depIter = pdfDepList->createIterator() ;
    RooAbsArg* dep ;
    while(dep=(RooAbsArg*)depIter->Next()) {
      if (!allVars.find(dep->GetName())) {
	fact=kFALSE ;
	allFact=kFALSE ;
      }
    }
    if (fact) {
      // Factorize, i.e. drop this component pdf
      analVars.add(*pdfDepList) ;
      cout << "RooProdPdf::getAI(" << GetName() << ") dropping pdf #" << n << " " << pdf->GetName() << endl ;
      subCode[n] = -1 ;
    } else {
      // Determine partial integration code
      RooArgSet subAnalVars ;
      subCode[n] = pdf->getAnalyticalIntegral(allVars,subAnalVars,normSet) ;      
      analVars.add(subAnalVars) ;
      cout << "RooProdPdf::getAI(" << GetName() << ") subCode(" << n << "," << pdf->GetName() << ") = " << subCode[n] << endl ;
    }
    delete depIter ;
    delete pdfDepList ;
    n++ ;
  }

  Int_t masterCode = _codeReg.store(subCode,_pdfList.getSize())+1 ;
  delete[] subCode ;

  // This PDF is by construction normalized
  return allFact?-1:masterCode ;
}


Double_t RooProdPdf::analyticalIntegral(Int_t code, const RooArgSet* normSet) const 
{
  // Return analytical integral defined by given scenario code
  //cout << "RooProdPdf::aI(" << GetName() << ") code = " << code << " normSet = " << normSet << endl ;

  // No integration scenario
  if (code==0) {
    return getVal() ;
  }

  // Full integration scenario
  if (code==-1) {
    return 1.0 ;
  }

  // Partial integration scenarios
  RooAbsReal* pdf ;
  _pdfIter->Reset() ;
  Int_t n(0) ;
  Double_t value(1) ;
  const Int_t* subCode = _codeReg.retrieve(code-1) ;

  // Calculate running product of pdfs, skipping factorized components
  while(pdf=(RooAbsReal*)_pdfIter->Next()) {    
    if (subCode[n]>=0) {
      value *= pdf->analyticalIntegral(subCode[n],normSet) ;
    }
    if (value<_cutOff) break ;
    n++ ;
  }

  return value ;
}


