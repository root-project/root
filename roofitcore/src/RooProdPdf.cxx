/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitTools
 *    File: $Id: RooProdPdf.cc,v 1.13 2001/10/05 07:01:50 verkerke Exp $
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
// explicit normalization itself. No PDF may share any dependents with any other PDF. 
// 
// To construct a product of PDFs that share dependents, and thus require explicit
// normalization of the product, use RooGenericPdf.

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
  _codeReg(10),
  _extendedIndex(-1)
{
  // Dummy constructor
}


RooProdPdf::RooProdPdf(const char *name, const char *title,
		       RooAbsPdf& pdf1, RooAbsPdf& pdf2, Double_t cutOff) : 
  RooAbsPdf(name,title), 
  _pdfList("_pdfList","List of PDFs",this),
  _pdfIter(_pdfList.createIterator()), 
  _cutOff(cutOff),
  _codeReg(10),
  _extendedIndex(-1)
{
  // Constructor with 2 PDFs (most frequent use case).
  // 
  // The optional cutOff parameter can be used as a speed optimization if
  // one or more of the PDF have sizable regions with very small values,
  // which would pull the entire product of PDFs to zero in those regions.
  //
  // After each PDF multiplication, the running product is compared with
  // the cutOff parameter. If the running product is smaller than the
  // cutOff value, the product series is terminated and remaining PDFs
  // are not evaluated.
  //
  // There is no magic value of the cutOff, the user should experiment
  // to find the appropriate balance between speed and precision.
  // If a cutoff is specified, the PDFs most likely to be small should
  // be put first in the product. The default cutOff value is zero.

  _pdfList.add(pdf1) ;
  if (pdf1.canBeExtended()) {
    _extendedIndex = _pdfList.index(&pdf1) ;
  }

  _pdfList.add(pdf2) ;
  if (pdf2.canBeExtended()) {
    if (_extendedIndex>=0) {
      // Protect against multiple extended terms
      cout << "RooProdPdf::RooProdPdf(" << GetName() 
	   << ") multiple components with extended terms detected,"
	   << " product will not be extendible." << endl ;
      _extendedIndex=-1 ;
    } else {
      _extendedIndex=_pdfList.index(&pdf2) ;
    }
  }

}



RooProdPdf::RooProdPdf(const char* name, const char* title, RooArgList& pdfList, Double_t cutOff) :
  RooAbsPdf(name,title), 
  _pdfList("_pdfList","List of PDFs",this),
  _pdfIter(_pdfList.createIterator()), 
  _cutOff(cutOff),
  _codeReg(10),
  _extendedIndex(-1)
{
  // Constructor from a list of PDFs
  // 
  // The optional cutOff parameter can be used as a speed optimization if
  // one or more of the PDF have sizable regions with very small values,
  // which would pull the entire product of PDFs to zero in those regions.
  //
  // After each PDF multiplication, the running product is compared with
  // the cutOff parameter. If the running product is smaller than the
  // cutOff value, the product series is terminated and remaining PDFs
  // are not evaluated.
  //
  // There is no magic value of the cutOff, the user should experiment
  // to find the appropriate balance between speed and precision.
  // If a cutoff is specified, the PDFs most likely to be small should
  // be put first in the product. The default cutOff value is zero.

  TIterator* iter = pdfList.createIterator() ;
  RooAbsPdf* pdf ;
  Int_t numExtended(0) ;
  while(pdf=(RooAbsPdf*)iter->Next()) {
    if (!dynamic_cast<RooAbsPdf*>(pdf)) {
      cout << "RooProdPdf::RooProdPdf(" << GetName() << ") list arg " 
	   << pdf->GetName() << " is not a PDF, ignored" << endl ;
      continue ;
    }
    _pdfList.add(*pdf) ;
    if (pdf->canBeExtended()) {
      _extendedIndex = _pdfList.index(pdf) ;
      numExtended++ ;
    }
  }

  // Protect against multiple extended terms
  if (numExtended>1) {
    cout << "RooProdPdf::RooProdPdf(" << GetName() 
	 << ") WARNING: multiple components with extended terms detected,"
	 << " product will not be extendible." << endl ;
    _extendedIndex = -1 ;
  }

  delete iter ;
}


RooProdPdf::RooProdPdf(const RooProdPdf& other, const char* name) :
  RooAbsPdf(other,name), 
  _pdfList("_pdfList",this,other._pdfList),
  _pdfIter(_pdfList.createIterator()), 
  _cutOff(other._cutOff),
  _codeReg(other._codeReg),
  _extendedIndex(other._extendedIndex)
{
  // Copy constructor
}


RooProdPdf::~RooProdPdf()
{
  // Destructor

  delete _pdfIter ;
}


Double_t RooProdPdf::evaluate() const 
{
  // Calculate current unnormalized value of object

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



Bool_t RooProdPdf::canBeExtended() const
{
  return (_extendedIndex>=0) ;
}



Double_t RooProdPdf::expectedEvents() const 
{
  assert(_extendedIndex>=0) ;
  return ((RooAbsPdf*)_pdfList.at(_extendedIndex))->expectedEvents() ;
}



Int_t RooProdPdf::getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& analVars, const RooArgSet* normSet) const 
{
  // Determine which part (if any) of given integral can be performed analytically.
  // If any analytical integration is possible, return integration scenario code.
  //
  // RooProdPdf implements two strategies in implementing analytical integrals
  //
  // First, PDF components whose entire set of dependents are requested to be integrated
  // can be dropped from the product, as they will integrate out to 1 by construction
  //
  // Second, RooProdPdf queries each remaining component PDF for its analytical integration 
  // capability of the requested set ('allVars'). It finds the largest common set of variables 
  // that can be integrated by all remaining components. If such a set exists, it reconfirms that 
  // each component is capable of analytically integrating the common set, and combines the components 
  // individual integration codes into a single integration code valid for RooProdPdf.

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
//       cout << "RooProdPdf::getAI(" << GetName() << ") dropping pdf #" << n << " " << pdf->GetName() << endl ;
      subCode[n] = -1 ;
    } else {
      // Determine partial integration code
      RooArgSet subAnalVars ;
      subCode[n] = pdf->getAnalyticalIntegralWN(allVars,subAnalVars,normSet) ;      
      analVars.add(subAnalVars) ;
//       cout << "RooProdPdf::getAI(" << GetName() << ") subCode(" << n << "," << pdf->GetName() << ") = " << subCode[n] << endl ;
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


Double_t RooProdPdf::analyticalIntegralWN(Int_t code, const RooArgSet* normSet) const 
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
      value *= pdf->analyticalIntegralWN(subCode[n],normSet) ;
    }
    if (value<_cutOff) break ;
    n++ ;
  }

  return value ;
}


