/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitTools
 *    File: $Id: RooAddPdf.cc,v 1.9 2001/09/18 02:03:45 verkerke Exp $
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
// RooAddPdf is an efficient implementation of a sum of PDFs of the form 
//
//  (c_1*PDF_1 + c_2*PDF_2 + ... (1-sum(c_1...c_n-1))*PDF_n 
//
// The coefficient of the last PDF is calculated automatically from the
// normalization condition. RooAddPdf relies on each component PDF
// to be normalized and will perform no normalization other than calculating
// the proper last coefficient c_n.
// An additional condition for this is that each coefficient c_k may
// not overlap (i.e. share servers) with pdf_k.

#include "TIterator.h"
#include "TList.h"
#include "RooFitCore/RooAddPdf.hh"
#include "RooFitCore/RooRealProxy.hh"

ClassImp(RooAddPdf)
;


RooAddPdf::RooAddPdf(const char *name, const char *title) :
  RooAbsPdf(name,title), 
  _coefList("coefList","List of coefficients",this),
  _pdfList("pdfList","List of PDFs",this),
  _clArr(0)
{
  // Dummy constructor 
  _pdfIter  = _pdfList.createIterator() ;
  _coefIter = _coefList.createIterator() ;
}


RooAddPdf::RooAddPdf(const char *name, const char *title,
		     RooAbsPdf& pdf1, RooAbsPdf& pdf2, RooAbsReal& coef1) : 
  RooAbsPdf(name,title),
  _coefList("coefList","List of coefficients",this),
  _pdfList("pdfProxyList","List of PDFs",this),
  _clArr(0)
{
  _pdfIter  = _pdfList.createIterator() ;
  _coefIter = _coefList.createIterator() ;

  // Constructor with two PDFs
  addPdf(pdf1,coef1) ;
  addLastPdf(pdf2) ;    
}


RooAddPdf::RooAddPdf(const RooAddPdf& other, const char* name) :
  RooAbsPdf(other,name),
  _coefList("coefList",this,other._coefList),
  _pdfList("pdfProxyList",this,other._pdfList),
  _clArr(0)
{
  // Copy constructor

  _pdfIter  = _pdfList.createIterator() ;
  _coefIter = _coefList.createIterator() ;

  // Copy code-list array if other PDF has one
  if (other._clArr) {
    _clArr = new pInt_t[10] ;
    Int_t i(0),j ;
    while(other._clArr[i] && i<10) {
      _clArr[i] = new Int_t[_pdfList.getSize()] ;
      for (j=0 ; i<_pdfList.getSize() ; j++) {
	_clArr[i][j] = other._clArr[i][j] ;
      }
      i++ ;
    }
  }

}


RooAddPdf::~RooAddPdf()
{
  // Destructor
  delete _pdfIter ;
  delete _coefIter ;

  // Delete code list array, if allocated
  if (_clArr) {
    Int_t i(0) ;
    while(_clArr[i] && i<10) delete[] _clArr[i++] ;
    delete[] _clArr ;
  }
}



void RooAddPdf::addPdf(RooAbsPdf& pdf, RooAbsReal& coef) 
{  
  // Add a PDF/coefficient pair to the PDF sum
  _pdfList.add(pdf) ;
  _coefList.add(coef) ;
}


void RooAddPdf::addLastPdf(RooAbsPdf& pdf) 
{
  // Specify the last PDF, whose coefficient is automatically 
  // calculated from the normalization requirement
//   RooRealProxy *pdfProxy = new RooRealProxy("pdf","pdf",this,pdf) ;
  _pdfList.add(pdf) ;
}


Double_t RooAddPdf::evaluate() const 
{
  // Calculate the current value of this object
  const RooArgSet* nset = _pdfList.nset() ;
  
  Double_t value(0) ;
  Double_t lastCoef(1) ;

  // Do running sum of coef/pdf pairs, calculate lastCoef.
  _pdfIter->Reset() ;
  _coefIter->Reset() ;
  RooAbsReal* coef ;
  RooAbsReal* pdf ;
  while(coef=(RooAbsReal*)_coefIter->Next()) {
    // WVE check if coef>epsilon before multiplying with pdf
    pdf = (RooAbsReal*)_pdfIter->Next() ;
    value += pdf->getVal(nset)*coef->getVal(nset) ;
    lastCoef -= coef->getVal(nset) ;
  }

  // Add last pdf with correct coefficient
  pdf = (RooAbsReal*) _pdfIter->Next() ;
  value += pdf->getVal(nset)*lastCoef;

  // Warn about coefficient degeneration
  if (lastCoef<0 || lastCoef>1) {
    cout << "RooAddPdf::evaluate(" << GetName() 
	 << " WARNING: sum of PDF coefficients not in range [0-1], value=" 
	 << 1-lastCoef << endl ;
  } 

  return value ;
}


Bool_t RooAddPdf::checkDependents(const RooArgSet* nset) const 
{
  // Check if PDF is valid with dependent configuration given by specified data set

  // Coeffient and PDF should be non-overlapping, but coef/pdf pairs can overlap each other
  Bool_t ret(kFALSE) ;

  _pdfIter->Reset() ;
  _coefIter->Reset() ;
  RooAbsReal* coef ;
  RooAbsReal* pdf ;
  while(coef=(RooAbsReal*)_coefIter->Next()) {
    pdf = (RooAbsReal*)_pdfIter->Next() ;
    ret |= pdf->checkDependents(nset) ;
    ret |= coef->checkDependents(nset) ;
    if (pdf->dependentOverlaps(nset,*coef)) {
      cout << "RooAddPdf::checkDependents(" << GetName() << "): ERROR: coefficient " << coef->GetName() 
	   << " and PDF " << pdf->GetName() << " have one or more dependents in common" << endl ;
      ret = kTRUE ;
    }
  }
  
  return ret ;
}


Int_t RooAddPdf::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const RooArgSet* normSet) const 
{
  // Determine which part (if any) of given integral can be performed analytically.
  // If any analytical integration is possible, return integration scenario code
    // This PDF is by construction normalized
  _pdfIter->Reset() ;
  RooAbsPdf* pdf ;
  RooArgSet allAnalVars(allVars) ;
  TIterator* avIter = allVars.createIterator() ;

  Int_t n(0) ;
  // First iteration, determine what each component can integrate analytically
  while(pdf=(RooAbsPdf*)_pdfIter->Next()) {
    RooArgSet subAnalVars ;
    Int_t subCode = pdf->getAnalyticalIntegral(allVars,subAnalVars,normSet) ;
    
    // If a dependent is not supported by any of the components, 
    // it is dropped from the combined analytic list
    avIter->Reset() ;
    RooAbsArg* arg ;
    while(arg=(RooAbsArg*)avIter->Next()) {
      if (!subAnalVars.find(arg->GetName())) {
	allAnalVars.remove(*arg) ;
      }
    }
    n++ ;
  }

  if (allAnalVars.getSize()==0) {
    delete avIter ;
    return 0 ;
  }

  // Now retrieve the component codes for the common set of analytic dependents 
  _pdfIter->Reset() ;
  n=0 ;
  Int_t* subCode = new Int_t[_pdfList.getSize()] ;
  Bool_t allOK(kTRUE) ;
  while(pdf=(RooAbsPdf*)_pdfIter->Next()) {
    RooArgSet subAnalVars ;
    subCode[n] = pdf->getAnalyticalIntegral(allAnalVars,subAnalVars,normSet) ;
    if (subCode[n]==0) {
      cout << "RooAddPdf::getAnalyticalIntegral(" << GetName() << ") WARNING: component PDF " << pdf->GetName() 
	   << "   advertises inconsistent set of integrals (e.g. (X,Y) but not X or Y individually."
	   << "   Distributed analytical integration disabled. Please fix PDF" << endl ;
      allOK = kFALSE ;
    }
    n++ ;
  }  
  if (!allOK) return 0 ;

  analVars.add(allAnalVars) ;
  Int_t masterCode = registerAICodeList(subCode)+1 ;

  delete[] subCode ;
  delete avIter ;
  return masterCode ;
}


Double_t RooAddPdf::analyticalIntegral(Int_t code) const 
{
  // Return analytical integral defined by given scenario code
  if (code==0) return getVal() ;

  const Int_t* subCode = retrieveAICodeList(code-1) ;
  if (!subCode) {
    cout << "RooAddPdf::analyticalIntegral(" << GetName() << "): ERROR unrecognized integration code, " << code << endl ;
    assert(0) ;    
  }

  // Calculate the current value of this object
  const RooArgSet* nset = _pdfList.nset() ;
  
  Double_t value(0) ;
  Double_t lastCoef(1) ;

  // Do running sum of coef/pdf pairs, calculate lastCoef.
  _pdfIter->Reset() ;
  _coefIter->Reset() ;
  RooAbsReal* coef ;
  RooAbsReal* pdf ;
  Int_t i(0) ;
  while(coef=(RooAbsReal*)_coefIter->Next()) {
    pdf = (RooAbsReal*)_pdfIter->Next() ;
    value += pdf->analyticalIntegral(subCode[i])*coef->getVal(nset) ;
    lastCoef -= coef->getVal(nset) ;
    i++ ;
  }

  // Add last pdf with correct coefficient
  pdf = (RooAbsReal*) _pdfIter->Next() ;
  value += pdf->analyticalIntegral(subCode[i])*lastCoef;

  // Warn about coefficient degeneration
  if (lastCoef<0 || lastCoef>1) {
    cout << "RooAddPdf::analyticalIntegral(" << GetName() 
	 << " WARNING: sum of PDF coefficients not in range [0-1], value=" 
	 << 1-lastCoef << endl ;
  } 

  return value ;
}


Int_t RooAddPdf::registerAICodeList(Int_t* codeList) const
{
  Int_t i,j ;

  // If code list array has never been used, allocate and initialize here
  if (!_clArr) {
    _clArr = new pInt_t[10] ;
    for (i=0 ; i<10 ; i++) _clArr[i]=0 ;    
  }

  // Loop over code-list array
  for (i=0 ; i<10 ; i++) {
    if (_clArr[i]==0) {
      // Empty slot, store code list and return index
      _clArr[i] = new Int_t(_pdfList.getSize()) ;
      for (j=0 ; j<_pdfList.getSize() ; j++) _clArr[i][j] = codeList[j] ;
      return i ;
    } else {
      // Existing slot, compare with current list, if matched return index
      Bool_t match(kTRUE) ;
      for (j=0 ; j<_pdfList.getSize() ; j++) {
	if (_clArr[i][j] != codeList[j]) match=kFALSE ;
      }
      if (match) return i ;
    }
  }

  assert(0) ;
  return 0 ;
}


const Int_t* RooAddPdf::retrieveAICodeList(Int_t masterCode) const 
{
  return _clArr[masterCode] ;
}
