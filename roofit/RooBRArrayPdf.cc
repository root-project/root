/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitModels
 *    File: $Id: RooBRArrayPdf.cc,v 1.3 2001/08/23 01:23:34 verkerke Exp $
 * Authors:
 *   JB, John Back, Queen Mary College, Univ London, jback@slac.stanford.edu
 * History:
 *   19-Jun 2001 JB Port to RooFitCore/RooFitModels
 *
 * Copyright (C) 2001 Queen Mary College, Univ London.
 *****************************************************************************/
// -- CLASS DESCRIPTION --
// RooBRArrayPdf is an efficient implementation of the total extended likelihood
// function, L, that is used to calculate branching ratios:
//
// L = sum(c_i)^N exp^(-sum(c_i)) Prod_{N}(c_1*PDF_1 + c_2*PDF_2 + ...c_n*PDF_n)/N!
//
// where c_i is the number of events for the ith hypothesis (signal or background) and
// N is the total number of events in the fit. Each PDF is fixed, and we just want to 
// find the number of events of each signal/background hypothesis, c_i. The coefficients 
// c_i are allowed to float in the fit. RooBRArrayPdf relies on each component PDF 
// to be normalized and will perform no normalization on its own. An additional condition
// is that each coefficient c_i may not overlap (i.e. share servers) with pdf_i. 
// To use this class properly, the "e" option needs to be specified in the fit option
// (so that the extended term is included).

#include "TIterator.h"
#include "RooFitModels/RooBRArrayPdf.hh"
#include "RooFitCore/RooRealProxy.hh"

#include <iostream.h>
#include <assert.h>

ClassImp(RooBRArrayPdf)

RooBRArrayPdf::RooBRArrayPdf(const char *name, const char *title)
{
  // Dummy constructor 
}


RooBRArrayPdf::RooBRArrayPdf(const char *name, const char *title,
			     const RooArgSet& pdfArray, const RooArgSet& coefArray):
  RooAbsPdf(name, title)
{
  // add the parameters and dependents of all PDFs
  Int_t i(0);
  Int_t numberOfPdfs = pdfArray.getSize();
  Int_t numberOfCoeffs = coefArray.getSize();

  if (numberOfPdfs != numberOfCoeffs) {
    cout<<"Error. Number of Pdfs != number of coeffs. Aborting."<<endl;
    return;
  }

  for (i = 0; i < numberOfPdfs; i++) {

    RooAbsPdf* thePdf = (RooAbsPdf*) pdfArray.At(i);
    RooAbsReal* theCoeff = (RooAbsReal*) coefArray.At(i);

    if (thePdf != 0 && theCoeff != 0) {
      addPdf(*thePdf, *theCoeff);
    } else {
      cout<<"Error. Pdf and/or coeff are null"<<endl;
    }

  }
}


RooBRArrayPdf::RooBRArrayPdf(const RooBRArrayPdf& other, const char* name) :
  RooAbsPdf(other,name)
{
  // Copy constructor

  // Copy proxy lists
  TIterator *iter = other._coefProxyList.MakeIterator() ;
  RooRealProxy* proxy ;
  while(proxy=(RooRealProxy*)iter->Next()) {
    _coefProxyList.Add(new RooRealProxy("coef",this,*proxy)) ;
  }
  delete iter ;

  iter = other._pdfProxyList.MakeIterator() ;
  while(proxy=(RooRealProxy*)iter->Next()) {
    _pdfProxyList.Add(new RooRealProxy("pdf",this,*proxy)) ;
  }
  delete iter ;

}


RooBRArrayPdf::~RooBRArrayPdf()
{
  // Destructor

  // Delete all owned proxies 
  _coefProxyList.Delete() ;
  _pdfProxyList.Delete() ;
}



void RooBRArrayPdf::addPdf(RooAbsPdf& pdf, RooAbsReal& coef) 
{  
  // Add a PDF/coefficient pair to the PDF sum

  RooRealProxy *pdfProxy = new RooRealProxy("pdf","pdf",this,pdf) ;
  RooRealProxy *coefProxy = new RooRealProxy("coef","coef",this,coef) ;

  _pdfProxyList.Add(pdfProxy) ;
  _coefProxyList.Add(coefProxy) ;
}

Double_t RooBRArrayPdf::evaluate() const {

  // Calculate the current value of this object
  TIterator *pIter = _pdfProxyList.MakeIterator() ;
  TIterator *cIter = _coefProxyList.MakeIterator() ;
  
  Double_t value(0.0), coefSum(0.0) ;

  RooRealProxy* coef ;
  RooRealProxy* pdf ;

  // Do running sum of coef*pdf values to return likelihood
  while (coef = (RooRealProxy*) cIter->Next()) {

    pdf = (RooRealProxy*)pIter->Next();
    value += (*pdf)*(*coef);

  }

  delete pIter ;
  delete cIter ;

  return value ;
}


Bool_t RooBRArrayPdf::checkDependents(const RooArgSet* nset) const 
{
  // Check if PDF is valid with dependent configuration given by specified data set

  // Special, more lenient dependent checking: Coeffient and PDF should
  // be non-overlapping, but coef/pdf pairs can
  Bool_t ret(kFALSE) ;

  TIterator *pIter = _pdfProxyList.MakeIterator() ;
  TIterator *cIter = _coefProxyList.MakeIterator() ;

  RooRealProxy* coef ;
  RooRealProxy* pdf ;
  while(coef=(RooRealProxy*)cIter->Next()) {
    pdf = (RooRealProxy*)pIter->Next() ;
    ret |= pdf->arg().checkDependents(nset) ;
    ret |= coef->arg().checkDependents(nset) ;
    if (pdf->arg().dependentOverlaps(nset,coef->arg())) {
      cout << "RooBRArrayPdf::checkDependents(" << GetName() << "): ERROR: coefficient " << coef->arg().GetName() 
	   << " and PDF " << pdf->arg().GetName() << " have one or more dependents in common" << endl ;
      ret = kTRUE ;
    }
  }
  
  
  delete pIter ;
  delete cIter ;

  return ret ;
}


Int_t RooBRArrayPdf::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& numVars) const 
{
  // Determine which part (if any) of given integral can be performed analytically.
  // If any analytical integration is possible, return integration scenario code

  // This PDF is by construction normalized
  return 0 ;
}


Double_t RooBRArrayPdf::analyticalIntegral(Int_t code) const 
{
  // Return analytical integral defined by given scenario code

  // This PDF is by construction normalized
  return 1.0 ;
}

Double_t RooBRArrayPdf::expectedEvents() const {

  // Calculate the current value of this object
  TIterator *cIter = _coefProxyList.MakeIterator() ;
  
  Double_t expectedTotal(0.0);

  // Do running sum of coef - for expectedTotal
  RooRealProxy* coef ;
  while(coef=(RooRealProxy*)cIter->Next()) {
    expectedTotal += (*coef);
  }   

  delete cIter;

  return expectedTotal;
}

Double_t RooBRArrayPdf::extendedTerm(UInt_t observedEvents) const {

  // Return the expected number of events = sum(c_i)
  return expectedEvents();
}
