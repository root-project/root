/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitTools
 *    File: $Id: RooAddPdf.cc,v 1.3 2001/05/17 00:43:15 verkerke Exp $
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


RooAddPdf::RooAddPdf(const char *name, const char *title)
{
  // Dummy constructor 
}


RooAddPdf::RooAddPdf(const char *name, const char *title,
		     RooAbsPdf& pdf1, RooAbsPdf& pdf2, RooAbsReal& coef1) : 
  RooAbsPdf(name,title)
{
  // Constructor with two PDFs
  addPdf(pdf1,coef1) ;
  addLastPdf(pdf2) ;    
}


RooAddPdf::RooAddPdf(const RooAddPdf& other, const char* name) :
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


RooAddPdf::~RooAddPdf()
{
  // Destructor

  // Delete all owned proxies 
  _coefProxyList.Delete() ;
  _pdfProxyList.Delete() ;
}



void RooAddPdf::addPdf(RooAbsPdf& pdf, RooAbsReal& coef) 
{  
  // Add a PDF/coefficient pair to the PDF sum

  RooRealProxy *pdfProxy = new RooRealProxy("pdf","pdf",this,pdf) ;
  RooRealProxy *coefProxy = new RooRealProxy("coef","coef",this,coef) ;
  
  _pdfProxyList.Add(pdfProxy) ;
  _coefProxyList.Add(coefProxy) ;
}


void RooAddPdf::addLastPdf(RooAbsPdf& pdf) 
{
  // Specify the last PDF, whose coefficient is automatically 
  // calculated from the normalization requirement
  RooRealProxy *pdfProxy = new RooRealProxy("pdf","pdf",this,pdf) ;
  _pdfProxyList.Add(pdfProxy) ;
}


Double_t RooAddPdf::evaluate(const RooDataSet* dset) const 
{
  // Calculate the current value of this object
  TIterator *pIter = _pdfProxyList.MakeIterator() ;
  TIterator *cIter = _coefProxyList.MakeIterator() ;
  
  Double_t value(0) ;
  Double_t lastCoef(1) ;

  // Do running sum of coef/pdf pairs, calculate lastCoef.
  RooRealProxy* coef ;
  RooRealProxy* pdf ;
  while(coef=(RooRealProxy*)cIter->Next()) {
    pdf = (RooRealProxy*)pIter->Next() ;
    value += (*pdf)*(*coef) ;
    lastCoef -= (*coef) ;
  }

  // Add last pdf with correct coefficient
  pdf = (RooRealProxy*) pIter->Next() ;
  value += (*pdf)*lastCoef ;

  // Warn about coefficient degeneration
  if (lastCoef<0 || lastCoef>1) {
    cout << "RooAddPdf::evaluate(" << GetName() 
	 << " WARNING: sum of PDF coefficients not in range [0-1], value=" 
	 << 1-lastCoef << endl ;
  } 

  delete pIter ;
  delete cIter ;

  return value ;
}


Bool_t RooAddPdf::checkDependents(const RooDataSet* set) const 
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
    ret |= pdf->arg().checkDependents(set) ;
    ret |= coef->arg().checkDependents(set) ;
    if (pdf->arg().dependentOverlaps(set,coef->arg())) {
      cout << "RooAddPdf::checkDependents(" << GetName() << "): ERROR: coefficient " << coef->arg().GetName() 
	   << " and PDF " << pdf->arg().GetName() << " have one or more dependents in common" << endl ;
      ret = kTRUE ;
    }
  }
  
  
  delete pIter ;
  delete cIter ;

  return ret ;
}


Int_t RooAddPdf::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& numVars) const 
{
  // Determine which part (if any) of given integral can be performed analytically.
  // If any analytical integration is possible, return integration scenario code

  // This PDF is by construction normalized
  return 0 ;
}


Double_t RooAddPdf::analyticalIntegral(Int_t code) const 
{
  // Return analytical integral defined by given scenario code

  // This PDF is by construction normalized
  return 1.0 ;
}
