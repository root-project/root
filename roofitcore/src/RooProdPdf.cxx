/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitTools
 *    File: $Id: RooProdPdf.cc,v 1.1 2001/05/10 00:16:08 verkerke Exp $
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

#include "TIterator.h"
#include "TList.h"
#include "RooFitCore/RooProdPdf.hh"
#include "RooFitCore/RooRealProxy.hh"

ClassImp(RooProdPdf)
;


RooProdPdf::RooProdPdf(const char *name, const char *title)
{
  // Dummy constructor
}


RooProdPdf::RooProdPdf(const char *name, const char *title,
		     RooAbsPdf& pdf1, RooAbsPdf& pdf2) : 
  RooAbsPdf(name,title)
{
  // Constructor with 2 PDFs
  addPdf(pdf1) ;
  addPdf(pdf2) ;    
}


RooProdPdf::RooProdPdf(const RooProdPdf& other, const char* name) :
  RooAbsPdf(other,name)
{
  // Copy constructor

  // Copy proxy lists
  RooRealProxy* proxy ;
  TIterator *iter = other._pdfProxyList.MakeIterator() ;
  while(proxy=(RooRealProxy*)iter->Next()) {
    _pdfProxyList.Add(new RooRealProxy("pdf",this,*proxy)) ;
  }
  delete iter ;
}


RooProdPdf::~RooProdPdf()
{
  // Destructor

  // Delete all owned proxies 
  _pdfProxyList.Delete() ;
}



void RooProdPdf::addPdf(RooAbsPdf& pdf) 
{  
  // Add PDF to product of PDFs
  RooRealProxy *pdfProxy = new RooRealProxy("pdf","pdf",this,pdf) ;  
  _pdfProxyList.Add(pdfProxy) ;
}


Double_t RooProdPdf::evaluate() const 
{
  // Calculate current value of object

  TIterator *pIter = _pdfProxyList.MakeIterator() ;
  Double_t value(1) ;
    
  // Calculate running product of pdfs
  RooRealProxy* pdf ;
  while(pdf=(RooRealProxy*)pIter->Next()) {
    value *= (*pdf) ;
  }

  delete pIter ;
  return value ;
}


Int_t RooProdPdf::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& numVars) const 
{
  // Determine which part (if any) of given integral can be performed analytically.
  // If any analytical integration is possible, return integration scenario code

  // This PDF is by construction normalized
  return 0 ;
}


Double_t RooProdPdf::analyticalIntegral(Int_t code) const 
{
  // Return analytical integral defined by given scenario code

  // This PDF is by construction normalized
  return 1.0 ;
}
