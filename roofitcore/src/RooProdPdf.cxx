/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitTools
 *    File: $Id: RooProdPdf.cc,v 1.4 2001/06/08 05:51:05 verkerke Exp $
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
#include "TList.h"
#include "RooFitCore/RooProdPdf.hh"
#include "RooFitCore/RooRealProxy.hh"

ClassImp(RooProdPdf)
;


RooProdPdf::RooProdPdf(const char *name, const char *title, Double_t cutOff) :
  RooAbsPdf(name,title), _pdfProxyIter(_pdfProxyList.MakeIterator()), _cutOff(cutOff)
{
  // Dummy constructor
}


RooProdPdf::RooProdPdf(const char *name, const char *title,
		     RooAbsPdf& pdf1, RooAbsPdf& pdf2, Double_t cutOff) : 
  RooAbsPdf(name,title), _pdfProxyIter(_pdfProxyList.MakeIterator()), _cutOff(cutOff)
{
  // Constructor with 2 PDFs
  addPdf(pdf1) ;
  addPdf(pdf2) ;    
}


RooProdPdf::RooProdPdf(const RooProdPdf& other, const char* name) :
  RooAbsPdf(other,name), _pdfProxyIter(_pdfProxyList.MakeIterator()), 
  _cutOff(other._cutOff)
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
  delete _pdfProxyIter ;
  _pdfProxyList.Delete() ;
}



void RooProdPdf::addPdf(RooAbsPdf& pdf) 
{  
  // Add PDF to product of PDFs
  RooRealProxy *pdfProxy = new RooRealProxy("pdf","pdf",this,pdf) ;  
  _pdfProxyList.Add(pdfProxy) ;
}


Double_t RooProdPdf::evaluate(const RooDataSet* dset) const 
{
  // Calculate current value of object

  Double_t value(1) ;
    
  // Calculate running product of pdfs
  RooRealProxy* pdf ;
  _pdfProxyIter->Reset() ;
  while(pdf=(RooRealProxy*)_pdfProxyIter->Next()) {    
    value *= (*pdf) ;
    if (value<_cutOff) break ;
  }

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
