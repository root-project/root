/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooSimultaneous.cc,v 1.8 2001/08/18 02:13:11 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   25-Jun-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

#include "TObjString.h"
#include "RooFitCore/RooSimultaneous.hh"
#include "RooFitCore/RooAbsCategoryLValue.hh"

ClassImp(RooSimultaneous)
;


RooSimultaneous::RooSimultaneous(const char *name, const char *title, 
				 RooAbsCategoryLValue& indexCat) : 
  RooAbsPdf(name,title), _numPdf(0.),
  _indexCat("indexCat","Index category",this,indexCat)
{
}

RooSimultaneous::RooSimultaneous(const RooSimultaneous& other, const char* name) : 
  RooAbsPdf(other,name),
  _indexCat("indexCat",this,other._indexCat), _numPdf(other._numPdf)
{
  // Copy proxy list 
  TIterator* pIter = other._pdfProxyList.MakeIterator() ;
  RooRealProxy* proxy ;
  while (proxy=(RooRealProxy*)pIter->Next()) {
    _pdfProxyList.Add(new RooRealProxy(proxy->GetName(),this,*proxy)) ;
  }
  delete pIter ;
}


RooSimultaneous::~RooSimultaneous() 
{
  _pdfProxyList.Delete() ;
}



const RooFitResult* RooSimultaneous::fitTo(RooDataSet& data, Option_t *fitOpt, Option_t *optOpt) 
{
  TString opts = optOpt ;
  opts.ToLower() ;

  if (!opts.Contains("s")) {
  // Fit this PDF to given data set using a regular fit context    
    return RooAbsPdf::fitTo(data,fitOpt,optOpt) ;
  } 

  // Fit this PDF to given data set using a SimFit context
  RooSimFitContext context(&data,this) ;
  return context.fit(fitOpt,optOpt) ;  
}



Bool_t RooSimultaneous::addPdf(const RooAbsPdf& pdf, const char* catLabel)
{
  // PDFs cannot overlap with the index category
  if (pdf.dependsOn(_indexCat.arg())) {
    cout << "RooSimultaneous::addPdf(" << GetName() << "): ERROR, PDF " << pdf.GetName() 
	 << " overlaps with index category " << _indexCat.arg().GetName() << endl ;
    return kTRUE ;
  }

  // Each index state can only have one PDF associated with it
  if (_pdfProxyList.FindObject(catLabel)) {
    cout << "RooSimultaneous::addPdf(" << GetName() << "): ERROR, index state " 
	 << catLabel << " has already an associated PDF" << endl ;
    return kTRUE ;
  }


  // Create a proxy named after the associated index state
  TObject* proxy = new RooRealProxy(catLabel,catLabel,this,(RooAbsPdf&)pdf) ;
  _pdfProxyList.Add(proxy) ;
  _numPdf += 1.0 ;

  return kFALSE ;
}



Double_t RooSimultaneous::evaluate(const RooArgSet* nset) const
{
//   // Require that all states have an associated PDF
//   if (_pdfProxyList.GetSize() != _indexCat.arg().numTypes()) {
//     cout << "RooSimultaneous::evaluate(" << GetName() 
// 	 << "): ERROR, number of PDFs and number of index states do not match" << endl ;
//     return 0 ;
//   }

  // Retrieve the proxy by index name
  RooRealProxy* proxy = (RooRealProxy*) _pdfProxyList.FindObject((const char*) _indexCat) ;
  
  assert(proxy!=0) ;

  // Return the selected PDF value, normalized by the number of index states
  return ((RooAbsPdf*)(proxy->absArg()))->getVal(nset) / _numPdf ;
}

