/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAddGenContext.cc,v 1.2 2001/10/13 00:38:53 david Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   11-Oct-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [AUX} --
// RooAddGenContext is an efficient implementation of the generator context
// specific for RooAddPdf PDFs. The sim-context owns a list of
// component generator contexts that are used to generate the events
// for each component PDF sequentially


#include "RooFitCore/RooAddGenContext.hh"
#include "RooFitCore/RooAddPdf.hh"
#include "RooFitCore/RooDataSet.hh"
#include "RooFitCore/RooRandom.hh"

ClassImp(RooAddGenContext)
;
  
RooAddGenContext::RooAddGenContext(const RooAddPdf &model, const RooArgSet &vars, 
				   const RooDataSet *prototype, Bool_t verbose) :
  RooAbsGenContext(model,vars,prototype,verbose), _pdf(&model)
{
  // Constructor. Build an array of generator contexts for each product component PDF
  model._pdfIter->Reset() ;
  RooAbsPdf* pdf ;
  _nComp = model._pdfList.getSize() ;
  _coefThresh = new Double_t[_nComp+1] ;

  Int_t i=1 ;
  _coefThresh[0] = 0 ;
  while(pdf=(RooAbsPdf*)model._pdfIter->Next()) {
    RooAbsGenContext* cx = pdf->genContext(vars,prototype,verbose) ;
    _gcList.Add(cx) ;

    // Store the fraction of this component
    Bool_t accum(kTRUE) ;
    if (model._allExtendable) {
      // All PDFs extendable: coefficient is expected number of events
      _coefThresh[i] = pdf->expectedEvents() ;
    } else {
      // N or N-1 coefficients
      RooAbsReal* coef = (RooAbsReal*) model._coefList.at(i-1) ;
      _coefThresh[i] = coef ? coef->getVal() : 1.0 ;
      // If last coef is missing, skip accumulation and keep coefSum at 1.0 
      accum = coef?kTRUE:kFALSE ;
    }

    // Accumulate coefficient
    if (i>0 && accum) {
      _coefThresh[i] += _coefThresh[i-1] ;
    }
    i++ ;
  }  
  
  // Normalize coefThresh array
  if (_coefThresh[_nComp]!=1.0) {
    for (i=1 ; i<_nComp ; i++) {
      _coefThresh[i] /= _coefThresh[_nComp] ;
    }      
  }
}



RooAddGenContext::~RooAddGenContext()
{
  // Destructor. Delete all owned subgenerator contexts
  delete[] _coefThresh ;
  _gcList.Delete() ;
}



void RooAddGenContext::initGenerator(const RooArgSet &theEvent)
{
  // Forward initGenerator call to all components
  TIterator* iter = _gcList.MakeIterator() ;
  RooAbsGenContext* gc ;
  while(gc=(RooAbsGenContext*)iter->Next()){
    gc->initGenerator(theEvent) ;
  }
  delete iter ;
}



void RooAddGenContext::generateEvent(RooArgSet &theEvent, Int_t remaining)
{
  // Throw a random number to determin which component to generate
  Double_t rand = RooRandom::uniform() ;
  Int_t i=0 ;
  for (i=0 ; i<_nComp ; i++) {
    if (rand>_coefThresh[i] && rand<_coefThresh[i+1]) {
      ((RooAbsGenContext*)_gcList.At(i))->generateEvent(theEvent,remaining) ;
      return ;
    }
  }
}


