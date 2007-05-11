/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAddGenContext.cc,v 1.18 2005/12/01 16:10:20 wverkerke Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

// -- CLASS DESCRIPTION [AUX} --
// RooAddGenContext is an efficient implementation of the generator context
// specific for RooAddPdf PDFs. The sim-context owns a list of
// component generator contexts that are used to generate the events
// for each component PDF sequentially


#include "RooFit.h"

#include "RooAddGenContext.h"
#include "RooAddGenContext.h"
#include "RooAddPdf.h"
#include "RooDataSet.h"
#include "RooRandom.h"

ClassImp(RooAddGenContext)
;
  
RooAddGenContext::RooAddGenContext(const RooAddPdf &model, const RooArgSet &vars, 
				   const RooDataSet *prototype, const RooArgSet* auxProto,
				   Bool_t verbose) :
  RooAbsGenContext(model,vars,prototype,auxProto,verbose)
{
  // Constructor. Build an array of generator contexts for each product component PDF
  _pdfSet = (RooArgSet*) RooArgSet(model).snapshot(kTRUE) ;
  _pdf = (RooAddPdf*) _pdfSet->find(model.GetName()) ;


  // Fix normalization set of this RooAddPdf
  if (prototype) 
    {
      RooArgSet coefNSet(vars) ;
      coefNSet.add(*prototype->get()) ;
      _pdf->fixAddCoefNormalization(coefNSet) ;
    }

  model._pdfIter->Reset() ;
  RooAbsPdf* pdf ;
  _nComp = model._pdfList.getSize() ;
  _coefThresh = new Double_t[_nComp+1] ;
  _vars = (RooArgSet*) vars.snapshot(kFALSE) ;

  while((pdf=(RooAbsPdf*)model._pdfIter->Next())) {
    RooAbsGenContext* cx = pdf->genContext(vars,prototype,auxProto,verbose) ;
    _gcList.Add(cx) ;
  }  

  _pdf->syncCoefProjList(_vars) ;
  _pdf->recursiveRedirectServers(*_theEvent) ;
}



RooAddGenContext::~RooAddGenContext()
{
  // Destructor. Delete all owned subgenerator contexts
  delete[] _coefThresh ;
  _gcList.Delete() ;
  delete _vars ;
  delete _pdfSet ;
}



void RooAddGenContext::initGenerator(const RooArgSet &theEvent)
{
  _pdf->recursiveRedirectServers(theEvent) ;

  // Forward initGenerator call to all components
  TIterator* iter = _gcList.MakeIterator() ;
  RooAbsGenContext* gc ;
  while((gc=(RooAbsGenContext*)iter->Next())){
    gc->initGenerator(theEvent) ;
  }
  delete iter ;
}



void RooAddGenContext::generateEvent(RooArgSet &theEvent, Int_t remaining)
{
  // Throw a random number to determin which component to generate
  updateThresholds() ;
  Double_t rand = RooRandom::uniform() ;
  Int_t i=0 ;
  for (i=0 ; i<_nComp ; i++) {
    if (rand>_coefThresh[i] && rand<_coefThresh[i+1]) {
      ((RooAbsGenContext*)_gcList.At(i))->generateEvent(theEvent,remaining) ;
      return ;
    }
  }
}


void RooAddGenContext::updateThresholds()
{
  _pdf->updateCoefCache(_vars,_vars,0) ;  
  _coefThresh[0] = 0. ;
  Int_t i ;
  for (i=0 ; i<_nComp ; i++) {
    _coefThresh[i+1] = _pdf->_coefCache[i] ;
    _coefThresh[i+1] += _coefThresh[i] ;
  }
}


void RooAddGenContext::setProtoDataOrder(Int_t* lut)
{
  RooAbsGenContext::setProtoDataOrder(lut) ;
  Int_t i ;
  for (i=0 ; i<_nComp ; i++) {
    ((RooAbsGenContext*)_gcList.At(i))->setProtoDataOrder(lut) ;
  }
}
