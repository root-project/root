/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooProdGenContext.cc,v 1.5 2002/09/05 04:33:49 verkerke Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2002, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

// -- CLASS DESCRIPTION [AUX} --
// RooProdGenContext is an efficient implementation of the generator context
// specific for RooProdPdf PDFs. The sim-context owns a list of
// component generator contexts that are used to generate the dependents
// for each component PDF sequentially. 

#include "RooFitCore/RooProdGenContext.hh"
#include "RooFitCore/RooProdPdf.hh"
#include "RooFitCore/RooDataSet.hh"
#include "RooFitCore/RooCategory.hh"
#include "RooFitCore/RooSuperCategory.hh"
#include "RooFitCore/RooRandom.hh"

ClassImp(RooProdGenContext)
;
  
RooProdGenContext::RooProdGenContext(const RooProdPdf &model, const RooArgSet &vars, 
				   const RooDataSet *prototype, Bool_t verbose) :
  RooAbsGenContext(model,vars,prototype,verbose), _pdf(&model), _ccdSuper(0), _ccdCloneSet(0),
  _pdfCloneSet(0),_pdfClone(0), _pdfCcdInt(0), _ccdTable(0)
{
  // Constructor. Build an array of generator contexts for each product component PDF

  // Take eventual common category dependents out of vars
  RooArgSet goodVars(vars) ;
  RooArgList* commonCatDepSet = model.getCategoryOverlapDeps(&vars) ;
  goodVars.remove(*commonCatDepSet,kTRUE,kTRUE) ;
  _commonCats.addClone(*commonCatDepSet) ;
  delete commonCatDepSet ;

  // Determine if we need to refresh CCD tables every event
  _ccdRefresh=kFALSE ;
  if (_commonCats.getSize()>0 && prototype) {
    RooArgSet* deps = model.getDependents(RooArgSet()) ;
    RooArgSet* protoDeps = (RooArgSet*) deps->selectCommon(*prototype->get()) ;
    if (protoDeps->getSize()>0) {
      _ccdRefresh=kTRUE ;
    }
    delete deps ;
    delete protoDeps ;
  }

  model._pdfIter->Reset() ;
  RooAbsPdf* pdf ;
  while(pdf=(RooAbsPdf*)model._pdfIter->Next()) {
    RooArgSet* pdfDep = pdf->getDependents(&goodVars) ;
    if (pdfDep->getSize()>0) {
      RooAbsGenContext* cx = pdf->genContext(*pdfDep,prototype,verbose) ;
      _gcList.Add(cx) ;
    } 
    delete pdfDep ;
  }
  _gcIter = _gcList.MakeIterator() ;
}



RooProdGenContext::~RooProdGenContext()
{
  // Destructor. Delete all owned subgenerator contexts
  delete _gcIter ;
  delete _pdfCcdInt ;
  delete _pdfCloneSet ;
  delete _ccdSuper ;
  delete _ccdCloneSet ;
  delete _ccdTable ;
  _gcList.Delete() ;  
}


void RooProdGenContext::initGenerator(const RooArgSet &theEvent)
{
  // Forward initGenerator call to all components
  RooAbsGenContext* gc ;
  _gcIter->Reset() ;
  while(gc=(RooAbsGenContext*)_gcIter->Next()){
    gc->initGenerator(theEvent) ;
  }

  // Replace commonCatDeps with those in theEvent
  RooArgSet* tmp = (RooArgSet*) theEvent.selectCommon(_commonCats) ;
  _commonCats.removeAll() ;
  _commonCats.add(*tmp) ;
  delete tmp ;


  if (_commonCats.getSize()>0) {
    _ccdCloneSet = (RooArgSet*) _commonCats.snapshot(kTRUE) ;
    _ccdSuper = new RooSuperCategory("ccdSuper","ccdSuper",*_ccdCloneSet) ;
    _pdfCloneSet = (RooArgSet*) RooArgSet(*_pdf).snapshot(kTRUE) ;
    _pdfClone = (RooAbsPdf*) _pdfCloneSet->find(_pdf->GetName()) ;
    _pdfClone->recursiveRedirectServers(theEvent) ;
    _pdfClone->recursiveRedirectServers(*_ccdCloneSet) ;
    _ccdTable = new Double_t[_ccdSuper->numTypes()+1] ;

    RooArgSet intSet(theEvent) ;
    intSet.remove(_commonCats,kTRUE,kTRUE) ;
    _pdfCcdInt = (RooRealIntegral*) _pdfClone->createIntegral(intSet) ;
    
    if (!_ccdRefresh) updateCCDTable() ;
  }

}



void RooProdGenContext::updateCCDTable()
{
  _ccdTable[0] = 0. ;
  
  // Fill table with integrals
  TIterator* iter = _ccdSuper->typeIterator() ;
  RooCatType* type ;
  Int_t i=1 ;
  while(type=(RooCatType*)iter->Next()) {
    _ccdSuper->setLabel(type->GetName()) ;
    
    _ccdTable[i]  = _pdfCcdInt->getVal() ;
    _ccdTable[i] += _ccdTable[i-1] ;
    i++ ;
  }
  delete iter ;
  
  // Normalize table to sum of integrals
  Int_t n=_ccdSuper->numTypes() ;
  for (i=1 ; i<=n ; i++) {
    _ccdTable[i]/=_ccdTable[n] ;
  }
}



void RooProdGenContext::generateEvent(RooArgSet &theEvent, Int_t remaining)
{
  // Generate a single event of the product by generating the components
  // of the products sequentially

  if (_ccdSuper) {

    if (_ccdRefresh) updateCCDTable() ;

    // Throw random number and select ccd state according to weight
    Double_t rand = RooRandom::uniform() ;
    Int_t i,n=_ccdSuper->numTypes() ;
    for (i=0 ; i<n ; i++) {
      if (rand>_ccdTable[i] && rand<_ccdTable[i+1]) {
	_ccdSuper->setIndex(i) ;
	theEvent = *_ccdCloneSet ;
      }
    }
  }


  // Loop over the component generators
  TList compData ;
  RooAbsGenContext* gc ;
  _gcIter->Reset() ;
  while(gc=(RooAbsGenContext*)_gcIter->Next()) {

    // Generate component 
    gc->generateEvent(theEvent,remaining) ;
  }
}
