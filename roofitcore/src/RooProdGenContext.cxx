/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooProdGenContext.cc,v 1.7 2002/09/30 00:57:29 verkerke Exp $
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

ClassImp(RooProdGenContext)
;
  
RooProdGenContext::RooProdGenContext(const RooProdPdf &model, const RooArgSet &vars, 
				   const RooDataSet *prototype, Bool_t verbose) :
  RooAbsGenContext(model,vars,prototype,verbose), _pdf(&model)
{
  // Constructor. Build an array of generator contexts for each product component PDF

  // Make full list of dependents (generated & proto)
  RooArgSet deps(vars) ;
  if (prototype) {
    RooArgSet* protoDeps = model.getDependents(*prototype->get()) ;
    deps.add(*protoDeps) ;
    delete protoDeps ;
  }

  // Factorize product in irreducible terms
  TList* termList = model.factorizeProduct(deps) ;
  TIterator* termIter = termList->MakeIterator() ;

  RooAbsPdf* pdf ;
  RooArgSet* term ;
  while(term=(RooArgSet*)termIter->Next()) {
    TIterator* pdfIter = term->createIterator() ;
    
    if (term->getSize()==1) {
      // Simple term

      pdf = (RooAbsPdf*) pdfIter->Next() ;
      RooArgSet* pdfDep = pdf->getDependents(&vars) ;
      if (pdfDep->getSize()>0) {
	RooAbsGenContext* cx = pdf->genContext(*pdfDep,prototype,verbose) ;
	_gcList.Add(cx) ;
      } 
      delete pdfDep ;

    } else {

      // Composite term
      RooArgSet termDeps ;
      while(pdf=(RooAbsPdf*) pdfIter->Next()) {
	RooArgSet* pdfDep = pdf->getDependents(&vars) ;
	termDeps.add(*pdfDep,kFALSE) ;
	delete pdfDep ;
      }
      if (termDeps.getSize()>0) {
	const char* name = model.makeRGPPName("PRODGEN_",*term,RooArgSet(),RooArgSet()) ;      

	RooProdPdf* multiPdf = new RooProdPdf(name,name,*term) ;
	multiPdf->useDefaultGen(kTRUE) ;
	_ownedMultiProds.addOwned(*multiPdf) ;

	RooAbsGenContext* cx = multiPdf->genContext(termDeps,prototype,verbose) ;
	_gcList.Add(cx) ;
      }
    }
    
    delete pdfIter ;
  }
  _gcIter = _gcList.MakeIterator() ;
}



RooProdGenContext::~RooProdGenContext()
{
  // Destructor. Delete all owned subgenerator contexts
  delete _gcIter ;
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
}



void RooProdGenContext::generateEvent(RooArgSet &theEvent, Int_t remaining)
{
  // Generate a single event of the product by generating the components
  // of the products sequentially

  // Loop over the component generators
  TList compData ;
  RooAbsGenContext* gc ;
  _gcIter->Reset() ;
  while(gc=(RooAbsGenContext*)_gcIter->Next()) {

    // Generate component 
    gc->generateEvent(theEvent,remaining) ;
  }
}


void RooProdGenContext::printToStream(ostream &os, PrintOption opt, TString indent) const 
{
  RooAbsGenContext::printToStream(os,opt,indent) ;
  TString indent2(indent) ;
  indent2.Append("    ") ;
  RooAbsGenContext* gc ;
  _gcIter->Reset() ;
  while(gc=(RooAbsGenContext*)_gcIter->Next()) {
    gc->printToStream(os,opt,indent2) ;
  }  
}
