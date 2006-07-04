/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooProdGenContext.cc,v 1.23 2005/12/01 16:10:20 wverkerke Exp $
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
// RooProdGenContext is an efficient implementation of the generator context
// specific for RooProdPdf PDFs. The sim-context owns a list of
// component generator contexts that are used to generate the dependents
// for each component PDF sequentially. 

#include "RooFitCore/RooFit.hh"

#include "RooFitCore/RooProdGenContext.hh"
#include "RooFitCore/RooProdGenContext.hh"
#include "RooFitCore/RooProdPdf.hh"
#include "RooFitCore/RooDataSet.hh"
#include "RooFitCore/RooRealVar.hh"
#include "RooFitCore/RooGlobalFunc.hh"

ClassImp(RooProdGenContext)
;
  
RooProdGenContext::RooProdGenContext(const RooProdPdf &model, const RooArgSet &vars, 
				     const RooDataSet *prototype, const RooArgSet* auxProto, Bool_t verbose) :
  RooAbsGenContext(model,vars,prototype,auxProto,verbose), _pdf(&model)
{

  // Constructor. Build an array of generator contexts for each product component PDF

  // Make full list of dependents (generated & proto)
  RooArgSet deps(vars) ;
  if (prototype) {
    RooArgSet* protoDeps = model.getObservables(*prototype->get()) ;
    deps.remove(*protoDeps,kTRUE,kTRUE) ;
    delete protoDeps ;
  }

  // Factorize product in irreducible terms
  RooLinkedList termList,depsList,impDepList,crossDepList,intList ;
  model.factorizeProduct(deps,RooArgSet(),termList,depsList,impDepList,crossDepList,intList) ;
  TIterator* termIter = termList.MakeIterator() ;
  TIterator* normIter = depsList.MakeIterator() ;
  TIterator* impIter = impDepList.MakeIterator() ;

  RooArgSet genDeps ;
  // First add terms that do not import observables
  
  Bool_t working = kTRUE ;
  Int_t nSkip=0 ;
  while(working) {
    working = kFALSE ;

    RooAbsPdf* pdf ;
    RooArgSet* term ;
    RooArgSet* impDeps ;
    RooArgSet* termDeps ;

    termIter->Reset() ;
    impIter->Reset() ;
    normIter->Reset() ;

    while((term=(RooArgSet*)termIter->Next())) {
      impDeps = (RooArgSet*)impIter->Next() ;
      termDeps = (RooArgSet*)normIter->Next() ;

//        cout << "considering term " ; term->Print("1") ;
//        cout << "deps to be generated: " ; termDeps->Print("1") ;
//        cout << "imported dependents are " ; impDeps->Print("1") ;

      // Add this term if we have no imported dependents, or imported dependents are
      // already generated
      RooArgSet neededDeps(*impDeps) ;
      neededDeps.remove(genDeps,kTRUE,kTRUE) ;

//    cout << "needed imported dependents are " ; neededDeps.Print("1") ;
      if (neededDeps.getSize()>0) {
//   	cout << "skipping this term for now because it needs imported dependents that are not generated yet" << endl ;
	if (++nSkip<100) {
	  working = kTRUE ;
	} else {
	  cout << "RooProdGenContext ERROR: Generations is requested of observables that are conditional observables of the entire product expression: " ; neededDeps.Print("1") ;
	  _isValid = kFALSE ;
	  return ;
	}
	continue ;
      }

      // Check if this component has any dependents that need to be generated
      // e.g. it can happen that there are none if all dependents of this component are prototyped
      if (termDeps->getSize()==0) {
//   	cout << "no dependents to be generated for this term, removing it from list" << endl ;
	termList.Remove(term) ;
	depsList.Remove(termDeps) ;
	impDepList.Remove(impDeps) ;
	delete term ;
	delete termDeps ;
	delete impDeps ;
	continue ;
      }

      working = kTRUE ;	
      TIterator* pdfIter = term->createIterator() ;      
      if (term->getSize()==1) {
	// Simple term
	
	pdf = (RooAbsPdf*) pdfIter->Next() ;
	RooArgSet* pdfDep = pdf->getObservables(termDeps) ;
	if (pdfDep->getSize()>0) {
// 	  cout << "RooProdGenContext(" << model.GetName() << "): creating subcontext for " << pdf->GetName() << " with depSet " ; pdfDep->Print("1") ;
	  RooArgSet* auxProto = impDeps ? pdf->getObservables(impDeps) : 0 ;
	  RooAbsGenContext* cx = pdf->genContext(*pdfDep,prototype,auxProto,verbose) ;
	  _gcList.Add(cx) ;
	} 

// 	cout << "adding following dependents to list of generated observables: " ; pdfDep->Print("1") ;
	genDeps.add(*pdfDep) ;

	delete pdfDep ;
	
      } else {
	
	// Composite term
	if (termDeps->getSize()>0) {
	  const char* name = model.makeRGPPName("PRODGEN_",*term,RooArgSet(),RooArgSet(),0) ;      
	  
	  // Construct auxiliary PDF expressing product of composite terms, 
	  // following Conditional component specification of input model
	  RooLinkedList cmdList ;
	  RooLinkedList pdfSetList ;
	  pdfIter->Reset() ;
	  RooArgSet fullPdfSet ;
	  while((pdf=(RooAbsPdf*)pdfIter->Next())) {

	    RooArgSet* pdfnset = model.findPdfNSet(*pdf) ;
	    RooArgSet* pdfSet = new RooArgSet(*pdf) ;
	    pdfSetList.Add(pdfSet) ;

	    if (pdfnset && pdfnset->getSize()>0) {
	      // This PDF requires a Conditional() construction
	      cmdList.Add(RooFit::Conditional(*pdfSet,*pdfnset).Clone()) ;
//   	      cout << "Conditional " << pdf->GetName() << " " ; pdfnset->Print("1") ;
	    } else {
	      fullPdfSet.add(*pdfSet) ;
	    }
	    
	  }
	  RooProdPdf* multiPdf = new RooProdPdf(name,name,fullPdfSet,cmdList) ;
	  cmdList.Delete() ;
	  pdfSetList.Delete() ;

	  multiPdf->useDefaultGen(kTRUE) ;
	  _ownedMultiProds.addOwned(*multiPdf) ;
	  
//   	  cout << "RooProdGenContext(" << model.GetName() << "): creating subcontext for composite " << multiPdf->GetName() << " with depSet " ; termDeps->Print("1") ;
	  RooAbsGenContext* cx = multiPdf->genContext(*termDeps,prototype,auxProto,verbose) ;
	  _gcList.Add(cx) ;

//   	  cout << "adding following dependents to list of generated observables: " ; termDeps->Print("1") ;
	  genDeps.add(*termDeps) ;

	}
      }
      
      delete pdfIter ;

//        cout << "added generator for this term, removing from list" << endl ;


      termList.Remove(term) ;
      depsList.Remove(termDeps) ;
      impDepList.Remove(impDeps) ;
      delete term ;
      delete termDeps ;
      delete impDeps ;
      
    }
  }

  // Check if there are any left over terms that cannot be generated 
  // separately due to cross dependency of observables
  if (termList.GetSize()>0) {
//      cout << "there are left-over terms that need to be generated separately" << endl ;

    RooAbsPdf* pdf ;
    RooArgSet* term ;

    // Concatenate remaining terms
    termIter->Reset() ;
    normIter->Reset() ;
    RooArgSet trailerTerm ;
    RooArgSet trailerTermDeps ;
    while((term=(RooArgSet*)termIter->Next())) {
      RooArgSet* termDeps = (RooArgSet*)normIter->Next() ;
      trailerTerm.add(*term) ;
      trailerTermDeps.add(*termDeps) ;
    }

    const char* name = model.makeRGPPName("PRODGEN_",trailerTerm,RooArgSet(),RooArgSet(),0) ;      
      
    // Construct auxiliary PDF expressing product of composite terms, 
    // following Partial/Full component specification of input model
    RooLinkedList cmdList ;
    RooLinkedList pdfSetList ;
    RooArgSet fullPdfSet ;

    TIterator* pdfIter = trailerTerm.createIterator() ;
    while((pdf=(RooAbsPdf*)pdfIter->Next())) {
	
      RooArgSet* pdfnset = model.findPdfNSet(*pdf) ;
      RooArgSet* pdfSet = new RooArgSet(*pdf) ;
      pdfSetList.Add(pdfSet) ;
      
      if (pdfnset && pdfnset->getSize()>0) {
	// This PDF requires a Conditional() construction
	  cmdList.Add(RooFit::Conditional(*pdfSet,*pdfnset).Clone()) ;
      } else {
	fullPdfSet.add(*pdfSet) ;
      }
      
    }
//     cmdList.Print("v") ;
    RooProdPdf* multiPdf = new RooProdPdf(name,name,fullPdfSet,cmdList) ;
    cmdList.Delete() ;
    pdfSetList.Delete() ;
    
    multiPdf->useDefaultGen(kTRUE) ;
    _ownedMultiProds.addOwned(*multiPdf) ;
    
//     cout << "RooProdGenContext(" << model.GetName() << "): creating context for trailer composite " << multiPdf->GetName() << " with depSet " ; trailerTermDeps.Print("1") ;
    RooAbsGenContext* cx = multiPdf->genContext(trailerTermDeps,prototype,auxProto,verbose) ;
    _gcList.Add(cx) ;    
  }


  delete termIter ;
  delete impIter ;
  delete normIter ;

  _gcIter = _gcList.MakeIterator() ;


  // We own contents of lists filled by factorizeProduct() 
  termList.Delete() ;
  depsList.Delete() ;
  impDepList.Delete() ;
  crossDepList.Delete() ;
  intList.Delete() ;
}



RooProdGenContext::~RooProdGenContext()
{
  // Destructor. Delete all owned subgenerator contexts
  delete _gcIter ;
  _gcList.Delete() ;  
}


void RooProdGenContext::initGenerator(const RooArgSet &theEvent)
{
//   cout << "RooProdGenContext::initGenerator(" << GetName() << ") theEvent = " << endl ;
//   theEvent.Print("v") ;


  // Forward initGenerator call to all components
  RooAbsGenContext* gc ;
  _gcIter->Reset() ;
  while((gc=(RooAbsGenContext*)_gcIter->Next())){
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

//   cout << "generateEvent" << endl ;
//   ((RooRealVar*)theEvent.find("x"))->setVal(0) ;
//   ((RooRealVar*)theEvent.find("y"))->setVal(0) ;

//   cout << "theEvent before generation cycle:" << endl ;
//   theEvent.Print("v") ;

  while((gc=(RooAbsGenContext*)_gcIter->Next())) {

    // Generate component 
//     cout << endl << endl << "calling generator component " << gc->GetName() << endl ;
    gc->generateEvent(theEvent,remaining) ;
//     cout << "theEvent is after this generation call is" << endl ;
//     theEvent.Print("v") ;
// //     theEvent.Print("v") ;
  }
}


void RooProdGenContext::setProtoDataOrder(Int_t* lut)
{
  RooAbsGenContext::setProtoDataOrder(lut) ;
  _gcIter->Reset() ;
  RooAbsGenContext* gc ;
  while((gc=(RooAbsGenContext*)_gcIter->Next())) {
    gc->setProtoDataOrder(lut) ;
  }
}


void RooProdGenContext::printToStream(ostream &os, PrintOption opt, TString indent) const 
{
  RooAbsGenContext::printToStream(os,opt,indent) ;
  TString indent2(indent) ;
  indent2.Append("    ") ;
  RooAbsGenContext* gc ;
  _gcIter->Reset() ;
  while((gc=(RooAbsGenContext*)_gcIter->Next())) {
    gc->printToStream(os,opt,indent2) ;
  }  
}
