/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooProdGenContext.cc,v 1.2 2001/10/13 00:38:54 david Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   11-Oct-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
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
  model._pdfIter->Reset() ;
  RooAbsPdf* pdf ;
  while(pdf=(RooAbsPdf*)model._pdfIter->Next()) {
    RooArgSet* pdfDep = pdf->getDependents(&vars) ;
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
