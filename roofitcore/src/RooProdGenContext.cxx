/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
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
// for each component PDF sequentially. At the end, the component dataset are 
// merged into a single dataset

#include "RooFitCore/RooProdGenContext.hh"
#include "RooFitCore/RooProdPdf.hh"
#include "RooFitCore/RooDataSet.hh"

ClassImp(RooProdGenContext)
;
  
RooProdGenContext::RooProdGenContext(const RooProdPdf &model, const RooArgSet &vars, 
				   const RooDataSet *prototype, Bool_t verbose) :
  RooAbsGenContext(model,verbose), _prototype(prototype), _pdf(&model)
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
}



RooProdGenContext::~RooProdGenContext()
{
  // Destructor. Delete all owned subgenerator contexts
  _gcList.Delete() ;
}



RooDataSet* RooProdGenContext::generate(Int_t nEvents) const
{
  // Generate dependents of each PDF product component independently
  // and merge results into a single data set

  if(!isValid()) {
    cout << ClassName() << "::" << GetName() << ": context is not valid" << endl;
    return 0;
  }

  // Calculate the expected number of events if necessary
  if(nEvents <= 0) {
    if(_prototype) {
      nEvents= (Int_t)_prototype->numEntries();
    }
    else {
      nEvents= (Int_t)(_pdf->expectedEvents() + 0.5);
    }
    if(nEvents <= 0) {
      cout << ClassName() << "::" << GetName()
	   << ":generate: cannot calculate expected number of events" << endl;
      return 0;
    }
    else if(_verbose) {
      cout << ClassName() << "::" << GetName() << ":generate: will generate "
	   << nEvents << " events" << endl;
    }
  }  

  // Loop over the component generators
  TList compData ;
  TIterator* iter = _gcList.MakeIterator() ;
  RooAbsGenContext* gc ;
  while(gc=(RooAbsGenContext*)iter->Next()) {
    if (_verbose) {
      cout << "RooProdGenContext::generate: generating component PDF " << gc->GetName() << endl ;
    }

    // Generate component 
    RooDataSet *data = gc->generate(nEvents) ;

    // Check if generation was successfull
    if (!data) {
      cout << "RooProdGenContext::generate(" << GetName() 
	   << ") unable to generator component " << gc->GetName() << endl ;
      compData.Delete() ;
      return 0 ;
    }

    //Add component data to output list
    compData.Add(data) ;
  }
  delete iter ;

  // Pull first component from the list ;
  RooDataSet* combiData = (RooDataSet*) compData.At(0) ;
  compData.Remove(combiData) ;

  // Merge other datasets of other components with first component
  combiData->merge(compData) ;

  return combiData ;
}


