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
// RooAddGenContext is an efficient implementation of the generator context
// specific for RooAddPdf PDFs. The sim-context owns a list of
// component generator contexts that are used to generate the events
// for each component PDF sequentially


#include "RooFitCore/RooAddGenContext.hh"
#include "RooFitCore/RooAddPdf.hh"
#include "RooFitCore/RooDataSet.hh"

ClassImp(RooAddGenContext)
;
  
RooAddGenContext::RooAddGenContext(const RooAddPdf &model, const RooArgSet &vars, 
				   const RooDataSet *prototype, Bool_t verbose) :
  RooAbsGenContext(model,verbose), _pdf(&model)
{
  // Constructor. Build an array of generator contexts for each product component PDF
  model._pdfIter->Reset() ;
  RooAbsPdf* pdf ;
  while(pdf=(RooAbsPdf*)model._pdfIter->Next()) {
    RooAbsGenContext* cx = pdf->genContext(vars,prototype,verbose) ;
    _gcList.Add(cx) ;
  }  
}



RooAddGenContext::~RooAddGenContext()
{
  // Destructor. Delete all owned subgenerator contexts
  _gcList.Delete() ;
}



RooDataSet* RooAddGenContext::generate(Int_t nEvents) const
{
  // Generate dependents of each PDF product component independently
  // and merge results into a single data set

  if(!isValid()) {
    cout << ClassName() << "::" << GetName() << ": context is not valid" << endl;
    return 0;
  }

  // Loop over the component generators
  TIterator* iter = _gcList.MakeIterator() ;
  TIterator* cIter = _pdf->_coefList.createIterator() ;
  

  RooAbsGenContext* gc ;
  Int_t nEvtSum(0) ;
  RooDataSet* sumData(0) ;
  while(gc=(RooAbsGenContext*)iter->Next()) {
    
    if (_verbose) {
      cout << "RooProdGenContext::generate: generating component PDF " << gc->GetName() << endl ;
    }

    // Calculate number of events for this component
    Int_t nEvtComp(0) ;
    if (nEvents>0) {
      // Regular mode Nevt = Ntot*coef
      RooAbsReal* coef = (RooAbsReal*) cIter->Next() ;
      if (coef) {
	nEvtComp = Int_t(nEvents*coef->getVal()+0.5) ;
	nEvtSum += nEvtComp ;
      } else {	
	// Last component: take exact remainder of number of events
	nEvtComp = nEvents - nEvtSum ;	

	// If remainder is zero or negative due to rounding, 
	// skip generation of last component altogether.
	if (nEvtComp<=0) {
	  if (_verbose) {
	    cout << "RooProdGenContext::generate(" << GetName() << ") number of events to generate"
		 << " for last component is " << nEvtComp << ", skipping last component" << endl ;
	  }
	  continue ;
	}
      }
    } else {
      // Extended mode: Nevt is calculated by component
      nEvtComp = 0 ;
    }

    // Generate component 
    RooDataSet *data = gc->generate(nEvtComp) ;

    // Check if generation was successfull
    if (!data) {
      cout << "RooProdGenContext::generate(" << GetName() 
	   << ") unable to generator component " << gc->GetName() << endl ;
      delete data ;
      delete sumData ;
      return 0 ;
    }

    // Append component data to output data set
    if (sumData) {
      sumData->append(*data) ;
      delete data ;
    } else {
      sumData = data ;
    }

  }
  delete iter ;
  delete cIter ;

  return sumData ;
}


