/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooSimGenContext.cc,v 1.1 2001/10/12 01:48:46 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   11-Oct-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [AUX} --
// RooSimGenContext is an efficient implementation of the generator context
// specific for RooSimultaneous PDFs. The sim-context owns a list of
// component generator contexts that are used to generate the events
// for each sim-state sequentially.

#include "RooFitCore/RooSimGenContext.hh"
#include "RooFitCore/RooSimultaneous.hh"
#include "RooFitCore/RooRealProxy.hh"
#include "RooFitCore/RooDataSet.hh"
#include "RooFitCore/Roo1DTable.hh"
#include "RooFitCore/RooCategory.hh"


ClassImp(RooSimGenContext)
;
  
RooSimGenContext::RooSimGenContext(const RooSimultaneous &model, const RooArgSet &vars, 
				   const RooDataSet *prototype, Bool_t verbose) :
  RooAbsGenContext(model,vars,prototype,verbose), _pdf(&model)
{
  // Constructor. Build an array of generator contexts for each component PDF

  // Determine if we are requested to generate the index category
  RooAbsCategory *idxCat = (RooAbsCategory*) model._indexCat.absArg() ;
  RooArgSet pdfVars(vars) ;
  _doGenIdx = pdfVars.remove(*idxCat,kTRUE,kTRUE) ;
  
  if (_doGenIdx) {

    // Generate index category and all registered PDFS
    TIterator* iter = model._pdfProxyList.MakeIterator() ;
    RooRealProxy* proxy ;
    RooAbsPdf* pdf ;
    while(proxy=(RooRealProxy*)iter->Next()) {
      pdf=(RooAbsPdf*)proxy->absArg() ;
      RooAbsGenContext* cx = pdf->genContext(pdfVars,prototype,verbose) ;
      _gcList.Add(cx) ;
    }   
    delete iter ;

  } else {

    // Generate only current PDF
    RooRealProxy* proxy=(RooRealProxy*) _pdf->_pdfProxyList.FindObject(model._indexCat.arg().getLabel()) ;
    if (!proxy) {
      cout << "RooSimGenContext::ctor(" << GetName() << ") ERROR: current state ("
	   << model._indexCat.arg().getLabel() << ") has no associated PDF" << endl ;
      assert(0) ;
    }
    RooAbsPdf* pdf=(RooAbsPdf*)proxy->absArg() ;
    RooAbsGenContext* cx = pdf->genContext(pdfVars,prototype,verbose) ;
    _gcList.Add(cx) ;    

  }
}



RooSimGenContext::~RooSimGenContext()
{
  // Destructor. Delete all owned subgenerator contexts
  _gcList.Delete() ;
}


void RooSimGenContext::initGenerator(const RooArgSet &theEvent)
{
}

void RooSimGenContext::generateEvent(RooArgSet &theEvent, Int_t remaining)
{
}


RooDataSet* RooSimGenContext::__generate(Int_t nEvents) const
{
  // Generate dependents of each PDF product component independently
  // and merge results into a single data set

  if(!isValid()) {
    cout << ClassName() << "::" << GetName() << ": context is not valid" << endl;
    return 0;
  }


  RooAbsCategory* protoCat = _prototype ? (RooAbsCategory*) _prototype->get()->find(_pdf->_indexCat.arg().GetName()) : 0 ;

  // The number of event of each component must come from either the
  // extended mode of each component PDF, or from the prototype data set
  if (_doGenIdx && nEvents>0 && !protoCat) {
    cout << "RooSimGenContext::generate(" << GetName() << ") ERROR: Need either extended mode"
	 << " or prototype data to calculate number of events per category" << endl ;
    return 0 ;
  }

  // Retrieve event counters from prototype
  Roo1DTable* wTable(0) ;
  if (protoCat) wTable = _prototype->table(*protoCat) ;

  // Loop over the component generators
  TIterator* iter = _gcList.MakeIterator() ;
  TIterator* pIter = _pdf->_pdfProxyList.MakeIterator() ;
  RooAbsGenContext* gc ;
  Int_t nEvtSum(0) ;
  RooDataSet* sumData(0) ;
  RooCategory* indexFund = (RooCategory*) _pdf->_indexCat.arg().createFundamental() ;

  while(gc=(RooAbsGenContext*)iter->Next()) {
    RooRealProxy* proxy = (RooRealProxy*)pIter->Next() ;
    
    if (_verbose) {
      cout << "RooProdGenContext::generate: generating component PDF " << gc->GetName() << endl ;
    }

    // Calculate number of events for this component
    Int_t nEvtComp(0) ;
    if (nEvents>0) {
      // Regular mode: take number of events from prototype
      nEvtComp = _doGenIdx ? Int_t(wTable->get(proxy->name())+0.5) : nEvents ;
    } else {
      // Extended mode: Nevt is calculated by component
      nEvtComp = 0 ;
    }

    // Generate component 
    RooDataSet *data = gc->generate(nEvtComp) ;

    // Check if generation was successful
    if (!data) {
      cout << "RooProdGenContext::generate(" << GetName() 
	   << ") unable to generator component " << gc->GetName() << endl ;
      delete data ;
      delete sumData ;
      return 0 ;
    }

    // Add index column, if requested
    if (_doGenIdx) {
      indexFund->setLabel(proxy->name()) ;
      data->addColumn(*indexFund) ;
    }

    // Append component data to output data set
    if (sumData) {
      sumData->append(*data) ;
      delete data ;
    } else {
      sumData = data ;
    }

  }

  delete indexFund ;
  delete iter ;
  delete pIter ;
  delete wTable ;

  return sumData ;
}


