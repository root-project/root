/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooSimGenContext.cc,v 1.3 2001/10/14 07:11:42 verkerke Exp $
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
// specific for RooSimultaneous PDFs when generating more than one of the
// component pdfs.

#include "RooFitCore/RooSimGenContext.hh"
#include "RooFitCore/RooSimultaneous.hh"
#include "RooFitCore/RooRealProxy.hh"
#include "RooFitCore/RooDataSet.hh"
#include "RooFitCore/Roo1DTable.hh"
#include "RooFitCore/RooCategory.hh"
#include "RooFitCore/RooRandom.hh"


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
  Bool_t doGenIdx = pdfVars.remove(*idxCat,kTRUE,kTRUE) ;
  if (!doGenIdx) {
    cout << "RooSimGenContext::ctor(" << GetName() << ") ERROR: This context must"
	 << " generate the index category" << endl ;
    _isValid = kFALSE ;
    return ;
  }

  // We must either have the prototype or extended likelihood to determined
  // the relative fractions of the components
  _haveIdxProto = prototype ? prototype->get()->find(idxCat->GetName())?kTRUE:kFALSE : 0 ;
  _idxCatName = idxCat->GetName() ;
  if (!_haveIdxProto && !model.canBeExtended()) {
    cout << "RooSimGenContext::ctor(" << GetName() << ") ERROR: Need either extended mode"
	 << " or prototype data to calculate number of events per category" << endl ;
    _isValid = kFALSE ;
    return ;
  }

  // Initialize fraction threshold array (used only in extended mode)
  _numPdf = model._pdfProxyList.GetSize() ;
  _fracThresh = new Double_t[_numPdf+1] ;
  _fracThresh[0] = 0 ;
  
  // Generate index category and all registered PDFS
  TIterator* iter = model._pdfProxyList.MakeIterator() ;
  RooRealProxy* proxy ;
  RooAbsPdf* pdf ;
  Int_t i(1) ;
  while(proxy=(RooRealProxy*)iter->Next()) {
    pdf=(RooAbsPdf*)proxy->absArg() ;

    // Create generator context for this PDF
    RooAbsGenContext* cx = pdf->genContext(pdfVars,prototype,verbose) ;

    // Name the context after the associated state and add to list
    cx->SetName(proxy->name()) ;
    _gcList.Add(cx) ;

    // Fill fraction threshold array
    _fracThresh[i] = _fracThresh[i-1] + (_haveIdxProto?0:pdf->expectedEvents()) ;
    i++ ;
  }   
  delete iter ;
    
  // Normalize fraction threshold array
  if (!_haveIdxProto) {
    for(i=0 ; i<_numPdf ; i++) 
      _fracThresh[i] /= _fracThresh[_numPdf] ;
  }
  
}



RooSimGenContext::~RooSimGenContext()
{
  // Destructor. Delete all owned subgenerator contexts
  delete[] _fracThresh ;
  _gcList.Delete() ;
}



void RooSimGenContext::initGenerator(const RooArgSet &theEvent)
{
  // Forward initGenerator call to all components
  RooAbsGenContext* gc ;
  TIterator* iter = _gcList.MakeIterator() ;
  while(gc=(RooAbsGenContext*)iter->Next()){
    gc->initGenerator(theEvent) ;
  }
  delete iter;

}



void RooSimGenContext::generateEvent(RooArgSet &theEvent, Int_t remaining)
{
  // Generate event appropriate for current index state. 
  // The index state is taken either from the prototype
  // or generated from the fraction threshold table.

  if (_haveIdxProto) {

    // Lookup pdf from selected prototype index state
    RooAbsCategory* cat = (RooAbsCategory*) theEvent.find(_idxCatName) ;
    ((RooAbsGenContext*)_gcList.FindObject(cat->getLabel()))->generateEvent(theEvent,remaining) ;
    
  
  } else {

    // Throw a random number and select PDF from fraction threshold table
    Double_t rand = RooRandom::uniform() ;
    Int_t i=0 ;
    for (i=0 ; i<_numPdf ; i++) {
      if (rand>_fracThresh[i] && rand<_fracThresh[i+1]) {
	RooAbsGenContext* gen= ((RooAbsGenContext*)_gcList.At(i)) ;
	gen->generateEvent(theEvent,remaining) ;
	((RooCategory*)theEvent.find(_idxCatName))->setLabel(gen->GetName()) ;
	return ;
      }
    }

  }
}

