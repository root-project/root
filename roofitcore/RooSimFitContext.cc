/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooSimFitContext.cc,v 1.14 2001/11/27 23:19:05 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [AUX] --

#include "RooFitCore/RooSimFitContext.hh"
#include "RooFitCore/RooSimultaneous.hh"
#include "RooFitCore/RooAbsCategoryLValue.hh"
#include "RooFitCore/RooAbsData.hh"
#include "RooFitCore/RooFormulaVar.hh"
#include "RooFitCore/RooArgList.hh"

ClassImp(RooSimFitContext)
;

RooSimFitContext::RooSimFitContext(const RooAbsData* data, const RooSimultaneous* simpdf, const RooArgSet* projDeps) : 
  RooFitContext(data,simpdf,kFALSE,kTRUE,projDeps), _nGlobEvents(-1)
{
  RooAbsCategoryLValue& simCat = (RooAbsCategoryLValue&) simpdf->_indexCat.arg() ;

  _nCtx = simCat.numTypes() ;
  _nCtxFilled = 0 ;
  _ctxArray = new pRooFitContext[_nCtx] ;
  _dsetArray = new pRooAbsData[_nCtx] ;
  _offArray = new Double_t[_nCtx] ;
  _nllArray = new Double_t[_nCtx] ;
  _dirtyArray = new Bool_t[_nCtx] ;

  TString simCatName(simCat.GetName()) ;

  // WVE clone simcat and attach to dataset!
  
  // Create array of regular fit contexts, containing subset of data and single fitCat PDF
  Int_t n(0) ;
  RooCatType* type ;
  TIterator* catIter = simCat.typeIterator() ;
  while(type=(RooCatType*)catIter->Next()){
    simCat.setIndex(type->getVal()) ;

    // Retrieve the PDF for this simCat state
    RooRealProxy* proxy = (RooRealProxy*) simpdf->_pdfProxyList.FindObject((const char*) simpdf->_indexCat) ;
    if (proxy) {
      RooAbsPdf* pdf = (RooAbsPdf*)proxy->absArg() ;

      //Refine a dataset containing only events for this simCat state
      char cutSpec[1024] ;
      sprintf(cutSpec,"%s==%d",simCatName.Data(),simCat.getIndex()) ;
      RooAbsData* dset = _dataClone->reduce(RooFormulaVar("simCatCut",cutSpec,simCat)) ;
      if (dset->numEntries()>0) {
	cout << "RooSimFitContext::RooSimFitContext: creating fit sub-context for state " << type->GetName() 
	     << " (" << dset->numEntries() << " dataset entries)" << endl ;
	_dsetArray[n] = dset ;
	_ctxArray[n] = new RooFitContext(dset,pdf,kFALSE,kTRUE,projDeps) ;
	_dirtyArray[n] = kTRUE ;
	_nCtxFilled++ ;
      } else {
	cout << "RooSimFitContext::RooSimFitContext: state " << type->GetName() 
	     << " has no data entries, sub-context not created" << endl ;
	delete dset ;
	_dsetArray[n] = 0 ;
	_ctxArray[n] = 0 ;
	_dirtyArray[n] = kFALSE ;
      }
    } else {
      _dsetArray[n] = 0 ;
      _ctxArray[n] = 0 ;
      _dirtyArray[n] = kFALSE ;
    }

    _nllArray[n] = 0 ;
    n++ ;
  }

  // Precalculate NLL offsets for each category
  Int_t i ;
  Double_t logNF = log(_nCtxFilled) ;
  for (i=0 ; i<_nCtx ; i++) {
    if (_ctxArray[i]) {
      _offArray[i] = _ctxArray[i]->_dataClone->numEntries()*logNF ;
    } else {
      _offArray[i] = 0 ;
    }
  }

  delete catIter ;

}


RooSimFitContext::~RooSimFitContext() 
{
  Int_t i ;
  for (i=0 ; i<_nCtx ; i++) {
    if (_ctxArray[i]) delete _ctxArray[i] ;
    if (_dsetArray[i]) delete _dsetArray[i] ;
  }

  delete[] _dsetArray ;
  delete[] _ctxArray ;
  delete[] _nllArray ;
  delete[] _offArray ;
  delete[] _dirtyArray ;
}


RooFitResult* RooSimFitContext::fit(Option_t *fitOptions, Option_t* optOptions) 
{
  // Setup and perform MINUIT fit of PDF to dataset
  if (_zombie) return 0 ;

  // Parse our fit options string
  TString fitOpts= fitOptions;
  fitOpts.ToLower();
  
  Bool_t extendedMode = fitOpts.Contains("e") ;
  Bool_t globExtMode  = fitOpts.Contains("g") ;

  if (globExtMode) {
    if (extendedMode) {
      _nGlobEvents = _dataClone->numEntries() ;
      cout << "nGlobEvents = " << _dataClone->numEntries() << endl ;
    } else {
      cout << "RooSimFitContext::fit: WARNING option 'g' meaningless without option 'e'" << endl ;
    }    
  }

  return RooFitContext::fit(fitOptions,optOptions) ;
}

Double_t RooSimFitContext::nLogLikelihood(Bool_t dummy, Int_t nObserved) const 
{
  Double_t nllSum(0) ;
  // Update likelihood from subcontexts that changed
  Double_t offSet(log(_nCtxFilled)) ;
  Int_t i ;
  for (i=0 ; i<_nCtx ; i++) {
    if (_ctxArray[i]) {
      if (_dirtyArray[i]) {
	Bool_t extend = (_extendedMode && _ctxArray[i]->_pdfClone->canBeExtended()) ;
	_nllArray[i] = _ctxArray[i]->nLogLikelihood(extend,_nGlobEvents) + _offArray[i] ;
	_dirtyArray[i] = kFALSE ;	

	// If any sub-context returns zero (i.e. events with zero or neg. prob) return zero
	if (_nllArray[i] == 0.) {
	  return 0 ;
	}

      }
      nllSum += _nllArray[i] ;
    }
  }
  
  // Return sum of NLL from subcontexts
  return nllSum ;
}


Bool_t RooSimFitContext::optimize(Bool_t doPdf,Bool_t doData, Bool_t doCache) 
{
  Bool_t ret(kFALSE) ;
  Int_t i ;
  for (i=0 ; i<_nCtx ; i++) {
    if (_ctxArray[i]) {
      cout << "RooSimFitContext::optimize: forwarding call to subContext " << i << endl ;
      if (_ctxArray[i]->optimize(doPdf,doData,doCache)) ret=kTRUE ;
    }
  }
  return ret ;
}


Bool_t RooSimFitContext::setPdfParamVal(Int_t index, Double_t value, Bool_t verbose) 
{
  Int_t i ;
  // First check if variable actually changes 
  if (!RooFitContext::setPdfParamVal(index,value,verbose)) return kFALSE;

  // Forward parameter change to sub-contexts
  for (i=0 ; i<_nCtx ; i++) {
    if (_ctxArray[i]) {
      RooAbsArg* par = _floatParamList->at(index) ;
      if (!par) {
	cout << "RooSimFitContext::setPdfParamVal: cannot find parameter with index " << index << endl ;
	assert(0) ;
      }      
      Int_t subIdx = _ctxArray[i]->_floatParamList->index(par) ;

      // Mark subcontexts as dirty
      if (subIdx!=-1) {	  
	_ctxArray[i]->setPdfParamVal(subIdx,value,kFALSE) ;
	_dirtyArray[i] = kTRUE ;
      }
    }
  }

  return kTRUE ;
}

