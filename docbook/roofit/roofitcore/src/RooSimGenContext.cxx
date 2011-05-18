/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
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

//////////////////////////////////////////////////////////////////////////////
//
// BEGIN_HTML
// RooSimGenContext is an efficient implementation of the generator context
// specific for RooSimultaneous PDFs when generating more than one of the
// component pdfs.
// END_HTML
//

#include "RooFit.h"
#include "Riostream.h"

#include "RooSimGenContext.h"
#include "RooSimultaneous.h"
#include "RooRealProxy.h"
#include "RooDataSet.h"
#include "Roo1DTable.h"
#include "RooCategory.h"
#include "RooMsgService.h"
#include "RooRandom.h"

#include <string>

ClassImp(RooSimGenContext)
;
  

//_____________________________________________________________________________
RooSimGenContext::RooSimGenContext(const RooSimultaneous &model, const RooArgSet &vars, 
				   const RooDataSet *prototype, const RooArgSet* auxProto, Bool_t verbose) :
  RooAbsGenContext(model,vars,prototype,auxProto,verbose), _pdf(&model)
{
  // Constructor of specialized generator context for RooSimultaneous p.d.f.s. This
  // context creates a dedicated context for each component p.d.f.s and delegates
  // generation of events to the appropriate component generator context

  // Determine if we are requested to generate the index category
  RooAbsCategory *idxCat = (RooAbsCategory*) model._indexCat.absArg() ;
  RooArgSet pdfVars(vars) ;

  RooArgSet allPdfVars(pdfVars) ;
  if (prototype) allPdfVars.add(*prototype->get(),kTRUE) ;

  if (!idxCat->isDerived()) {
    pdfVars.remove(*idxCat,kTRUE,kTRUE) ;
    Bool_t doGenIdx = allPdfVars.find(idxCat->GetName())?kTRUE:kFALSE ;

    if (!doGenIdx) {
      oocoutE(_pdf,Generation) << "RooSimGenContext::ctor(" << GetName() << ") ERROR: This context must"
			       << " generate the index category" << endl ;
      _isValid = kFALSE ;
      _numPdf = 0 ;
      _haveIdxProto = kFALSE ;
      return ;
    }
  } else {
    TIterator* sIter = idxCat->serverIterator() ;
    RooAbsArg* server ;
    Bool_t anyServer(kFALSE), allServers(kTRUE) ;
    while((server=(RooAbsArg*)sIter->Next())) {
      if (vars.find(server->GetName())) {
	anyServer=kTRUE ;
	pdfVars.remove(*server,kTRUE,kTRUE) ;
      } else {
	allServers=kFALSE ;
      }
    }
    delete sIter ;    

    if (anyServer && !allServers) {
      oocoutE(_pdf,Generation) << "RooSimGenContext::ctor(" << GetName() << ") ERROR: This context must"
			       << " generate all components of a derived index category" << endl ;
      _isValid = kFALSE ;
      _numPdf = 0 ;
      _haveIdxProto = kFALSE ;
      return ;
    }
  }

  // We must either have the prototype or extended likelihood to determined
  // the relative fractions of the components
  _haveIdxProto = prototype ? kTRUE : kFALSE ;
  _idxCatName = idxCat->GetName() ;
  if (!_haveIdxProto && !model.canBeExtended()) {
    oocoutE(_pdf,Generation) << "RooSimGenContext::ctor(" << GetName() << ") ERROR: Need either extended mode"
			     << " or prototype data to calculate number of events per category" << endl ;
    _isValid = kFALSE ;
    _numPdf = 0 ;
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
  while((proxy=(RooRealProxy*)iter->Next())) {
    pdf=(RooAbsPdf*)proxy->absArg() ;

    // Create generator context for this PDF
    RooAbsGenContext* cx = pdf->genContext(pdfVars,prototype,auxProto,verbose) ;

    // Name the context after the associated state and add to list
    cx->SetName(proxy->name()) ;
    _gcList.Add(cx) ;

    // Fill fraction threshold array
    _fracThresh[i] = _fracThresh[i-1] + (_haveIdxProto?0:pdf->expectedEvents(&allPdfVars)) ;
    i++ ;
  }   
  delete iter ;
    
  // Normalize fraction threshold array
  if (!_haveIdxProto) {
    for(i=0 ; i<_numPdf ; i++) 
      _fracThresh[i] /= _fracThresh[_numPdf] ;
  }
  

  // Clone the index category
  _idxCatSet = (RooArgSet*) RooArgSet(model._indexCat.arg()).snapshot(kTRUE) ;
  if (!_idxCatSet) {
    oocoutE(_pdf,Generation) << "RooSimGenContext::RooSimGenContext(" << GetName() << ") Couldn't deep-clone index category, abort," << endl ;
    throw std::string("RooSimGenContext::RooSimGenContext() Couldn't deep-clone index category, abort") ;
  }
  
  _idxCat = (RooAbsCategoryLValue*) _idxCatSet->find(model._indexCat.arg().GetName()) ;
}



//_____________________________________________________________________________
RooSimGenContext::~RooSimGenContext()
{
  // Destructor. Delete all owned subgenerator contexts

  delete[] _fracThresh ;
  delete _idxCatSet ;
  _gcList.Delete() ;
}



//_____________________________________________________________________________
void RooSimGenContext::attach(const RooArgSet& args) 
{
  // Attach the index category clone to the given event buffer

  if (_idxCat->isDerived()) {
    _idxCat->recursiveRedirectServers(args,kTRUE) ;
  }

  // Forward initGenerator call to all components
  RooAbsGenContext* gc ;
  TIterator* iter = _gcList.MakeIterator() ;
  while((gc=(RooAbsGenContext*)iter->Next())){
    gc->attach(args) ;
  }
  delete iter;
  
}


//_____________________________________________________________________________
void RooSimGenContext::initGenerator(const RooArgSet &theEvent)
{
  // Perform one-time initialization of generator context

  // Attach the index category clone to the event
  if (_idxCat->isDerived()) {
    _idxCat->recursiveRedirectServers(theEvent,kTRUE) ;
  } else {
    _idxCat = (RooAbsCategoryLValue*) theEvent.find(_idxCat->GetName()) ;
  }

  // Forward initGenerator call to all components
  RooAbsGenContext* gc ;
  TIterator* iter = _gcList.MakeIterator() ;
  while((gc=(RooAbsGenContext*)iter->Next())){
    gc->initGenerator(theEvent) ;
  }
  delete iter;

}



//_____________________________________________________________________________
void RooSimGenContext::generateEvent(RooArgSet &theEvent, Int_t remaining)
{
  // Generate event appropriate for current index state. 
  // The index state is taken either from the prototype
  // or is generated from the fraction threshold table.

  if (_haveIdxProto) {

    // Lookup pdf from selected prototype index state
    const char* label = _idxCat->getLabel() ;
    RooAbsGenContext* cx = (RooAbsGenContext*)_gcList.FindObject(label) ;
    if (cx) {      
      cx->generateEvent(theEvent,remaining) ;
    } else {
      oocoutW(_pdf,Generation) << "RooSimGenContext::generateEvent: WARNING, no PDF to generate event of type " << label << endl ;
    }    

  
  } else {

    // Throw a random number and select PDF from fraction threshold table
    Double_t rand = RooRandom::uniform() ;
    Int_t i=0 ;
    for (i=0 ; i<_numPdf ; i++) {
      if (rand>_fracThresh[i] && rand<_fracThresh[i+1]) {
	RooAbsGenContext* gen= ((RooAbsGenContext*)_gcList.At(i)) ;
	gen->generateEvent(theEvent,remaining) ;
	_idxCat->setLabel(gen->GetName()) ;
	return ;
      }
    }

  }
}


//_____________________________________________________________________________
void RooSimGenContext::setProtoDataOrder(Int_t* lut)
{
  // Set the traversal order of the prototype data to that in the
  // given lookup table. This information is passed to all
  // component generator contexts

  RooAbsGenContext::setProtoDataOrder(lut) ;

  TIterator* iter = _gcList.MakeIterator() ;
  RooAbsGenContext* gc ;
  while((gc=(RooAbsGenContext*)iter->Next())) {
    gc->setProtoDataOrder(lut) ;
  }
  delete iter ;
}


//_____________________________________________________________________________
void RooSimGenContext::printMultiline(ostream &os, Int_t content, Bool_t verbose, TString indent) const 
{
  // Detailed printing interface

  RooAbsGenContext::printMultiline(os,content,verbose,indent) ;
  os << indent << "--- RooSimGenContext ---" << endl ;
  os << indent << "Using PDF ";
  _pdf->printStream(os,kName|kArgs|kClassName,kSingleLine,indent);
  os << indent << "List of component generators" << endl ;

  TString indent2(indent) ;
  indent2.Append("    ") ;

  TIterator* iter = _gcList.MakeIterator() ;
  RooAbsGenContext* gc ;
  while((gc=(RooAbsGenContext*)iter->Next())) {
    gc->printMultiline(os,content,verbose,indent2);
  }
  delete iter ;
}
