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

/**
\file RooSimGenContext.cxx
\class RooSimGenContext
\ingroup Roofitcore

Efficient implementation of the generator context
specific for RooSimultaneous PDFs when generating more than one of the
component pdfs.
It runs in two modes:
- Proto data with category entries are given: An event from the same category as
in the proto data is created.
- No proto data: A category is chosen randomly.
\note This requires that the PDFs are extended, to determine the relative probabilities
that an event originates from a certain category.
**/

#include "RooSimGenContext.h"
#include "RooSimultaneous.h"
#include "RooRealProxy.h"
#include "RooDataSet.h"
#include "Roo1DTable.h"
#include "RooCategory.h"
#include "RooMsgService.h"
#include "RooRandom.h"
#include "RooGlobalFunc.h"

#include <iostream>
#include <string>


////////////////////////////////////////////////////////////////////////////////
/// Constructor of specialized generator context for RooSimultaneous p.d.f.s. This
/// context creates a dedicated context for each component p.d.f.s and delegates
/// generation of events to the appropriate component generator context

RooSimGenContext::RooSimGenContext(const RooSimultaneous &model, const RooArgSet &vars,
               const RooDataSet *prototype, const RooArgSet* auxProto, bool verbose) :
  RooAbsGenContext(model,vars,prototype,auxProto,verbose), _pdf(&model)
{
  // Determine if we are requested to generate the index category
  RooAbsCategoryLValue const& idxCat = model.indexCat();
  RooArgSet pdfVars(vars) ;

  RooArgSet allPdfVars(pdfVars) ;
  if (prototype) allPdfVars.add(*prototype->get(),true) ;

  RooArgSet catsAmongAllVars;
  allPdfVars.selectCommon(model.flattenedCatList(), catsAmongAllVars);

  if(catsAmongAllVars.size() != model.flattenedCatList().size()) {
      oocoutE(_pdf,Generation) << "RooSimGenContext::ctor(" << GetName() << ") ERROR: This context must"
                << " generate all components of the index category" << std::endl ;
      _isValid = false ;
      _numPdf = 0 ;
      _haveIdxProto = false ;
      return ;
  }

  // We must either have the prototype or extended likelihood to determined
  // the relative fractions of the components
  _haveIdxProto = prototype ? true : false ;
  _idxCatName = idxCat.GetName() ;
  if (!_haveIdxProto && !model.canBeExtended()) {
    oocoutE(_pdf,Generation) << "RooSimGenContext::ctor(" << GetName() << ") ERROR: Need either extended mode"
              << " or prototype data to calculate number of events per category" << std::endl ;
    _isValid = false ;
    _numPdf = 0 ;
    return ;
  }

  // Initialize fraction threshold array (used only in extended mode)
  _numPdf = model._pdfProxyList.GetSize() ;
  _fracThresh.resize(_numPdf+1);
  _fracThresh[0] = 0 ;

  // Generate index category and all registered PDFS
  _allVarsPdf.add(allPdfVars) ;
  Int_t i(1) ;
  for(auto * proxy : static_range_cast<RooRealProxy*>(model._pdfProxyList)) {
    auto* pdf = static_cast<RooAbsPdf*>(proxy->absArg());

    // Name the context after the associated state and add to list
    _gcList.emplace_back(pdf->genContext(pdfVars,prototype,auxProto,verbose)) ;

    // Create generator context for this PDF
    _gcList.back()->SetName(proxy->name()) ;

    _gcIndex.push_back(idxCat.lookupIndex(proxy->name()));

    // Fill fraction threshold array
    _fracThresh[i] = _fracThresh[i-1] + (_haveIdxProto?0:pdf->expectedEvents(&allPdfVars)) ;
    i++ ;
  }

  // Normalize fraction threshold array
  if (!_haveIdxProto) {
    for(i=0 ; i<_numPdf ; i++)
      _fracThresh[i] /= _fracThresh[_numPdf] ;
  }


  // Clone the index category
  _idxCatSet = std::make_unique<RooArgSet>();
  RooArgSet(model.indexCat()).snapshot(*_idxCatSet, true);
  if (!_idxCatSet) {
    oocoutE(_pdf,Generation) << "RooSimGenContext::RooSimGenContext(" << GetName() << ") Couldn't deep-clone index category, abort," << std::endl ;
    throw std::string("RooSimGenContext::RooSimGenContext() Couldn't deep-clone index category, abort") ;
  }

  _idxCat = static_cast<RooAbsCategoryLValue*>(_idxCatSet->find(model.indexCat().GetName()));
}

RooSimGenContext::~RooSimGenContext() = default;

////////////////////////////////////////////////////////////////////////////////
/// Attach the index category clone to the given event buffer

void RooSimGenContext::attach(const RooArgSet& args)
{
  if (_idxCat->isDerived()) {
    _idxCat->recursiveRedirectServers(args) ;
  }

  // Forward initGenerator call to all components
  for(auto& item : _gcList) {
    item->attach(args) ;
  }

}


////////////////////////////////////////////////////////////////////////////////
/// Perform one-time initialization of generator context

void RooSimGenContext::initGenerator(const RooArgSet &theEvent)
{
  // Attach the index category clone to the event
  if (_idxCat->isDerived()) {
    _idxCat->recursiveRedirectServers(theEvent) ;
  } else {
    _idxCat = static_cast<RooAbsCategoryLValue*>(theEvent.find(_idxCat->GetName())) ;
  }

  // Update fractions reflecting possible new parameter values
  updateFractions() ;

  // Forward initGenerator call to all components
  for(auto& item : _gcList) {
    item->initGenerator(theEvent) ;
  }

}


////////////////////////////////////////////////////////////////////////////////
/// Create an empty dataset to hold the events that will be generated

RooDataSet* RooSimGenContext::createDataSet(const char* name, const char* title, const RooArgSet& obs)
{

  // If the observables do not contain the index, make a plain dataset
  if (!obs.contains(*_idxCat)) {
    return new RooDataSet(name,title,obs) ;
  }

  if (!_protoData) {
    std::map<std::string,RooAbsData*> dmap ;
    for (const auto& nameIdx : *_idxCat) {
      RooAbsPdf* slicePdf = _pdf->getPdf(nameIdx.first.c_str());
      std::unique_ptr<RooArgSet> sliceObs{slicePdf->getObservables(obs)};
      std::string sliceName = name + ("_slice_" + nameIdx.first);
      std::string sliceTitle = title + (" (index slice " + nameIdx.first + ")");
      dmap[nameIdx.first] = new RooDataSet(sliceName,sliceTitle,*sliceObs);
    }
    using namespace RooFit;
    _protoData = std::make_unique<RooDataSet>(name, title, obs, Index(static_cast<RooCategory&>(*_idxCat)), Link(dmap), OwnLinked()) ;
  }

  return new RooDataSet(*_protoData,name) ;
}





////////////////////////////////////////////////////////////////////////////////
/// Generate event appropriate for current index state.
/// The index state is taken either from the prototype
/// or is generated from the fraction threshold table.

void RooSimGenContext::generateEvent(RooArgSet &theEvent, Int_t remaining)
{
  if (_haveIdxProto) {

    // Lookup pdf from selected prototype index state
    Int_t gidx(0);
    Int_t cidx = _idxCat->getCurrentIndex();
    for (Int_t i=0 ; i<(Int_t)_gcIndex.size() ; i++) {
      if (_gcIndex[i]==cidx) { gidx = i ; break ; }
    }
    RooAbsGenContext* cx = _gcList[gidx].get();
    if (cx) {
      cx->generateEvent(theEvent,remaining) ;
    } else {
      oocoutW(_pdf,Generation) << "RooSimGenContext::generateEvent: WARNING, no PDF to generate event of type " << cidx << std::endl ;
    }


  } else {

    // Throw a random number and select PDF from fraction threshold table
    double rand = RooRandom::uniform() ;
    Int_t i=0 ;
    for (i=0 ; i<_numPdf ; i++) {
      if (rand>_fracThresh[i] && rand<_fracThresh[i+1]) {
        RooAbsGenContext* gen=_gcList[i].get();
        gen->generateEvent(theEvent,remaining) ;
        //Write through to sub-categories because they might be written to dataset:
        _idxCat->setIndex(_gcIndex[i]);
        return ;
      }
    }

  }
}



////////////////////////////////////////////////////////////////////////////////
/// No action needed if we have a proto index

void RooSimGenContext::updateFractions()
{
  if (_haveIdxProto) return ;

  // Generate index category and all registered PDFS
  Int_t i(1) ;
  for(auto * proxy : static_range_cast<RooRealProxy*>(_pdf->_pdfProxyList)) {
    auto* pdf = static_cast<RooAbsPdf*>(proxy->absArg());

    // Fill fraction threshold array
    _fracThresh[i] = _fracThresh[i-1] + (_haveIdxProto?0:pdf->expectedEvents(&_allVarsPdf)) ;
    i++ ;
  }

  // Normalize fraction threshold array
  if (!_haveIdxProto) {
    for(i=0 ; i<_numPdf ; i++)
      _fracThresh[i] /= _fracThresh[_numPdf] ;
  }

}



////////////////////////////////////////////////////////////////////////////////
/// Set the traversal order of the prototype data to that in the
/// given lookup table. This information is passed to all
/// component generator contexts

void RooSimGenContext::setProtoDataOrder(Int_t* lut)
{
  RooAbsGenContext::setProtoDataOrder(lut) ;

  for (auto& item : _gcList) {
    item->setProtoDataOrder(lut) ;
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Detailed printing interface

void RooSimGenContext::printMultiline(std::ostream &os, Int_t content, bool verbose, TString indent) const
{
  RooAbsGenContext::printMultiline(os,content,verbose,indent) ;
  os << indent << "--- RooSimGenContext ---" << std::endl ;
  os << indent << "Using PDF ";
  _pdf->printStream(os,kName|kArgs|kClassName,kSingleLine,indent);
  os << indent << "List of component generators" << std::endl ;

  TString indent2(indent) ;
  indent2.Append("    ") ;

  for (auto& item : _gcList) {
    item->printMultiline(os,content,verbose,indent2);
  }
}
