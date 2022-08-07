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
\file RooAddGenContext.cxx
\class RooAddGenContext
\ingroup Roofitcore

RooAddGenContext is an efficient implementation of the
generator context specific for RooAddPdf PDFs. The strategy
of RooAddGenContext is to defer generation of each component
to a dedicated generator context for that component and to
randomly choose one of those context to generate an event,
with a probability proportional to its associated coefficient.
**/

#include "RooAddGenContext.h"

#include "Riostream.h"
#include "TClass.h"

#include "RooDataSet.h"
#include "RooRandom.h"

#include <sstream>


ClassImp(RooAddGenContext);


////////////////////////////////////////////////////////////////////////////////
/// Constructor

RooAddGenContext::RooAddGenContext(const RooAddPdf &model, const RooArgSet &vars,
               const RooDataSet *prototype, const RooArgSet* auxProto,
               bool verbose) :
  RooAbsGenContext(model,vars,prototype,auxProto,verbose), _isModel(false)
{
  cxcoutI(Generation) << "RooAddGenContext::ctor() setting up event special generator context for sum p.d.f. " << model.GetName()
         << " for generation of observable(s) " << vars ;
  if (prototype) ccxcoutI(Generation) << " with prototype data for " << *prototype->get() ;
  if (auxProto && !auxProto->empty())  ccxcoutI(Generation) << " with auxiliary prototypes " << *auxProto ;
  ccxcoutI(Generation) << std::endl;

  // Constructor. Build an array of generator contexts for each product component PDF
  _pdfSet.reset(static_cast<RooArgSet*>(RooArgSet(model).snapshot(true)));
  _pdf = (RooAddPdf*) _pdfSet->find(model.GetName()) ;
  _pdf->setOperMode(RooAbsArg::ADirty,true) ;

  // Fix normalization set of this RooAddPdf
  if (prototype)
    {
      RooArgSet coefNSet(vars) ;
      coefNSet.add(*prototype->get()) ;
      _pdf->fixAddCoefNormalization(coefNSet,false) ;
    }

  _nComp = model._pdfList.getSize() ;
  _coefThresh.resize(_nComp+1);
  _vars.reset(static_cast<RooArgSet*>(vars.snapshot(false)));

  for (const auto arg : model._pdfList) {
    auto pdf = dynamic_cast<const RooAbsPdf *>(arg);
    if (!pdf) {
      coutF(Generation) << "Cannot generate events from an object that is not a PDF.\n\t"
          << "The offending object is a " << arg->ClassName() << " named '" << arg->GetName() << "'." << std::endl;
      throw std::invalid_argument("Trying to generate events from on object that is not a PDF.");
    }

    _gcList.emplace_back(pdf->genContext(vars,prototype,auxProto,verbose));
  }

  ((RooAddPdf*)_pdf)->getProjCache(_vars.get()) ;
  _pdf->recursiveRedirectServers(_theEvent) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Constructor

RooAddGenContext::RooAddGenContext(const RooAddModel &model, const RooArgSet &vars,
               const RooDataSet *prototype, const RooArgSet* auxProto,
               bool verbose) :
  RooAbsGenContext(model,vars,prototype,auxProto,verbose), _isModel(true)
{
  cxcoutI(Generation) << "RooAddGenContext::ctor() setting up event special generator context for sum resolution model " << model.GetName()
         << " for generation of observable(s) " << vars ;
  if (prototype) ccxcoutI(Generation) << " with prototype data for " << *prototype->get() ;
  if (auxProto && !auxProto->empty())  ccxcoutI(Generation) << " with auxiliary prototypes " << *auxProto ;
  ccxcoutI(Generation) << std::endl;

  // Constructor. Build an array of generator contexts for each product component PDF
  _pdfSet.reset(static_cast<RooArgSet*>(RooArgSet(model).snapshot(true)));
  _pdf = (RooAbsPdf*) _pdfSet->find(model.GetName()) ;

  _nComp = model._pdfList.getSize() ;
  _coefThresh.resize(_nComp+1);
  _vars.reset(static_cast<RooArgSet*>(vars.snapshot(false)));

  for (const auto obj : model._pdfList) {
    auto pdf = static_cast<RooAbsPdf*>(obj);
    _gcList.emplace_back(pdf->genContext(vars,prototype,auxProto,verbose));
  }

  ((RooAddModel*)_pdf)->getProjCache(_vars.get()) ;
  _pdf->recursiveRedirectServers(_theEvent) ;
}


////////////////////////////////////////////////////////////////////////////////
/// Attach given set of variables to internal p.d.f. clone

void RooAddGenContext::attach(const RooArgSet& args)
{
  _pdf->recursiveRedirectServers(args) ;

  // Forward initGenerator call to all components
  for(auto& gc : _gcList) {
    gc->attach(args) ;
  }
}



////////////////////////////////////////////////////////////////////////////////
/// One-time initialization of generator contex. Attach theEvent
/// to internal p.d.f clone and forward initialization call to
/// the component generators

void RooAddGenContext::initGenerator(const RooArgSet &theEvent)
{
  _pdf->recursiveRedirectServers(theEvent) ;

  if (_isModel) {
    RooAddModel* amod = (RooAddModel*) _pdf ;
    _pcache = amod->getProjCache(_vars.get()) ;
  } else {
    RooAddPdf* apdf = (RooAddPdf*) _pdf ;
    _pcache = apdf->getProjCache(_vars.get(),nullptr,"FULL_RANGE_ADDGENCONTEXT") ;
  }

  // Forward initGenerator call to all components
  for(auto& gc : _gcList) {
    gc->initGenerator(theEvent) ;
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Randomly choose one of the component contexts to generate this event,
/// with a probability proportional to its coefficient

void RooAddGenContext::generateEvent(RooArgSet &theEvent, Int_t remaining)
{
  // Throw a random number to determin which component to generate
  updateThresholds() ;
  double rand = RooRandom::uniform() ;
  for (Int_t i=0 ; i<_nComp ; i++) {
    if (rand>_coefThresh[i] && rand<_coefThresh[i+1]) {
      _gcList[i]->generateEvent(theEvent,remaining) ;
      return ;
    }
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Update the cumulative threshold table from the current coefficient
/// values

void RooAddGenContext::updateThresholds()
{
  // Templated lambda to support RooAddModel and RooAddPdf
  auto updateThresholdsImpl = [&](auto* pdf, auto * cache) {
    pdf->updateCoefficients(*cache,_vars.get()) ;

    _coefThresh[0] = 0. ;
    for (Int_t i=0 ; i<_nComp ; i++) {
      double coef = pdf->_coefCache[i];
      if(coef < 0.0) {
        std::stringstream errMsgStream;
        errMsgStream << "RooAddGenContext::updateThresholds(): coefficient number " << i << " of the "
                     << pdf->ClassName() << " \"" << pdf->GetName() <<  "\"" << " is negative!"
                     << " The current RooAddGenConext doesn't support negative coefficients."
                     << " Please recreate a new generator context with " << pdf->ClassName() << "::genContext()";
        auto const errMsg = errMsgStream.str();
        cxcoutE(Generation) << errMsg << std::endl;
        throw std::runtime_error(errMsg);
      }
      _coefThresh[i+1] = coef + _coefThresh[i];
    }
  };

  _isModel ? updateThresholdsImpl(static_cast<RooAddModel*>(_pdf), _pcache)
           : updateThresholdsImpl(static_cast<RooAddPdf*>(_pdf), _pcache);
}


////////////////////////////////////////////////////////////////////////////////
/// Forward the setProtoDataOrder call to the component generator contexts

void RooAddGenContext::setProtoDataOrder(Int_t* lut)
{
  RooAbsGenContext::setProtoDataOrder(lut) ;
  for(auto& gc : _gcList) {
    gc->setProtoDataOrder(lut) ;
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Print the details of the context

void RooAddGenContext::printMultiline(std::ostream &os, Int_t content, bool verbose, TString indent) const
{
  RooAbsGenContext::printMultiline(os,content,verbose,indent) ;
  os << indent << "--- RooAddGenContext ---" << std::endl;
  os << indent << "Using PDF ";
  _pdf->printStream(os,kName|kArgs|kClassName,kSingleLine,indent);

  os << indent << "List of component generators" << std::endl;
  TString indent2(indent) ;
  indent2.Append("    ") ;
  for(auto& gc : _gcList) {
    gc->printMultiline(os,content,verbose,indent2) ;
  }
}
