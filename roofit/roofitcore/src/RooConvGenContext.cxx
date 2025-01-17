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
\file RooConvGenContext.cxx
\class RooConvGenContext
\ingroup Roofitcore

Efficient implementation of the generator context
specific for RooAbsAnaConvPdf objects. The physics model is generated
with a truth resolution model and the requested resolution model is generated
separately as a PDF. The convolution variable of the physics model is
subsequently explicitly smeared with the resolution model distribution.
**/

#include "RooMsgService.h"
#include "RooErrorHandler.h"
#include "RooConvGenContext.h"
#include "RooAbsAnaConvPdf.h"
#include "RooNumConvPdf.h"
#include "RooFFTConvPdf.h"
#include "RooProdPdf.h"
#include "RooDataSet.h"
#include "RooArgSet.h"
#include "RooTruthModel.h"
#include "Riostream.h"


using std::ostream;

ClassImp(RooConvGenContext);

////////////////////////////////////////////////////////////////////////////////
/// Constructor for specialized generator context for analytical convolutions.
///
/// Builds a generator for the physics PDF convoluted with the truth model
/// and a generator for the resolution model as PDF. Events are generated
/// by sampling events from the p.d.f and smearings from the resolution model
/// and adding these to obtain a distribution of events consistent with the
/// convolution of these two. The advantage of this procedure is so that
/// both p.d.f and resolution model can take advantage of any internal
/// generators that may be defined.

RooConvGenContext::RooConvGenContext(const RooAbsAnaConvPdf &model, const RooArgSet &vars,
                 const RooDataSet *prototype, const RooArgSet* auxProto, bool verbose) :
  RooAbsGenContext(model,vars,prototype,auxProto,verbose)
{
  cxcoutI(Generation) << "RooConvGenContext::ctor() setting up special generator context for analytical convolution p.d.f. " << model.GetName()
            << " for generation of observable(s) " << vars << std::endl ;

  // Clone PDF and change model to internal truth model
  _pdfCloneSet = std::make_unique<RooArgSet>();
  RooArgSet(model).snapshot(*_pdfCloneSet, true);
  if (!_pdfCloneSet) {
    coutE(Generation) << "RooConvGenContext::RooConvGenContext(" << GetName() << ") Couldn't deep-clone PDF, abort," << std::endl ;
    RooErrorHandler::softAbort() ;
  }

  RooAbsAnaConvPdf* pdfClone = static_cast<RooAbsAnaConvPdf*>(_pdfCloneSet->find(model.GetName())) ;
  RooTruthModel truthModel("truthModel","Truth resolution model",*pdfClone->convVar()) ;
  pdfClone->changeModel(truthModel) ;
  auto convV = dynamic_cast<RooRealVar*>(pdfClone->convVar());
  if (!convV) {
    throw std::runtime_error("RooConvGenContext only works with convolution variables of type RooRealVar.");
  }
  convV->removeRange();

  // Create generator for physics X truth model
  _pdfVars = std::unique_ptr<RooArgSet>{pdfClone->getObservables(&vars)};
  _pdfGen.reset(pdfClone->genContext(*_pdfVars,prototype,auxProto,verbose));

  // Clone resolution model and use as normal PDF
  _modelCloneSet = std::make_unique<RooArgSet>();
  RooArgSet(*model._convSet.at(0)).snapshot(*_modelCloneSet, true);
  if (!_modelCloneSet) {
    coutE(Generation) << "RooConvGenContext::RooConvGenContext(" << GetName() << ") Couldn't deep-clone resolution model, abort," << std::endl;
    RooErrorHandler::softAbort() ;
  }
  std::unique_ptr<RooResolutionModel> modelClone{static_cast<RooResolutionModel*>(_modelCloneSet->find(model._convSet.at(0)->GetName())->Clone("smearing"))};
  modelClone->changeBasis(nullptr) ;
  convV = dynamic_cast<RooRealVar*>(&modelClone->convVar());
  if (!convV) {
    throw std::runtime_error("RooConvGenContext only works with convolution variables of type RooRealVar.");
  }
  convV->removeRange();

  // Create generator for resolution model as PDF
  _modelVars = std::unique_ptr<RooArgSet>{modelClone->getObservables(&vars)};

  _modelVars->add(modelClone->convVar()) ;
  _convVarName = modelClone->convVar().GetName() ;
  _modelGen.reset(modelClone->genContext(*_modelVars,prototype,auxProto,verbose));

  _modelCloneSet->addOwned(std::move(modelClone));

  if (prototype) {
    _pdfVars->add(*prototype->get()) ;
    _modelVars->add(*prototype->get()) ;
  }

  // WVE ADD FOR DEBUGGING
  if (auxProto) {
    _pdfVars->add(*auxProto) ;
    _modelVars->add(*auxProto) ;
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Constructor for specialized generator context for numerical convolutions.
///
/// Builds a generator for the physics PDF convoluted with the truth model
/// and a generator for the resolution model as PDF. Events are generated
/// by sampling events from the p.d.f and smearings from the resolution model
/// and adding these to obtain a distribution of events consistent with the
/// convolution of these two. The advantage of this procedure is so that
/// both p.d.f and resolution model can take advantage of any internal
/// generators that may be defined.

RooConvGenContext::RooConvGenContext(const RooNumConvPdf &model, const RooArgSet &vars,
                 const RooDataSet *prototype, const RooArgSet* auxProto, bool verbose) :
  RooAbsGenContext(model,vars,prototype,auxProto,verbose)
{
  cxcoutI(Generation) << "RooConvGenContext::ctor() setting up special generator context for numeric convolution p.d.f. " << model.GetName()
         << " for generation of observable(s) " << vars << std::endl ;

  // Create generator for physics X truth model
  {
    RooArgSet clonedPdfObservables;
    model.conv().clonePdf().getObservables(&vars, clonedPdfObservables);
    _pdfVarsOwned = std::make_unique<RooArgSet>();
    clonedPdfObservables.snapshot(*_pdfVarsOwned, true);
  }
  _pdfVars = std::make_unique<RooArgSet>(*_pdfVarsOwned) ;
  _pdfGen.reset(static_cast<RooAbsPdf&>(model.conv().clonePdf()).genContext(*_pdfVars,prototype,auxProto,verbose));

  // Create generator for resolution model as PDF
  {
    RooArgSet clonedModelObservables;
    model.conv().cloneModel().getObservables(&vars, clonedModelObservables);
    _modelVarsOwned = std::make_unique<RooArgSet>();
    clonedModelObservables.snapshot(*_modelVarsOwned, true);
  }
  _modelVars = std::make_unique<RooArgSet>(*_modelVarsOwned) ;
  _convVarName = model.conv().cloneVar().GetName() ;
  _modelGen.reset(static_cast<RooAbsPdf&>(model.conv().cloneModel()).genContext(*_modelVars,prototype,auxProto,verbose));
  _modelCloneSet = std::make_unique<RooArgSet>();
  _modelCloneSet->add(model.conv().cloneModel()) ;

  if (prototype) {
    _pdfVars->add(*prototype->get()) ;
    _modelVars->add(*prototype->get()) ;
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Constructor for specialized generator context for FFT numerical convolutions.
///
/// Builds a generator for the physics PDF convoluted with the truth model
/// and a generator for the resolution model as PDF. Events are generated
/// by sampling events from the p.d.f and smearings from the resolution model
/// and adding these to obtain a distribution of events consistent with the
/// convolution of these two. The advantage of this procedure is so that
/// both p.d.f and resolution model can take advantage of any internal
/// generators that may be defined.

RooConvGenContext::RooConvGenContext(const RooFFTConvPdf &model, const RooArgSet &vars,
                 const RooDataSet *prototype, const RooArgSet* auxProto, bool verbose) :
  RooAbsGenContext(model,vars,prototype,auxProto,verbose)
{
  cxcoutI(Generation) << "RooConvGenContext::ctor() setting up special generator context for fft convolution p.d.f. " << model.GetName()
         << " for generation of observable(s) " << vars << std::endl ;

  _convVarName = model._x.arg().GetName() ;

  // Create generator for physics model
  _pdfCloneSet = std::make_unique<RooArgSet>();
  RooArgSet(model._pdf1.arg()).snapshot(*_pdfCloneSet, true);
  RooAbsPdf* pdfClone = static_cast<RooAbsPdf*>(_pdfCloneSet->find(model._pdf1.arg().GetName())) ;
  RooRealVar* cvPdf = static_cast<RooRealVar*>(_pdfCloneSet->find(model._x.arg().GetName())) ;
  cvPdf->removeRange() ;
  RooArgSet tmp1;
  pdfClone->getObservables(&vars, tmp1) ;
  _pdfVarsOwned = std::make_unique<RooArgSet>();
  tmp1.snapshot(*_pdfVarsOwned, true);
  _pdfVars = std::make_unique<RooArgSet>(*_pdfVarsOwned) ;
  _pdfGen.reset(pdfClone->genContext(*_pdfVars,prototype,auxProto,verbose));

  // Create generator for resolution model
  _modelCloneSet = std::make_unique<RooArgSet>();
  RooArgSet(model._pdf2.arg()).snapshot(*_modelCloneSet, true);
  RooAbsPdf* modelClone = static_cast<RooAbsPdf*>(_modelCloneSet->find(model._pdf2.arg().GetName())) ;
  RooRealVar* cvModel = static_cast<RooRealVar*>(_modelCloneSet->find(model._x.arg().GetName())) ;
  cvModel->removeRange() ;
  RooArgSet tmp2;
  modelClone->getObservables(&vars, tmp2) ;
  _modelVarsOwned = std::make_unique<RooArgSet>();
  tmp2.snapshot(*_modelVarsOwned, true);
  _modelVars = std::make_unique<RooArgSet>(*_modelVarsOwned) ;
  _modelGen.reset(modelClone->genContext(*_pdfVars,prototype,auxProto,verbose));

  if (prototype) {
    _pdfVars->add(*prototype->get()) ;
    _modelVars->add(*prototype->get()) ;
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Attach given set of arguments to internal clones of
/// pdf and resolution model

void RooConvGenContext::attach(const RooArgSet& args)
{
  // Find convolution variable in input and output sets
  auto* cvModel = static_cast<RooRealVar*>(_modelVars->find(_convVarName));
  auto* cvPdf   = static_cast<RooRealVar*>(_pdfVars->find(_convVarName));

  // Replace all servers in _pdfVars and _modelVars with those in theEvent, except for the convolution variable
  std::unique_ptr<RooArgSet> pdfCommon{args.selectCommon(*_pdfVars)};
  pdfCommon->remove(*cvPdf,true,true) ;

  std::unique_ptr<RooArgSet> modelCommon{args.selectCommon(*_modelVars)};
  modelCommon->remove(*cvModel,true,true) ;

  _pdfGen->attach(*pdfCommon) ;
  _modelGen->attach(*modelCommon) ;
}


////////////////////////////////////////////////////////////////////////////////
/// One-time initialization of generator context, attaches
/// the context to the supplied event container

void RooConvGenContext::initGenerator(const RooArgSet &theEvent)
{
  // Find convolution variable in input and output sets
  _cvModel = static_cast<RooRealVar*>(_modelVars->find(_convVarName)) ;
  _cvPdf   = static_cast<RooRealVar*>(_pdfVars->find(_convVarName)) ;
  _cvOut   = static_cast<RooRealVar*>(theEvent.find(_convVarName)) ;

  // Replace all servers in _pdfVars and _modelVars with those in theEvent, except for the convolution variable
  std::unique_ptr<RooArgSet> pdfCommon{theEvent.selectCommon(*_pdfVars)};
  pdfCommon->remove(*_cvPdf,true,true) ;
  _pdfVars->replace(*pdfCommon) ;

  std::unique_ptr<RooArgSet> modelCommon{theEvent.selectCommon(*_modelVars)};
  modelCommon->remove(*_cvModel,true,true) ;
  _modelVars->replace(*modelCommon) ;

  // Initialize component generators
  _pdfGen->initGenerator(*_pdfVars) ;
  _modelGen->initGenerator(*_modelVars) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Generate a single event

void RooConvGenContext::generateEvent(RooArgSet &theEvent, Int_t remaining)
{
  while(true) {

    // Generate pdf and model data
    _modelGen->generateEvent(*_modelVars,remaining) ;
    _pdfGen->generateEvent(*_pdfVars,remaining) ;

    // Construct smeared convolution variable
    double convValSmeared = _cvPdf->getVal() + _cvModel->getVal() ;
    if (_cvOut->isValidReal(convValSmeared)) {
      // Smeared value in acceptance range, transfer values to output set
      theEvent.assign(*_modelVars) ;
      theEvent.assign(*_pdfVars) ;
      _cvOut->setVal(convValSmeared) ;
      return ;
    }
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Set the traversal order for events in the prototype dataset
/// The argument is a array of integers with a size identical
/// to the number of events in the prototype dataset. Each element
/// should contain an integer in the range 1-N.

void RooConvGenContext::setProtoDataOrder(Int_t* lut)
{
  RooAbsGenContext::setProtoDataOrder(lut) ;
  _modelGen->setProtoDataOrder(lut) ;
  _pdfGen->setProtoDataOrder(lut) ;
}


////////////////////////////////////////////////////////////////////////////////
/// Print the details of this generator context

void RooConvGenContext::printMultiline(ostream &os, Int_t content, bool verbose, TString indent) const
{
  RooAbsGenContext::printMultiline(os,content,verbose,indent) ;
  os << indent << "--- RooConvGenContext ---" << std::endl ;
  os << indent << "List of component generators" << std::endl ;

  TString indent2(indent) ;
  indent2.Append("    ") ;

  _modelGen->printMultiline(os,content,verbose,indent2);
  _pdfGen->printMultiline(os,content,verbose,indent2);
}
