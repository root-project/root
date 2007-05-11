/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooConvGenContext.cc,v 1.21 2005/12/05 14:58:16 wverkerke Exp $
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

// -- CLASS DESCRIPTION [AUX} --
// RooConvGenContext is an efficient implementation of the generator context
// specific for RooAbsAnaConvPdf objects. The physics model is generated
// with a truth resolution model and the requested resolution model is generated
// separately as a PDF. The convolution variable of the physics model is 
// subsequently explicitly smeared with the resolution model distribution.

#include "RooFit.h"

#include "RooConvGenContext.h"
#include "RooConvGenContext.h"
#include "RooAbsAnaConvPdf.h"
#include "RooNumConvPdf.h"
#include "RooProdPdf.h"
#include "RooDataSet.h"
#include "RooArgSet.h"
#include "RooTruthModel.h"
#include "Riostream.h"


ClassImp(RooConvGenContext)
;
  
RooConvGenContext::RooConvGenContext(const RooAbsAnaConvPdf &model, const RooArgSet &vars, 
	 			     const RooDataSet *prototype, const RooArgSet* auxProto, Bool_t verbose) :
  RooAbsGenContext(model,vars,prototype,auxProto,verbose)
{
  // Constructor for analytical convolutions. 
  // 
  // Build a generator for the physics PDF convoluted with the truth model
  // and a generator for the resolution model as PDF.

  // Clone PDF and change model to internal truth model
  _pdfCloneSet = (RooArgSet*) RooArgSet(model).snapshot(kTRUE) ;
  if (!_pdfCloneSet) {
    cout << "RooConvGenContext::RooConvGenContext(" << GetName() << ") Couldn't deep-clone PDF, abort," << endl ;
    RooErrorHandler::softAbort() ;
  }

  RooAbsAnaConvPdf* pdfClone = (RooAbsAnaConvPdf*) _pdfCloneSet->find(model.GetName()) ;
  RooTruthModel truthModel("truthModel","Truth resolution model",(RooRealVar&)*pdfClone->convVar()) ;
  pdfClone->changeModel(truthModel) ;
  ((RooRealVar*)pdfClone->convVar())->removeRange() ;

  // Create generator for physics X truth model
  _pdfVars = (RooArgSet*) pdfClone->getObservables(&vars) ; ;
  _pdfGen = pdfClone->genContext(*_pdfVars,prototype,auxProto,verbose) ;  

  // Clone resolution model and use as normal PDF
  _modelCloneSet = (RooArgSet*) RooArgSet(*model._convSet.at(0)).snapshot(kTRUE) ;
  if (!_modelCloneSet) {
    cout << "RooConvGenContext::RooConvGenContext(" << GetName() << ") Couldn't deep-clone resolution model, abort," << endl ;
    RooErrorHandler::softAbort() ;
  }
  RooResolutionModel* modelClone = (RooResolutionModel*) 
    _modelCloneSet->find(model._convSet.at(0)->GetName())->Clone("smearing") ;
  _modelCloneSet->addOwned(*modelClone) ;
  modelClone->changeBasis(0) ;
  modelClone->convVar().removeRange() ;

  // Create generator for resolution model as PDF
  _modelVars = (RooArgSet*) modelClone->getObservables(&vars) ;

  _modelVars->add(modelClone->convVar()) ;
  _convVarName = modelClone->convVar().GetName() ;
  _modelGen = modelClone->genContext(*_modelVars,prototype,auxProto,verbose) ;

  if (prototype) {
    _pdfVars->add(*prototype->get()) ;
    _modelVars->add(*prototype->get()) ;  
  }

  // WVE ADD FOR DEBUGGING
  if (auxProto) {
    _pdfVars->add(*auxProto) ;
    _modelVars->add(*auxProto) ;
  }

//   cout << "RooConvGenContext::ctor(" << this << "," << GetName() << ") _pdfVars = " << _pdfVars << " "  ; _pdfVars->Print("1") ;
//   cout << "RooConvGenContext::ctor(" << this << "," << GetName() << ") _modelVars = " << _modelVars << " " ; _modelVars->Print("1") ;
}



RooConvGenContext::RooConvGenContext(const RooNumConvPdf &model, const RooArgSet &vars, 
	 			     const RooDataSet *prototype, const RooArgSet* auxProto, Bool_t verbose) :
  RooAbsGenContext(model,vars,prototype,auxProto,verbose)
{
  // Constructor for numeric convolutions.
  //
  // Build a generator for the physics PDF convoluted with the truth model
  // and a generator for the resolution model as PDF.

  // Create generator for physics X truth model
  _pdfVars = (RooArgSet*) model.conv().clonePdf().getObservables(&vars)->snapshot(kTRUE) ;
  _pdfGen = ((RooAbsPdf&)model.conv().clonePdf()).genContext(*_pdfVars,prototype,auxProto,verbose) ;  
  _pdfCloneSet = 0 ;

  // Create generator for resolution model as PDF
  _modelVars = (RooArgSet*) model.conv().cloneModel().getObservables(&vars)->snapshot(kTRUE) ;
  _convVarName = model.conv().cloneVar().GetName() ;
  _modelGen = ((RooAbsPdf&)model.conv().cloneModel()).genContext(*_modelVars,prototype,auxProto,verbose) ;
  _modelCloneSet = 0 ;

  if (prototype) {
    _pdfVars->add(*prototype->get()) ;
    _modelVars->add(*prototype->get()) ;  
  }
}



RooConvGenContext::~RooConvGenContext()
{
//   cout << "RooConvGenContext::dtor(" << this << "," << GetName() << ")" << endl ;

  // Destructor. Delete all owned subgenerator contexts
  delete _pdfGen ;
  delete _modelGen ;
  delete _pdfCloneSet ;
  delete _modelCloneSet ;
  delete _modelVars ;
  delete _pdfVars ;
}


void RooConvGenContext::printToStream(ostream &os, PrintOption opt, TString indent) const
{
  RooAbsGenContext::printToStream(os,opt,indent);
}


void RooConvGenContext::initGenerator(const RooArgSet &theEvent)
{
  // Initialize genertor for this event holder

//   cout << "RooConvGenContext::initGenerator(" << this << "," << GetName() << ") _pdfVars = " << _pdfVars << endl ; 
//   _pdfVars->Print("v") ;
//   cout << "RooConvGenContext::initGenerator(" << this << "," << GetName() << ") _modelVars = " << _modelVars << endl ; 
//   _modelVars->Print("v") ;
//   cout << "RooConvGenContext::initGenerator(" << this << "," << GetName() << ") theEvent = " << &theEvent << endl ; 
//   theEvent.Print("v") ;

  // Find convolution variable in input and output sets
  _cvModel = (RooRealVar*) _modelVars->find(_convVarName) ;
  _cvPdf   = (RooRealVar*) _pdfVars->find(_convVarName) ;
  _cvOut   = (RooRealVar*) theEvent.find(_convVarName) ;

  // Replace all servers in _pdfVars and _modelVars with those in theEvent, except for the convolution variable  
  RooArgSet* pdfCommon = (RooArgSet*) theEvent.selectCommon(*_pdfVars) ;
  pdfCommon->remove(*_cvPdf,kTRUE,kTRUE) ;
  _pdfVars->replace(*pdfCommon) ;
  delete pdfCommon ;

  RooArgSet* modelCommon = (RooArgSet*) theEvent.selectCommon(*_modelVars) ;
  modelCommon->remove(*_cvModel,kTRUE,kTRUE) ;
  _modelVars->replace(*modelCommon) ;
  delete modelCommon ;


  // Initialize component generators
  _pdfGen->initGenerator(*_pdfVars) ;
  _modelGen->initGenerator(*_modelVars) ;

}



void RooConvGenContext::generateEvent(RooArgSet &theEvent, Int_t remaining)
{
  // Generate a single event of the product by generating the components
  // of the products sequentially

//   cout << "RooConvGenContext::generateEvent()" << endl ;

  while(1) {

    // Generate pdf and model data
//     cout << "RooConvGenContext::generateEvent() calling _modelGen->generateEvent" << endl ;
    _modelGen->generateEvent(*_modelVars,remaining) ;
//     cout << "RooConvGenContext::generateEvent() calling _pdfGen->generateEvent" << endl ;
    _pdfGen->generateEvent(*_pdfVars,remaining) ;    
    
    // Construct smeared convolution variable
//     cout << "RooConvGenContext: evaluating sum" << endl ;
//     cout << "_cvPdf = " << _cvPdf << endl ;
//     cout << "_cvModel = " << _cvModel << endl ;
    Double_t convValSmeared = _cvPdf->getVal() + _cvModel->getVal() ;
//     cout << " and the value = " << convValSmeared << endl ;
    if (_cvOut->isValidReal(convValSmeared)) {
//       cout << "event accepted, returning" << endl ;
      // Smeared value in acceptance range, transfer values to output set
      theEvent = *_modelVars ;
      theEvent = *_pdfVars ;
      _cvOut->setVal(convValSmeared) ;
      return ;
    }
//     cout << "event not accepted, reiterating" << endl ;
  }
}


void RooConvGenContext::setProtoDataOrder(Int_t* lut)
{
  RooAbsGenContext::setProtoDataOrder(lut) ;
  _modelGen->setProtoDataOrder(lut) ;
  _pdfGen->setProtoDataOrder(lut) ;
}
