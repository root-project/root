/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooConvGenContext.cc,v 1.4 2002/03/22 22:43:54 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   11-Oct-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [AUX} --
// RooConvGenContext is an efficient implementation of the generator context
// specific for RooConvolutedPdf objects. The physics model is generated
// with a truth resolution model and the requested resolution model is generated
// separately as a PDF. The convolution variable of the physics model is 
// subsequently explicitly smeared with the resolution model distribution.

#include "RooFitCore/RooConvGenContext.hh"
#include "RooFitCore/RooConvolutedPdf.hh"
#include "RooFitCore/RooProdPdf.hh"
#include "RooFitCore/RooDataSet.hh"
#include "RooFitCore/RooArgSet.hh"
#include "RooFitCore/RooTruthModel.hh"


ClassImp(RooConvGenContext)
;
  
RooConvGenContext::RooConvGenContext(const RooConvolutedPdf &model, const RooArgSet &vars, 
				   const RooDataSet *prototype, Bool_t verbose) :
  RooAbsGenContext(model,vars,prototype,verbose)
{
  // Constructor. Build a generator for the physics PDF convoluted with the truth model
  // and a generator for the resolution model as PDF.

  // Clone PDF and change model to internal truth model
  _pdfCloneSet = (RooArgSet*) RooArgSet(model).snapshot(kTRUE) ;
  if (!_pdfCloneSet) {
    cout << "RooConvGenContext::RooConvGenContext(" << GetName() << ") Couldn't deep-clone PDF, abort," << endl ;
    RooErrorHandler::softAbort() ;
  }

  RooConvolutedPdf* pdfClone = (RooConvolutedPdf*) _pdfCloneSet->find(model.GetName()) ;
  RooTruthModel truthModel("truthModel","Truth resolution model",(RooRealVar&)*pdfClone->convVar()) ;
  pdfClone->changeModel(truthModel) ;
  ((RooRealVar*)pdfClone->convVar())->removeFitRange() ;

  // Create generator for physics X truth model
  _pdfVars = (RooArgSet*) pdfClone->getDependents(&vars) ;
  _pdfGen = pdfClone->genContext(*_pdfVars,prototype,verbose) ;  

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
  modelClone->convVar().removeFitRange() ;

  // Create generator for resolution model as PDF
  _modelVars = (RooArgSet*) modelClone->getDependents(&vars) ;
  _modelVars->add(modelClone->convVar()) ;
  _convVarName = modelClone->convVar().GetName() ;
  _modelGen = modelClone->genContext(*_modelVars,prototype,verbose) ;

  if (prototype) {
    _pdfVars->add(*prototype->get()) ;
    _modelVars->add(*prototype->get()) ;  
  }
}



RooConvGenContext::~RooConvGenContext()
{
  // Destructor. Delete all owned subgenerator contexts
  delete _pdfGen ;
  delete _modelGen ;
  delete _pdfCloneSet ;
  delete _modelCloneSet ;
  delete _modelVars ;
  delete _pdfVars ;
}


void RooConvGenContext::initGenerator(const RooArgSet &theEvent)
{
  // Initialize genertor for this event holder

  // Initialize component generators
  _pdfGen->initGenerator(*_pdfVars) ;
  _modelGen->initGenerator(*_modelVars) ;

  // Find convolution variable in input and output sets
  _cvModel = (RooRealVar*) _modelVars->find(_convVarName) ;
  _cvPdf   = (RooRealVar*) _pdfVars->find(_convVarName) ;
  _cvOut   = (RooRealVar*) theEvent.find(_convVarName) ;
}



void RooConvGenContext::generateEvent(RooArgSet &theEvent, Int_t remaining)
{
  // Generate a single event of the product by generating the components
  // of the products sequentially

  while(1) {
    // Generate pdf and model data
    _modelGen->generateEvent(*_modelVars,remaining) ;
    _pdfGen->generateEvent(*_pdfVars,remaining) ;    
    
    // Construct smeared convolution variable
    Double_t convValSmeared = _cvPdf->getVal() + _cvModel->getVal() ;
    if (_cvOut->isValidReal(convValSmeared)) {
      // Smeared value in acceptance range, transfer values to output set
      theEvent = *_modelVars ;
      theEvent = *_pdfVars ;
      _cvOut->setVal(convValSmeared) ;
      return ;
    }
  }
}
