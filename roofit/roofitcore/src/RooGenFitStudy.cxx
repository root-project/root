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
\file RooGenFitStudy.cxx
\class RooGenFitStudy
\ingroup Roofitcore

RooGenFitStudy is an abstract base class for RooStudyManager modules

**/



#include "RooFit.h"
#include "Riostream.h"

#include "RooGenFitStudy.h"
#include "RooWorkspace.h"
#include "RooMsgService.h"
#include "RooDataSet.h"
#include "RooAbsPdf.h"
#include "RooRealVar.h"
#include "RooGlobalFunc.h"
#include "RooFitResult.h"


using namespace std ;

ClassImp(RooGenFitStudy);
  ;


////////////////////////////////////////////////////////////////////////////////
/// Constructor

RooGenFitStudy::RooGenFitStudy(const char* name, const char* title) : 
  RooAbsStudy(name?name:"RooGenFitStudy",title?title:"RooGenFitStudy"), 
  _genPdf(0), 
  _fitPdf(0), 
  _genSpec(0),
  _nllVar(0),
  _ngenVar(0),
  _params(0),
  _initParams(0)
{  
}



////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooGenFitStudy::RooGenFitStudy(const RooGenFitStudy& other) : 
  RooAbsStudy(other),
  _genPdfName(other._genPdfName),
  _genObsName(other._genObsName),
  _fitPdfName(other._fitPdfName),
  _fitObsName(other._fitObsName),
  _genPdf(0),
  _fitPdf(0),
  _genSpec(0),
  _nllVar(0),
  _ngenVar(0),
  _params(0),
  _initParams(0)
{  
  for(TObject * o : other._genOpts) _genOpts.Add(o->Clone());
  for(TObject * o : other._fitOpts) _fitOpts.Add(o->Clone());
}



////////////////////////////////////////////////////////////////////////////////

RooGenFitStudy::~RooGenFitStudy()
{
  if (_params) delete _params ;
}



////////////////////////////////////////////////////////////////////////////////
/// Function called after insertion into workspace

Bool_t RooGenFitStudy::attach(RooWorkspace& w) 
{ 
  Bool_t ret = kFALSE ;

  RooAbsPdf* pdf = w.pdf(_genPdfName.c_str()) ;
  if (pdf) {
    _genPdf = pdf ;
  } else {
    coutE(InputArguments) << "RooGenFitStudy(" << GetName() << ") ERROR: generator p.d.f named " << _genPdfName << " not found in workspace " << w.GetName() << endl ;
    ret = kTRUE ;
  }

  _genObs.add(w.argSet(_genObsName.c_str())) ;
  if (_genObs.getSize()==0) {
    coutE(InputArguments) << "RooGenFitStudy(" << GetName() << ") ERROR: no generator observables defined" << endl ;
    ret = kTRUE ;
  }

  pdf = w.pdf(_fitPdfName.c_str()) ;
  if (pdf) {
    _fitPdf = pdf ;
  } else {
    coutE(InputArguments) << "RooGenFitStudy(" << GetName() << ") ERROR: fitting p.d.f named " << _fitPdfName << " not found in workspace " << w.GetName() << endl ;
    ret = kTRUE ;
  }

  _fitObs.add(w.argSet(_fitObsName.c_str())) ;
  if (_fitObs.getSize()==0) {
    coutE(InputArguments) << "RooGenFitStudy(" << GetName() << ") ERROR: no fitting observables defined" << endl ;
    ret = kTRUE ;
  }

  return ret ; 
} 



////////////////////////////////////////////////////////////////////////////////

void RooGenFitStudy::setGenConfig(const char* pdfName, const char* obsName, const RooCmdArg& arg1,const RooCmdArg& arg2,const RooCmdArg& arg3) 
{
  _genPdfName = pdfName ;
  _genObsName = obsName ;
  _genOpts.Add(arg1.Clone()) ;
  _genOpts.Add(arg2.Clone()) ;
  _genOpts.Add(arg3.Clone()) ;
}



////////////////////////////////////////////////////////////////////////////////

void RooGenFitStudy::setFitConfig(const char* pdfName, const char* obsName, const RooCmdArg& arg1,const RooCmdArg& arg2,const RooCmdArg& arg3) 
{
  _fitPdfName = pdfName ;
  _fitObsName = obsName ;
  _fitOpts.Add(arg1.Clone()) ;
  _fitOpts.Add(arg2.Clone()) ;
  _fitOpts.Add(arg3.Clone()) ;
}



////////////////////////////////////////////////////////////////////////////////
/// One-time initialization of study 

Bool_t RooGenFitStudy::initialize() 
{ 
  _nllVar = new RooRealVar("NLL","-log(Likelihood)",0) ;
  _ngenVar = new RooRealVar("ngen","number of generated events",0) ;
  
  _params = _fitPdf->getParameters(_genObs) ;
  RooArgSet modelParams(*_params) ;
  _initParams = (RooArgSet*) _params->snapshot() ;
  _params->add(*_nllVar) ;
  _params->add(*_ngenVar) ;

  _genSpec = _genPdf->prepareMultiGen(_genObs,(RooCmdArg&)*_genOpts.At(0),(RooCmdArg&)*_genOpts.At(1),(RooCmdArg&)*_genOpts.At(2)) ;

  registerSummaryOutput(*_params,modelParams) ;
  return kFALSE ;
} 



////////////////////////////////////////////////////////////////////////////////
/// Execute one study iteration

Bool_t RooGenFitStudy::execute() 
{ 
  _params->assign(*_initParams) ;
  RooDataSet* data = _genPdf->generate(*_genSpec) ;
  RooFitResult* fr  = _fitPdf->fitTo(*data,RooFit::Save(kTRUE),(RooCmdArg&)*_fitOpts.At(0),(RooCmdArg&)*_fitOpts.At(1),(RooCmdArg&)*_fitOpts.At(2)) ;

  if (fr->status()==0) {
    _ngenVar->setVal(data->sumEntries()) ;
    _nllVar->setVal(fr->minNll()) ;
    storeSummaryOutput(*_params) ;
    storeDetailedOutput(*fr) ;
  }

  delete data ;
  return kFALSE ;
} 



////////////////////////////////////////////////////////////////////////////////
/// Finalization of study

Bool_t RooGenFitStudy::finalize() 
{ 
  delete _params ;
  delete _nllVar ;
  delete _ngenVar ;
  delete _initParams ;
  delete _genSpec ;
  _params = 0 ;
  _nllVar = 0 ;
  _ngenVar = 0 ;
  _initParams = 0 ;
  _genSpec = 0 ;
  

  return kFALSE ; 
} 


////////////////////////////////////////////////////////////////////////////////

void RooGenFitStudy::Print(Option_t* /*options*/) const
{
}


