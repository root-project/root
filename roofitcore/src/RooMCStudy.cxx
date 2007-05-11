/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooMCStudy.cc,v 1.32 2006/10/06 11:51:26 wverkerke Exp $
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

// -- CLASS DESCRIPTION [MISC] --
// RooMCStudy is a help class to facilitate Monte Carlo studies
// such as 'goodness-of-fit' studies, that involve fitting a PDF 
// to multiple toy Monte Carlo sets generated from the same PDF 
// or another PDF.
//
// Given a fit PDF and a generator PDF, RooMCStudy can produce
// large numbers of toyMC samples and/or fit these samples
// and acculumate the final parameters of each fit in a dataset.
//
// Additional plotting routines simplify the task of plotting
// the distribution of the minimized likelihood, each parameters fitted value, 
// fitted error and pull distribution.



#include "RooFit.h"

#include "RooMCStudy.h"
#include "RooAbsMCStudyModule.h"

#include "RooGenContext.h"
#include "RooAbsPdf.h"
#include "RooDataSet.h"
#include "RooDataHist.h"
#include "RooRealVar.h"
#include "RooFitResult.h"
#include "RooErrorVar.h"
#include "RooFormulaVar.h"
#include "RooArgList.h"
#include "RooPlot.h"
#include "RooGenericPdf.h"
#include "RooRandom.h"
#include "RooCmdConfig.h"
#include "RooGlobalFunc.h"
#include "RooPullVar.h"


ClassImp(RooMCStudy)
  ;

RooMCStudy::RooMCStudy(const RooAbsPdf& model, const RooArgSet& observables,
   		       RooCmdArg arg1, RooCmdArg arg2,
   		       RooCmdArg arg3,RooCmdArg arg4,RooCmdArg arg5,
   		       RooCmdArg arg6,RooCmdArg arg7,RooCmdArg arg8) 

  // Construct Monte Carlo Study Manager. This class automates generating data from a given PDF,
  // fitting the PDF to that data and accumulating the fit statistics.
  //
  // The constructor accepts the following arguments
  //
  // model       -- The PDF to be studied
  // observables -- The variables of the PDF to be considered the observables
  //
  // FitModel(const RooAbsPdf&)        -- The PDF for fitting, if it is different from the PDF for generating
  // ConditionalObservables
  //           (const RooArgSet& set)  -- The set of observables that the PDF should _not_ be normalized over
  // Binned(Bool_t flag)               -- Bin the dataset before fitting it. Speeds up fitting of large data samples
  // FitOptions(const char*)           -- Classic fit options, provided for backward compatibility
  // FitOptions(....)                  -- Options to be used for fitting. All named arguments inside FitOptions()
  //                                                   are passed to RooAbsPdf::fitTo();
  // Verbose(Bool_t flag)              -- Activate informational messages in event generation phase
  // Extended(Bool_t flag)             -- Determine number of events for each sample anew from a Poisson distribution
  // ProtoData(const RooDataSet&, 
  //                 Bool_t randOrder) -- Prototype data for the event generation. If the randOrder flag is
  //                                      set, the order of the dataset will be re-randomized for each generation
  //                                      cycle to protect against systematic biases if the number of generated
  //                                      events does not exactly match the number of events in the prototype dataset
  //                                      at the cost of reduced precision
  //                                      with mu equal to the specified number of events
{
  // Stuff all arguments in a list
  RooLinkedList cmdList;
  cmdList.Add(const_cast<RooCmdArg*>(&arg1)) ;  cmdList.Add(const_cast<RooCmdArg*>(&arg2)) ;
  cmdList.Add(const_cast<RooCmdArg*>(&arg3)) ;  cmdList.Add(const_cast<RooCmdArg*>(&arg4)) ;
  cmdList.Add(const_cast<RooCmdArg*>(&arg5)) ;  cmdList.Add(const_cast<RooCmdArg*>(&arg6)) ;
  cmdList.Add(const_cast<RooCmdArg*>(&arg7)) ;  cmdList.Add(const_cast<RooCmdArg*>(&arg8)) ;

  // Select the pdf-specific commands 
  RooCmdConfig pc(Form("RooMCStudy::RooMCStudy(%s)",model.GetName())) ;
  
  pc.defineObject("fitModel","FitModel",0,0) ;
  pc.defineObject("condObs","ProjectedDependents",0,0) ;
  pc.defineObject("protoData","PrototypeData",0,0) ;
  pc.defineInt("randProtoData","PrototypeData",0,0) ;
  pc.defineInt("verboseGen","Verbose",0,0) ;
  pc.defineInt("extendedGen","Extended",0,0) ;
  pc.defineInt("binGenData","Binned",0,0) ;
  pc.defineString("fitOpts","FitOptions",0,"") ;
  pc.defineInt("dummy","FitOptArgs",0,0) ;
  pc.defineMutex("FitOptions","FitOptArgs") ;
  
  // Process and check varargs 
  pc.process(cmdList) ;
  if (!pc.ok(kTRUE)) {
    // WVE do something here
    return ;
  }
  
  // Save fit command options
  if (pc.hasProcessed("FitOptArgs")) {
    RooCmdArg* fitOptArg = static_cast<RooCmdArg*>(cmdList.FindObject("FitOptArgs")) ;
    for (Int_t i=0 ; i<fitOptArg->subArgs().GetSize() ;i++) {
      _fitOptList.Add(new RooCmdArg(static_cast<RooCmdArg&>(*fitOptArg->subArgs().At(i)))) ;
    }
  }

  // Decode command line arguments
  _verboseGen = pc.getInt("verboseGen") ;
  _extendedGen = pc.getInt("extendedGen") ;
  _binGenData = pc.getInt("binGenData") ;
  _randProto = pc.getInt("randProtoData") ;
  
  _genModel = const_cast<RooAbsPdf*>(&model) ;
  RooAbsPdf* fitModel = static_cast<RooAbsPdf*>(pc.getObject("fitModel",0)) ;
  _fitModel = fitModel ? fitModel : _genModel ;
  
  _genProtoData = static_cast<RooDataSet*>(pc.getObject("protoData",0)) ;
  if (pc.getObject("condObs",0)) {
    _projDeps.add(static_cast<RooArgSet&>(*pc.getObject("condObs",0))) ;
  }
  
  _dependents.add(observables) ;
     
  _allDependents.add(_dependents) ;  
  _fitOptions = pc.getString("fitOpts") ;
  _canAddFitResults = kTRUE ;
  
  if (_extendedGen && _genProtoData && !_randProto) {
    cout << "RooMCStudy::RooMCStudy: WARNING Using generator option 'e' (Poisson distribution of #events) together " << endl
	 << "                        with a prototype dataset implies incomplete sampling or oversampling of proto data." << endl
	 << "                        Use option \"r\" to randomize prototype dataset order and thus to randomize" << endl
	 << "                        the set of over/undersampled prototype events for each generation cycle." << endl ;
  }
  
  _genContext = _genModel->genContext(_dependents,_genProtoData,0,_verboseGen) ;
  _genParams = _genModel->getParameters(&_dependents) ;
  _genInitParams = (RooArgSet*) _genParams->snapshot(kFALSE) ;
  
  // Store list of parameters and save initial values separately
  _fitParams = _fitModel->getParameters(&_dependents) ;
  _fitInitParams = (RooArgSet*) _fitParams->snapshot(kTRUE) ;
  
  _nExpGen = _extendedGen ? _genModel->expectedEvents(&_dependents) : 0 ;
  
  // Place holder for NLL
  _nllVar = new RooRealVar("NLL","-log(Likelihood)",0) ;
  
  // Create data set containing parameter values, errors and pulls
  RooArgSet tmp2(*_fitParams) ;
  tmp2.add(*_nllVar) ;
  
  // Mark all variable to store their errors in the dataset
  tmp2.setAttribAll("StoreError",kTRUE) ;
  tmp2.setAttribAll("StoreAsymError",kTRUE) ;
  _fitParData = new RooDataSet("fitParData","Fit Parameters DataSet",tmp2) ;
  tmp2.setAttribAll("StoreError",kFALSE) ;
  tmp2.setAttribAll("StoreAsymError",kFALSE) ;
  
  // Append proto variables to allDependents
  if (_genProtoData) {
    _allDependents.add(*_genProtoData->get(),kTRUE) ;
  }

  // Call module initializers
  list<RooAbsMCStudyModule*>::iterator iter ;
  for (iter=_modList.begin() ; iter!= _modList.end() ; ++iter) {
    Bool_t ok = (*iter)->doInitializeInstance(*this) ;
    if (!ok) {
      cout << "RooMCStudy::ctor: removing study module " << (*iter)->GetName() << " from analysis chain because initialization failed" << endl ;
      iter = _modList.erase(iter) ;
    }
  }
  
}

RooMCStudy::RooMCStudy(const RooAbsPdf& genModel, const RooAbsPdf& fitModel, 
   		       const RooArgSet& dependents, const char* genOptions, 
   		       const char* fitOptions, const RooDataSet* genProtoData, 
   		       const RooArgSet& projDeps) :
  _genModel((RooAbsPdf*)&genModel), 
  _genProtoData(genProtoData),
  _projDeps(projDeps),
  _dependents(dependents), 
  _allDependents(dependents), 
  _fitModel((RooAbsPdf*)&fitModel), 
  _fitOptions(fitOptions),
  _canAddFitResults(kTRUE)
{
  // Constructor with a generator and fit model. Both models may point
  // to the same object. The 'dependents' set of variables is generated 
  // in the generator phase. The optional prototype dataset is passed to
  // the generator
  //
  // Available generator options
  //  v  - Verbose
  //  e  - Extended: use Poisson distribution for Nevts generated
  //
  // Available fit options
  //  See RooAbsPdf::fitTo()
  //
  
  // Decode generator options
  TString genOpt(genOptions) ;
  genOpt.ToLower() ;
  _verboseGen = genOpt.Contains("v") ;
  _extendedGen = genOpt.Contains("e") ;
  _binGenData = genOpt.Contains("b") ;
  _randProto = genOpt.Contains("r") ;
  
  if (_extendedGen && genProtoData && !_randProto) {
    cout << "RooMCStudy::RooMCStudy: WARNING Using generator option 'e' (Poisson distribution of #events) together " << endl
   	 << "                        with a prototype dataset implies incomplete sampling or oversampling of proto data." << endl
   	 << "                        Use option \"r\" to randomize prototype dataset order and thus to randomize" << endl
	 << "                        the set of over/undersampled prototype events for each generation cycle." << endl ;
  }
  
  _genContext = genModel.genContext(dependents,genProtoData,0,_verboseGen) ;
  RooArgSet* tmp = genModel.getParameters(&dependents) ;
  _genInitParams = (RooArgSet*) tmp->snapshot(kFALSE) ;
  delete tmp ;
  
  // Store list of parameters and save initial values separately
  _fitParams = fitModel.getParameters(&dependents) ;
  _fitInitParams = (RooArgSet*) _fitParams->snapshot(kTRUE) ;
  
  _nExpGen = _extendedGen ? genModel.expectedEvents(&dependents) : 0 ;
  
  // Place holder for NLL
  _nllVar = new RooRealVar("NLL","-log(Likelihood)",0) ;
  
  // Create data set containing parameter values, errors and pulls
  RooArgSet tmp2(*_fitParams) ;
  tmp2.add(*_nllVar) ;
  
  // Mark all variable to store their errors in the dataset
  tmp2.setAttribAll("StoreError",kTRUE) ;
  tmp2.setAttribAll("StoreAsymError",kTRUE) ;
  _fitParData = new RooDataSet("fitParData","Fit Parameters DataSet",tmp2) ;
  tmp2.setAttribAll("StoreError",kFALSE) ;
  tmp2.setAttribAll("StoreAsymError",kFALSE) ;
  
  // Append proto variables to allDependents
  if (genProtoData) {
    _allDependents.add(*genProtoData->get(),kTRUE) ;
  }

  // Call module initializers
  list<RooAbsMCStudyModule*>::iterator iter ;
  for (iter=_modList.begin() ; iter!= _modList.end() ; ++iter) {
    Bool_t ok = (*iter)->doInitializeInstance(*this) ;
    if (!ok) {
      cout << "RooMCStudy::ctor: removing study module " << (*iter)->GetName() << " from analysis chain because initialization failed" << endl ;
      iter = _modList.erase(iter) ;
    }
  }
  
}



RooMCStudy::~RooMCStudy() 
{  
  // Destructor 
  
  _genDataList.Delete() ;
  _fitResList.Delete() ;
  _fitOptList.Delete() ;
  delete _fitParData ;
  delete _fitInitParams ;
  delete _fitParams ;
  delete _genInitParams ;
  delete _genParams ;
  delete _genContext ;
  delete _nllVar ;
}



void RooMCStudy::addModule(RooAbsMCStudyModule& module) 
{
  // Method to add study modules
  module.doInitializeInstance(*this) ;
  _modList.push_back(&module) ;        
}



Bool_t RooMCStudy::run(Bool_t generate, Bool_t fit, Int_t nSamples, Int_t nEvtPerSample, Bool_t keepGenData, const char* asciiFilePat) 
{
  // Run engine. Generate and/or fit, according to flags, 'nSamples' samples of 'nEvtPerSample' events.
  // If keepGenData is set, all generated data sets will be kept in memory and can be accessed
  // later via genData().
  //
  // When generating, data sets will be written out in ascii form if the pattern string is supplied
  // The pattern, which is a template for sprintf, should look something like "data/toymc_%04d.dat"
  // and should contain one integer field that encodes the sample serial number.
  //
  // When fitting only, data sets may optionally be read from ascii files, using the same file
  // pattern.
  //

  list<RooAbsMCStudyModule*>::iterator iter ;
  for (iter=_modList.begin() ; iter!= _modList.end() ; ++iter) {
    (*iter)->initializeRun(nSamples) ;
  }  

  while(nSamples--) {
    
    cout << "RooMCStudy::run: " ;
    if (generate) cout << "Generating " ;
    if (generate && fit) cout << "and " ;
    if (fit) cout << "fitting " ;
    cout << "sample " << nSamples << endl ;
    
    _genSample = 0;
    Bool_t existingData = kFALSE ;
    if (generate) {
      // Generate sample
      Int_t nEvt(nEvtPerSample) ;

      // Reset generator parameters to initial values
      *_genParams = *_genInitParams ;

      // Call module before-generation hook
      list<RooAbsMCStudyModule*>::iterator iter ;
      for (iter=_modList.begin() ; iter!= _modList.end() ; ++iter) {
	(*iter)->processBeforeGen(nSamples) ;
      }  

      // Calculate the number of (extended) events for this run
      if (_extendedGen) {
	_nExpGen = _genModel->expectedEvents(&_dependents) ;
	nEvt = RooRandom::randomGenerator()->Poisson(nEvtPerSample==0?_nExpGen:nEvtPerSample) ;
      }
      
      // Optional randomization of protodata for this run
      if (_randProto && _genProtoData && _genProtoData->numEntries()!=nEvt) {
	cout << "RooMCStudy: (Re)randomizing event order in prototype dataset (Nevt=" << nEvt << ")" << endl ;
          Int_t* newOrder = _genModel->randomizeProtoOrder(_genProtoData->numEntries(),nEvt) ;
          _genContext->setProtoDataOrder(newOrder) ;
          delete[] newOrder ;
      }
      
      // Actual generation of events
      _genSample = _genContext->generate(nEvt) ;
      
    } else if (asciiFilePat && &asciiFilePat) {

      // Load sample from ASCII file
      char asciiFile[1024] ;
      sprintf(asciiFile,asciiFilePat,nSamples) ;
      RooArgList depList(_allDependents) ;
      _genSample = RooDataSet::read(asciiFile,depList,"q") ;      
      
    } else {
      
      // Load sample from internal list
      _genSample = (RooDataSet*) _genDataList.At(nSamples) ;
      existingData = kTRUE ;
      if (!_genSample) {
   	cout << "RooMCStudy::run: WARNING: Sample #" << nSamples << " not loaded, skipping" << endl ;
   	continue ;
      }
    }

    // Call module between generation and fitting hook
    list<RooAbsMCStudyModule*>::iterator iter ;
    for (iter=_modList.begin() ; iter!= _modList.end() ; ++iter) {
      (*iter)->processBetweenGenAndFit(nSamples) ;
    }  
    
    if (fit) fitSample(_genSample) ;

    // Call module between generation and fitting hook
    for (iter=_modList.begin() ; iter!= _modList.end() ; ++iter) {
      (*iter)->processAfterFit(nSamples) ;
    }  
    
    // Optionally write to ascii file
    if (generate && asciiFilePat && *asciiFilePat) {
      char asciiFile[1024] ;
      sprintf(asciiFile,asciiFilePat,nSamples) ;
      _genSample->write(asciiFile) ;
    }
    
    // Add to list or delete
    if (!existingData) {
      if (keepGenData) {
	_genDataList.Add(_genSample) ;
      } else {
	delete _genSample ;
      }
    }
  }

  for (iter=_modList.begin() ; iter!= _modList.end() ; ++iter) {
    RooDataSet* auxData = (*iter)->finalizeRun() ;
    if (auxData) {
      _fitParData->merge(auxData) ;
    }
  }  

  _canAddFitResults = kFALSE ;
  if (fit) calcPulls() ;
  return kFALSE ;
}






Bool_t RooMCStudy::generateAndFit(Int_t nSamples, Int_t nEvtPerSample, Bool_t keepGenData, const char* asciiFilePat) 
{
  // Generate and fit 'nSamples' samples of 'nEvtPerSample' events.
  // If keepGenData is set, all generated data sets will be kept in memory and can be accessed
  // later via genData().
  //
  // Data sets will be written out is ascii form if the pattern string is supplied.
  // The pattern, which is a template for sprintf, should look something like "data/toymc_%04d.dat"
  // and should contain one integer field that encodes the sample serial number.
  //
  
  // Clear any previous data in memory
  _fitResList.Delete() ;
  _genDataList.Delete() ;
  _fitParData->reset() ;
  
  return run(kTRUE,kTRUE,nSamples,nEvtPerSample,keepGenData,asciiFilePat) ;
}


Bool_t RooMCStudy::generate(Int_t nSamples, Int_t nEvtPerSample, Bool_t keepGenData, const char* asciiFilePat) 
{
  // Generate 'nSamples' samples of 'nEvtPerSample' events.
  // If keepGenData is set, all generated data sets will be kept in memory 
  // and can be accessed later via genData().
  //
  // Data sets will be written out in ascii form if the pattern string is supplied.
  // The pattern, which is a template for sprintf, should look something like "data/toymc_%04d.dat"
  // and should contain one integer field that encodes the sample serial number.
  //
  
  // Clear any previous data in memory
  _genDataList.Delete() ;
  
  return run(kTRUE,kFALSE,nSamples,nEvtPerSample,keepGenData,asciiFilePat) ;
}


Bool_t RooMCStudy::fit(Int_t nSamples, const char* asciiFilePat) 
{
  // Fit 'nSamples' datasets, which are read from ASCII files.
  //
  // The ascii file pattern, which is a template for sprintf, should look something like "data/toymc_%04d.dat"
  // and should contain one integer field that encodes the sample serial number.
  //
  
  // Clear any previous data in memory
  _fitResList.Delete() ;
  _fitParData->reset() ;
  
  return run(kFALSE,kTRUE,nSamples,0,kFALSE,asciiFilePat) ;
}


Bool_t RooMCStudy::fit(Int_t nSamples, TList& dataSetList) 
{
  // Fit 'nSamples' datasets, as supplied in 'dataSetList'
  // 
  
  // Clear any previous data in memory
  _fitResList.Delete() ;
  _genDataList.Delete() ;
  _fitParData->reset() ;
  
  // Load list of data sets
  TIterator* iter = dataSetList.MakeIterator() ;
  RooAbsData* gset ;
  while((gset=(RooAbsData*)iter->Next())) {
    _genDataList.Add(gset) ;
  }
  delete iter ;
  
  return run(kFALSE,kTRUE,nSamples,0,kTRUE,0) ;
}



void RooMCStudy::resetFitParams()
{
  *_fitParams = *_fitInitParams ;
}



RooFitResult* RooMCStudy::doFit(RooAbsData* genSample)
{
  // Perform actual fit according to specifications

  // Fit model to data set
  TString fitOpt2(_fitOptions) ; fitOpt2.Append("r") ;
  
  // Optionally bin dataset before fitting
  RooAbsData* data ;
  if (_binGenData) {    
    RooArgSet* depList = _fitModel->getObservables(genSample) ;
    data = new RooDataHist(genSample->GetName(),genSample->GetTitle(),*depList,*genSample) ;
    delete depList ;
  } else {
    data = genSample ;
  }
  
  RooFitResult* fr ;
  if (_fitOptList.GetSize()==0) {
    if (_projDeps.getSize()>0) {
      fr = (RooFitResult*) _fitModel->fitTo(*data,_projDeps,fitOpt2) ;
    } else {
      fr = (RooFitResult*) _fitModel->fitTo(*data,fitOpt2) ;
    }
  } else {
    RooCmdArg save  = RooFit::Save() ;
    RooCmdArg condo = RooFit::ConditionalObservables(_projDeps) ;
    RooLinkedList fitOptList(_fitOptList) ;
    fitOptList.Add(&save) ;
    if (_projDeps.getSize()>0) {
      fitOptList.Add(&condo) ;
    }
    fr = (RooFitResult*) _fitModel->fitTo(*data,fitOptList) ;
  }

  if (_binGenData) delete data ;

  return fr ;
}



RooFitResult* RooMCStudy::refit(RooAbsData* genSample) 
{
  if (!genSample) {
    genSample = _genSample ;
  }

  RooFitResult* fr = doFit(genSample) ;
    
  return fr ;
}



Bool_t RooMCStudy::fitSample(RooAbsData* genSample) 
{  
  // Fit given dataset with fit model. If fit
  // converges (TMinuit status code zero)
  // The fit results are appended to the fit results
  // dataset
  //
  // If the fit option "r" is supplied, the RooFitResult
  // objects will always be saved, regardless of the
  // fit status. RooFitResults objects can be retrieved
  // later via fitResult().
  //  
  
  // Reset all fit parameters to their initial values  
  resetFitParams() ;

  // Perform actual fit
  RooFitResult* fr = doFit(genSample) ;

  // If fit converged, store parameters and NLL
  Bool_t ok = (fr->status()==0) ;
  if (ok) {
    _nllVar->setVal(fr->minNll()) ;
    RooArgSet tmp(*_fitParams) ;
    tmp.add(*_nllVar) ;
    _fitParData->add(tmp) ;
  }
  
  // Store fit result if requested by user
  Bool_t userSaveRequest = kFALSE ;
  if (_fitOptList.GetSize()>0) {
    if (_fitOptList.FindObject("Save")) userSaveRequest = kTRUE ;
  } else {
    if (_fitOptions.Contains("r")) userSaveRequest = kTRUE ;
  }

  if (userSaveRequest) {
    _fitResList.Add(fr) ;
  } else {
    delete fr ;
  }
    
  return !ok ;
}


Bool_t RooMCStudy::addFitResult(const RooFitResult& fr) 
{  
  if (!_canAddFitResults) {
    cout << "RooMCStudy::addFitResult: ERROR cannot add fit results in current state" << endl ;
    return kTRUE ;
  }
  
  // Transfer contents of fit result to fitParams ;
  *_fitParams = RooArgSet(fr.floatParsFinal()) ;
  
  // If fit converged, store parameters and NLL
  Bool_t ok = (fr.status()==0) ;
  if (ok) {
    _nllVar->setVal(fr.minNll()) ;
    RooArgSet tmp(*_fitParams) ;
    tmp.add(*_nllVar) ;
    _fitParData->add(tmp) ;
  }
  
  // Store fit result if requested by user
  if (_fitOptions.Contains("r")) {
    _fitResList.Add((TObject*)&fr) ;
  }  
  
  return kFALSE ;
}



void RooMCStudy::calcPulls() 
{
  // Calculate the pulls for all fit parameters in
  // the fit results data set, and add them to that dataset
  
  TIterator* iter = _fitParams->createIterator()  ;
  RooRealVar* par ;
  while((par=(RooRealVar*)iter->Next())) {
    
    RooErrorVar* err = par->errorVar() ;
    _fitParData->addColumn(*err) ;
    
    TString name(par->GetName()), title(par->GetTitle()) ;
    name.Append("pull") ;
    title.Append(" Pull") ;
    RooAbsReal* genParOrig = (RooAbsReal*)_genInitParams->find(par->GetName()) ;
    if (genParOrig) {
      RooAbsReal* genPar = (RooAbsReal*) genParOrig->Clone("truth") ;
      RooPullVar pull(name,title,*par,*genPar) ;
      
      _fitParData->addColumn(pull) ;
      delete genPar ;
    }    
  }
  delete iter ;
  
}




const RooDataSet& RooMCStudy::fitParDataSet()
{
  // Return the fit parameter dataset
  if (_canAddFitResults) {
    calcPulls() ;  
    _canAddFitResults = kFALSE ; 
  }
  
  return *_fitParData ;
}




const RooArgSet* RooMCStudy::fitParams(Int_t sampleNum) const 
{
  // Return an argset with the fit parameters for the given sample number
  // NB: The fit parameters are only stored for successfull fits,
  //     thus the maximum sampleNum can be less that the number
  //     of generated samples and if so, the indeces will
  //     be out of synch with genData() and fitResult()
  
  // Check if sampleNum is in range
  if (sampleNum<0 || sampleNum>=_fitParData->numEntries()) {
    cout << "RooMCStudy::fitParams: ERROR, invalid sample number: " << sampleNum << endl ;    
    return 0 ;
  }
  
  return _fitParData->get(sampleNum) ;
}



const RooFitResult* RooMCStudy::fitResult(Int_t sampleNum) const
{
  // Return the fit result object of the fit to given sample
  
  // Check if sampleNum is in range
  if (sampleNum<0 || sampleNum>=_fitResList.GetSize()) {
    cout << "RooMCStudy::fitResult: ERROR, invalid sample number: " << sampleNum << endl ;    
    return 0 ;
  }
  
  // Retrieve fit result object
  const RooFitResult* fr = (RooFitResult*) _fitResList.At(sampleNum) ;
  if (fr) {
    return fr ;
  } else {
    cout << "RooMCStudy::fitResult: ERROR, no fit result saved for sample " 
   	 << sampleNum << ", did you use the 'r; fit option?" << endl ;
  }
  return 0 ;
}


const RooDataSet* RooMCStudy::genData(Int_t sampleNum) const 
{
  // Return the given generated dataset 
  
  // Check that generated data was saved
  if (_genDataList.GetSize()==0) {
    cout << "RooMCStudy::genData() ERROR, generated data was not saved" << endl ;
    return 0 ;
  }
  
  // Check if sampleNum is in range
  if (sampleNum<0 || sampleNum>=_genDataList.GetSize()) {
    cout << "RooMCStudy::genData() ERROR, invalid sample number: " << sampleNum << endl ;    
    return 0 ;
  }
  
  return (RooDataSet*) _genDataList.At(sampleNum) ;
}


RooPlot* RooMCStudy::plotParamOn(RooPlot* frame, const RooCmdArg& arg1, const RooCmdArg& arg2, const RooCmdArg& arg3, const RooCmdArg& arg4, 
   				 const RooCmdArg& arg5, const RooCmdArg& arg6, const RooCmdArg& arg7, const RooCmdArg& arg8) 
{
  // Plot the distribution of the fitted value of the given parameter on the specified frame
  // Any specified named argument is passed to the RooAbsData::plotOn() call. See that function for allowed options
  
  _fitParData->plotOn(frame,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8) ;
  return frame ;
}



RooPlot* RooMCStudy::plotParam(const char* paramName, const RooCmdArg& arg1, const RooCmdArg& arg2, const RooCmdArg& arg3, const RooCmdArg& arg4, 
   			       const RooCmdArg& arg5, const RooCmdArg& arg6, const RooCmdArg& arg7, const RooCmdArg& arg8) 
{
  // Plot the distribution of the fitted value of the given parameter on a newly created frame.
  //
  // This function accepts the following optional arguments
  // FrameRange(double lo, double hi) -- Set range of frame to given specification
  // FrameBins(int bins)              -- Set default number of bins of frame to given number
  // Frame(...)                       -- Pass supplied named arguments to RooAbsRealLValue::frame() function. See frame() function
  //                                     for list of allowed arguments
  //
  // If no frame specifications are given, the AutoRange() feature will be used to set the range
  // Any other named argument is passed to the RooAbsData::plotOn() call. See that function for allowed options


  // Find parameter in fitParDataSet
  RooRealVar* param = static_cast<RooRealVar*>(_fitParData->get()->find(paramName)) ;
  if (!param) {
    cout << "RooMCStudy::plotParam: ERROR: no parameter defined with name " << paramName << endl ;  
    return 0 ;
  }

  // Forward to implementation below
  return plotParam(*param,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8) ;
}


RooPlot* RooMCStudy::plotParam(const RooRealVar& param, const RooCmdArg& arg1, const RooCmdArg& arg2, const RooCmdArg& arg3, const RooCmdArg& arg4, 
   			       const RooCmdArg& arg5, const RooCmdArg& arg6, const RooCmdArg& arg7, const RooCmdArg& arg8) 
{
  // Plot the distribution of the fitted value of the given parameter on a newly created frame.
  //
  // This function accepts the following optional arguments
  // FrameRange(double lo, double hi) -- Set range of frame to given specification
  // FrameBins(int bins)              -- Set default number of bins of frame to given number
  // Frame(...)                       -- Pass supplied named arguments to RooAbsRealLValue::frame() function. See frame() function
  //                                     for list of allowed arguments
  //
  // If no frame specifications are given, the AutoRange() feature will be used to set the range
  // Any other named argument is passed to the RooAbsData::plotOn() call. See that function for allowed options

  // Stuff all arguments in a list
  RooLinkedList cmdList;
  cmdList.Add(const_cast<RooCmdArg*>(&arg1)) ;  cmdList.Add(const_cast<RooCmdArg*>(&arg2)) ;
  cmdList.Add(const_cast<RooCmdArg*>(&arg3)) ;  cmdList.Add(const_cast<RooCmdArg*>(&arg4)) ;
  cmdList.Add(const_cast<RooCmdArg*>(&arg5)) ;  cmdList.Add(const_cast<RooCmdArg*>(&arg6)) ;
  cmdList.Add(const_cast<RooCmdArg*>(&arg7)) ;  cmdList.Add(const_cast<RooCmdArg*>(&arg8)) ;
  
  RooPlot* frame = makeFrameAndPlotCmd(param, cmdList) ;
  if (frame) {
    _fitParData->plotOn(frame, cmdList) ;
  }

  return frame ;
}


RooPlot* RooMCStudy::plotNLL(const RooCmdArg& arg1, const RooCmdArg& arg2,
                     const RooCmdArg& arg3, const RooCmdArg& arg4,
                     const RooCmdArg& arg5, const RooCmdArg& arg6,
                     const RooCmdArg& arg7, const RooCmdArg& arg8) 
{
  // Plot the distribution of the -log(l) values on a newly created frame.
  //
  // This function accepts the following optional arguments
  // FrameRange(double lo, double hi) -- Set range of frame to given specification
  // FrameBins(int bins)              -- Set default number of bins of frame to given number
  // Frame(...)                       -- Pass supplied named arguments to RooAbsRealLValue::frame() function. See frame() function
  //                                     for list of allowed arguments
  //
  // If no frame specifications are given, the AutoRange() feature will be used to set the range
  // Any other named argument is passed to the RooAbsData::plotOn() call. See that function for allowed options

  return plotParam(*_nllVar,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8) ;
}


RooPlot* RooMCStudy::plotError(const RooRealVar& param, const RooCmdArg& arg1, const RooCmdArg& arg2,
                     const RooCmdArg& arg3, const RooCmdArg& arg4,
                     const RooCmdArg& arg5, const RooCmdArg& arg6,
                     const RooCmdArg& arg7, const RooCmdArg& arg8) 
{
  // Plot the distribution of the fit errors for the specified parameter on a newly created frame.
  //
  // This function accepts the following optional arguments
  // FrameRange(double lo, double hi) -- Set range of frame to given specification
  // FrameBins(int bins)              -- Set default number of bins of frame to given number
  // Frame(...)                       -- Pass supplied named arguments to RooAbsRealLValue::frame() function. See frame() function
  //                                     for list of allowed arguments
  //
  // If no frame specifications are given, the AutoRange() feature will be used to set the range
  // Any other named argument is passed to the RooAbsData::plotOn() call. See that function for allowed options

  if (_canAddFitResults) {
    calcPulls() ;
    _canAddFitResults=kFALSE ;
  }

  RooErrorVar* evar = param.errorVar() ;
  RooRealVar* evar_rrv = static_cast<RooRealVar*>(evar->createFundamental()) ;
  RooPlot* frame = plotParam(*evar_rrv,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8) ;
  delete evar_rrv ;
  delete evar ;
  return frame ;
}

RooPlot* RooMCStudy::plotPull(const RooRealVar& param, const RooCmdArg& arg1, const RooCmdArg& arg2,
                     const RooCmdArg& arg3, const RooCmdArg& arg4,
                     const RooCmdArg& arg5, const RooCmdArg& arg6,
                     const RooCmdArg& arg7, const RooCmdArg& arg8) 
{
  // Plot the distribution of pull values for the specified parameter on a newly created frame. If asymmetric
  // errors are calculated in the fit (by MINOS) those will be used in the pull calculation
  //
  // This function accepts the following optional arguments
  // FrameRange(double lo, double hi) -- Set range of frame to given specification
  // FrameBins(int bins)              -- Set default number of bins of frame to given number
  // Frame(...)                       -- Pass supplied named arguments to RooAbsRealLValue::frame() function. See frame() function
  //                                     for list of allowed arguments
  // FitGauss(Bool_t flag)            -- Add a gaussian fit to the frame
  //
  // If no frame specifications are given, the AutoSymRange() feature will be used to set the range
  // Any other named argument is passed to the RooAbsData::plotOn() call. See that function for allowed options

  // Stuff all arguments in a list
  RooLinkedList cmdList;
  cmdList.Add(const_cast<RooCmdArg*>(&arg1)) ;  cmdList.Add(const_cast<RooCmdArg*>(&arg2)) ;
  cmdList.Add(const_cast<RooCmdArg*>(&arg3)) ;  cmdList.Add(const_cast<RooCmdArg*>(&arg4)) ;
  cmdList.Add(const_cast<RooCmdArg*>(&arg5)) ;  cmdList.Add(const_cast<RooCmdArg*>(&arg6)) ;
  cmdList.Add(const_cast<RooCmdArg*>(&arg7)) ;  cmdList.Add(const_cast<RooCmdArg*>(&arg8)) ;

  TString name(param.GetName()), title(param.GetTitle()) ;
  name.Append("pull") ; title.Append(" Pull") ;
  RooRealVar pvar(name,title,-100,100) ;
  pvar.setBins(100) ;


  RooPlot* frame = makeFrameAndPlotCmd(pvar, cmdList, kTRUE) ;
  if (frame) {

    // Pick up optonal FitGauss command from list
    RooCmdConfig pc(Form("RooMCStudy::plotPull(%s)",_genModel->GetName())) ;
    pc.defineInt("fitGauss","FitGauss",0,0) ;
    pc.allowUndefined() ;
    pc.process(cmdList) ;
    Bool_t fitGauss=pc.getInt("fitGauss") ;

    // Pass stripped command list to plotOn()
    pc.stripCmdList(cmdList,"FitGauss") ;
    _fitParData->plotOn(frame,cmdList) ;

    // Add Gaussian fit if requested
    if (fitGauss) {
      RooRealVar pullMean("pullMean","Mean of pull",0,-100,100) ;
      RooRealVar pullSigma("pullSigma","Width of pull",1,0,5) ;
      RooGenericPdf pullGauss("pullGauss","Gaussian of pull",
			      "exp(-0.5*(@0-@1)*(@0-@1)/(@2*@2))",
			      RooArgSet(pvar,pullMean,pullSigma)) ;
      pullGauss.fitTo(*_fitParData,RooFit::Minos(0),RooFit::PrintLevel(-1)) ;
      pullGauss.plotOn(frame) ;
      pullGauss.paramOn(frame,_fitParData) ;
    }
  }
  return frame ; ;
}


RooPlot* RooMCStudy::makeFrameAndPlotCmd(const RooRealVar& param, RooLinkedList& cmdList, Bool_t symRange) const 
{

  // Select the frame-specific commands 
  RooCmdConfig pc(Form("RooMCStudy::plotParam(%s)",_genModel->GetName())) ;
  pc.defineInt("nbins","FrameBins",0,0) ;
  pc.defineDouble("xlo","FrameRange",0,0) ;
  pc.defineDouble("xhi","FrameRange",1,0) ;
  pc.defineInt("dummy","FrameArgs",0,0) ;
  pc.defineMutex("FrameBins","FrameArgs") ;
  pc.defineMutex("FrameRange","FrameArgs") ;

  // Process and check varargs 
  pc.allowUndefined() ;
  pc.process(cmdList) ;
  if (!pc.ok(kTRUE)) {
    return 0 ;
  }
  
  // Make frame according to specs
  Int_t nbins = pc.getInt("nbins") ;
  Double_t xlo = pc.getDouble("xlo") ;
  Double_t xhi = pc.getDouble("xhi") ;
  RooPlot* frame ; 

  if (pc.hasProcessed("FrameArgs")) {
    // Explicit frame arguments are given, pass them on
    RooCmdArg* frameArg = static_cast<RooCmdArg*>(cmdList.FindObject("FrameArgs")) ;
    frame = param.frame(frameArg->subArgs()) ;
  } else {
    // FrameBins, FrameRange or none are given, build custom frame command list
    RooCmdArg bins = RooFit::Bins(nbins) ;
    RooCmdArg range = RooFit::Range(xlo,xhi) ;
    RooCmdArg autor = symRange ? RooFit::AutoSymRange(*_fitParData,0.2) : RooFit::AutoRange(*_fitParData,0.2) ;
    RooLinkedList frameCmdList ;

    if (pc.hasProcessed("FrameBins")) frameCmdList.Add(&bins) ;
    if (pc.hasProcessed("FrameRange")) {
      frameCmdList.Add(&range) ;
    } else {
      frameCmdList.Add(&autor) ;
    }
    frame = param.frame(frameCmdList) ;
  }
  
  // Filter frame command from list and pass on to plotOn() 
  pc.stripCmdList(cmdList,"FrameBins,FrameRange,FrameArgs") ;

  return frame ;
}


RooPlot* RooMCStudy::plotNLL(Double_t lo, Double_t hi, Int_t nBins) 
{
  // Create a RooPlot of the NLL distribution in the range lo-hi
  // with 'nBins' bins

  RooPlot* frame = _nllVar->frame(lo,hi,nBins) ;
  
  _fitParData->plotOn(frame) ;
  return frame ;
}



RooPlot* RooMCStudy::plotError(const RooRealVar& param, Double_t lo, Double_t hi, Int_t nbins) 
{
  // Create a RooPlot of the distribution of the fitted errors of the given parameter. 
  // The range lo-hi is plotted in nbins bins
  if (_canAddFitResults) {
    calcPulls() ;
    _canAddFitResults=kFALSE ;
  }

  RooErrorVar* evar = param.errorVar() ;
  RooPlot* frame = evar->frame(lo,hi,nbins) ;
  _fitParData->plotOn(frame) ;

  delete evar ;
  return frame ;
}



RooPlot* RooMCStudy::plotPull(const RooRealVar& param, Double_t lo, Double_t hi, Int_t nbins, Bool_t fitGauss) 
{
  // Create a RooPlot of the pull distribution for the given parameter.
  // The range lo-hi is plotted in nbins.
  // If fitGauss is set, an unbinned max. likelihood fit of the distribution to a Gaussian model 
  // is performed. The fit result is overlaid on the returned RooPlot and a box with the fitted
  // mean and sigma is added.

  if (_canAddFitResults) {
    calcPulls() ;
    _canAddFitResults=kFALSE ;
  }


  TString name(param.GetName()), title(param.GetTitle()) ;
  name.Append("pull") ; title.Append(" Pull") ;
  RooRealVar pvar(name,title,lo,hi) ;
  pvar.setBins(nbins) ;

  RooPlot* frame = pvar.frame() ;
  _fitParData->plotOn(frame) ;

  if (fitGauss) {
    RooRealVar pullMean("pullMean","Mean of pull",0,lo,hi) ;
    RooRealVar pullSigma("pullSigma","Width of pull",1,0,5) ;
    RooGenericPdf pullGauss("pullGauss","Gaussian of pull",
			    "exp(-0.5*(@0-@1)*(@0-@1)/(@2*@2))",
			    RooArgSet(pvar,pullMean,pullSigma)) ;
    pullGauss.fitTo(*_fitParData,"mh") ;
    pullGauss.plotOn(frame) ;
    pullGauss.paramOn(frame,_fitParData) ;
  }

  return frame ;
}



