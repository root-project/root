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
// RooMCStudy is a help class to facilitate Monte Carlo studies
// such as 'goodness-of-fit' studies, that involve fitting a PDF 
// to multiple toy Monte Carlo sets generated from the same PDF 
// or another PDF.
// <p>
// Given a fit PDF and a generator PDF, RooMCStudy can produce
// large numbers of toyMC samples and/or fit these samples
// and acculumate the final parameters of each fit in a dataset.
// <p>
// Additional plotting routines simplify the task of plotting
// the distribution of the minimized likelihood, each parameters fitted value, 
// fitted error and pull distribution.
// <p>
// Class RooMCStudy provides the option to insert add-in modules
// that modify the generate and fit cycle and allow to perform
// extra steps in the cycle. Output of these modules can be stored
// alongside the fit results in the aggregate results dataset.
// These study modules should derive from classs RooAbsMCStudyModel
//
// END_HTML
//



#include "RooFit.h"
#include "Riostream.h"

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
#include "RooMsgService.h"
#include "RooProdPdf.h"

using namespace std ;

ClassImp(RooMCStudy)
  ;


//_____________________________________________________________________________
RooMCStudy::RooMCStudy(const RooAbsPdf& model, const RooArgSet& observables,
   		       const RooCmdArg& arg1, const RooCmdArg& arg2,
   		       const RooCmdArg& arg3,const RooCmdArg& arg4,const RooCmdArg& arg5,
   		       const RooCmdArg& arg6,const RooCmdArg& arg7,const RooCmdArg& arg8) : TNamed("mcstudy","mcstudy")

{
  // Construct Monte Carlo Study Manager. This class automates generating data from a given PDF,
  // fitting the PDF to that data and accumulating the fit statistics.
  //
  // The constructor accepts the following arguments
  //
  // model       -- The PDF to be studied
  // observables -- The variables of the PDF to be considered the observables
  //
  // Silence()                         -- Suppress all RooFit messages during running below PROGRESS level
  // FitModel(const RooAbsPdf&)        -- The PDF for fitting, if it is different from the PDF for generating
  // ConditionalObservables
  //           (const RooArgSet& set)  -- The set of observables that the PDF should _not_ be normalized over
  // Binned(Bool_t flag)               -- Bin the dataset before fitting it. Speeds up fitting of large data samples
  // FitOptions(const char*)           -- Classic fit options, provided for backward compatibility
  // FitOptions(....)                  -- Options to be used for fitting. All named arguments inside FitOptions()
  //                                                   are passed to RooAbsPdf::fitTo();
  // Verbose(Bool_t flag)              -- Activate informational messages in event generation phase
  // Extended(Bool_t flag)             -- Determine number of events for each sample anew from a Poisson distribution
  // Constrain(const RooArgSet& pars)  -- Apply internal constraints on given parameters in fit and sample constrained parameter
  //                                      values from constraint p.d.f for each toy.
  // ExternalConstraints(const RooArgSet& ) -- Apply internal constraints on given parameters in fit and sample constrained parameter
  //                                      values from constraint p.d.f for each toy.
  // ProtoData(const RooDataSet&, 
  //                 Bool_t randOrder) -- Prototype data for the event generation. If the randOrder flag is
  //                                      set, the order of the dataset will be re-randomized for each generation
  //                                      cycle to protect against systematic biases if the number of generated
  //                                      events does not exactly match the number of events in the prototype dataset
  //                                      at the cost of reduced precision
  //                                      with mu equal to the specified number of events

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
  pc.defineSet("cPars","Constrain",0,0) ;
  pc.defineSet("extCons","ExternalConstraints",0,0) ;
  pc.defineInt("silence","Silence",0,0) ;
  pc.defineInt("randProtoData","PrototypeData",0,0) ;
  pc.defineInt("verboseGen","Verbose",0,0) ;
  pc.defineInt("extendedGen","Extended",0,0) ;
  pc.defineInt("binGenData","Binned",0,0) ;
  pc.defineString("fitOpts","FitOptions",0,"") ;
  pc.defineInt("dummy","FitOptArgs",0,0) ;
  pc.defineMutex("FitOptions","FitOptArgs") ; // can have either classic or new-style fit options
  pc.defineMutex("Constrain","FitOptions") ; // constraints only work with new-style fit options
  pc.defineMutex("ExternalConstraints","FitOptions") ; // constraints only work with new-style fit options
  
  // Process and check varargs 
  pc.process(cmdList) ;
  if (!pc.ok(kTRUE)) {
    // WVE do something here
    throw std::string("RooMCStudy::RooMCStudy() Error in parsing arguments passed to contructor") ;
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
  _silence = pc.getInt("silence") ;
  _verboseGen = pc.getInt("verboseGen") ;
  _extendedGen = pc.getInt("extendedGen") ;
  _binGenData = pc.getInt("binGenData") ;
  _randProto = pc.getInt("randProtoData") ;

  // Process constraints specifications
  const RooArgSet* cParsTmp = pc.getSet("cPars") ;
  const RooArgSet* extCons = pc.getSet("extCons") ;

  RooArgSet* cPars = new RooArgSet ;
  if (cParsTmp) {
    cPars->add(*cParsTmp) ;
  }
  
  // If constraints are specified, add to fit options
  if (cPars) {
    _fitOptList.Add(RooFit::Constrain(*cPars).Clone()) ;
  }
  if (extCons) {
    _fitOptList.Add(RooFit::ExternalConstraints(*extCons).Clone()) ;
  }

  // Make list of all constraints
  RooArgSet allConstraints ;
  RooArgSet consPars ;
  if (cPars) {
    RooArgSet* constraints = model.getAllConstraints(observables,*cPars,kTRUE) ;
    if (constraints) {
      allConstraints.add(*constraints) ;
      delete constraints ;
    }
  }
  
  // Construct constraint p.d.f
  if (allConstraints.getSize()>0) {
    _constrPdf = new RooProdPdf("mcs_constr_prod","RooMCStudy constraints product",allConstraints) ;

    if (cPars) {
      consPars.add(*cPars) ;
    } else {
      RooArgSet* params = model.getParameters(observables) ;
      RooArgSet* cparams = _constrPdf->getObservables(*params) ;
      consPars.add(*cparams) ;
      delete params ;
      delete cparams ;
    }
    _constrGenContext = _constrPdf->genContext(consPars,0,0,_verboseGen) ;

    _perExptGenParams = kTRUE ;

    coutI(Generation) << "RooMCStudy::RooMCStudy: INFO have pdf with constraints, will generate paramaters from constraint pdf for each experiment" << endl ;


  } else {
    _constrPdf = 0 ;
    _constrGenContext=0 ;

    _perExptGenParams = kFALSE ;
  }

  
  // Extract generator and fit models
  _genModel = const_cast<RooAbsPdf*>(&model) ;
  _genSample = 0 ;
  RooAbsPdf* fitModel = static_cast<RooAbsPdf*>(pc.getObject("fitModel",0)) ;
  _fitModel = fitModel ? fitModel : _genModel ;
  
  // Extract conditional observables and prototype data
  _genProtoData = static_cast<RooDataSet*>(pc.getObject("protoData",0)) ;
  if (pc.getObject("condObs",0)) {
    _projDeps.add(static_cast<RooArgSet&>(*pc.getObject("condObs",0))) ;
  }
  
  _dependents.add(observables) ;
     
  _allDependents.add(_dependents) ;  
  _fitOptions = pc.getString("fitOpts") ;
  _canAddFitResults = kTRUE ;
  
  if (_extendedGen && _genProtoData && !_randProto) {
    oocoutW(_fitModel,Generation) << "RooMCStudy::RooMCStudy: WARNING Using generator option 'e' (Poisson distribution of #events) together " << endl
				  << "                        with a prototype dataset implies incomplete sampling or oversampling of proto data." << endl
				  << "                        Use option \"r\" to randomize prototype dataset order and thus to randomize" << endl
				  << "                        the set of over/undersampled prototype events for each generation cycle." << endl ;
  }
  
  _genParams = _genModel->getParameters(&_dependents) ;
  if (!_binGenData) {
    _genContext = _genModel->genContext(_dependents,_genProtoData,0,_verboseGen) ;
    _genContext->attach(*_genParams) ;
  } else {
    _genContext = 0 ;
  }

  _genInitParams = (RooArgSet*) _genParams->snapshot(kFALSE) ;

  // Store list of parameters and save initial values separately
  _fitParams = _fitModel->getParameters(&_dependents) ;
  _fitInitParams = (RooArgSet*) _fitParams->snapshot(kTRUE) ;
  
  _nExpGen = _extendedGen ? _genModel->expectedEvents(&_dependents) : 0 ;
  
  // Place holder for NLL
  _nllVar = new RooRealVar("NLL","-log(Likelihood)",0) ;

  // Place holder for number of generated events
  _ngenVar = new RooRealVar("ngen","number of generated events",0) ;
  
  // Create data set containing parameter values, errors and pulls
  RooArgSet tmp2(*_fitParams) ;
  tmp2.add(*_nllVar) ;
  tmp2.add(*_ngenVar) ;

  // Mark all variable to store their errors in the dataset
  tmp2.setAttribAll("StoreError",kTRUE) ;
  tmp2.setAttribAll("StoreAsymError",kTRUE) ;
  TString fpdName ;
  if (_fitModel==_genModel) {
    fpdName = Form("fitParData_%s",_fitModel->GetName()) ;
  } else {
    fpdName= Form("fitParData_%s_%s",_fitModel->GetName(),_genModel->GetName()) ;
  }

  _fitParData = new RooDataSet(fpdName.Data(),"Fit Parameters DataSet",tmp2) ;
  tmp2.setAttribAll("StoreError",kFALSE) ;
  tmp2.setAttribAll("StoreAsymError",kFALSE) ;

  if (_perExptGenParams) {
    _genParData = new RooDataSet("genParData","Generated Parameters dataset",*_genParams) ;
  } else {
    _genParData = 0 ;
  }
  
  // Append proto variables to allDependents
  if (_genProtoData) {
    _allDependents.add(*_genProtoData->get(),kTRUE) ;
  }

  // Call module initializers
  list<RooAbsMCStudyModule*>::iterator iter ;
  for (iter=_modList.begin() ; iter!= _modList.end() ; ++iter) {
    Bool_t ok = (*iter)->doInitializeInstance(*this) ;
    if (!ok) {
      oocoutE(_fitModel,Generation) << "RooMCStudy::ctor: removing study module " << (*iter)->GetName() << " from analysis chain because initialization failed" << endl ;
      iter = _modList.erase(iter) ;
    }
  }
  
}


//_____________________________________________________________________________
RooMCStudy::RooMCStudy(const RooAbsPdf& genModel, const RooAbsPdf& fitModel, 
   		       const RooArgSet& dependents, const char* genOptions, 
   		       const char* fitOptions, const RooDataSet* genProtoData, 
   		       const RooArgSet& projDeps) :
  TNamed("mcstudy","mcstudy"),
  _genModel((RooAbsPdf*)&genModel), 
  _genProtoData(genProtoData),
  _projDeps(projDeps),
  _constrPdf(0),
  _constrGenContext(0),
  _dependents(dependents), 
  _allDependents(dependents), 
  _fitModel((RooAbsPdf*)&fitModel), 
  _nllVar(0),
  _ngenVar(0),
  _genParData(0),
  _fitOptions(fitOptions),
  _canAddFitResults(kTRUE),
  _perExptGenParams(0),
  _silence(kFALSE)
{
  // OBSOLETE, RETAINED FOR BACKWARD COMPATIBILY. PLEASE
  // USE CONSTRUCTOR WITH NAMED ARGUMENTS
  //
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
    oocoutE(_fitModel,Generation) << "RooMCStudy::RooMCStudy: WARNING Using generator option 'e' (Poisson distribution of #events) together " << endl
				  << "                        with a prototype dataset implies incomplete sampling or oversampling of proto data." << endl
				  << "                        Use option \"r\" to randomize prototype dataset order and thus to randomize" << endl
				  << "                        the set of over/undersampled prototype events for each generation cycle." << endl ;
  }
  
  if (!_binGenData) {
    _genContext = genModel.genContext(dependents,genProtoData,0,_verboseGen) ;
  } else {
    _genContext = 0 ;
  }
  _genParams = _genModel->getParameters(&_dependents) ;
  _genSample = 0 ;
  RooArgSet* tmp = genModel.getParameters(&dependents) ;
  _genInitParams = (RooArgSet*) tmp->snapshot(kFALSE) ;
  delete tmp ;
  
  // Store list of parameters and save initial values separately
  _fitParams = fitModel.getParameters(&dependents) ;
  _fitInitParams = (RooArgSet*) _fitParams->snapshot(kTRUE) ;
  
  _nExpGen = _extendedGen ? genModel.expectedEvents(&dependents) : 0 ;
  
  // Place holder for NLL
  _nllVar = new RooRealVar("NLL","-log(Likelihood)",0) ;
  
  // Place holder for number of generated events
  _ngenVar = new RooRealVar("ngen","number of generated events",0) ;
  
  // Create data set containing parameter values, errors and pulls
  RooArgSet tmp2(*_fitParams) ;
  tmp2.add(*_nllVar) ;
  tmp2.add(*_ngenVar) ;
  
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
      oocoutE(_fitModel,Generation) << "RooMCStudy::ctor: removing study module " << (*iter)->GetName() << " from analysis chain because initialization failed" << endl ;
      iter = _modList.erase(iter) ;
    }
  }
  
}



//_____________________________________________________________________________
RooMCStudy::~RooMCStudy() 
{  
  // Destructor 
  
  _genDataList.Delete() ;
  _fitResList.Delete() ;
  _fitOptList.Delete() ;
  delete _ngenVar ;
  delete _fitParData ;
  delete _genParData ;
  delete _fitInitParams ;
  delete _fitParams ;
  delete _genInitParams ;
  delete _genParams ;
  delete _genContext ;
  delete _nllVar ;
  delete _constrPdf ;
  delete _constrGenContext ;
}



//_____________________________________________________________________________
void RooMCStudy::addModule(RooAbsMCStudyModule& module) 
{
  // Insert given RooMCStudy add-on module to the processing chain
  // of this MCStudy object

  module.doInitializeInstance(*this) ;
  _modList.push_back(&module) ;        
}



//_____________________________________________________________________________
Bool_t RooMCStudy::run(Bool_t doGenerate, Bool_t DoFit, Int_t nSamples, Int_t nEvtPerSample, Bool_t keepGenData, const char* asciiFilePat) 
{
  // Run engine method. Generate and/or fit, according to flags, 'nSamples' samples of 'nEvtPerSample' events.
  // If keepGenData is set, all generated data sets will be kept in memory and can be accessed
  // later via genData().
  //
  // When generating, data sets will be written out in ascii form if the pattern string is supplied
  // The pattern, which is a template for snprintf, should look something like "data/toymc_%04d.dat"
  // and should contain one integer field that encodes the sample serial number.
  //
  // When fitting only, data sets may optionally be read from ascii files, using the same file
  // pattern.
  //

  RooFit::MsgLevel oldLevel(RooFit::FATAL) ;
  if (_silence) {
    oldLevel = RooMsgService::instance().globalKillBelow() ;
    RooMsgService::instance().setGlobalKillBelow(RooFit::PROGRESS) ;
  }

  list<RooAbsMCStudyModule*>::iterator iter ;
  for (iter=_modList.begin() ; iter!= _modList.end() ; ++iter) {
    (*iter)->initializeRun(nSamples) ;
  }  
  
  Int_t prescale = nSamples>100 ? Int_t(nSamples/100) : 1 ;

  while(nSamples--) {
    
    if (nSamples%prescale==0) {
      oocoutP(_fitModel,Generation) << "RooMCStudy::run: " ;
      if (doGenerate) ooccoutI(_fitModel,Generation) << "Generating " ;
      if (doGenerate && DoFit) ooccoutI(_fitModel,Generation) << "and " ;
      if (DoFit) ooccoutI(_fitModel,Generation) << "fitting " ;
      ooccoutP(_fitModel,Generation) << "sample " << nSamples << endl ;
    }

    _genSample = 0;
    Bool_t existingData = kFALSE ;
    if (doGenerate) {
      // Generate sample
      Int_t nEvt(nEvtPerSample) ;

      // Reset generator parameters to initial values
      *_genParams = *_genInitParams ;

      // If constraints are present, sample generator values from constraints
      if (_constrPdf) {
	RooDataSet* tmp = _constrGenContext->generate(1) ;
	*_genParams = *tmp->get() ;
	delete tmp ;
      }

      // Save generated parameters if required
      if (_genParData) {
	_genParData->add(*_genParams) ;
      }

      // Call module before-generation hook
      list<RooAbsMCStudyModule*>::iterator iter2 ;
      for (iter2=_modList.begin() ; iter2!= _modList.end() ; ++iter2) {
	(*iter2)->processBeforeGen(nSamples) ;
      }  

      if (_binGenData) {

	// Calculate the number of (extended) events for this run
	if (_extendedGen) {
	  _nExpGen = _genModel->expectedEvents(&_dependents) ;
	  nEvt = RooRandom::randomGenerator()->Poisson(nEvtPerSample==0?_nExpGen:nEvtPerSample) ;
	}	

	// Binned generation
	_genSample = _genModel->generateBinned(_dependents,nEvt) ;

      } else {

	// Calculate the number of (extended) events for this run
	if (_extendedGen) {
	  _nExpGen = _genModel->expectedEvents(&_dependents) ;
	  nEvt = RooRandom::randomGenerator()->Poisson(nEvtPerSample==0?_nExpGen:nEvtPerSample) ;
	}
	
	// Optional randomization of protodata for this run
	if (_randProto && _genProtoData && _genProtoData->numEntries()!=nEvt) {
	  oocoutI(_fitModel,Generation) << "RooMCStudy: (Re)randomizing event order in prototype dataset (Nevt=" << nEvt << ")" << endl ;
	  Int_t* newOrder = _genModel->randomizeProtoOrder(_genProtoData->numEntries(),nEvt) ;
	  _genContext->setProtoDataOrder(newOrder) ;
	  delete[] newOrder ;
	}
	
	// Actual generation of events
	if (nEvt>0) {
	  _genSample = _genContext->generate(nEvt) ;
	} else {
	  // Make empty dataset
	  _genSample = new RooDataSet("emptySample","emptySample",_dependents) ;
	}	
      } 

	
    //} else if (asciiFilePat && &asciiFilePat) { //warning: the address of 'asciiFilePat' will always evaluate as 'true'
    } else if (asciiFilePat) {

      // Load sample from ASCII file
      char asciiFile[1024] ;
      snprintf(asciiFile,1024,asciiFilePat,nSamples) ;
      RooArgList depList(_allDependents) ;
      _genSample = RooDataSet::read(asciiFile,depList,"q") ;      
      
    } else {
      
      // Load sample from internal list
      _genSample = (RooDataSet*) _genDataList.At(nSamples) ;
      existingData = kTRUE ;
      if (!_genSample) {
   	oocoutW(_fitModel,Generation) << "RooMCStudy::run: WARNING: Sample #" << nSamples << " not loaded, skipping" << endl ;
   	continue ;
      }
    }

    // Save number of generated events
    _ngenVar->setVal(_genSample->sumEntries()) ;

    // Call module between generation and fitting hook
    list<RooAbsMCStudyModule*>::iterator iter3 ;
    for (iter3=_modList.begin() ; iter3!= _modList.end() ; ++iter3) {
      (*iter3)->processBetweenGenAndFit(nSamples) ;
    }  
    
    if (DoFit) fitSample(_genSample) ;

    // Call module between generation and fitting hook
    for (iter3=_modList.begin() ; iter3!= _modList.end() ; ++iter3) {
      (*iter3)->processAfterFit(nSamples) ;
    }  
    
    // Optionally write to ascii file
    if (doGenerate && asciiFilePat && *asciiFilePat) {
      char asciiFile[1024] ;
      snprintf(asciiFile,1024,asciiFilePat,nSamples) ;
      RooDataSet* unbinnedData = dynamic_cast<RooDataSet*>(_genSample) ;
      if (unbinnedData) {
	unbinnedData->write(asciiFile) ;
      } else {
	coutE(InputArguments) << "RooMCStudy::run(" << GetName() << ") ERROR: ASCII writing of binned datasets is not supported" << endl ;
      }
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

  if (_genParData) {
    const RooArgSet* genPars = _genParData->get() ;
    TIterator* iter2 = genPars->createIterator() ;
    RooAbsArg* arg ;
    while((arg=(RooAbsArg*)iter2->Next())) {
      _genParData->changeObservableName(arg->GetName(),Form("%s_gen",arg->GetName())) ;
    }
    delete iter2 ;
    
    _fitParData->merge(_genParData) ;
  }

  if (DoFit) calcPulls() ;

  if (_silence) {
    RooMsgService::instance().setGlobalKillBelow(oldLevel) ;
  }

  return kFALSE ;
}






//_____________________________________________________________________________
Bool_t RooMCStudy::generateAndFit(Int_t nSamples, Int_t nEvtPerSample, Bool_t keepGenData, const char* asciiFilePat) 
{
  // Generate and fit 'nSamples' samples of 'nEvtPerSample' events.
  // If keepGenData is set, all generated data sets will be kept in memory and can be accessed
  // later via genData().
  //
  // Data sets will be written out is ascii form if the pattern string is supplied.
  // The pattern, which is a template for snprintf, should look something like "data/toymc_%04d.dat"
  // and should contain one integer field that encodes the sample serial number.
  //
  
  // Clear any previous data in memory
  _fitResList.Delete() ;
  _genDataList.Delete() ;
  _fitParData->reset() ;
  
  return run(kTRUE,kTRUE,nSamples,nEvtPerSample,keepGenData,asciiFilePat) ;
}



//_____________________________________________________________________________
Bool_t RooMCStudy::generate(Int_t nSamples, Int_t nEvtPerSample, Bool_t keepGenData, const char* asciiFilePat) 
{
  // Generate 'nSamples' samples of 'nEvtPerSample' events.
  // If keepGenData is set, all generated data sets will be kept in memory 
  // and can be accessed later via genData().
  //
  // Data sets will be written out in ascii form if the pattern string is supplied.
  // The pattern, which is a template for snprintf, should look something like "data/toymc_%04d.dat"
  // and should contain one integer field that encodes the sample serial number.
  //
  
  // Clear any previous data in memory
  _genDataList.Delete() ;
  
  return run(kTRUE,kFALSE,nSamples,nEvtPerSample,keepGenData,asciiFilePat) ;
}



//_____________________________________________________________________________
Bool_t RooMCStudy::fit(Int_t nSamples, const char* asciiFilePat) 
{
  // Fit 'nSamples' datasets, which are read from ASCII files.
  //
  // The ascii file pattern, which is a template for snprintf, should look something like "data/toymc_%04d.dat"
  // and should contain one integer field that encodes the sample serial number.
  //
  
  // Clear any previous data in memory
  _fitResList.Delete() ;
  _fitParData->reset() ;
  
  return run(kFALSE,kTRUE,nSamples,0,kFALSE,asciiFilePat) ;
}



//_____________________________________________________________________________
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



//_____________________________________________________________________________
void RooMCStudy::resetFitParams()
{
  // Reset all fit parameters to the initial model
  // parameters at the time of the RooMCStudy constructor

  *_fitParams = *_fitInitParams ;
}



//_____________________________________________________________________________
RooFitResult* RooMCStudy::doFit(RooAbsData* genSample)
{
  // Internal function. Performs actual fit according to specifications

  // Fit model to data set
  TString fitOpt2(_fitOptions) ; fitOpt2.Append("r") ;
  if (_silence) {
    fitOpt2.Append("b") ;
  }
  
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
      fr = (RooFitResult*) _fitModel->fitTo(*data,RooFit::ConditionalObservables(_projDeps),RooFit::FitOptions(fitOpt2)) ;
    } else {
      fr = (RooFitResult*) _fitModel->fitTo(*data,RooFit::FitOptions(fitOpt2)) ;
    }
  } else {
    RooCmdArg save  = RooFit::Save() ;
    RooCmdArg condo = RooFit::ConditionalObservables(_projDeps) ;
    RooCmdArg plevel = RooFit::PrintLevel(-1) ;
    RooLinkedList fitOptList(_fitOptList) ;
    fitOptList.Add(&save) ;
    if (_projDeps.getSize()>0) {
      fitOptList.Add(&condo) ;
    }
    if (_silence) {
      fitOptList.Add(&plevel) ;
    }
    fr = (RooFitResult*) _fitModel->fitTo(*data,fitOptList) ;
  }

  if (_binGenData) delete data ;

  return fr ;
}



//_____________________________________________________________________________
RooFitResult* RooMCStudy::refit(RooAbsData* genSample) 
{
  // Redo fit on 'current' toy sample, or if genSample is not NULL
  // do fit on given sample instead

  if (!genSample) {
    genSample = _genSample ;
  }

  RooFitResult* fr(0) ;
  if (genSample->sumEntries()>0) {
    fr = doFit(genSample) ;
  }

  return fr ;
}



//_____________________________________________________________________________
Bool_t RooMCStudy::fitSample(RooAbsData* genSample) 
{  
  // Internal method. Fit given dataset with fit model. If fit
  // converges (TMinuit status code zero) The fit results are appended
  // to the fit results dataset
  //
  // If the fit option "r" is supplied, the RooFitResult
  // objects will always be saved, regardless of the
  // fit status. RooFitResults objects can be retrieved
  // later via fitResult().
  //  
  
  // Reset all fit parameters to their initial values  
  resetFitParams() ;

  // Perform actual fit
  Bool_t ok ;
  RooFitResult* fr(0) ;
  if (genSample->sumEntries()>0) {
    fr = doFit(genSample) ;
    ok = (fr->status()==0) ;
  } else {
    ok = kFALSE ;
  }

  // If fit converged, store parameters and NLL
  if (ok) {
    _nllVar->setVal(fr->minNll()) ;
    RooArgSet tmp(*_fitParams) ;
    tmp.add(*_nllVar) ;
    tmp.add(*_ngenVar) ;
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



//_____________________________________________________________________________
Bool_t RooMCStudy::addFitResult(const RooFitResult& fr) 
{  
  // Utility function to add fit result from external fit to this RooMCStudy
  // and process its results through the standard RooMCStudy statistics gathering tools.
  // This function allows users to run the toy MC generation and/or fitting
  // in a distributed way and to collect and analyze the results in a RooMCStudy
  // as if they were run locally.
  //
  // This method is only functional if this RooMCStudy object is cleanm, i.e. it was not used
  // to generate and/or fit any samples.

  if (!_canAddFitResults) {
    oocoutE(_fitModel,InputArguments) << "RooMCStudy::addFitResult: ERROR cannot add fit results in current state" << endl ;
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
    tmp.add(*_ngenVar) ;
    _fitParData->add(tmp) ;
  }
  
  // Store fit result if requested by user
  if (_fitOptions.Contains("r")) {
    _fitResList.Add((TObject*)&fr) ;
  }  
  
  return kFALSE ;
}



//_____________________________________________________________________________
void RooMCStudy::calcPulls() 
{
  // Calculate the pulls for all fit parameters in
  // the fit results data set, and add them to that dataset
  
  TIterator* iter = _fitParams->createIterator()  ;
  RooRealVar* par ;
  while((par=(RooRealVar*)iter->Next())) {
    
    RooErrorVar* err = par->errorVar() ;
    _fitParData->addColumn(*err) ;
    delete err ;
    
    TString name(par->GetName()), title(par->GetTitle()) ;
    name.Append("pull") ;
    title.Append(" Pull") ;    

    // First look in fitParDataset to see if per-experiment generated value has been stored
    RooAbsReal* genParOrig = (RooAbsReal*) _fitParData->get()->find(Form("%s_gen",par->GetName())) ;    
    if (genParOrig && _perExptGenParams) {

      RooPullVar pull(name,title,*par,*genParOrig) ;
      _fitParData->addColumn(pull,kFALSE) ;

    } else {
      // If not use fixed generator value
      genParOrig = (RooAbsReal*)_genInitParams->find(par->GetName()) ;
      
      if (genParOrig) {
	RooAbsReal* genPar = (RooAbsReal*) genParOrig->Clone("truth") ;
	RooPullVar pull(name,title,*par,*genPar) ;
	
	_fitParData->addColumn(pull,kFALSE) ;
	delete genPar ;
	
      }

    }

  }
  delete iter ;
  
}




//_____________________________________________________________________________
const RooDataSet& RooMCStudy::fitParDataSet()
{
  // Return a RooDataSet the resulting fit parameters of each toy cycle.
  // This dataset also contains any additional output that was generated
  // by study modules that were added to this RooMCStudy

  if (_canAddFitResults) {
    calcPulls() ;  
    _canAddFitResults = kFALSE ; 
  }
  
  return *_fitParData ;
}



//_____________________________________________________________________________
const RooArgSet* RooMCStudy::fitParams(Int_t sampleNum) const 
{
  // Return an argset with the fit parameters for the given sample number
  //
  // NB: The fit parameters are only stored for successfull fits,
  //     thus the maximum sampleNum can be less that the number
  //     of generated samples and if so, the indeces will
  //     be out of synch with genData() and fitResult()
  
  // Check if sampleNum is in range
  if (sampleNum<0 || sampleNum>=_fitParData->numEntries()) {
    oocoutE(_fitModel,InputArguments) << "RooMCStudy::fitParams: ERROR, invalid sample number: " << sampleNum << endl ;    
    return 0 ;
  }
  
  return _fitParData->get(sampleNum) ;
}



//_____________________________________________________________________________
const RooFitResult* RooMCStudy::fitResult(Int_t sampleNum) const
{
  // Return the RooFitResult object of the fit to given sample 
  
  // Check if sampleNum is in range
  if (sampleNum<0 || sampleNum>=_fitResList.GetSize()) {
    oocoutE(_fitModel,InputArguments) << "RooMCStudy::fitResult: ERROR, invalid sample number: " << sampleNum << endl ;    
    return 0 ;
  }
  
  // Retrieve fit result object
  const RooFitResult* fr = (RooFitResult*) _fitResList.At(sampleNum) ;
  if (fr) {
    return fr ;
  } else {
    oocoutE(_fitModel,InputArguments) << "RooMCStudy::fitResult: ERROR, no fit result saved for sample " 
			  << sampleNum << ", did you use the 'r; fit option?" << endl ;
  }
  return 0 ;
}



//_____________________________________________________________________________
const RooAbsData* RooMCStudy::genData(Int_t sampleNum) const 
{
  // Return the given generated dataset. This method will only return datasets
  // if during the run cycle it was indicated that generator data should be saved.
  
  // Check that generated data was saved
  if (_genDataList.GetSize()==0) {
    oocoutE(_fitModel,InputArguments) << "RooMCStudy::genData() ERROR, generated data was not saved" << endl ;
    return 0 ;
  }
  
  // Check if sampleNum is in range
  if (sampleNum<0 || sampleNum>=_genDataList.GetSize()) {
    oocoutE(_fitModel,InputArguments) << "RooMCStudy::genData() ERROR, invalid sample number: " << sampleNum << endl ;    
    return 0 ;
  }
  
  return  (RooAbsData*) _genDataList.At(sampleNum) ;
}



//_____________________________________________________________________________
RooPlot* RooMCStudy::plotParamOn(RooPlot* frame, const RooCmdArg& arg1, const RooCmdArg& arg2, const RooCmdArg& arg3, const RooCmdArg& arg4, 
   				 const RooCmdArg& arg5, const RooCmdArg& arg6, const RooCmdArg& arg7, const RooCmdArg& arg8) 
{
  // Plot the distribution of fitted values of a parameter. The parameter shown is the one from which the RooPlot
  // was created, e.g.
  //
  // RooPlot* frame = param.frame(100,-10,10) ;
  // mcstudy.paramOn(frame,LineStyle(kDashed)) ;
  // 
  // Any named arguments passed to plotParamOn() are forwarded to the underlying plotOn() call
  
  _fitParData->plotOn(frame,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8) ;
  return frame ;
}



//_____________________________________________________________________________
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
    oocoutE(_fitModel,InputArguments) << "RooMCStudy::plotParam: ERROR: no parameter defined with name " << paramName << endl ;  
    return 0 ;
  }

  // Forward to implementation below
  return plotParam(*param,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8) ;
}



//_____________________________________________________________________________
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



//_____________________________________________________________________________
RooPlot* RooMCStudy::plotNLL(const RooCmdArg& arg1, const RooCmdArg& arg2,
                     const RooCmdArg& arg3, const RooCmdArg& arg4,
                     const RooCmdArg& arg5, const RooCmdArg& arg6,
                     const RooCmdArg& arg7, const RooCmdArg& arg8) 
{
  // Plot the distribution of the -log(L) values on a newly created frame.
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



//_____________________________________________________________________________
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



//_____________________________________________________________________________
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
      RooRealVar pullMean("pullMean","Mean of pull",0,-10,10) ;
      RooRealVar pullSigma("pullSigma","Width of pull",1,0.1,5) ;
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



//_____________________________________________________________________________
RooPlot* RooMCStudy::makeFrameAndPlotCmd(const RooRealVar& param, RooLinkedList& cmdList, Bool_t symRange) const 
{
  // Internal function. Construct RooPlot from given parameter and modify the list of named
  // arguments 'cmdList' to only contain the plot arguments that should be forwarded to 
  // RooAbsData::plotOn()

  // Select the frame-specific commands 
  RooCmdConfig pc(Form("RooMCStudy::plotParam(%s)",_genModel->GetName())) ;
  pc.defineInt("nbins","Bins",0,0) ;
  pc.defineDouble("xlo","Range",0,0) ;
  pc.defineDouble("xhi","Range",1,0) ;
  pc.defineInt("dummy","FrameArgs",0,0) ;
  pc.defineMutex("Bins","FrameArgs") ;
  pc.defineMutex("Range","FrameArgs") ;

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

    if (pc.hasProcessed("Bins")) frameCmdList.Add(&bins) ;
    if (pc.hasProcessed("Range")) {
      frameCmdList.Add(&range) ;
    } else {
      frameCmdList.Add(&autor) ;
    }
    frame = param.frame(frameCmdList) ;
  }
  
  // Filter frame command from list and pass on to plotOn() 
  pc.stripCmdList(cmdList,"FrameArgs,Bins,Range") ;

  return frame ;
}



//_____________________________________________________________________________
RooPlot* RooMCStudy::plotNLL(Double_t lo, Double_t hi, Int_t nBins) 
{
  // Create a RooPlot of the -log(L) distribution in the range lo-hi
  // with 'nBins' bins

  RooPlot* frame = _nllVar->frame(lo,hi,nBins) ;
  
  _fitParData->plotOn(frame) ;
  return frame ;
}



//_____________________________________________________________________________
RooPlot* RooMCStudy::plotError(const RooRealVar& param, Double_t lo, Double_t hi, Int_t nbins) 
{
  // Create a RooPlot of the distribution of the fitted errors of the given parameter. 
  // The frame is created with a range [lo,hi] and plotted data will be binned in 'nbins' bins

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



//_____________________________________________________________________________
RooPlot* RooMCStudy::plotPull(const RooRealVar& param, Double_t lo, Double_t hi, Int_t nbins, Bool_t fitGauss) 
{
  // Create a RooPlot of the pull distribution for the given
  // parameter.  The range lo-hi is plotted in nbins.  If fitGauss is
  // set, an unbinned ML fit of the distribution to a Gaussian p.d.f
  // is performed. The fit result is overlaid on the returned RooPlot
  // and a box with the fitted mean and sigma is added.

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



