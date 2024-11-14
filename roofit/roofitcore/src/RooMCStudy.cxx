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
\file RooMCStudy.cxx
\class RooMCStudy
\ingroup Roofitcore

Helper class to facilitate Monte Carlo studies
such as 'goodness-of-fit' studies, that involve fitting a PDF
to multiple toy Monte Carlo sets. These may be generated from either same PDF
or from a different PDF with similar parameters.

Given a fit and a generator PDF (they might be identical), RooMCStudy can produce
toyMC samples and/or fit these.
It accumulates the post-fit parameters of each iteration in a dataset. These can be
retrieved using fitParams() or fitParDataSet(). This dataset additionally contains the
variables
- NLL: The value of the negative log-likelihood for each run.
- ngen: The number of events generated for each run.

Additional plotting routines simplify the task of plotting
the distribution of the minimized likelihood, the fitted parameter values,
fitted error and pull distribution.

RooMCStudy provides the option to insert add-in modules
that modify the generate-and-fit cycle and allow to perform
extra steps in the cycle. Output of these modules can be stored
alongside the fit results in the aggregate results dataset.
These study modules should derive from the class RooAbsMCStudyModule.

Check the RooFit tutorials
- rf801_mcstudy.C
- rf802_mcstudy_addons.C
- rf803_mcstudy_addons2.C
- rf804_mcstudy_constr.C
for usage examples.
**/


#include <RooMCStudy.h>

#include <RooAbsMCStudyModule.h>
#include <RooAbsPdf.h>
#include <RooArgList.h>
#include <RooCmdConfig.h>
#include <RooDataHist.h>
#include <RooDataSet.h>
#include <RooErrorVar.h>
#include <RooFitResult.h>
#include <RooFormulaVar.h>
#include <RooGenContext.h>
#include <RooGlobalFunc.h>
#include <RooMsgService.h>
#include <RooPlot.h>
#include <RooProdPdf.h>
#include <RooPullVar.h>
#include <RooRandom.h>
#include <RooRealVar.h>
#include <RooWorkspace.h>

#include <snprintf.h>
#include <iostream>

ClassImp(RooMCStudy);

/**
Construct Monte Carlo Study Manager. This class automates generating data from a given PDF,
fitting the PDF to data and accumulating the fit statistics.

\param[in] model The PDF to be studied
\param[in] observables The variables of the PDF to be considered observables
\param[in] arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8 Optional arguments according to table below.

<table>
<tr><th> Optional arguments <th>
<tr><td> Silence()                         <td> Suppress all RooFit messages during running below PROGRESS level
<tr><td> FitModel(const RooAbsPdf&)        <td> The PDF for fitting if it is different from the PDF for generating.
<tr><td> ConditionalObservables(const RooArgSet& set)  <td> The set of observables that the PDF should _not_ be normalized over
<tr><td> Binned(bool flag)               <td> Bin the dataset before fitting it. Speeds up fitting of large data samples
<tr><td> FitOptions(....)                  <td> Options to be used for fitting. All named arguments inside FitOptions() are passed to RooAbsPdf::fitTo().
                                                `Save()` is especially interesting to be able to retrieve fit results of each run using fitResult().
<tr><td> Verbose(bool flag)              <td> Activate informational messages in event generation phase
<tr><td> Extended(bool flag)             <td> Determine number of events for each sample anew from a Poisson distribution
<tr><td> Constrain(const RooArgSet& pars)  <td> Apply internal constraints on given parameters in fit and sample constrained parameter values from constraint p.d.f for each toy.
<tr><td> ProtoData(const RooDataSet&, bool randOrder)
         <td> Prototype data for the event generation. If the randOrder flag is set, the order of the dataset will be re-randomized for each generation
              cycle to protect against systematic biases if the number of generated events does not exactly match the number of events in the prototype dataset
              at the cost of reduced precision with mu equal to the specified number of events
</table>
*/
RooMCStudy::RooMCStudy(const RooAbsPdf& model, const RooArgSet& observables,
                const RooCmdArg& arg1, const RooCmdArg& arg2,
                const RooCmdArg& arg3,const RooCmdArg& arg4,const RooCmdArg& arg5,
                const RooCmdArg& arg6,const RooCmdArg& arg7,const RooCmdArg& arg8) : TNamed("mcstudy","mcstudy")

{
  // Stuff all arguments in a list
  RooLinkedList cmdList;
  cmdList.Add(const_cast<RooCmdArg*>(&arg1)) ;  cmdList.Add(const_cast<RooCmdArg*>(&arg2)) ;
  cmdList.Add(const_cast<RooCmdArg*>(&arg3)) ;  cmdList.Add(const_cast<RooCmdArg*>(&arg4)) ;
  cmdList.Add(const_cast<RooCmdArg*>(&arg5)) ;  cmdList.Add(const_cast<RooCmdArg*>(&arg6)) ;
  cmdList.Add(const_cast<RooCmdArg*>(&arg7)) ;  cmdList.Add(const_cast<RooCmdArg*>(&arg8)) ;

  // Select the pdf-specific commands
  RooCmdConfig pc("RooMCStudy::RooMCStudy(" + std::string(model.GetName()) + ")");

  pc.defineObject("fitModel","FitModel",0,nullptr) ;
  pc.defineSet("condObs","ProjectedObservables",0,nullptr) ;
  pc.defineObject("protoData","PrototypeData",0,nullptr) ;
  pc.defineSet("cPars","Constrain",0,nullptr) ;
  pc.defineSet("extCons","ExternalConstraints",0,nullptr) ;
  pc.defineInt("silence","Silence",0,0) ;
  pc.defineInt("randProtoData","PrototypeData",0,0) ;
  pc.defineInt("verboseGen","Verbose",0,0) ;
  pc.defineInt("extendedGen","Extended",0,0) ;
  pc.defineInt("binGenData","Binned",0,0) ;
  pc.defineInt("dummy","FitOptArgs",0,0) ;

  // Process and check varargs
  pc.process(cmdList) ;
  if (!pc.ok(true)) {
    // WVE do something here
    throw std::string("RooMCStudy::RooMCStudy() Error in parsing arguments passed to constructor") ;
    return ;
  }

  // Save fit command options
  if (pc.hasProcessed("FitOptArgs")) {
    RooCmdArg* fitOptArg = static_cast<RooCmdArg*>(cmdList.FindObject("FitOptArgs")) ;
    for (int i=0 ; i<fitOptArg->subArgs().GetSize() ;i++) {
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

  auto cPars = std::make_unique<RooArgSet>();
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
    if (std::unique_ptr<RooArgSet> constraints{model.getAllConstraints(observables,*cPars,true)}) {
      allConstraints.add(*constraints) ;
    }
  }

  // Construct constraint p.d.f
  if (!allConstraints.empty()) {
    _constrPdf = std::make_unique<RooProdPdf>("mcs_constr_prod","RooMCStudy constraints product",allConstraints);

    if (cPars) {
      consPars.add(*cPars) ;
    } else {
      RooArgSet params;
      model.getParameters(&observables, params);
      RooArgSet cparams;
      _constrPdf->getObservables(&params, cparams);
      consPars.add(cparams) ;
    }
    _constrGenContext.reset(_constrPdf->genContext(consPars,nullptr,nullptr,_verboseGen));

    _perExptGenParams = true ;

    coutI(Generation) << "RooMCStudy::RooMCStudy: INFO have pdf with constraints, will generate parameters from constraint pdf for each experiment" << std::endl ;
  }


  // Extract generator and fit models
  _genModel = const_cast<RooAbsPdf*>(&model) ;
  RooAbsPdf* fitModel = static_cast<RooAbsPdf*>(pc.getObject("fitModel",nullptr)) ;
  _fitModel = fitModel ? fitModel : _genModel ;

  // Extract conditional observables and prototype data
  _genProtoData = static_cast<RooDataSet*>(pc.getObject("protoData",nullptr)) ;
  if (auto condObs = pc.getSet("condObs",nullptr)) {
    _projDeps.add(*condObs);
  }

  _dependents.add(observables) ;

  _allDependents.add(_dependents) ;
  _canAddFitResults = true ;

  if (_extendedGen && _genProtoData && !_randProto) {
    oocoutW(_fitModel,Generation) << "RooMCStudy::RooMCStudy: WARNING Using generator option 'e' (Poisson distribution of #events) together " << std::endl
              << "                        with a prototype dataset implies incomplete sampling or oversampling of proto data." << std::endl
              << "                        Use option \"r\" to randomize prototype dataset order and thus to randomize" << std::endl
              << "                        the set of over/undersampled prototype events for each generation cycle." << std::endl ;
  }

  _genModel->getParameters(&_dependents, _genParams);
  if (!_binGenData) {
    _genContext.reset(_genModel->genContext(_dependents,_genProtoData,nullptr,_verboseGen));
    _genContext->attach(_genParams) ;
  }

  _genParams.snapshot(_genInitParams, false) ;

  // Store list of parameters and save initial values separately
  _fitModel->getParameters(&_dependents, _fitParams);
  _fitParams.snapshot(_fitInitParams, true);

  _nExpGen = _extendedGen ? _genModel->expectedEvents(&_dependents) : 0 ;

  // Place holder for NLL
  _nllVar = std::make_unique<RooRealVar>("NLL","-log(Likelihood)",0);

  // Place holder for number of generated events
  _ngenVar = std::make_unique<RooRealVar>("ngen","number of generated events",0);

  // Create data set containing parameter values, errors and pulls
  RooArgSet tmp2(_fitParams) ;
  tmp2.add(*_nllVar) ;
  tmp2.add(*_ngenVar) ;

  // Mark all variable to store their errors in the dataset
  tmp2.setAttribAll("StoreError",true) ;
  tmp2.setAttribAll("StoreAsymError",true) ;
  std::string fpdName;
  if (_fitModel==_genModel) {
    fpdName = "fitParData_" + std::string(_fitModel->GetName());
  } else {
    fpdName= "fitParData_" + std::string(_fitModel->GetName()) + "_" + std::string(_genModel->GetName());
  }

  _fitParData = std::make_unique<RooDataSet>(fpdName,"Fit Parameters DataSet",tmp2);
  tmp2.setAttribAll("StoreError",false) ;
  tmp2.setAttribAll("StoreAsymError",false) ;

  if (_perExptGenParams) {
    _genParData = std::make_unique<RooDataSet>("genParData","Generated Parameters dataset",_genParams);
  }

  // Append proto variables to allDependents
  if (_genProtoData) {
    _allDependents.add(*_genProtoData->get(),true) ;
  }

  // Call module initializers
  for (auto iter=_modList.begin() ; iter!= _modList.end() ; ++iter) {
    bool ok = (*iter)->doInitializeInstance(*this) ;
    if (!ok) {
      oocoutE(_fitModel,Generation) << "RooMCStudy::ctor: removing study module " << (*iter)->GetName() << " from analysis chain because initialization failed" << std::endl ;
      iter = _modList.erase(iter) ;
    }
  }

}


////////////////////////////////////////////////////////////////////////////////

RooMCStudy::~RooMCStudy()
{
  _genDataList.Delete() ;
  _fitOptList.Delete() ;
}



////////////////////////////////////////////////////////////////////////////////
/// Insert given RooMCStudy add-on module to the processing chain
/// of this MCStudy object

void RooMCStudy::addModule(RooAbsMCStudyModule& module)
{
  module.doInitializeInstance(*this) ;
  _modList.push_back(&module) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Run engine method. Generate and/or fit, according to flags, 'nSamples' samples of 'nEvtPerSample' events.
/// If keepGenData is set, all generated data sets will be kept in memory and can be accessed
/// later via genData().
///
/// When generating, data sets will be written out in ascii form if the pattern string is supplied
/// The pattern, which is a template for snprintf, should look something like "data/toymc_%04d.dat"
/// and should contain one integer field that encodes the sample serial number.
///
/// When fitting only, data sets may optionally be read from ascii files, using the same file
/// pattern.
///

bool RooMCStudy::run(bool doGenerate, bool DoFit, Int_t nSamples, Int_t nEvtPerSample, bool keepGenData, const char* asciiFilePat)
{
  RooFit::MsgLevel oldLevel(RooFit::FATAL) ;
  if (_silence) {
    oldLevel = RooMsgService::instance().globalKillBelow() ;
    RooMsgService::instance().setGlobalKillBelow(RooFit::PROGRESS) ;
  }

  for (RooAbsMCStudyModule *mod : _modList) {
    mod->initializeRun(nSamples) ;
  }

  int prescale = nSamples>100 ? int(nSamples/100) : 1 ;

  while(nSamples--) {

    if (nSamples%prescale==0) {
      oocoutP(_fitModel,Generation) << "RooMCStudy::run: " ;
      if (doGenerate) ooccoutI(_fitModel,Generation) << "Generating " ;
      if (doGenerate && DoFit) ooccoutI(_fitModel,Generation) << "and " ;
      if (DoFit) ooccoutI(_fitModel,Generation) << "fitting " ;
      ooccoutP(_fitModel,Generation) << "sample " << nSamples << std::endl ;
    }

    std::unique_ptr<RooAbsData> ownedGenSample;
    _genSample = nullptr;
    bool existingData = false ;
    if (doGenerate) {
      // Generate sample
      int nEvt(nEvtPerSample) ;

      // Reset generator parameters to initial values
      _genParams.assign(_genInitParams) ;

      // If constraints are present, sample generator values from constraints
      if (_constrPdf) {
        _genParams.assign(*std::unique_ptr<RooDataSet>{_constrGenContext->generate(1)}->get());
      }

      // Save generated parameters if required
      if (_genParData) {
   _genParData->add(_genParams) ;
      }

      // Call module before-generation hook
      for (RooAbsMCStudyModule *mod : _modList) {
         mod->processBeforeGen(nSamples) ;
      }

      if (_binGenData) {

   // Calculate the number of (extended) events for this run
   if (_extendedGen) {
     _nExpGen = _genModel->expectedEvents(&_dependents) ;
     nEvt = RooRandom::randomGenerator()->Poisson(nEvtPerSample==0?_nExpGen:nEvtPerSample) ;
   }

   // Binned generation
   ownedGenSample = std::unique_ptr<RooDataHist>{_genModel->generateBinned(_dependents,nEvt)};

      } else {

   // Calculate the number of (extended) events for this run
   if (_extendedGen) {
     _nExpGen = _genModel->expectedEvents(&_dependents) ;
     nEvt = RooRandom::randomGenerator()->Poisson(nEvtPerSample==0?_nExpGen:nEvtPerSample) ;
   }

   // Optional randomization of protodata for this run
   if (_randProto && _genProtoData && _genProtoData->numEntries()!=nEvt) {
     oocoutI(_fitModel,Generation) << "RooMCStudy: (Re)randomizing event order in prototype dataset (Nevt=" << nEvt << ")" << std::endl ;
     Int_t* newOrder = _genModel->randomizeProtoOrder(_genProtoData->numEntries(),nEvt) ;
     _genContext->setProtoDataOrder(newOrder) ;
     delete[] newOrder ;
   }

   // Actual generation of events
   if (nEvt>0) {
     ownedGenSample = std::unique_ptr<RooAbsData>{_genContext->generate(nEvt)};
   } else {
     // Make empty dataset
     ownedGenSample = std::make_unique<RooDataSet>("emptySample","emptySample",_dependents);
   }
      }

   _genSample = ownedGenSample.get();

    //} else if (asciiFilePat && &asciiFilePat) { //warning: the address of 'asciiFilePat' will always evaluate as 'true'
    } else if (asciiFilePat) {

      // Load sample from ASCII file
      char asciiFile[1024] ;
      snprintf(asciiFile,1024,asciiFilePat,nSamples) ;
      RooArgList depList(_allDependents) ;
      ownedGenSample = std::unique_ptr<RooDataSet>{RooDataSet::read(asciiFile,depList,"q")};
      _genSample = ownedGenSample.get();

    } else {

      // Load sample from internal list
      _genSample = static_cast<RooDataSet*>(_genDataList.At(nSamples)) ;
      existingData = true ;
      if (!_genSample) {
      oocoutW(_fitModel,Generation) << "RooMCStudy::run: WARNING: Sample #" << nSamples << " not loaded, skipping" << std::endl ;
      continue ;
      }
    }

    // Save number of generated events
    _ngenVar->setVal(_genSample->sumEntries()) ;

    // Call module between generation and fitting hook
    for (RooAbsMCStudyModule *mod : _modList) {
      mod->processBetweenGenAndFit(nSamples) ;
    }

    if (DoFit) fitSample(_genSample) ;

    // Call module between generation and fitting hook
    for (RooAbsMCStudyModule *mod : _modList) {
      mod->processAfterFit(nSamples) ;
    }

    // Optionally write to ascii file
    if (doGenerate && asciiFilePat && *asciiFilePat) {
      char asciiFile[1024] ;
      snprintf(asciiFile,1024,asciiFilePat,nSamples) ;
      if (RooDataSet* unbinnedData = dynamic_cast<RooDataSet*>(_genSample)) {
   unbinnedData->write(asciiFile) ;
      } else {
   coutE(InputArguments) << "RooMCStudy::run(" << GetName() << ") ERROR: ASCII writing of binned datasets is not supported" << std::endl ;
      }
    }

    // Add to list or delete
    if (!existingData) {
      if (keepGenData) {
   _genDataList.Add(ownedGenSample.release()) ;
      }
    }
  }

  for (RooAbsMCStudyModule *mod : _modList) {
    if (RooDataSet* auxData = mod->finalizeRun()) {
      _fitParData->merge(auxData) ;
    }
  }

  _canAddFitResults = false ;

  if (_genParData) {
    for(RooAbsArg * arg : *_genParData->get()) {
      _genParData->changeObservableName(arg->GetName(),(std::string(arg->GetName()) + "_gen").c_str());
    }

    _fitParData->merge(_genParData.get());
  }

  if (DoFit) calcPulls() ;

  if (_silence) {
    RooMsgService::instance().setGlobalKillBelow(oldLevel) ;
  }

  return false ;
}






////////////////////////////////////////////////////////////////////////////////
/// Generate and fit 'nSamples' samples of 'nEvtPerSample' events.
/// If keepGenData is set, all generated data sets will be kept in memory and can be accessed
/// later via genData().
///
/// Data sets will be written out in ascii form if the pattern string is supplied.
/// The pattern, which is a template for snprintf, should look something like "data/toymc_%04d.dat"
/// and should contain one integer field that encodes the sample serial number.
///

bool RooMCStudy::generateAndFit(Int_t nSamples, Int_t nEvtPerSample, bool keepGenData, const char* asciiFilePat)
{
  // Clear any previous data in memory
  _fitResList.Delete() ; // even though the fit results are owned by gROOT, we still want to scratch them here.
  _genDataList.Delete() ;
  _fitParData->reset() ;

  return run(true,true,nSamples,nEvtPerSample,keepGenData,asciiFilePat) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Generate 'nSamples' samples of 'nEvtPerSample' events.
/// If keepGenData is set, all generated data sets will be kept in memory
/// and can be accessed later via genData().
///
/// Data sets will be written out in ascii form if the pattern string is supplied.
/// The pattern, which is a template for snprintf, should look something like "data/toymc_%04d.dat"
/// and should contain one integer field that encodes the sample serial number.
///

bool RooMCStudy::generate(Int_t nSamples, Int_t nEvtPerSample, bool keepGenData, const char* asciiFilePat)
{
  // Clear any previous data in memory
  _genDataList.Delete() ;

  return run(true,false,nSamples,nEvtPerSample,keepGenData,asciiFilePat) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Fit 'nSamples' datasets, which are read from ASCII files.
///
/// The ascii file pattern, which is a template for snprintf, should look something like "data/toymc_%04d.dat"
/// and should contain one integer field that encodes the sample serial number.
///

bool RooMCStudy::fit(Int_t nSamples, const char* asciiFilePat)
{
  // Clear any previous data in memory
  _fitResList.Delete() ; // even though the fit results are owned by gROOT, we still want to scratch them here.
  _fitParData->reset() ;

  return run(false,true,nSamples,0,false,asciiFilePat) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Fit 'nSamples' datasets, as supplied in 'dataSetList'
///

bool RooMCStudy::fit(Int_t nSamples, TList& dataSetList)
{
  // Clear any previous data in memory
  _fitResList.Delete() ; // even though the fit results are owned by gROOT, we still want to scratch them here.
  _genDataList.Delete() ;
  _fitParData->reset() ;

  // Load list of data sets
  for(auto * gset : static_range_cast<RooAbsData*>(dataSetList)) {
    _genDataList.Add(gset) ;
  }

  return run(false,true,nSamples,0,true,nullptr) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Reset all fit parameters to the initial model
/// parameters at the time of the RooMCStudy constructor

void RooMCStudy::resetFitParams()
{
  _fitParams.assign(_fitInitParams) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Internal function. Performs actual fit according to specifications

RooFit::OwningPtr<RooFitResult> RooMCStudy::doFit(RooAbsData* genSample)
{
  // Optionally bin dataset before fitting
  std::unique_ptr<RooDataHist> ownedDataHist;
  RooAbsData* data ;
  if (_binGenData) {
    RooArgSet depList;
    _fitModel->getObservables(genSample->get(), depList);
    ownedDataHist = std::make_unique<RooDataHist>(genSample->GetName(),genSample->GetTitle(),depList,*genSample) ;
    data = ownedDataHist.get();
  } else {
    data = genSample ;
  }

  RooCmdArg save  = RooFit::Save() ;
  RooCmdArg condo = RooFit::ConditionalObservables(_projDeps) ;
  RooCmdArg plevel = RooFit::PrintLevel(_silence ? -1 : 1) ;

  RooLinkedList fitOptList(_fitOptList) ;
  fitOptList.Add(&save) ;
  if (!_projDeps.empty()) {
    fitOptList.Add(&condo) ;
  }
  fitOptList.Add(&plevel) ;
  return _fitModel->fitTo(*data,fitOptList);
}



////////////////////////////////////////////////////////////////////////////////
/// Redo fit on 'current' toy sample, or if genSample is not nullptr
/// do fit on given sample instead

RooFit::OwningPtr<RooFitResult> RooMCStudy::refit(RooAbsData* genSample)
{
  if (!genSample) {
    genSample = _genSample ;
  }

  std::unique_ptr<RooFitResult> fr;
  if (genSample->sumEntries()>0) {
    fr = std::unique_ptr<RooFitResult>{doFit(genSample)};
  }

  return RooFit::makeOwningPtr(std::move(fr));
}



////////////////////////////////////////////////////////////////////////////////
/// Internal method. Fit given dataset with fit model. If fit
/// converges (TMinuit status code zero) The fit results are appended
/// to the fit results dataset
///
/// If the fit option "r" is supplied, the RooFitResult
/// objects will always be saved, regardless of the
/// fit status. RooFitResults objects can be retrieved
/// later via fitResult().
///

bool RooMCStudy::fitSample(RooAbsData* genSample)
{
  // Reset all fit parameters to their initial values
  resetFitParams() ;

  // Perform actual fit
  bool ok ;
  std::unique_ptr<RooFitResult> fr;
  if (genSample->sumEntries()>0) {
    fr = std::unique_ptr<RooFitResult>{doFit(genSample)};
    ok = (fr->status()==0) ;
  } else {
    ok = false ;
  }

  // If fit converged, store parameters and NLL
  if (ok) {
    _nllVar->setVal(fr->minNll()) ;
    RooArgSet tmp(_fitParams) ;
    tmp.add(*_nllVar) ;
    tmp.add(*_ngenVar) ;

    _fitParData->add(tmp) ;
  }

  // Store fit result if requested by user
  if (_fitOptList.FindObject("Save")) {
    _fitResList.Add(fr.release()) ;
  }

  return !ok ;
}



////////////////////////////////////////////////////////////////////////////////
/// Utility function to add fit result from external fit to this RooMCStudy
/// and process its results through the standard RooMCStudy statistics gathering tools.
/// This function allows users to run the toy MC generation and/or fitting
/// in a distributed way and to collect and analyze the results in a RooMCStudy
/// as if they were run locally.
///
/// This method is only functional if this RooMCStudy object is cleanm, i.e. it was not used
/// to generate and/or fit any samples.

bool RooMCStudy::addFitResult(const RooFitResult& fr)
{
  if (!_canAddFitResults) {
    oocoutE(_fitModel,InputArguments) << "RooMCStudy::addFitResult: ERROR cannot add fit results in current state" << std::endl ;
    return true ;
  }

  // Transfer contents of fit result to fitParams ;
  _fitParams.assign(RooArgSet(fr.floatParsFinal())) ;

  // If fit converged, store parameters and NLL
  bool ok = (fr.status()==0) ;
  if (ok) {
    _nllVar->setVal(fr.minNll()) ;
    RooArgSet tmp(_fitParams) ;
    tmp.add(*_nllVar) ;
    tmp.add(*_ngenVar) ;
    _fitParData->add(tmp) ;
  }

  // Store fit result if requested by user
  if (_fitOptList.FindObject("Save")) {
    _fitResList.Add((TObject*)&fr) ;
  }

  return false ;
}



////////////////////////////////////////////////////////////////////////////////
/// Calculate the pulls for all fit parameters in
/// the fit results data set, and add them to that dataset.

void RooMCStudy::calcPulls()
{
  for (auto it = _fitParams.begin(); it != _fitParams.end(); ++it) {
    const auto par = static_cast<RooRealVar*>(*it);
    _fitParData->addColumn(*std::unique_ptr<RooErrorVar>{par->errorVar()});

    TString name(par->GetName());
    TString title(par->GetTitle());
    name.Append("pull") ;
    title.Append(" Pull") ;

    if (!par->hasError(false)) {
      coutW(Generation) << "Fit parameter '" << par->GetName() << "' does not have an error."
          " A pull distribution cannot be generated. This might be caused by the parameter being constant or"
          " because the fits were not run." << std::endl;
      continue;
    }

    // First look in fitParDataset to see if per-experiment generated value has been stored
    auto genParOrig = static_cast<RooAbsReal*>(_fitParData->get()->find(Form("%s_gen",par->GetName())));
    if (genParOrig && _perExptGenParams) {

      RooPullVar pull(name,title,*par,*genParOrig) ;
      _fitParData->addColumn(pull,false) ;

    } else {
      // If not use fixed generator value
      genParOrig = static_cast<RooAbsReal*>(_genInitParams.find(par->GetName()));

      if (!genParOrig) {
        std::size_t index = it - _fitParams.begin();
        genParOrig = index < _genInitParams.size() ?
            static_cast<RooAbsReal*>(_genInitParams[index]) :
            nullptr;

        if (genParOrig) {
          coutW(Generation) << "The fit parameter '" << par->GetName() << "' is not in the model that was used to generate toy data. "
              "The parameter '" << genParOrig->GetName() << "'=" << genParOrig->getVal() << " was found at the same position in the generator model."
              " It will be used to compute pulls."
              "\nIf this is not desired, the parameters of the generator model need to be renamed or reordered." << std::endl;
        }
      }

      if (genParOrig) {
        std::unique_ptr<RooAbsReal> genPar(static_cast<RooAbsReal*>(genParOrig->Clone("truth")));
        RooPullVar pull(name,title,*par,*genPar);

        _fitParData->addColumn(pull,false) ;
      } else {
        coutE(Generation) << "Cannot generate pull distribution for the fit parameter '" << par->GetName() << "'."
            "\nNo similar parameter was found in the set of parameters that were used to generate toy data." << std::endl;
      }
    }
  }
}




////////////////////////////////////////////////////////////////////////////////
/// Return a RooDataSet containing the post-fit parameters of each toy cycle.
/// This dataset also contains any additional output that was generated
/// by study modules that were added to this RooMCStudy.
/// By default, the two following variables are added (apart from fit parameters):
/// - NLL: The value of the negative log-likelihood for each run.
/// - ngen: Number of events generated for each run.
const RooDataSet& RooMCStudy::fitParDataSet()
{
  if (_canAddFitResults) {
    calcPulls() ;
    _canAddFitResults = false ;
  }

  return *_fitParData ;
}



////////////////////////////////////////////////////////////////////////////////
/// Return an argset with the fit parameters for the given sample number
///
/// NB: The fit parameters are only stored for successful fits,
///     thus the maximum sampleNum can be less that the number
///     of generated samples and if so, the indices will
///     be out of synch with genData() and fitResult()

const RooArgSet* RooMCStudy::fitParams(Int_t sampleNum) const
{
  // Check if sampleNum is in range
  if (sampleNum<0 || sampleNum>=_fitParData->numEntries()) {
    oocoutE(_fitModel,InputArguments) << "RooMCStudy::fitParams: ERROR, invalid sample number: " << sampleNum << std::endl ;
    return nullptr ;
  }

  return _fitParData->get(sampleNum) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Return the RooFitResult of the fit with the given run number.
///
/// \note Fit results are not saved by default. This requires passing `FitOptions(Save(), ...)`
/// to the constructor.
const RooFitResult* RooMCStudy::fitResult(Int_t sampleNum) const
{
  // Check if sampleNum is in range
  if (sampleNum<0 || sampleNum>=_fitResList.GetSize()) {
    oocoutE(_fitModel,InputArguments) << "RooMCStudy::fitResult: ERROR, invalid sample number: " << sampleNum << std::endl ;
    return nullptr ;
  }

  // Retrieve fit result object
  const RooFitResult* fr = static_cast<RooFitResult*>(_fitResList.At(sampleNum)) ;
  if (fr) {
    return fr ;
  } else {
    oocoutE(_fitModel,InputArguments) << "RooMCStudy::fitResult: ERROR, no fit result saved for sample "
           << sampleNum << ", did you use the 'r; fit option?" << std::endl ;
  }
  return nullptr ;
}



////////////////////////////////////////////////////////////////////////////////
/// Return the given generated dataset. This method will only return datasets
/// if during the run cycle it was indicated that generator data should be saved.

RooAbsData* RooMCStudy::genData(Int_t sampleNum) const
{
  // Check that generated data was saved
  if (_genDataList.GetSize()==0) {
    oocoutE(_fitModel,InputArguments) << "RooMCStudy::genData() ERROR, generated data was not saved" << std::endl ;
    return nullptr ;
  }

  // Check if sampleNum is in range
  if (sampleNum<0 || sampleNum>=_genDataList.GetSize()) {
    oocoutE(_fitModel,InputArguments) << "RooMCStudy::genData() ERROR, invalid sample number: " << sampleNum << std::endl ;
    return nullptr ;
  }

  return  static_cast<RooAbsData*>(_genDataList.At(sampleNum)) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Plot the distribution of fitted values of a parameter. The parameter shown is the one from which the RooPlot
/// was created, e.g.
///
/// RooPlot* frame = param.frame(100,-10,10) ;
/// mcstudy.paramOn(frame,LineStyle(kDashed)) ;
///
/// Any named arguments passed to plotParamOn() are forwarded to the underlying plotOn() call

RooPlot* RooMCStudy::plotParamOn(RooPlot* frame, const RooCmdArg& arg1, const RooCmdArg& arg2, const RooCmdArg& arg3, const RooCmdArg& arg4,
                const RooCmdArg& arg5, const RooCmdArg& arg6, const RooCmdArg& arg7, const RooCmdArg& arg8)
{
  _fitParData->plotOn(frame,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8) ;
  return frame ;
}



////////////////////////////////////////////////////////////////////////////////
/// Plot the distribution of the fitted value of the given parameter on a newly created frame.
///
/// <table>
/// <tr><th> Optional arguments <th>
/// <tr><td> FrameRange(double lo, double hi) <td> Set range of frame to given specification
/// <tr><td> FrameBins(int bins)              <td> Set default number of bins of frame to given number
/// <tr><td> Frame()                       <td> Pass supplied named arguments to RooAbsRealLValue::frame() function. See there
///     for list of allowed arguments
/// </table>
/// If no frame specifications are given, the AutoRange() feature will be used to set the range
/// Any other named argument is passed to the RooAbsData::plotOn() call. See that function for allowed options

RooPlot* RooMCStudy::plotParam(const char* paramName, const RooCmdArg& arg1, const RooCmdArg& arg2, const RooCmdArg& arg3, const RooCmdArg& arg4,
                   const RooCmdArg& arg5, const RooCmdArg& arg6, const RooCmdArg& arg7, const RooCmdArg& arg8)
{

  // Find parameter in fitParDataSet
  RooRealVar* param = static_cast<RooRealVar*>(_fitParData->get()->find(paramName)) ;
  if (!param) {
    oocoutE(_fitModel,InputArguments) << "RooMCStudy::plotParam: ERROR: no parameter defined with name " << paramName << std::endl ;
    return nullptr ;
  }

  // Forward to implementation below
  return plotParam(*param,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Plot the distribution of the fitted value of the given parameter on a newly created frame.
/// \copydetails RooMCStudy::plotParam(const char* paramName, const RooCmdArg& arg1, const RooCmdArg& arg2, const RooCmdArg& arg3, const RooCmdArg& arg4,
/// const RooCmdArg& arg5, const RooCmdArg& arg6, const RooCmdArg& arg7, const RooCmdArg& arg8)

RooPlot* RooMCStudy::plotParam(const RooRealVar& param, const RooCmdArg& arg1, const RooCmdArg& arg2, const RooCmdArg& arg3, const RooCmdArg& arg4,
                   const RooCmdArg& arg5, const RooCmdArg& arg6, const RooCmdArg& arg7, const RooCmdArg& arg8)
{
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



////////////////////////////////////////////////////////////////////////////////
/// Plot the distribution of the -log(L) values on a newly created frame.
///
/// <table>
/// <tr><th> Optional arguments <th>
/// <tr><td> FrameRange(double lo, double hi) <td> Set range of frame to given specification
/// <tr><td> FrameBins(int bins)              <td> Set default number of bins of frame to given number
/// <tr><td> Frame()                       <td> Pass supplied named arguments to RooAbsRealLValue::frame() function. See there
///     for list of allowed arguments
/// </table>
///
/// If no frame specifications are given, the AutoRange() feature will be used to set the range.
/// Any other named argument is passed to the RooAbsData::plotOn() call. See that function for allowed options

RooPlot* RooMCStudy::plotNLL(const RooCmdArg& arg1, const RooCmdArg& arg2,
                     const RooCmdArg& arg3, const RooCmdArg& arg4,
                     const RooCmdArg& arg5, const RooCmdArg& arg6,
                     const RooCmdArg& arg7, const RooCmdArg& arg8)
{
  return plotParam(*_nllVar,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Plot the distribution of the fit errors for the specified parameter on a newly created frame.
///
/// <table>
/// <tr><th> Optional arguments <th>
/// <tr><td> FrameRange(double lo, double hi) <td> Set range of frame to given specification
/// <tr><td> FrameBins(int bins)              <td> Set default number of bins of frame to given number
/// <tr><td> Frame()                       <td> Pass supplied named arguments to RooAbsRealLValue::frame() function. See there
///     for list of allowed arguments
/// </table>
///
/// If no frame specifications are given, the AutoRange() feature will be used to set a default range.
/// Any other named argument is passed to the RooAbsData::plotOn() call. See that function for allowed options.

RooPlot* RooMCStudy::plotError(const RooRealVar& param, const RooCmdArg& arg1, const RooCmdArg& arg2,
                     const RooCmdArg& arg3, const RooCmdArg& arg4,
                     const RooCmdArg& arg5, const RooCmdArg& arg6,
                     const RooCmdArg& arg7, const RooCmdArg& arg8)
{
  if (_canAddFitResults) {
    calcPulls() ;
    _canAddFitResults=false ;
  }

  std::unique_ptr<RooErrorVar> evar{param.errorVar()};
  std::unique_ptr<RooAbsArg> evar_rrv{evar->createFundamental()};
  RooPlot* frame = plotParam(static_cast<RooRealVar&>(*evar_rrv),arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8) ;

  // To make sure the frame has no dangling pointer to evar_rrv.
  frame->createInternalPlotVarClone();

  return frame ;
}

namespace {

// Fits a Gaussian to the pull distribution, plots the fit and prints the fit
// parameters on the canvas. Implementation detail of RooMCStudy::plotPull().
void fitGaussToPulls(RooPlot& frame, RooDataSet& fitParData)
{
   // Build the Gaussian fit mode for the pulls, then fit it and plot it. We
   // have to use the RooWorkspace factory here, because different from the
   // RooMCStudy class, the RooGaussian is not in RooFitCore.
   RooWorkspace ws;
   auto plotVar = frame.getPlotVar();
   const std::string plotVarName = plotVar->GetName();
   ws.import(*plotVar);
   ws.factory("Gaussian::pullGauss(" + plotVarName  + ", pullMean[0.0, -10.0, 10.0], pullSigma[1.0, 0.1, 5.0])");

   RooRealVar& pullMean = *ws.var("pullMean");
   RooRealVar& pullSigma = *ws.var("pullSigma");
   RooAbsPdf& pullGauss = *ws.pdf("pullGauss");

   pullGauss.fitTo(fitParData, RooFit::Minos(false), RooFit::PrintLevel(-1)) ;
   pullGauss.plotOn(&frame) ;

   // Instead of using paramOn() without command arguments to plot the fit
   // parameters, we are building the parameter label ourselves for more
   // flexibility and pass this together with an appropriate layout
   // parametrization to paramOn().
   const int sigDigits = 2;
   const char * options = "ELU";
   std::stringstream ss;
   ss << "Fit parameters:\n"
      << "#mu: " << pullMean.format(sigDigits, options)
      << "\n#sigma: " << pullSigma.format(sigDigits, options);
   // We set the parameters constant to disable the default label. Still, we
   // use param() on as a wrapper for the text box generation.
   pullMean.setConstant(true);
   pullSigma.setConstant(true);
   pullGauss.paramOn(&frame, RooFit::Label(ss.str().c_str()), RooFit::Layout(0.60, 0.9, 0.9));
}

} // namespace


////////////////////////////////////////////////////////////////////////////////
/// Plot the distribution of pull values for the specified parameter on a newly created frame. If asymmetric
/// errors are calculated in the fit (by MINOS) those will be used in the pull calculation.
///
/// If the parameters of the models for generation and fit differ, simple heuristics are used to find the
/// corresponding parameters:
/// - Parameters have the same name: They will be used to compute pulls.
/// - Parameters have different names: The position of the fit parameter in the set of fit parameters will be
///   computed. The parameter at the same position in the set of generator parameters will be used.
///
/// Further options:
/// <table>
/// <tr><th> Arguments <th> Effect
/// <tr><td> FrameRange(double lo, double hi) <td> Set range of frame to given specification
/// <tr><td> FrameBins(int bins)              <td> Set default number of bins of frame to given number
/// <tr><td> Frame()                       <td> Pass supplied named arguments to RooAbsRealLValue::frame() function. See there
///     for list of allowed arguments
/// <tr><td> FitGauss(bool flag)            <td> Add a gaussian fit to the frame
/// </table>
///
/// If no frame specifications are given, the AutoSymRange() feature will be used to set a default range.
/// Any other named argument is passed to the RooAbsData::plotOn(). See that function for allowed options.
///
/// If you want to have more control over the Gaussian fit to the pull
/// distribution, you can also do it after the call to plotPull():
///
/// ~~~ {.cpp}
/// RooPlot *frame = mcstudy->plotPull(myVariable, RooFit::Bins(40), RooFit::FitGauss(false));
/// RooRealVar pullMean("pullMean","Mean of pull",0,-10,10) ;
/// RooRealVar pullSigma("pullSigma","Width of pull",1,0.1,5) ;
/// pullMean.setPlotLabel("pull #mu");     // optional (to get nicer plot labels if you want)
/// pullSigma.setPlotLabel("pull #sigma"); // optional
/// RooGaussian pullGauss("pullGauss","Gaussian of pull", *frame->getPlotVar(), pullMean, pullSigma);
/// pullGauss.fitTo(const_cast<RooDataSet&>(mcstudy->fitParDataSet()),
///                 RooFit::Minos(0), RooFit::PrintLevel(-1)) ;
/// pullGauss.plotOn(frame) ;
/// pullGauss.paramOn(frame, RooFit::Layout(0.65, 0.9, 0.9)); // optionally specify label position (xmin, xmax, ymax)
/// ~~~

RooPlot* RooMCStudy::plotPull(const RooRealVar& param, const RooCmdArg& arg1, const RooCmdArg& arg2,
                     const RooCmdArg& arg3, const RooCmdArg& arg4,
                     const RooCmdArg& arg5, const RooCmdArg& arg6,
                     const RooCmdArg& arg7, const RooCmdArg& arg8)
{
  // Stuff all arguments in a list
  RooLinkedList cmdList;
  cmdList.Add(const_cast<RooCmdArg*>(&arg1)) ;  cmdList.Add(const_cast<RooCmdArg*>(&arg2)) ;
  cmdList.Add(const_cast<RooCmdArg*>(&arg3)) ;  cmdList.Add(const_cast<RooCmdArg*>(&arg4)) ;
  cmdList.Add(const_cast<RooCmdArg*>(&arg5)) ;  cmdList.Add(const_cast<RooCmdArg*>(&arg6)) ;
  cmdList.Add(const_cast<RooCmdArg*>(&arg7)) ;  cmdList.Add(const_cast<RooCmdArg*>(&arg8)) ;

  TString name(param.GetName());
  TString title(param.GetTitle());
  name.Append("pull") ; title.Append(" Pull") ;
  RooRealVar pvar(name,title,-100,100) ;
  pvar.setBins(100) ;


  RooPlot* frame = makeFrameAndPlotCmd(pvar, cmdList, true) ;
  if (frame) {

    // Pick up optional FitGauss command from list
    RooCmdConfig pc("RooMCStudy::plotPull(" + std::string(_genModel->GetName()) + ")");
    pc.defineInt("fitGauss","FitGauss",0,0) ;
    pc.allowUndefined() ;
    pc.process(cmdList) ;
    bool fitGauss=pc.getInt("fitGauss") ;

    // Pass stripped command list to plotOn()
    RooCmdConfig::stripCmdList(cmdList,"FitGauss") ;
    const bool success = _fitParData->plotOn(frame,cmdList) ;

    if (!success) {
      coutF(Plotting) << "No pull distribution for the parameter '" << param.GetName() << "'. Check logs for errors." << std::endl;
      return frame;
    }

    // Add Gaussian fit if requested
    if (fitGauss) {
      fitGaussToPulls(*frame, *_fitParData);
    }

    // To make sure the frame has no dangling pointer to pvar.
    frame->createInternalPlotVarClone();
  }
  return frame;
}



////////////////////////////////////////////////////////////////////////////////
/// Internal function. Construct RooPlot from given parameter and modify the list of named
/// arguments 'cmdList' to only contain the plot arguments that should be forwarded to
/// RooAbsData::plotOn()

RooPlot* RooMCStudy::makeFrameAndPlotCmd(const RooRealVar& param, RooLinkedList& cmdList, bool symRange) const
{
  // Select the frame-specific commands
  RooCmdConfig pc("RooMCStudy::plotParam(" + std::string(_genModel->GetName()) + ")");
  pc.defineInt("nbins","Bins",0,0) ;
  pc.defineDouble("xlo","Range",0,0) ;
  pc.defineDouble("xhi","Range",1,0) ;
  pc.defineInt("dummy","FrameArgs",0,0) ;
  pc.defineMutex("Bins","FrameArgs") ;
  pc.defineMutex("Range","FrameArgs") ;

  // Process and check varargs
  pc.allowUndefined() ;
  pc.process(cmdList) ;
  if (!pc.ok(true)) {
    return nullptr ;
  }

  // Make frame according to specs
  Int_t nbins = pc.getInt("nbins") ;
  double xlo = pc.getDouble("xlo") ;
  double xhi = pc.getDouble("xhi") ;
  RooPlot* frame ;

  if (pc.hasProcessed("FrameArgs")) {
    // Explicit frame arguments are given, pass them on
    RooCmdArg* frameArg = static_cast<RooCmdArg*>(cmdList.FindObject("FrameArgs")) ;
    frame = param.frame(frameArg->subArgs()) ;
  } else {
    // FrameBins, FrameRange or none are given, build custom frame command list
    RooCmdArg bins = RooFit::Bins(nbins) ;
    RooCmdArg range = RooFit::Range(xlo,xhi) ;
    RooCmdArg autoRange = symRange ? RooFit::AutoSymRange(*_fitParData,0.2) : RooFit::AutoRange(*_fitParData,0.2) ;
    RooLinkedList frameCmdList ;

    if (pc.hasProcessed("Bins")) frameCmdList.Add(&bins) ;
    if (pc.hasProcessed("Range")) {
      frameCmdList.Add(&range) ;
    } else {
      frameCmdList.Add(&autoRange) ;
    }
    frame = param.frame(frameCmdList) ;
  }

  // Filter frame command from list and pass on to plotOn()
  RooCmdConfig::stripCmdList(cmdList,"FrameArgs,Bins,Range") ;

  return frame ;
}



////////////////////////////////////////////////////////////////////////////////
/// Create a RooPlot of the -log(L) distribution in the range lo-hi
/// with 'nBins' bins

RooPlot* RooMCStudy::plotNLL(double lo, double hi, Int_t nBins)
{
  RooPlot* frame = _nllVar->frame(lo,hi,nBins) ;

  _fitParData->plotOn(frame) ;
  return frame ;
}



////////////////////////////////////////////////////////////////////////////////
/// Create a RooPlot of the distribution of the fitted errors of the given parameter.
/// The frame is created with a range [lo,hi] and plotted data will be binned in 'nbins' bins

RooPlot* RooMCStudy::plotError(const RooRealVar& param, double lo, double hi, Int_t nbins)
{
  if (_canAddFitResults) {
    calcPulls() ;
    _canAddFitResults=false ;
  }

  std::unique_ptr<RooErrorVar> evar{param.errorVar()};
  RooPlot* frame = evar->frame(lo,hi,nbins) ;
  _fitParData->plotOn(frame) ;

  return frame ;
}



////////////////////////////////////////////////////////////////////////////////
/// Create a RooPlot of the pull distribution for the given
/// parameter.  The range lo-hi is plotted in nbins.  If fitGauss is
/// set, an unbinned ML fit of the distribution to a Gaussian p.d.f
/// is performed. The fit result is overlaid on the returned RooPlot
/// and a box with the fitted mean and sigma is added.
///
/// If the parameters of the models for generation and fit differ, simple heuristics are used to find the
/// corresponding parameters:
/// - Parameters have the same name: They will be used to compute pulls.
/// - Parameters have different names: The position of the fit parameter in the set of fit parameters will be
///   computed. The parameter at the same position in the set of generator parameters will be used.

RooPlot* RooMCStudy::plotPull(const RooRealVar& param, double lo, double hi, Int_t nbins, bool fitGauss)
{
  if (_canAddFitResults) {
    calcPulls() ;
    _canAddFitResults=false ;
  }

  TString name(param.GetName());
  TString title(param.GetTitle());
  name.Append("pull") ; title.Append(" Pull") ;
  RooRealVar pvar(name,title,lo,hi) ;
  pvar.setBins(nbins) ;

  RooPlot* frame = pvar.frame() ;
  const bool success = _fitParData->plotOn(frame);

  if (!success) {
    coutF(Plotting) << "No pull distribution for the parameter '" << param.GetName() << "'. Check logs for errors." << std::endl;
    return frame;
  }

  if (fitGauss) {
    fitGaussToPulls(*frame, *_fitParData);
  }

  return frame ;
}


////////////////////////////////////////////////////////////////////////////////
/// If one of the TObject we have a referenced to is deleted, remove the
/// reference.

void RooMCStudy::RecursiveRemove(TObject *obj)
{
   _fitResList.RecursiveRemove(obj);
   _genDataList.RecursiveRemove(obj);
   _fitOptList.RecursiveRemove(obj);
   if (_ngenVar.get() == obj) _ngenVar.reset();

   if (_fitParData) _fitParData->RecursiveRemove(obj);
   if (_fitParData.get() == obj) _fitParData.reset();

   if (_genParData) _genParData->RecursiveRemove(obj);
   if (_genParData.get() == obj) _genParData.reset();
}

