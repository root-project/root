/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooMCStudy.cc,v 1.4 2001/12/01 08:12:47 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   09-Oct-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
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



#include "RooFitCore/RooMCStudy.hh"

#include "RooFitCore/RooGenContext.hh"
#include "RooFitCore/RooAbsPdf.hh"
#include "RooFitCore/RooDataSet.hh"
#include "RooFitCore/RooRealVar.hh"
#include "RooFitCore/RooFitResult.hh"
#include "RooFitCore/RooErrorVar.hh"
#include "RooFitCore/RooFormulaVar.hh"
#include "RooFitCore/RooArgList.hh"
#include "RooFitCore/RooPlot.hh"
#include "RooFitCore/RooGenericPdf.hh"


ClassImp(RooMCStudy)
;

RooMCStudy::RooMCStudy(const RooAbsPdf& genModel, const RooAbsPdf& fitModel, 
		       const RooArgSet& dependents, const char* genOptions, 
		       const char* fitOptions, const RooDataSet* genProtoData) :
  _genModel((RooAbsPdf*)&genModel), 
  _fitModel((RooAbsPdf*)&fitModel), 
  _dependents(dependents), 
  _fitOptions(fitOptions),
  _genProtoData(genProtoData)
{
  // Constructor with a generator and fit model. Both models may point
  // to the same object. The 'dependents' set of variables is generated 
  // in the generator phase. The optional prototype dataset is passed to
  // the generator
  //
  // Available generator options
  //  v  - Verbose
  //
  // Available fit options
  //  See RooAbsPdf::fitTo()
  //

  // Decode generator options
  TString genOpt(genOptions) ;
  genOpt.ToLower() ;
  Bool_t verboseGen = genOpt.Contains("v") ;

  _genContext = genModel.genContext(dependents,genProtoData,verboseGen) ;
  RooArgSet* tmp = genModel.getParameters(&dependents) ;
  _genParams = (RooArgSet*) tmp->snapshot(kFALSE) ;
  delete tmp ;

  // Store list of parameters and save initial values separately
  _fitParams = fitModel.getParameters(&dependents) ;
  _fitInitParams = (RooArgSet*) _fitParams->snapshot(kTRUE) ;

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
}



RooMCStudy::~RooMCStudy() 
{  
  // Destructor 

  _genDataList.Delete() ;
  _fitResList.Delete() ;
  delete _fitParData ;
  delete _fitParams ;
  delete _genParams ;
  delete _genContext ;
  delete _nllVar ;
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

  while(nSamples--) {

    cout << "RooMCStudy::run: " ;
    if (generate) cout << "Generating " ;
    if (generate && fit) cout << "and " ;
    if (fit) cout << "fitting " ;
    cout << "sample " << nSamples << endl ;

    RooDataSet* genSample(0) ;
    if (generate) {
      // Generate sample
      genSample = _genContext->generate(nEvtPerSample) ;
    } else if (asciiFilePat && &asciiFilePat) {
      // Load sample from ASCII file
      char asciiFile[1024] ;
      sprintf(asciiFile,asciiFilePat,nSamples) ;
      RooArgList depList(_dependents) ;
      genSample = RooDataSet::read(asciiFile,depList,"q") ;      
    } else {
      // Load sample from internal list
      genSample = (RooDataSet*) _genDataList.At(nSamples) ;
      if (!genSample) {
	cout << "RooMCStudy::run: WARNING: Sample #" << nSamples << " not loaded, skipping" << endl ;
	continue ;
      }
    }

    if (fit) fitSample(genSample) ;

    // Optionally write to ascii file
    if (generate && asciiFilePat && *asciiFilePat) {
      char asciiFile[1024] ;
      sprintf(asciiFile,asciiFilePat,nSamples) ;
      genSample->write(asciiFile) ;
    }

    // Add to list or delete
    if (keepGenData) {
      _genDataList.Add(genSample) ;
    } else {
      delete genSample ;
    }
  }

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
  RooDataSet* gset ;
  while(gset=(RooDataSet*)iter->Next()) {
    _genDataList.Add(gset) ;
  }
  delete iter ;

  return run(kFALSE,kTRUE,nSamples,0,kTRUE,0) ;
}




Bool_t RooMCStudy::fitSample(RooDataSet* genSample) 
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
  *_fitParams = *_fitInitParams ;

  // Fit model to data set
  TString fitOpt2(_fitOptions) ; fitOpt2.Append("r") ;
  RooFitResult* fr = (RooFitResult*) _fitModel->fitTo(*genSample,fitOpt2) ;

  // If fit converged, store parameters and NLL
  Bool_t ok = (fr->status()==0) ;
  if (ok) {
    _nllVar->setVal(fr->minNll()) ;
    RooArgSet tmp(*_fitParams) ;
    tmp.add(*_nllVar) ;
    _fitParData->add(tmp) ;
  }

  // Store fit result if requested by user
  if (_fitOptions.Contains("r")) {
    _fitResList.Add(fr) ;
  } else {
    delete fr ;
  }
  
  return !ok ;
}




void RooMCStudy::calcPulls() 
{
  // Calculate the pulls for all fit parameters in
  // the fit results data set, and add them to that dataset

  TIterator* iter = _fitParams->createIterator()  ;
  RooRealVar* par ;
  while(par=(RooRealVar*)iter->Next()) {

    RooErrorVar* err = par->errorVar() ;
    _fitParData->addColumn(*err) ;

    TString name(par->GetName()), title(par->GetTitle()) ;
    name.Append("pull") ;
    title.Append(" Pull") ;
    RooAbsReal* genPar = (RooAbsReal*) _genParams->find(par->GetName())->Clone("truth") ;
    RooFormulaVar pull(name,title,"(@0-@1)/@2",RooArgList(*par,*genPar,*err)) ;

    _fitParData->addColumn(pull) ;

    delete genPar ;
    
  }
  delete iter ;

}




const RooDataSet& RooMCStudy::fitParDataSet() const 
{
  // Return the fit parameter dataset
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
  if (sampleNum<0 || sampleNum>=_genDataList.GetSize()) {
    cout << "RooMCStudy::fitParams: ERROR, invalid sample number: " << sampleNum << endl ;    
    return 0 ;
  }

  return _fitParData->get(sampleNum) ;
}



const RooFitResult* RooMCStudy::fitResult(Int_t sampleNum) const
{
  // Return the fit result object of the fit to given sample

  // Check if sampleNum is in range
  if (sampleNum<0 || sampleNum>=_genDataList.GetSize()) {
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

  // Check if sampleNum is in range
  if (sampleNum<0 || sampleNum>=_genDataList.GetSize()) {
    cout << "RooMCStudy::fitResult: ERROR, invalid sample number: " << sampleNum << endl ;    
    return 0 ;
  }

  return (RooDataSet*) _genDataList.At(sampleNum) ;
}


RooPlot* RooMCStudy::plotParamOn(RooPlot* frame) 
{
  // Plot the distribution of the fitted value
  // of the given parameter. 

  _fitParData->plotOn(frame) ;
  return frame ;
}



RooPlot* RooMCStudy::plotParam(const RooRealVar& param) 
{
  // Create a RooPlot of the distribution of the fitted value
  // of the given parameter. The plot range and binning
  // of the supplied parameter will be used
  RooPlot* frame = param.frame() ;
  _fitParData->plotOn(frame) ;
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

  TString name(param.GetName()), title(param.GetTitle()) ;
  name.Append("pull") ; title.Append(" Pull") ;
  RooRealVar pvar(name,title,lo,hi) ;
  pvar.setFitBins(nbins) ;

  RooPlot* frame = pvar.frame() ;
  _fitParData->plotOn(frame) ;

  if (fitGauss) {
    RooRealVar pullMean("mean","Mean of pull",0,lo,hi) ;
    RooRealVar pullSigma("sigma","Width of pull",1,0,5) ;
    RooGenericPdf pullGauss("pullGauss","Gaussian of pull",
			    "exp(-0.5*(@0-@1)*(@0-@1)/(@2*@2))",
			    RooArgSet(pvar,pullMean,pullSigma)) ;
    pullGauss.fitTo(*_fitParData,"mh") ;
    pullGauss.plotOn(frame) ;
    pullGauss.paramOn(frame,_fitParData) ;
  }

  return frame ;
}



