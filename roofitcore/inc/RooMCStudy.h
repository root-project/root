/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooMCStudy.rdl,v 1.1 2001/10/11 01:28:50 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   09-Oct-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_MC_STUDY
#define ROO_MC_STUDY

#include "TList.h"
#include "RooFitCore/RooArgSet.hh"
class RooAbsPdf;
class RooDataSet ;
class RooGenContext ;
class RooFitResult ;
class RooPlot ;
class RooRealVar ;

class RooMCStudy {
public:

  RooMCStudy(const RooAbsPdf& genModel, const RooAbsPdf& fitModel, 
	     const RooArgSet& dependents, const char* genOptions="",
	     const char* fitOptions="", const RooDataSet* genProtoData=0) ;
  virtual ~RooMCStudy() ;
  
  // Run methods
  Bool_t generateAndFit(Int_t nSamples, Int_t nEvtPerSample, Bool_t keepGenData=kFALSE, const char* asciiFilePat=0) ;
  Bool_t generate(Int_t nSamples, Int_t nEvtPerSample, Bool_t keepGenData=kFALSE, const char* asciiFilePat=0) ;
  Bool_t fit(Int_t nSamples, const char* asciiFilePat) ;
  Bool_t fit(Int_t nSamples, TList& dataSetList) ;

  // Result accessors
  const RooArgSet* fitParams(Int_t sampleNum) const ;
  const RooFitResult* fitResult(Int_t sampleNum) const ;
  const RooDataSet* genData(Int_t sampleNum) const ;
  const RooDataSet& fitParDataSet() const ;

  // Plot methods
  RooPlot* plotNLL(Double_t lo, Double_t hi, Int_t nBins=100) ;
  RooPlot* plotParam(const RooRealVar& param) ;
  RooPlot* plotError(const RooRealVar& param, Double_t lo, Double_t hi, Int_t nbins=100) ;
  RooPlot* plotPull(const RooRealVar& param, Double_t lo, Double_t hi, Int_t nbins=100, Bool_t fitGauss=kFALSE) ;
    
protected:

  Bool_t run(Bool_t generate, Bool_t fit, Int_t nSamples, Int_t nEvtPerSample, Bool_t keepGenData, const char* asciiFilePat) ;
  Bool_t fitSample(RooDataSet* genSample) ;
  void calcPulls() ;
    
  RooAbsPdf*     _genModel ;    // Generator model 
  RooGenContext* _genContext ;  // Generator context 
  RooArgSet*     _genParams ;   // List of fit parameters
  const RooDataSet* _genProtoData ;// Generator prototype data set

  RooArgSet    _dependents ;    // List of dependents 
  RooAbsPdf*   _fitModel ;      // Fit model 
  RooArgSet*   _fitInitParams ; // List of initial values of fit parameters
  RooArgSet*   _fitParams ;     // List of fit parameters
  RooRealVar*  _nllVar ;

  TList       _genDataList ;    // List of generated data sample
  TList       _fitResList ;     // List of RooFitResult fit output objects
  RooDataSet* _fitParData ;     // Data set of fit parameters of each sample
  TString     _fitOptions ;     // Fit options string

private:
  RooMCStudy(const RooMCStudy&) ;
	
  ClassDef(RooMCStudy,0) // Monte Carlo study manager
} ;


#endif

