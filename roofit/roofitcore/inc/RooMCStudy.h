/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooMCStudy.h,v 1.18 2007/05/11 10:14:56 verkerke Exp $
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
#ifndef ROO_MC_STUDY
#define ROO_MC_STUDY

#include "TList.h"
#include "TNamed.h"
#include "RooArgSet.h"
#include <list>
class RooAbsPdf;
class RooDataSet ;
class RooAbsData ;
class RooAbsGenContext ;
class RooFitResult ;
class RooPlot ;
class RooRealVar ;
class RooAbsMCStudyModule ;

class RooMCStudy : public TNamed {
public:

  RooMCStudy(const RooAbsPdf& model, const RooArgSet& observables,
        const RooCmdArg& arg1=RooCmdArg::none(), const RooCmdArg& arg2=RooCmdArg::none(),
             const RooCmdArg& arg3=RooCmdArg::none(), const RooCmdArg& arg4=RooCmdArg::none(), const RooCmdArg& arg5=RooCmdArg::none(),
             const RooCmdArg& arg6=RooCmdArg::none(), const RooCmdArg& arg7=RooCmdArg::none(), const RooCmdArg& arg8=RooCmdArg::none()) ;

  ~RooMCStudy() override ;

  // Method to add study modules
  void addModule(RooAbsMCStudyModule& module) ;


  // Run methods
  bool generateAndFit(Int_t nSamples, Int_t nEvtPerSample=0, bool keepGenData=false, const char* asciiFilePat=nullptr) ;
  bool generate(Int_t nSamples, Int_t nEvtPerSample=0, bool keepGenData=false, const char* asciiFilePat=nullptr) ;
  bool fit(Int_t nSamples, const char* asciiFilePat) ;
  bool fit(Int_t nSamples, TList& dataSetList) ;
  bool addFitResult(const RooFitResult& fr) ;

  // Result accessors
  const RooArgSet* fitParams(Int_t sampleNum) const ;
  const RooFitResult* fitResult(Int_t sampleNum) const ;
        RooAbsData* genData(Int_t sampleNum) const ;
  const RooDataSet& fitParDataSet() ;
  /// Return dataset with generator parameters for each toy. When constraints are used these
  /// may generally not be the same as the fitted parameters.
  const RooDataSet* genParDataSet() const {
    return _genParData ;
  }

  // Plot methods
  RooPlot* plotParamOn(RooPlot* frame, const RooCmdArg& arg1=RooCmdArg::none(), const RooCmdArg& arg2=RooCmdArg::none(),
                       const RooCmdArg& arg3=RooCmdArg::none(), const RooCmdArg& arg4=RooCmdArg::none(),
                       const RooCmdArg& arg5=RooCmdArg::none(), const RooCmdArg& arg6=RooCmdArg::none(),
                       const RooCmdArg& arg7=RooCmdArg::none(), const RooCmdArg& arg8=RooCmdArg::none()) ;
  RooPlot* plotParam(const RooRealVar& param, const RooCmdArg& arg1=RooCmdArg::none(), const RooCmdArg& arg2=RooCmdArg::none(),
                     const RooCmdArg& arg3=RooCmdArg::none(), const RooCmdArg& arg4=RooCmdArg::none(),
                     const RooCmdArg& arg5=RooCmdArg::none(), const RooCmdArg& arg6=RooCmdArg::none(),
                     const RooCmdArg& arg7=RooCmdArg::none(), const RooCmdArg& arg8=RooCmdArg::none()) ;
  RooPlot* plotParam(const char* paramName, const RooCmdArg& arg1=RooCmdArg::none(), const RooCmdArg& arg2=RooCmdArg::none(),
                     const RooCmdArg& arg3=RooCmdArg::none(), const RooCmdArg& arg4=RooCmdArg::none(),
                     const RooCmdArg& arg5=RooCmdArg::none(), const RooCmdArg& arg6=RooCmdArg::none(),
                     const RooCmdArg& arg7=RooCmdArg::none(), const RooCmdArg& arg8=RooCmdArg::none()) ;
  RooPlot* plotNLL(const RooCmdArg& arg1=RooCmdArg::none(), const RooCmdArg& arg2=RooCmdArg::none(),
                     const RooCmdArg& arg3=RooCmdArg::none(), const RooCmdArg& arg4=RooCmdArg::none(),
                     const RooCmdArg& arg5=RooCmdArg::none(), const RooCmdArg& arg6=RooCmdArg::none(),
                     const RooCmdArg& arg7=RooCmdArg::none(), const RooCmdArg& arg8=RooCmdArg::none()) ;
  RooPlot* plotError(const RooRealVar& param, const RooCmdArg& arg1=RooCmdArg::none(), const RooCmdArg& arg2=RooCmdArg::none(),
                     const RooCmdArg& arg3=RooCmdArg::none(), const RooCmdArg& arg4=RooCmdArg::none(),
                     const RooCmdArg& arg5=RooCmdArg::none(), const RooCmdArg& arg6=RooCmdArg::none(),
                     const RooCmdArg& arg7=RooCmdArg::none(), const RooCmdArg& arg8=RooCmdArg::none()) ;
  RooPlot* plotPull(const RooRealVar& param, const RooCmdArg& arg1, const RooCmdArg& arg2=RooCmdArg::none(),
                     const RooCmdArg& arg3=RooCmdArg::none(), const RooCmdArg& arg4=RooCmdArg::none(),
                     const RooCmdArg& arg5=RooCmdArg::none(), const RooCmdArg& arg6=RooCmdArg::none(),
                     const RooCmdArg& arg7=RooCmdArg::none(), const RooCmdArg& arg8=RooCmdArg::none()) ;


  RooPlot* plotNLL(double lo, double hi, Int_t nBins=100) ;
  RooPlot* plotError(const RooRealVar& param, double lo, double hi, Int_t nbins=100) ;
  RooPlot* plotPull(const RooRealVar& param, double lo=-3.0, double hi=3.0, Int_t nbins=25, bool fitGauss=false) ;

protected:

  friend class RooAbsMCStudyModule ;

  RooPlot* makeFrameAndPlotCmd(const RooRealVar& param, RooLinkedList& cmdList, bool symRange=false) const ;

  bool run(bool generate, bool fit, Int_t nSamples, Int_t nEvtPerSample, bool keepGenData, const char* asciiFilePat) ;
  bool fitSample(RooAbsData* genSample) ;
  RooFitResult* doFit(RooAbsData* genSample) ;

  void calcPulls() ;

  RooAbsData*       _genSample ;       ///< Currently generated sample
  RooAbsPdf*        _genModel ;        ///< Generator model
  RooAbsGenContext* _genContext ;      ///< Generator context
  RooArgSet*        _genInitParams ;   ///< List of original generator parameters
  RooArgSet*        _genParams ;       ///< List of actual generator parameters
  const RooDataSet* _genProtoData ;    ///< Generator prototype data set
  RooArgSet         _projDeps ;        ///< List of projected dependents in fit

  RooAbsPdf*        _constrPdf ;        ///< Constraints p.d.f
  RooAbsGenContext* _constrGenContext ; ///< Generator context for constraints p.d.f

  RooArgSet    _dependents ;    ///< List of dependents
  RooArgSet    _allDependents ; ///< List of generate + prototype dependents
  RooAbsPdf*   _fitModel ;      ///< Fit model
  RooArgSet*   _fitInitParams ; ///< List of initial values of fit parameters
  RooArgSet*   _fitParams ;     ///< List of actual fit parameters
  RooRealVar*  _nllVar ;
  RooRealVar*  _ngenVar ;

  TList       _genDataList ;    // List of generated data sample
  TList       _fitResList ;     // List of RooFitResult fit output objects
  RooDataSet* _genParData ;     // List of of generated parameters of each sample
  RooDataSet* _fitParData ;     // Data set of fit parameters of each sample
  RooLinkedList _fitOptList ;   // Fit option command list
  bool      _extendedGen ;    // Add poisson term to number of events to generate?
  bool      _binGenData ;     // Bin data between generating and fitting
  double    _nExpGen ;        // Number of expected events to generate in extended mode
  bool      _randProto ;      // Randomize order of prototype data access

  bool      _canAddFitResults ; ///< Allow adding of external fit results?
  bool      _verboseGen       ; ///< Verbose generation?
  bool      _perExptGenParams ; ///< Do generation parameter change per event?
  bool      _silence          ; ///< Silent running mode?

  std::list<RooAbsMCStudyModule*> _modList ; ///< List of additional study modules ;

  // Utilities for modules ;
  RooFitResult* refit(RooAbsData* genSample=nullptr) ;
  void resetFitParams() ;
  void RecursiveRemove(TObject *obj) override;

private:

  RooMCStudy(const RooMCStudy&) ;

  ClassDefOverride(RooMCStudy,0) // A general purpose toy Monte Carlo study manager
} ;


#endif

