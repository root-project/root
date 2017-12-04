// @(#)root/tmva $Id$
// Author: Omar Zapata, Thomas James Stevenson and Pourya Vakilipourtakalou
// Modified: Kim Albertsson 2017

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : Factory                                                               *
 *                                                                                *
 * Copyright (c) 2016-2018:                                                       *
 *      CERN, Switzerland                                                         *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_CROSS_EVALUATION
#define ROOT_TMVA_CROSS_EVALUATION

#include "TString.h"
#include "TMultiGraph.h"

#include "TMVA/IMethod.h"
#include "TMVA/Configurable.h"
#include "TMVA/Types.h"
#include "TMVA/DataSet.h"
#include "TMVA/Event.h"
#include <TMVA/Results.h>
#include <TMVA/Factory.h>
#include <TMVA/DataLoader.h>
#include <TMVA/OptionMap.h>
#include <TMVA/Envelope.h>

/*! \class TMVA::CrossValidationResult
 * Class to save the results of cross validation,
 * the metric for the classification ins ROC and you can ROC curves
 * ROC integrals, ROC average and ROC standard deviation.
\ingroup TMVA
*/

/*! \class TMVA::CrossValidation
 * Class to perform cross validation, splitting the dataloader into folds.
\ingroup TMVA
*/

namespace TMVA {

   class CvSplitCrossValidation;

   using EventCollection_t = std::vector<Event *>;
   using EventTypes_t      = std::vector<Bool_t>;
   using EventOutputs_t    = std::vector<Float_t>;
   using EventOutputsMulticlass_t = std::vector< std::vector<Float_t> >;

   // Used internally to keep per-fold aggregate statistics
   // such as ROC curves, ROC integrals and efficiencies.
   class CrossValidationResult {
      friend class CrossValidation;

   private:
      std::map<UInt_t,Float_t> fROCs;
      std::shared_ptr<TMultiGraph> fROCCurves;

      std::vector<Double_t> fSigs;
      std::vector<Double_t> fSeps;
      std::vector<Double_t> fEff01s;
      std::vector<Double_t> fEff10s;
      std::vector<Double_t> fEff30s;
      std::vector<Double_t> fEffAreas;
      std::vector<Double_t> fTrainEff01s;
      std::vector<Double_t> fTrainEff10s;
      std::vector<Double_t> fTrainEff30s;

   public:
      CrossValidationResult();
      CrossValidationResult(const CrossValidationResult &);
      ~CrossValidationResult(){fROCCurves=nullptr;}

      std::map<UInt_t,Float_t> GetROCValues(){return fROCs;}
      Float_t GetROCAverage() const;
      Float_t GetROCStandardDeviation() const;
      TMultiGraph *GetROCCurves(Bool_t fLegend=kTRUE);
      void Print() const ;

      TCanvas* Draw(const TString name="CrossValidation") const;

      std::vector<Double_t> GetSigValues() {return fSigs;}
      std::vector<Double_t> GetSepValues() {return fSeps;}
      std::vector<Double_t> GetEff01Values() {return fEff01s;}
      std::vector<Double_t> GetEff10Values() {return fEff10s;}
      std::vector<Double_t> GetEff30Values() {return fEff30s;}
      std::vector<Double_t> GetEffAreaValues() {return fEffAreas;}
      std::vector<Double_t> GetTrainEff01Values() {return fTrainEff01s;}
      std::vector<Double_t> GetTrainEff10Values() {return fTrainEff10s;}
      std::vector<Double_t> GetTrainEff30Values() {return fTrainEff30s;}
   };

   class CrossValidation : public Envelope {

   public:
      explicit CrossValidation(TString jobName, TMVA::DataLoader *dataloader, TString options);
      explicit CrossValidation(TString jobName, TMVA::DataLoader *dataloader, TFile * outputFile, TString options);
      ~CrossValidation();

      void InitOptions();
      void ParseOptions();

      void SetNumFolds(UInt_t i);
      void SetSplitExpr(TString splitExpr);

      UInt_t GetNumFolds() {return fNumFolds;}
      TString GetSplitExpr() {return fSplitExprString;}

      Factory & GetFactory() {return *fFactory;}

      const std::vector<CrossValidationResult> &GetResults() const;

      void Evaluate();

   private:
      void ProcessFold(UInt_t iFold, UInt_t iMethod);
      void MergeFolds();

      Types::EAnalysisType fAnalysisType;
      TString fAnalysisTypeStr;
      Bool_t fCorrelations;
      TString fCvFactoryOptions;
      Bool_t fDrawProgressBar;
      Bool_t fFoldFileOutput;  //! If true: generate output file for each fold
      Bool_t fFoldStatus; //! If true: dataset is prepared
      TString fJobName;
      UInt_t fNumFolds; //! Number of folds to prepare
      TString fOutputFactoryOptions;
      TString fOutputEnsembling; //! How to combine output of individual folds
      TFile * fOutputFile;
      Bool_t fSilent;
      TString fSplitExprString;
      std::vector<CrossValidationResult> fResults; //!
      Bool_t fROC;
      TString fTransformations;
      Bool_t fVerbose;
      TString fVerboseLevel;

      std::unique_ptr<Factory> fFoldFactory;
      std::unique_ptr<Factory> fFactory;
      std::unique_ptr<CvSplitCrossValidation> fSplit;

      ClassDef(CrossValidation, 0);
   };

} // namespace TMVA

#endif // ROOT_TMVA_CROSS_EVALUATION
