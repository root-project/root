// @(#)root/tmva $Id$
// Author: Kim Albertsson

/*************************************************************************
 * Copyright (C) 2017, Kim Albertsson                                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////////
///
//////////////////////////////////////////////////////////////////////////////

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

namespace TMVA {

   class CvSplitCrossEvaluation;

   using EventCollection_t = std::vector<Event *>;
   using EventTypes_t      = std::vector<Bool_t>;
   using EventOutputs_t    = std::vector<Float_t>;
   using EventOutputsMulticlass_t = std::vector< std::vector<Float_t> >;

   class CrossEvaluation : public Envelope {
   public:
      explicit CrossEvaluation(TString jobName, TMVA::DataLoader *dataloader, TString options);
      explicit CrossEvaluation(TString jobName, TMVA::DataLoader *dataloader, TFile * outputFile, TString options);
      ~CrossEvaluation();

      void InitOptions();
      void ParseOptions();

      void SetNumFolds(UInt_t i);
      void SetSplitExpr(TString splitExpr);

      UInt_t GetNumFolds() {return fNumFolds;}
      TString GetSplitExpr() {return fSplitExprString;}

      Factory & GetFactory() {return *fFactory;}

      void Evaluate();

   private:
      void ProcessFold(UInt_t iFold);
      void MergeFolds();

      Types::EAnalysisType fAnalysisType;    //! Indicates 
      TString              fAnalysisTypeStr; //! Indicates
      Bool_t               fCorrelations;
      TString              fCvFactoryOptions;
      Bool_t               fDrawProgressBar;
      Bool_t               fFoldFileOutput;  //! If true: generate output file for each fold
      Bool_t               fFoldStatus;      //!
      TString              fJobName;
      UInt_t               fNumFolds;        //!
      TString              fOutputFactoryOptions;
      TString              fOutputEnsembling; //! How to combine output of individual folds
      TFile *              fOutputFile;
      Bool_t               fSilent;
      TString              fSplitExprString;
      Bool_t               fROC;
      TString              fTransformations;
      Bool_t               fVerbose;
      TString              fVerboseLevel;

      std::unique_ptr<Factory> fFoldFactory;
      std::unique_ptr<Factory> fFactory;
      std::unique_ptr<CvSplitCrossEvaluation> fSplit;

      ClassDef(CrossEvaluation, 0);
   };

} // namespace TMVA

#endif // ROOT_TMVA_CROSS_EVALUATION
