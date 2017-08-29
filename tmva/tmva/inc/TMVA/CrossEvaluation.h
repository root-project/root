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
      explicit CrossEvaluation(TMVA::DataLoader *dataloader, TString options);
      explicit CrossEvaluation(TMVA::DataLoader *dataloader, TFile * outputFile, TString options);
      ~CrossEvaluation();

      void SetNumFolds(UInt_t i);
      UInt_t GetNumFolds() {return fNumFolds;}

      Factory & GetFactory() {return *fFactory;}

      void Evaluate();

   private:

      void StoreResults(MethodBase * smethod);
      void MergeResults(MethodBase * smethod);

      void StoreResultsMulticlass(MethodBase * smethod);
      void MergeResultsMulticlass(MethodBase * smethod);

      void ProcessFold(UInt_t iFold);
      void MergeFolds();

      
      

      Types::EAnalysisType fAnalysisType;     //! Indicates 
      Bool_t               fFoldStatus;       //!
      UInt_t               fNumFolds;         //!
      TFile *              fOutputFile;
      TString              fSplitSpectator;
      TString              fTransformations;
      Bool_t               fVerbose;
      TString              fVerboseLevel;

      std::unique_ptr<Factory> fClassifier;
      std::unique_ptr<Factory> fFactory;
      std::unique_ptr<CvSplitCrossEvaluation> fSplit;

      std::vector<EventTypes_t> fClassesPerFold;
      std::vector<EventOutputs_t> fOutputsPerFold;
      std::vector<EventOutputsMulticlass_t> fOutputsPerFoldMulticlass;

      ClassDef(CrossEvaluation, 0);
   };

} // namespace TMVA

#endif // ROOT_TMVA_CROSS_EVALUATION
