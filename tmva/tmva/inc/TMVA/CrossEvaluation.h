// @(#)root/tmva $Id$
// Author: Kim Albertsson 2017

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

   using EventCollection_t = std::vector<Event *>;
   using EventTypes_t      = std::vector<Bool_t>;
   using EventOutputs_t    = std::vector<Float_t>;
   using EventOutputsMulticlass_t = std::vector< std::vector<Float_t> >;

   class CrossEvaluation : public Envelope {
      UInt_t                 fNumFolds;     //!
      Bool_t                 fFoldStatus;   //!
   public:
      explicit CrossEvaluation(TMVA::DataLoader *dataloader, TString splitSpectator, Types::EAnalysisType analysisType);
      explicit CrossEvaluation(TMVA::DataLoader *dataloader, TFile * outputFile, TString splitSpectator, Types::EAnalysisType analysisType);
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

      TString fSplitSpectator;
      Types::EAnalysisType fAnalysisType;

      std::unique_ptr<Factory> fClassifier;
      std::unique_ptr<Factory> fFactory;

      std::vector<EventTypes_t> fClassesPerFold;
      std::vector<EventOutputs_t> fOutputsPerFold;
      std::vector<EventOutputsMulticlass_t> fOutputsPerFoldMulticlass;

      ClassDef(CrossEvaluation, 0);
   };

} // namespace TMVA

#endif // ROOT_TMVA_CROSS_EVALUATION
