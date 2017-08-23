// @(#)root/tmva $Id$
// Author: Omar Zapata, Thomas James Stevenson.

#include "TMVA/HyperParameterOptimisation.h"

#include "TMVA/Configurable.h"
#include "TMVA/DataSet.h"
#include "TMVA/Event.h"
#include "TMVA/MethodBase.h"
#include "TMVA/ResultsClassification.h"
#include "TMVA/Types.h"

#include "TGraph.h"
#include "TMultiGraph.h"
#include "TString.h"
#include "TSystem.h"
#include "ROOT/TProcessExecutor.hxx"

#include <iostream>
#include <memory>
#include <vector>

using namespace std;
/*! \class TMVA::HyperParameterOptimisationResult
\ingroup TMVA

*/

/*! \class TMVA::HyperParameterOptimisation
\ingroup TMVA

*/

TMVA::HyperParameterOptimisationResult::HyperParameterOptimisationResult()
   : fROCAVG(0.0), fROCCurves(std::make_shared<TMultiGraph>())
{
}

TMVA::HyperParameterOptimisationResult::~HyperParameterOptimisationResult()
{
}

TMultiGraph *TMVA::HyperParameterOptimisationResult::GetROCCurves(Bool_t /* fLegend */)
{

    return fROCCurves.get();
}

void TMVA::HyperParameterOptimisationResult::Print() const
{
    TMVA::MsgLogger::EnableOutput();
    TMVA::gConfig().SetSilent(kFALSE);

    MsgLogger fLogger("HyperParameterOptimisation");

    for(UInt_t j=0; j<fFoldParameters.size(); ++j) {
        fLogger<<kHEADER<< "===========================================================" << Endl;
        fLogger<<kINFO<< "Optimisation for " << fMethodName << " fold " << j+1 << Endl;

        for(auto &it : fFoldParameters.at(j)) {
            fLogger<<kINFO<< it.first << "     " << it.second << Endl;
        }
    }

    TMVA::gConfig().SetSilent(kTRUE);

}

TMVA::HyperParameterOptimisation::HyperParameterOptimisation(TMVA::DataLoader *dataloader)
   : Envelope("HyperParameterOptimisation", dataloader), fFomType("Separation"), fFitType("Minuit"), fNumFolds(4),
     fResults(), fClassifier(new TMVA::Factory(
                    "HyperParameterOptimisation",
                    "!V:!ROC:Silent:!ModelPersistence:!Color:!DrawProgressBar:AnalysisType=Classification"))
{
    fFoldStatus=kFALSE;
}

TMVA::HyperParameterOptimisation::~HyperParameterOptimisation()
{
    fClassifier=nullptr;
}

void TMVA::HyperParameterOptimisation::SetNumFolds(UInt_t i)
{
    fNumFolds=i;
    fDataLoader->MakeKFoldDataSet(fNumFolds);
    fFoldStatus=kTRUE;
}

void TMVA::HyperParameterOptimisation::Evaluate()
{
   cout << "Number of Workers : " << TMVA::gConfig().NWorkers() << endl;
   TString methodName = fMethod.GetValue<TString>("MethodName");
   TString methodTitle = fMethod.GetValue<TString>("MethodTitle");
   TString methodOptions = fMethod.GetValue<TString>("MethodOptions");

   if (!fFoldStatus) {
      fDataLoader->MakeKFoldDataSet(fNumFolds);
      fFoldStatus = kTRUE;
   }
   fResults.fMethodName = methodName;
   auto workItem = [&](UInt_t workerID) {

      TString foldTitle = methodTitle;

      foldTitle += "_opt";
      foldTitle += workerID + 1;

      Event::SetIsTraining(kTRUE);
      fDataLoader->PrepareFoldDataSet(workerID, TMVA::Types::kTraining);

      auto smethod = fClassifier->BookMethod(fDataLoader.get(), methodName, methodTitle, methodOptions);

      auto params = smethod->OptimizeTuningParameters(fFomType, fFitType);

      smethod->Data()->DeleteResults(smethod->GetMethodName(), Types::kTraining, Types::kClassification);

      fClassifier->DeleteAllMethods();

      fClassifier->fMethodsMap.clear();

      return params;

   };
   vector<map<TString, Double_t>> res;
   auto nWorkers = TMVA::gConfig().NWorkers();
   if (nWorkers > 1) {
      ROOT::TProcessExecutor workers(nWorkers);
      res = workers.Map(workItem, ROOT::TSeqI(fNumFolds));
   } else {
      for (UInt_t i = 0; i < fNumFolds; ++i) {
         res.push_back(workItem(i));
      }
   }
   for (auto results : res) {
      fResults.fFoldParameters.push_back(results);
   }
}
