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

#include <iostream>
#include <vector>

/*! \class TMVA::HyperParameterOptimisationResult
\ingroup TMVA

*/

/*! \class TMVA::HyperParameterOptimisation
\ingroup TMVA

*/

TMVA::HyperParameterOptimisationResult::HyperParameterOptimisationResult():fROCCurves(new TMultiGraph())
{
}

TMVA::HyperParameterOptimisationResult::~HyperParameterOptimisationResult()
{
    fROCCurves=nullptr;
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

TMVA::HyperParameterOptimisation::HyperParameterOptimisation(TMVA::DataLoader *dataloader):Envelope("HyperParameterOptimisation",dataloader),
    fFomType("Separation"),
    fFitType("Minuit"),
    fNumFolds(5),
    fResults(),
    fClassifier(new TMVA::Factory("HyperParameterOptimisation","!V:!ROC:Silent:!ModelPersistence:!Color:!DrawProgressBar:AnalysisType=Classification"))
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
    TString methodName    = fMethod.GetValue<TString>("MethodName");
    TString methodTitle   = fMethod.GetValue<TString>("MethodTitle");
    TString methodOptions = fMethod.GetValue<TString>("MethodOptions");

    if(!fFoldStatus)
    {
        fDataLoader->MakeKFoldDataSet(fNumFolds);
        fFoldStatus=kTRUE;
    }
    fResults.fMethodName = methodName;

    for(UInt_t i = 0; i < fNumFolds; ++i) {

        TString foldTitle = methodTitle;
        foldTitle += "_opt";
        foldTitle += i+1;

        Event::SetIsTraining(kTRUE);
        fDataLoader->PrepareFoldDataSet(i, TMVA::Types::kTraining);

        auto smethod = fClassifier->BookMethod(fDataLoader.get(), methodName, methodTitle, methodOptions);

        auto params=smethod->OptimizeTuningParameters(fFomType,fFitType);
        fResults.fFoldParameters.push_back(params);

        smethod->Data()->DeleteResults(smethod->GetMethodName(), Types::kTraining, Types::kClassification);

        fClassifier->DeleteAllMethods();

        fClassifier->fMethodsMap.clear();

    }

}
