// @(#)root/tmva $Id$
// Author: Omar Zapata, Thomas James Stevenson.

#include "TMVA/CrossValidation.h"

#include "TMVA/Config.h"
#include "TMVA/DataSet.h"
#include "TMVA/Event.h"
#include "TMVA/MethodBase.h"
#include "TMVA/MsgLogger.h"
#include "TMVA/ResultsClassification.h"
#include "TMVA/tmvaglob.h"
#include "TMVA/Types.h"

#include "TSystem.h"
#include "TAxis.h"
#include "TCanvas.h"
#include "TGraph.h"

#include <iostream>
#include <memory>

TMVA::CrossValidationResult::CrossValidationResult():fROCCurves(new TMultiGraph())
{
}

TMVA::CrossValidationResult::CrossValidationResult(const CrossValidationResult &obj)
{
    fROCs=obj.fROCs;
    fROCCurves = obj.fROCCurves;
}


TMultiGraph *TMVA::CrossValidationResult::GetROCCurves(Bool_t /*fLegend*/)
{
    return fROCCurves.get();
}

Float_t TMVA::CrossValidationResult::GetROCAverage() const
{
    Float_t avg=0;
    for(auto &roc:fROCs) avg+=roc.second;
    return avg/fROCs.size();
}


void TMVA::CrossValidationResult::Print() const
{
    TMVA::MsgLogger::EnableOutput();
    TMVA::gConfig().SetSilent(kFALSE);   
    
    MsgLogger fLogger("CrossValidation");
    fLogger<<kHEADER<<" ==== Results ===="<<Endl;
    for(auto &item:fROCs)
        fLogger<<kINFO<<Form("Fold  %i ROC-Int : %f",item.first,item.second)<<std::endl;
    
    fLogger<<kINFO<<Form("Average ROC-Int : %f",GetROCAverage())<<Endl;

    TMVA::gConfig().SetSilent(kTRUE);   
    
}


TCanvas* TMVA::CrossValidationResult::Draw(const TString name) const
{
    TCanvas *c=new TCanvas(name.Data());
    fROCCurves->Draw("AL");
    fROCCurves->GetXaxis()->SetTitle(" Signal Efficiency ");
    fROCCurves->GetYaxis()->SetTitle(" Background Rejection ");
    Float_t adjust=1+fROCs.size()*0.01;
    c->BuildLegend(0.15,0.15,0.4*adjust,0.5*adjust);
    c->SetTitle("Cross Validation ROC Curves");
    c->Draw();
    return c;
}

TMVA::CrossValidation::CrossValidation(TMVA::DataLoader *dataloader):TMVA::Envelope("CrossValidation",dataloader),
fNumFolds(5),fClassifier(new TMVA::Factory("CrossValidation","!V:!ROC:Silent:!ModelPersistence:!Color:!DrawProgressBar:AnalysisType=Classification"))
{
    fFoldStatus=kFALSE;
}

TMVA::CrossValidation::~CrossValidation()
{
    fClassifier=nullptr;
}

void TMVA::CrossValidation::SetNumFolds(UInt_t i)
{
    fNumFolds=i;
    fDataLoader->MakeKFoldDataSet(fNumFolds);
    fFoldStatus=kTRUE;
}


void TMVA::CrossValidation::Evaluate()
{
    TString methodName    = fMethod.GetValue<TString>("MethodName");
    TString methodTitle   = fMethod.GetValue<TString>("MethodTitle");
    TString methodOptions = fMethod.GetValue<TString>("MethodOptions");
    if(!fFoldStatus)
    {
        fDataLoader->MakeKFoldDataSet(fNumFolds);
        fFoldStatus=kTRUE;
    }
  
      for(UInt_t i = 0; i < fNumFolds; ++i){    
        TString foldTitle = methodTitle;
        foldTitle += "_fold";
        foldTitle += i+1;
    
        fDataLoader->PrepareFoldDataSet(i, TMVA::Types::kTesting);

    
        auto smethod=fClassifier->BookMethod(fDataLoader.get(), methodName, methodTitle, methodOptions);

        Event::SetIsTraining(kTRUE);
        smethod->TrainMethod();

        Event::SetIsTraining(kFALSE);
        smethod->AddOutput(Types::kTesting, smethod->GetAnalysisType());
        smethod->TestClassification();

        
        fResults.fROCs[i]=fClassifier->GetROCIntegral(fDataLoader->GetName(),methodTitle);

        auto  gr=fClassifier->GetROCCurve(fDataLoader->GetName(), methodTitle, true);
    
        gr->SetLineColor(i+1);
        gr->SetLineWidth(2);
        gr->SetTitle(foldTitle.Data());
    
        fResults.fROCCurves->Add(gr);
    
        fResults.fSigs.push_back(smethod->GetSignificance());
        fResults.fSeps.push_back(smethod->GetSeparation());
        
        Double_t err;
        fResults.fEff01s.push_back(smethod->GetEfficiency("Efficiency:0.01",Types::kTesting, err));
        fResults.fEff10s.push_back(smethod->GetEfficiency("Efficiency:0.10",Types::kTesting,err));
        fResults.fEff30s.push_back(smethod->GetEfficiency("Efficiency:0.30",Types::kTesting,err));
        fResults.fEffAreas.push_back(smethod->GetEfficiency(""             ,Types::kTesting,err));
        fResults.fTrainEff01s.push_back(smethod->GetTrainingEfficiency("Efficiency:0.01"));
        fResults.fTrainEff10s.push_back(smethod->GetTrainingEfficiency("Efficiency:0.10"));
        fResults.fTrainEff30s.push_back(smethod->GetTrainingEfficiency("Efficiency:0.30"));
    
        smethod->Data()->DeleteResults(smethod->GetMethodName(), Types::kTesting, Types::kClassification);
        smethod->Data()->DeleteResults(smethod->GetMethodName(), Types::kTraining, Types::kClassification);

        fClassifier->DeleteAllMethods();
        fClassifier->fMethodsMap.clear();
        }
        TMVA::MsgLogger::EnableOutput();
        TMVA::gConfig().SetSilent(kFALSE);   
        Log()<<kINFO<<"Evaluation done."<<Endl;
        TMVA::gConfig().SetSilent(kTRUE);   
        

}
