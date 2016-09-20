// @(#)root/tmva $Id$
// Author: Omar Zapata, Thomas James Stevenson, Pourya Vakilipourtakalou.

#include <iostream>

#include "TMVA/CrossValidation.h"
#include "TMVA/MethodBase.h"
#include "TMVA/ResultsClassification.h"
#include "TSystem.h"
#include "TAxis.h"
#include "TCanvas.h"
#include "TGraph.h"
#include <memory>
#include "TMVA/tmvaglob.h"

TMVA::CrossValidationResult::CrossValidationResult():fROCCurves(new TMultiGraph())
{
}

TMVA::CrossValidationResult::CrossValidationResult(const CrossValidationResult &obj)
{
    fROCs=obj.fROCs;
    fROCCurves = obj.fROCCurves;
}

TMVA::CrossValidationResult::~CrossValidationResult()
{
    fROCCurves=nullptr;
}

std::shared_ptr<TMultiGraph> &TMVA::CrossValidationResult::GetROCCurves()
{
    return fROCCurves;
}

void TMVA::CrossValidationResult::SetROCValue(UInt_t fold,Float_t rocint)
{
    fROCs[fold]=rocint;
}


Float_t TMVA::CrossValidationResult::GetROCAverage() const
{
    Float_t avg=0;
    for(auto &roc:fROCs) avg+=roc.second;
    return avg/fROCs.size();
}


void TMVA::CrossValidationResult::Print() const
{    
    MsgLogger fLogger("CrossValidation");
    for(auto &item:fROCs)
        fLogger<<kINFO<<Form("Fold  %i ROC-Int : %f",item.first,item.second)<<std::endl;
    
    fLogger<<kINFO<<Form("Average ROC-Int : %f",GetROCAverage())<<Endl;

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

//CrossValidation class stuff                                                                                                                                                        
// ClassImp(TMVA::CrossValidation)//serialization is not support yet in so many class TMVA                                                                                           

TMVA::CrossValidation::CrossValidation():Configurable( ),
					 fDataLoader(0)
{
  fClassifier=new TMVA::Factory("CrossValidation","!V:Silent:Color:DrawProgressBar:AnalysisType=Classification");
}


TMVA::CrossValidation::CrossValidation(TMVA::DataLoader *loader):Configurable(),
								 fDataLoader(loader)
{
  fClassifier=new TMVA::Factory("CrossValidation","!V:Silent:Color:DrawProgressBar:AnalysisType=Classification");
}

TMVA::CrossValidation::~CrossValidation()
{
  if(fClassifier) delete fClassifier;
}

TMVA::CrossValidationResult* TMVA::CrossValidation::CrossValidate( TString theMethodName, TString methodTitle, TString theOption, int NumFolds)
{
  
  CrossValidationResult * result = new CrossValidationResult();

  fDataLoader->MakeKFoldDataSet(NumFolds);

  for(Int_t i = 0; i < NumFolds; ++i){

    TString foldTitle = methodTitle;
    foldTitle += "_fold";
    foldTitle += i+1;

    fDataLoader->PrepareFoldDataSet(i, TMVA::Types::kTesting);

    fClassifier->BookMethod(fDataLoader, theMethodName, methodTitle, theOption);

    TMVA::MethodBase * smethod = dynamic_cast<TMVA::MethodBase*>(fClassifier->fMethodsMap[fDataLoader->GetName()][0][0]);

    Event::SetIsTraining(kTRUE);
    smethod->TrainMethod();

    Event::SetIsTraining(kFALSE);
    smethod->AddOutput(Types::kTesting, smethod->GetAnalysisType());
    smethod->TestClassification();

    result->SetROCValue(i,fClassifier->GetROCIntegral(fDataLoader->GetName(), methodTitle));
    auto  gr=fClassifier->GetROCCurve(fDataLoader->GetName(), methodTitle, true);
    gr->SetLineColor(i+1);
    gr->SetLineWidth(2);
    gr->SetTitle(fDataLoader->GetName());
        
    result->GetROCCurves()->Add(gr);

    result->fSigs.push_back(smethod->GetSignificance());
    result->fSeps.push_back(smethod->GetSeparation());
    Double_t err;
    result->fEff01s.push_back(smethod->GetEfficiency("Efficiency:0.01",Types::kTesting, err));
    result->fEff10s.push_back(smethod->GetEfficiency("Efficiency:0.10",Types::kTesting,err));
    result->fEff30s.push_back(smethod->GetEfficiency("Efficiency:0.30",Types::kTesting,err));
    result->fEffAreas.push_back(smethod->GetEfficiency(""             ,Types::kTesting,err));
    result->fTrainEff01s.push_back(smethod->GetTrainingEfficiency("Efficiency:0.01"));
    result->fTrainEff10s.push_back(smethod->GetTrainingEfficiency("Efficiency:0.10"));
    result->fTrainEff30s.push_back(smethod->GetTrainingEfficiency("Efficiency:0.30"));

    smethod->Data()->DeleteResults(smethod->GetMethodName(), Types::kTesting, Types::kClassification);
    smethod->Data()->DeleteResults(smethod->GetMethodName(), Types::kTraining, Types::kClassification);

    fClassifier->DeleteAllMethods();

    fClassifier->fMethodsMap.clear();

  }
  
  for(int r = 0; r < NumFolds; ++r){
    result->fROCAVG += result->fROCs.at(r);
  }
  result->fROCAVG /= NumFolds;

  return result;
}
