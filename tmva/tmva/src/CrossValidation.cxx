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
#include "TMath.h"

#include <iostream>
#include <memory>

//_______________________________________________________________________
TMVA::CrossValidationResult::CrossValidationResult():fROCCurves(new TMultiGraph())
{
}

//_______________________________________________________________________
TMVA::CrossValidationResult::CrossValidationResult(const CrossValidationResult &obj)
{
   fROCs=obj.fROCs;
   fROCCurves = obj.fROCCurves;
}

//_______________________________________________________________________
TMultiGraph *TMVA::CrossValidationResult::GetROCCurves(Bool_t /*fLegend*/)
{
   return fROCCurves.get();
}

//_______________________________________________________________________
Float_t TMVA::CrossValidationResult::GetROCAverage() const
{
   Float_t avg=0;
   for(auto &roc:fROCs) avg+=roc.second;
   return avg/fROCs.size();
}

//_______________________________________________________________________
Float_t TMVA::CrossValidationResult::GetROCStandardDeviation() const
{
   // NOTE: We are using here the unbiased estimation of the standard deviation.
   Float_t std=0;
   Float_t avg=GetROCAverage();
   for(auto &roc:fROCs) std+=TMath::Power(roc.second-avg, 2);
   return TMath::Sqrt(std/float(fROCs.size()-1.0));
}

//_______________________________________________________________________
void TMVA::CrossValidationResult::Print() const
{
   TMVA::MsgLogger::EnableOutput();
   TMVA::gConfig().SetSilent(kFALSE);

   MsgLogger fLogger("CrossValidation");
   fLogger << kHEADER << " ==== Results ====" << Endl;
   for(auto &item:fROCs)
      fLogger << kINFO << Form("Fold  %i ROC-Int : %.4f",item.first,item.second) << std::endl;

   fLogger << kINFO << "------------------------" << Endl;
   fLogger << kINFO << Form("Average ROC-Int : %.4f",GetROCAverage()) << Endl;
   fLogger << kINFO << Form("Std-Dev ROC-Int : %.4f",GetROCStandardDeviation()) << Endl;

   TMVA::gConfig().SetSilent(kTRUE);
}

//_______________________________________________________________________
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

//_______________________________________________________________________
TMVA::CrossValidation::CrossValidation(TMVA::DataLoader *dataloader):TMVA::Envelope("CrossValidation",dataloader),
fNumFolds(5),fClassifier(new TMVA::Factory("CrossValidation","!V:!ROC:Silent:!ModelPersistence:!Color:!DrawProgressBar:AnalysisType=Classification"))
{
   fFoldStatus=kFALSE;
   ParseOptions();
}

//_______________________________________________________________________
TMVA::CrossValidation::~CrossValidation()
{
   fClassifier=nullptr;
}

//_______________________________________________________________________
void TMVA::CrossValidation::SetNumFolds(UInt_t i)
{
   fNumFolds=i;
   fDataLoader->MakeKFoldDataSet(fNumFolds);
   fFoldStatus=kTRUE;
}

//_______________________________________________________________________
void TMVA::CrossValidation::Evaluate()
{
   fResults.resize(fMethods.size());
   for (UInt_t j = 0; j < fMethods.size(); j++) {

      TString methodName = fMethods[j].GetValue<TString>("MethodName");
      TString methodTitle = fMethods[j].GetValue<TString>("MethodTitle");
      TString methodOptions = fMethods[j].GetValue<TString>("MethodOptions");
      if (methodName == "")
         Log() << kFATAL << "No method booked for cross-validation" << Endl;

      TMVA::MsgLogger::EnableOutput();
      TMVA::gConfig().SetSilent(kFALSE);
      Log() << kINFO << "Evaluate method: " << methodTitle << Endl;
      TMVA::gConfig().SetSilent(kTRUE);

      // Generate K folds on given dataset
      if (!fFoldStatus) {
         fDataLoader->MakeKFoldDataSet(fNumFolds);
         fFoldStatus = kTRUE;
      }

      // Process K folds
      for (UInt_t i = 0; i < fNumFolds; ++i) {
         Log() << kDEBUG << "Fold (" << methodTitle << "): " << i << Endl;
         // Get specific fold of dataset and setup method
         TString foldTitle = methodTitle;
         foldTitle += "_fold";
         foldTitle += i + 1;

         fDataLoader->PrepareFoldDataSet(i, TMVA::Types::kTesting);
         MethodBase *smethod = fClassifier->BookMethod(fDataLoader.get(), methodName, methodTitle, methodOptions);

         // Train method
         Event::SetIsTraining(kTRUE);
         smethod->TrainMethod();

         // Test method
         Event::SetIsTraining(kFALSE);
         smethod->AddOutput(Types::kTesting, smethod->GetAnalysisType());
         smethod->TestClassification();

         // Store results
         fResults[j].fROCs[i] = fClassifier->GetROCIntegral(fDataLoader->GetName(), methodTitle);

         TGraph *gr = fClassifier->GetROCCurve(fDataLoader->GetName(), methodTitle, true);
         gr->SetLineColor(i + 1);
         gr->SetLineWidth(2);
         gr->SetTitle(foldTitle.Data());
         fResults[j].fROCCurves->Add(gr);

         fResults[j].fSigs.push_back(smethod->GetSignificance());
         fResults[j].fSeps.push_back(smethod->GetSeparation());

         Double_t err;
         fResults[j].fEff01s.push_back(smethod->GetEfficiency("Efficiency:0.01", Types::kTesting, err));
         fResults[j].fEff10s.push_back(smethod->GetEfficiency("Efficiency:0.10", Types::kTesting, err));
         fResults[j].fEff30s.push_back(smethod->GetEfficiency("Efficiency:0.30", Types::kTesting, err));
         fResults[j].fEffAreas.push_back(smethod->GetEfficiency("", Types::kTesting, err));
         fResults[j].fTrainEff01s.push_back(smethod->GetTrainingEfficiency("Efficiency:0.01"));
         fResults[j].fTrainEff10s.push_back(smethod->GetTrainingEfficiency("Efficiency:0.10"));
         fResults[j].fTrainEff30s.push_back(smethod->GetTrainingEfficiency("Efficiency:0.30"));

         // Clean-up for this fold
         smethod->Data()->DeleteResults(smethod->GetMethodName(), Types::kTesting, Types::kClassification);
         smethod->Data()->DeleteResults(smethod->GetMethodName(), Types::kTraining, Types::kClassification);
         fClassifier->DeleteAllMethods();
         fClassifier->fMethodsMap.clear();
      }
   }
   TMVA::gConfig().SetSilent(kFALSE);
   Log() << kINFO << "Evaluation done." << Endl;
   TMVA::gConfig().SetSilent(kTRUE);
}

//_______________________________________________________________________
const std::vector<TMVA::CrossValidationResult> &TMVA::CrossValidation::GetResults() const
{
   if (fResults.size() == 0)
      Log() << kFATAL << "No cross-validation results available" << Endl;
   return fResults;
}
