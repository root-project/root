// @(#)root/tmva $Id$
// Author: Omar Zapata and Sergei Gleyzer

/*! \class TMVA::VariableImportanceResult
\ingroup TMVA
*/

/*! \class TMVA::VariableImportance
\ingroup TMVA
*/

#include "TMVA/VariableImportance.h"

#include "TMVA/Config.h"
#include "TMVA/DataSetInfo.h"
#include "TMVA/Envelope.h"
#include "TMVA/Factory.h"
#include "TMVA/OptionMap.h"
#include "TMVA/MethodBase.h"
#include "TMVA/MethodCategory.h"
#include "TMVA/MsgLogger.h"
#include "TMVA/Types.h"
#include "TMVA/VarTransformHandler.h"
#include "ROOT/TProcessExecutor.hxx"

#include "TAxis.h"
#include "TGraph.h"
#include "TCanvas.h"
#include "TH1.h"
#include "TRandom3.h"
#include "TStyle.h"
#include "TSystem.h"
#include "TFile.h"

#include <bitset>
#include <iostream>
#include <memory>
#include <utility>
#include <unordered_map>

//number of bits for bitset
#define NBITS 32
using namespace std;
////////////////////////////////////////////////////////////////////////////////

TMVA::VariableImportanceResult::VariableImportanceResult():fImportanceValues("VariableImportance"),
                                                           fImportanceHist(nullptr)
{

}

////////////////////////////////////////////////////////////////////////////////

TMVA::VariableImportanceResult::VariableImportanceResult(const VariableImportanceResult &obj)
{
    fImportanceValues = obj.fImportanceValues;
    fImportanceHist   = obj.fImportanceHist;
}

////////////////////////////////////////////////////////////////////////////////

void TMVA::VariableImportanceResult::Print() const
{
    TMVA::MsgLogger::EnableOutput();
    TMVA::gConfig().SetSilent(kFALSE);

    MsgLogger fLogger("VariableImportance");
    if(fType==VIType::kShort)
    {
        fLogger<<kINFO<<"Variable Importance Results (Short)"<<Endl;
    }else if(fType==VIType::kAll)
    {
        fLogger<<kINFO<<"Variable Importance Results (All)"<<Endl;
    }else{
        fLogger<<kINFO<<"Variable Importance Results (Random)"<<Endl;
    }

    fImportanceValues.Print();
    TMVA::gConfig().SetSilent(kTRUE);
}

////////////////////////////////////////////////////////////////////////////////

TCanvas* TMVA::VariableImportanceResult::Draw(const TString name) const
{
    TCanvas *c=new TCanvas(name.Data());
    fImportanceHist->Draw("");
    fImportanceHist->GetXaxis()->SetTitle(" Variable Names ");
    fImportanceHist->GetYaxis()->SetTitle(" Importance (%) ");
    c->Draw();
    return c;
}

////////////////////////////////////////////////////////////////////////////////

TMVA::VariableImportance::VariableImportance(TMVA::DataLoader *dataloader):TMVA::Envelope("VariableImportance",dataloader,nullptr),fType(VIType::kShort)
{
    fClassifier=std::unique_ptr<Factory>(new TMVA::Factory("VariableImportance","!V:!ROC:!ModelPersistence:Silent:Color:!DrawProgressBar:AnalysisType=Classification"));
}

////////////////////////////////////////////////////////////////////////////////

TMVA::VariableImportance::~VariableImportance()
{
    fClassifier=nullptr;
}

////////////////////////////////////////////////////////////////////////////////

void TMVA::VariableImportance::Evaluate()
{
    TString methodName    = fMethod.GetValue<TString>("MethodName");
    TString methodTitle   = fMethod.GetValue<TString>("MethodTitle");
    TString methodOptions = fMethod.GetValue<TString>("MethodOptions");

    //NOTE: Put the type of VI Algorithm in the results Print
    if(fType==VIType::kShort)
    {
        EvaluateImportanceShort();
    }else if(fType==VIType::kAll)
    {
        EvaluateImportanceAll();
    }else{
        UInt_t nbits=fDataLoader->GetDefaultDataSetInfo().GetNVariables();
        if(nbits<10)
            Log()<<kERROR<<"Running variable importance with less that 10 varibales in Random mode "<<
                           "can to produce inconsisten results"<<Endl;
        EvaluateImportanceRandom(pow(nbits,2));
    }
    fResults.fType = fType;
    TMVA::MsgLogger::EnableOutput();
    TMVA::gConfig().SetSilent(kFALSE);
    Log()<<kINFO<<"Evaluation done."<<Endl;
    TMVA::gConfig().SetSilent(kTRUE);
}

////////////////////////////////////////////////////////////////////////////////

ULong_t TMVA::VariableImportance::Sum(ULong_t i)
{
    ULong_t sum=0;
    for(ULong_t n=0;n<i;n++) sum+=pow(2,n);
    return sum;
}

////////////////////////////////////////////////////////////////////////////////

TH1F* TMVA::VariableImportance::GetImportance(const UInt_t nbits,std::vector<Float_t> &importances,std::vector<TString> &varNames)
{
    TH1F *vihist  = new TH1F("vihist", "", nbits, 0, nbits);

    gStyle->SetOptStat(000000);

    Float_t normalization = 0.0;
    for (UInt_t i = 0; i < nbits; i++) normalization += importances[i];

    Float_t roc = 0.0;

    gStyle->SetTitleXOffset(0.4);
    gStyle->SetTitleXOffset(1.2);


    for (UInt_t i = 1; i < nbits + 1; i++) {
        roc = 100.0 * importances[i - 1] / normalization;
        vihist->GetXaxis()->SetBinLabel(i, varNames[i - 1].Data());
        vihist->SetBinContent(i, roc);
    }

    vihist->LabelsOption("v >", "X");
    vihist->SetBarWidth(0.97);
    vihist->SetFillColor(TColor::GetColor("#006600"));

    vihist->GetXaxis()->SetTitle(" Variable Names ");
    vihist->GetXaxis()->SetTitleSize(0.045);
    vihist->GetXaxis()->CenterTitle();
    vihist->GetXaxis()->SetTitleOffset(1.24);

    vihist->GetYaxis()->SetTitle(" Importance (%)");
    vihist->GetYaxis()->SetTitleSize(0.045);
    vihist->GetYaxis()->CenterTitle();
    vihist->GetYaxis()->SetTitleOffset(1.24);

    vihist->GetYaxis()->SetRangeUser(-7, 50);
    vihist->SetDirectory(0);

    return vihist;
}

////////////////////////////////////////////////////////////////////////////////

void TMVA::VariableImportance::EvaluateImportanceShort()
{
   TString methodName = fMethod.GetValue<TString>("MethodName");
   TString methodTitle = fMethod.GetValue<TString>("MethodTitle");
   TString methodOptions = fMethod.GetValue<TString>("MethodOptions");

   uint32_t x = 0;
   uint32_t y = 0;
   // getting number of variables and variable names from loader
   const UInt_t nbits = fDataLoader->GetDefaultDataSetInfo().GetNVariables();
   std::vector<TString> varNames = fDataLoader->GetDefaultDataSetInfo().GetListOfVariables();

   ULong_t range = Sum(nbits);

   // vector to save importances
   std::vector<Float_t> importances(nbits);
   for (UInt_t i = 0; i < nbits; i++)
      importances[i] = 0;

   Float_t SROC, SSROC; // computed ROC value for every Seed and SubSeed

   x = range;

   std::bitset<NBITS> xbitset(x);
   if (x == 0)
      Log() << kFATAL << "Error: need at least one variable."; // dataloader need at least one variable
   // creating loader for seed
   TMVA::DataLoader *seeddl = new TMVA::DataLoader(xbitset.to_string());

   // adding variables from seed
   for (UInt_t index = 0; index < nbits; index++) {
      if (xbitset[index])
         seeddl->AddVariable(varNames[index], 'F');
   }

   DataLoaderCopy(seeddl, fDataLoader.get());

   seeddl->PrepareTrainingAndTestTree(fDataLoader->GetDefaultDataSetInfo().GetCut("Signal"),
                                      fDataLoader->GetDefaultDataSetInfo().GetCut("Background"),
                                      fDataLoader->GetDefaultDataSetInfo().GetSplitOptions());
   // Booking Seed
   fClassifier->BookMethod(seeddl, methodName, methodTitle, methodOptions);

   // Train/Test/Evaluation
   fClassifier->TrainAllMethods();
   fClassifier->TestAllMethods();
   fClassifier->EvaluateAllMethods();

   // getting ROC
   SROC = fClassifier->GetROCIntegral(xbitset.to_string(), methodTitle);

   delete seeddl;

   fClassifier->DeleteAllMethods();
   fClassifier->fMethodsMap.clear();

   auto workItem = [&](UInt_t workerID) {
      uint32_t i = workerID;
      if (x & (1 << i)) {
         y = x & ~(1 << i);
         std::bitset<NBITS> ybitset(y);
         // need at least one variable
         // NOTE: if subssed is zero then is the special case
         // that count in xbitset is 1
         Double_t ny = log(x - y) / 0.693147;
         if (y == 0) {
            return make_pair(ny, 0.5);
         }

         // creating loader for subseed
         TMVA::DataLoader *subseeddl = new TMVA::DataLoader(ybitset.to_string());

         // adding variables from subseed
         for (UInt_t index = 0; index < nbits; index++) {
            if (ybitset[index])
               subseeddl->AddVariable(varNames[index], 'F');
         }

         // Loading Dataset
         std::vector<std::shared_ptr<TFile>> files = DataLoaderCopyMP(subseeddl, fDataLoader.get());
         subseeddl->PrepareTrainingAndTestTree(fDataLoader->GetDefaultDataSetInfo().GetCut("Signal"),
                                               fDataLoader->GetDefaultDataSetInfo().GetCut("Background"),
                                               fDataLoader->GetDefaultDataSetInfo().GetSplitOptions());
         // Booking SubSeed
         fClassifier->BookMethod(subseeddl, methodName, methodTitle, methodOptions);

         // Train/Test/Evaluation
         fClassifier->TrainAllMethods();
         fClassifier->TestAllMethods();
         fClassifier->EvaluateAllMethods();

         // getting ROC
         SSROC = fClassifier->GetROCIntegral(ybitset.to_string(), methodTitle);
         // importances[ny] += SROC - SSROC;

         delete subseeddl;
         fClassifier->DeleteAllMethods();
         fClassifier->fMethodsMap.clear();
         DataLoaderCopyMPCloseFiles(files);

         return make_pair((double)ny, (double)SSROC);
      } else
         return make_pair(-1., (double)0.);
   };
   vector<pair<double, double>> results;
   if (TMVA::gConfig().NWorkers() > 1) {
      ROOT::TProcessExecutor workers(TMVA::gConfig().NWorkers());
      results = workers.Map(workItem, ROOT::TSeqI(32));
   } else {
      for (int i = 0; i < 32; ++i) {
         auto res = workItem(i);
         results.push_back(res);
      }
   }
   for (auto res_pair : results) {
      if (res_pair.first >= 0)
         importances[res_pair.first] += SROC - res_pair.second;
   }
   Float_t normalization = 0.0;
   for (UInt_t i = 0; i < nbits; i++)
      normalization += importances[i];

   for (UInt_t i = 0; i < nbits; i++) {
      // adding values
      fResults.fImportanceValues[varNames[i]] = (100.0 * importances[i] / normalization);
      // adding sufix
      fResults.fImportanceValues[varNames[i]] = fResults.fImportanceValues.GetValue<TString>(varNames[i]) + " % ";
   }
   fResults.fImportanceHist = std::shared_ptr<TH1F>(GetImportance(nbits, importances, varNames));
}

////////////////////////////////////////////////////////////////////////////////

void TMVA::VariableImportance::EvaluateImportanceRandom(UInt_t seeds)
{
   TString methodName = fMethod.GetValue<TString>("MethodName");
   TString methodTitle = fMethod.GetValue<TString>("MethodTitle");
   TString methodOptions = fMethod.GetValue<TString>("MethodOptions");

   TRandom3 *rangen = new TRandom3(0); // Random Gen.

   uint32_t y = 0;

   // getting number of variables and variable names from loader
   const UInt_t nbits = fDataLoader->GetDefaultDataSetInfo().GetNVariables();
   std::vector<TString> varNames = fDataLoader->GetDefaultDataSetInfo().GetListOfVariables();

   ULong_t range = pow(2, nbits);

   // vector to save importances
   std::vector<Float_t> importances(nbits);

   for (UInt_t i = 0; i < nbits; i++)
      importances[i] = 0;

   Float_t SROC, SSROC; // computed ROC value for every Seed and SubSeed

   std::unordered_map<int, int> used;
   auto workItem = [&](UInt_t workerID) {

      while (true) {
         workerID = rangen->Integer(range);
         if (!used[workerID] && workerID != 0)
            break;
      }
      std::bitset<NBITS> xbitset(workerID); // dataloader need at least one variable

      used[workerID] = 1;
      // creating loader for seed
      TMVA::DataLoader *seeddl = new TMVA::DataLoader(xbitset.to_string());
      // adding variables from seed
      for (UInt_t index = 0; index < nbits; index++) {
         if (xbitset[index])
            seeddl->AddVariable(varNames[index], 'F');
      }

      // Loading Dataset
      std::vector<std::shared_ptr<TFile>> files = DataLoaderCopyMP(seeddl, fDataLoader.get());

      seeddl->PrepareTrainingAndTestTree(fDataLoader->GetDefaultDataSetInfo().GetCut("Signal"),
                                         fDataLoader->GetDefaultDataSetInfo().GetCut("Background"),
                                         fDataLoader->GetDefaultDataSetInfo().GetSplitOptions());

      // Booking Seed
      fClassifier->BookMethod(seeddl, methodName, methodTitle, methodOptions);

      // Train/Test/Evaluation
      fClassifier->TrainAllMethods();
      fClassifier->TestAllMethods();
      fClassifier->EvaluateAllMethods();

      // getting ROC
      SROC = fClassifier->GetROCIntegral(xbitset.to_string(), methodTitle);

      delete seeddl;

      fClassifier->DeleteAllMethods();
      fClassifier->fMethodsMap.clear();
      DataLoaderCopyMPCloseFiles(files);

      return make_pair(SROC, workerID);
   };

   vector<pair<float, UInt_t>> SROC_results;
   ROOT::TProcessExecutor workers(TMVA::gConfig().NWorkers());

   // Fill the pool with work
   if (TMVA::gConfig().NWorkers() > 1) {
      SROC_results = workers.Map(workItem, ROOT::TSeqI(std::min(range - 1, ULong_t(seeds))));
   } else {
      for (UInt_t i = 0; i < std::min(range - 1, ULong_t(seeds)); ++i) {
         auto res = workItem(i);
         SROC_results.push_back(res);
      }
   }

   for (auto res : SROC_results) {
      auto xx = res.second;
      auto SROC_ = res.first;
      auto workItemsub = [&](UInt_t workerIDsub) {
         uint32_t i = workerIDsub;
         if (xx & (1 << i)) {
            std::bitset<NBITS> ybitset(y);
            // need at least one variable
            // NOTE: if subssed is zero then is the special case
            // that count in xbitset is 1
            Double_t ny = log(xx - y) / 0.693147;
            if (y == 0) {
               return make_pair(ny, .5);
            }
            //creating loader for subseed
            TMVA::DataLoader *subseeddl = new TMVA::DataLoader(ybitset.to_string());

            //adding variables from subseed
            for (UInt_t index = 0; index < nbits; index++) {
               if (ybitset[index])
                  subseeddl->AddVariable(varNames[index], 'F');
            }
            // Loading Dataset
            std::vector<std::shared_ptr<TFile>> files = DataLoaderCopyMP(subseeddl, fDataLoader.get());
            subseeddl->PrepareTrainingAndTestTree(fDataLoader->GetDefaultDataSetInfo().GetCut("Signal"),
                                                  fDataLoader->GetDefaultDataSetInfo().GetCut("Background"),
                                                  fDataLoader->GetDefaultDataSetInfo().GetSplitOptions());
            // Booking SubSeed
            fClassifier->BookMethod(subseeddl, methodName, methodTitle, methodOptions);

            // Train/Test/Evaluation
            fClassifier->TrainAllMethods();
            fClassifier->TestAllMethods();
            fClassifier->EvaluateAllMethods();

            // getting ROC
            SSROC = fClassifier->GetROCIntegral(ybitset.to_string(), methodTitle);
            delete subseeddl;

            fClassifier->DeleteAllMethods();
            fClassifier->fMethodsMap.clear();
            DataLoaderCopyMPCloseFiles(files);
            return make_pair((double)ny, (double)SSROC);
         } else
            return make_pair(-1., (double)0.);
      };

      vector<pair<double, double>> results;
      if (TMVA::gConfig().NWorkers() > 1) {
         ROOT::TProcessExecutor workers_sub(TMVA::gConfig().NWorkers());
         // Fill the pool with work
         results = workers_sub.Map(workItemsub, ROOT::TSeqI(32));
      } else {
         for (int i = 0; i < 32; ++i) {
            auto res_sub = workItemsub(i);
            results.push_back(res_sub);
         }
      }
      for (auto res_pair : results) {
         importances[res_pair.first] += SROC_ - res_pair.second;
      }
   }

   Float_t normalization = 0.0;
   for (UInt_t i = 0; i < nbits; i++)
      normalization += importances[i];

   for (UInt_t i = 0; i < nbits; i++) {
      // adding values
      fResults.fImportanceValues[varNames[i]] = (100.0 * importances[i] / normalization);
      // adding sufix
      fResults.fImportanceValues[varNames[i]] = fResults.fImportanceValues.GetValue<TString>(varNames[i]) + " % ";
   }
   fResults.fImportanceHist = std::shared_ptr<TH1F>(GetImportance(nbits, importances, varNames));
   delete rangen;
}

////////////////////////////////////////////////////////////////////////////////

void TMVA::VariableImportance::EvaluateImportanceAll()
{
   TString methodName = fMethod.GetValue<TString>("MethodName");
   TString methodTitle = fMethod.GetValue<TString>("MethodTitle");
   TString methodOptions = fMethod.GetValue<TString>("MethodOptions");

   uint32_t x = 0;
   uint32_t y = 0;

   // getting number of variables and variable names from loader
   const UInt_t nbits = fDataLoader->GetDefaultDataSetInfo().GetNVariables();
   std::vector<TString> varNames = fDataLoader->GetDefaultDataSetInfo().GetListOfVariables();

   ULong_t range = pow(2, nbits);

   // vector to save importances
   std::vector<Float_t> importances(nbits);

   for (UInt_t i = 0; i < nbits; i++)
      importances[i] = 0;

   Float_t SROC, SSROC; // computed ROC value

   auto workItem = [&](UInt_t workerID) {
      Float_t ROC;
      ROC = 0.5;
      std::bitset<NBITS> xbitset(workerID);

      if (workerID == 0)
         return ROC;
      // creating loader for seed
      TMVA::DataLoader *seeddl = new TMVA::DataLoader(xbitset.to_string());
      // adding variables from seed
      for (UInt_t index = 0; index < nbits; index++) {
         if (xbitset[index])
            seeddl->AddVariable(varNames[index], 'F');
      }
      std::vector<std::shared_ptr<TFile>> files = DataLoaderCopyMP(seeddl, fDataLoader.get());
      seeddl->PrepareTrainingAndTestTree(fDataLoader->GetDefaultDataSetInfo().GetCut("Signal"),
                                         fDataLoader->GetDefaultDataSetInfo().GetCut("Background"),
                                         fDataLoader->GetDefaultDataSetInfo().GetSplitOptions());

      TMVA::gConfig().SetSilent(kFALSE);
      auto classifier = std::unique_ptr<Factory>(
         new TMVA::Factory("VariableImportanceworker",
                           "!V:!ROC:!ModelPersistence:Silent:Color:!DrawProgressBar:AnalysisType=Classification"));
      classifier->BookMethod(seeddl, methodName, methodTitle, methodOptions);

      classifier->TrainAllMethods();
      classifier->TestAllMethods();
      classifier->EvaluateAllMethods();
      // getting ROC
      ROC = classifier->GetROCIntegral(xbitset.to_string(), methodTitle);

      delete seeddl;
      classifier->DeleteAllMethods();
      classifier->fMethodsMap.clear();
      DataLoaderCopyMPCloseFiles(files);
      return ROC;
   };

   vector<float> ROC_result;
   if (TMVA::gConfig().NWorkers() > 1) {
      ROOT::TProcessExecutor workers(TMVA::gConfig().NWorkers());
      // Fill the pool with work
      ROC_result = workers.Map(workItem, ROOT::TSeqI(range));
   } else {
      for (UInt_t i = 0; i < range; ++i) {
         auto res = workItem(i);
         ROC_result.push_back(res);
      }
   }
   for (x = 0; x < range; x++) {
      SROC = ROC_result[x];
      for (uint32_t i = 0; i < NBITS; ++i) {
         if (x & (1 << i)) {
            y = x & ~(1 << i);
            std::bitset<NBITS> ybitset(y);

            Float_t ny = log(x - y) / 0.693147;
            if (y == 0) {
               importances[ny] = SROC - 0.5;
               continue;
            }

            // getting ROC
            SSROC = ROC_result[y];
            importances[ny] += SROC - SSROC;
         }
      }
   }
   Float_t normalization = 0.0;
   for (UInt_t i = 0; i < nbits; i++)
      normalization += importances[i];

   for (UInt_t i = 0; i < nbits; i++) {
      // adding values
      fResults.fImportanceValues[varNames[i]] = (100.0 * importances[i] / normalization);
      // adding sufix
      fResults.fImportanceValues[varNames[i]] = fResults.fImportanceValues.GetValue<TString>(varNames[i]) + " % ";
   }

   fResults.fImportanceHist = std::shared_ptr<TH1F>(GetImportance(nbits, importances, varNames));
}
