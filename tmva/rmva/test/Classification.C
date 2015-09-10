#include <cstdlib>
#include <iostream>
#include <map>
#include <string>

#include "TChain.h"
#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TObjString.h"
#include "TSystem.h"
#include "TROOT.h"


#include "TMVA/Factory.h"
#include "TMVA/Tools.h"
#include<TMVA/MethodRXGB.h>
#include<TMVA/MethodRSNNS.h>


void Classification()
{
   TMVA::Tools::Instance();
   ROOT::R::TRInterface &r = ROOT::R::TRInterface::Instance();

   TString outfileName("TMVA.root");
   TFile *outputFile = TFile::Open(outfileName, "RECREATE");

   TMVA::Factory *factory = new TMVA::Factory("RMVAClassification", outputFile,
         "!V:!Silent:Color:DrawProgressBar:Transformations=I;D;P;G,D:AnalysisType=Classification");

   factory->AddVariable("myvar1 := var1+var2", 'F');
   factory->AddVariable("myvar2 := var1-var2", "Expression 2", "", 'F');
   factory->AddVariable("var3",                "Variable 3", "units", 'F');
   factory->AddVariable("var4",                "Variable 4", "units", 'F');
   factory->AddSpectator("spec1 := var1*2",  "Spectator 1", "units", 'F');
   factory->AddSpectator("spec2 := var1*3",  "Spectator 2", "units", 'F');

   TString fname = "./tmva_class_example.root";

   if (gSystem->AccessPathName(fname))    // file does not exist in local directory
      gSystem->Exec("curl -O http://root.cern.ch/files/tmva_class_example.root");

   TFile *input = TFile::Open(fname);

   std::cout << "--- TMVAClassification       : Using input file: " << input->GetName() << std::endl;

   // --- Register the training and test trees

   TTree *tsignal     = (TTree *)input->Get("TreeS");
   TTree *tbackground = (TTree *)input->Get("TreeB");

   // global event weights per tree (see below for setting event-wise weights)
   Double_t signalWeight     = 1.0;
   Double_t backgroundWeight = 1.0;

   // You can add an arbitrary number of signal or background trees
   factory->AddSignalTree(tsignal,     signalWeight);
   factory->AddBackgroundTree(tbackground, backgroundWeight);


   // Set individual event weights (the variables must exist in the original TTree)
   factory->SetBackgroundWeightExpression("weight");


   // Apply additional cuts on the signal and background samples (can be different)
   TCut mycuts = ""; // for example: TCut mycuts = "abs(var1)<0.5 && abs(var2-0.5)<1";
   TCut mycutb = ""; // for example: TCut mycutb = "abs(var1)<0.5";

   // Tell the factory how to use the training and testing events
   factory->PrepareTrainingAndTestTree(mycuts, mycutb,
                                       "nTrain_Signal=0:nTrain_Background=0:nTest_Signal=0:nTest_Background=0:SplitMode=Random:NormMode=NumEvents:!V");

   //R TMVA Methods
   factory->BookMethod(TMVA::Types::kC50, "C50",
                       "!H:NTrials=10:Rules=kFALSE:ControlSubSet=kFALSE:ControlBands=0:ControlWinnow=kFALSE:ControlNoGlobalPruning=kTRUE:ControlCF=0.25:ControlMinCases=2:ControlFuzzyThreshold=kTRUE:ControlSample=0:ControlEarlyStopping=kTRUE:!V");

   factory->BookMethod(TMVA::Types::kRXGB, "RXGB", "!V:NRounds=80:MaxDepth=2:Eta=1");

   factory->BookMethod(TMVA::Types::kRSNNS, "RMLP", "!H:VarTransform=N:Size=c(5):Maxit=200:InitFunc=Randomize_Weights:LearnFunc=Std_Backpropagation:LearnFuncParams=c(0.2,0):!V");

   factory->BookMethod(TMVA::Types::kRSVM, "RSVM", "!H:Kernel=linear:Type=C-classification:VarTransform=Norm:Probability=kTRUE:Tolerance=0.1:!V");


   // Train MVAs using the set of training events
   factory->TrainAllMethods();

   // ---- Evaluate all MVAs using the set of test events
   factory->TestAllMethods();

   // ----- Evaluate and compare performance of all configured MVAs
   factory->EvaluateAllMethods();
   // --------------------------------------------------------------

   // Save the output
   outputFile->Close();

   std::cout << "==> Wrote root file: " << outputFile->GetName() << std::endl;
   std::cout << "==> TMVAClassification is done!" << std::endl;

   //   delete factory;
   r.SetVerbose(1);

}
