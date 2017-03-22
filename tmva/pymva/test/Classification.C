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
#include "TMVA/DataLoader.h"
#include "TMVA/MethodPyRandomForest.h"

void Classification()
{
   TMVA::Tools::Instance();

   TString outfileName("TMVA.root");
   TFile *outputFile = TFile::Open(outfileName, "RECREATE");

   TMVA::Factory *factory =
      new TMVA::Factory("TMVAClassification", outputFile,
                        "!V:!Silent:Color:DrawProgressBar:Transformations=I;D;P;G,D:AnalysisType=Classification");

   TMVA::DataLoader dataloader("dl");
   dataloader.AddVariable("myvar1 := var1+var2", 'F');
   dataloader.AddVariable("myvar2 := var1-var2", "Expression 2", "", 'F');
   dataloader.AddVariable("var3", "Variable 3", "units", 'F');
   dataloader.AddVariable("var4", "Variable 4", "units", 'F');

   dataloader.AddSpectator("spec1 := var1*2", "Spectator 1", "units", 'F');
   dataloader.AddSpectator("spec2 := var1*3", "Spectator 2", "units", 'F');

   TString fname = "./tmva_class_example.root";

   if (gSystem->AccessPathName(fname)) // file does not exist in local directory
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
   dataloader.AddSignalTree(tsignal, signalWeight);
   dataloader.AddBackgroundTree(tbackground, backgroundWeight);

   // Set individual event weights (the variables must exist in the original TTree)
   dataloader.SetBackgroundWeightExpression("weight");

   // Apply additional cuts on the signal and background samples (can be different)
   TCut mycuts = ""; // for example: TCut mycuts = "abs(var1)<0.5 && abs(var2-0.5)<1";
   TCut mycutb = ""; // for example: TCut mycutb = "abs(var1)<0.5";

   // Tell the factory how to use the training and testing events
   dataloader.PrepareTrainingAndTestTree(
      mycuts, mycutb,
      "nTrain_Signal=0:nTrain_Background=0:nTest_Signal=0:nTest_Background=0:SplitMode=Random:NormMode=NumEvents:!V");

   ///////////////////
   // Booking         //
   ///////////////////
   //     PyMVA methods
   factory->BookMethod(&dataloader, TMVA::Types::kPyRandomForest, "PyRandomForest",
                       "!V:NEstimators=100:Criterion=gini:MaxFeatures=auto:MaxDepth=6:MinSamplesLeaf=1:"
                       "MinWeightFractionLeaf=0:Bootstrap=kTRUE");

   factory->BookMethod(&dataloader, TMVA::Types::kPyAdaBoost, "PyAdaBoost", "!V:NEstimators=1000");

   factory->BookMethod(&dataloader, TMVA::Types::kPyGTB, "PyGTB", "!V:NEstimators=150");

   factory->BookMethod(&dataloader, TMVA::Types::kPyRFOneVsRest, "PyRFOneVsRest", "!V");

   // factory->BookMethod(&dataloader, TMVA::Types::kPyKMeans, "PyKMeans","!V:NClusters=5" );

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
}