
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

#include "TMVAGui.C"

#ifndef __CINT__
#include "TMVA/Tools.h"
#include "TMVA/Factory.h"
#endif

using namespace TMVA;

void Boost(){
   TString outfileName = "boost.root";
   TFile* outputFile = TFile::Open( outfileName, "RECREATE" );
   TMVA::Factory *factory = new TMVA::Factory( "TMVAClassification", outputFile,
                                               "!V:!Silent:Color:DrawProgressBar:Transformations=I;D;P;G,D" );
   factory->AddVariable( "var0", 'F' );
   factory->AddVariable( "var1", 'F' );
   TFile *input(0);
   TString fname = "./data.root";
   if (!gSystem->AccessPathName( fname )) {
      // first we try to find tmva_example.root in the local directory
      std::cout << "--- BOOST       : Accessing " << fname << std::endl;
      input = TFile::Open( fname );
   }
   else {
      gROOT->LoadMacro( "./createData.C");
      create_circ(20000);
      cout << " created data.root with data and circle arranged in half circles"<<endl;
      input = TFile::Open( fname );
   }
   if (!input) {
      std::cout << "ERROR: could not open data file" << std::endl;
      exit(1);
   }
   TTree *signal     = (TTree*)input->Get("TreeS");
   TTree *background = (TTree*)input->Get("TreeB");
   Double_t signalWeight     = 1.0;
   Double_t backgroundWeight = 1.0;
   
   gROOT->cd( outfileName+TString(":/") );
   factory->AddSignalTree    ( signal,     signalWeight     );
   factory->AddBackgroundTree( background, backgroundWeight );
   factory->PrepareTrainingAndTestTree( "", "",
                                        "nTrain_Signal=0:nTrain_Background=0:SplitMode=Random:NormMode=NumEvents:!V" );

   TString fisher="H:!V";
   factory->BookMethod( TMVA::Types::kFisher, "Fisher", fisher );
   factory->BookMethod( TMVA::Types::kFisher, "FisherBoost", fisher+":Boost_Num=100:Boost_Type=AdaBoost" );
   factory->BookMethod( TMVA::Types::kFisher, "FisherBoostLog", fisher+":Boost_Num=100:Boost_Transform=log:Boost_Type=AdaBoost:Boost_AdaBoostBeta=1.0" );
   factory->BookMethod( TMVA::Types::kFisher, "FisherBoostLog2", fisher+":Boost_Num=100:Boost_Transform=log:Boost_Type=AdaBoost:Boost_AdaBoostBeta=2.0" );
   factory->BookMethod( TMVA::Types::kFisher, "FisherBoostStep", fisher+":Boost_Num=100:Boost_Transform=step:Boost_Type=AdaBoost:Boost_AdaBoostBeta=1.0" );
   factory->BookMethod( TMVA::Types::kFisher, "FisherBoostStep2", fisher+":Boost_Num=100:Boost_Transform=step:Boost_Type=AdaBoost:Boost_AdaBoostBeta=1.2" );
   factory->BookMethod( TMVA::Types::kFisher, "FisherBoostStep3", fisher+":Boost_Num=100:Boost_Transform=step:Boost_Type=AdaBoost:Boost_AdaBoostBeta=1.5" );

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
   
   delete factory;
   
   // Launch the GUI for the root macros
   if (!gROOT->IsBatch()) TMVAGui( outfileName );
   
   
}
