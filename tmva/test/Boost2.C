
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

void Boost2(){
   TString outfileName = "boost.root";
   TFile* outputFile = TFile::Open( outfileName, "RECREATE" );
   TMVA::Factory *factory = new TMVA::Factory( "TMVAClassification", outputFile,
                                               "!V:!Silent:Color:DrawProgressBar:Transformations=I;D;P;G,D" );
   factory->AddVariable( "var0", 'F' );
   factory->AddVariable( "var1", 'F' );
   TFile *input(0);
   TString fname = "./circledata.root";
   if (!gSystem->AccessPathName( fname )) {
      // first we try to find data.root in the local directory
      std::cout << "--- BOOST       : Accessing " << fname << std::endl;
      input = TFile::Open( fname );
   }
   else {
      gROOT->LoadMacro( "./createData.C");
      create_fullcirc(20000);
      cout << " created circledata.root with data and circle arranged in circles"<<endl;
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
                                        "nTrain_Signal=10000:nTrain_Background=10000:SplitMode=Random:NormMode=NumEvents:!V" );

   TString fisher="!H:!V";
   factory->BookMethod( TMVA::Types::kFisher, "Fisher", fisher );
   factory->BookMethod( TMVA::Types::kBDT, "BDTMitFisher", "!H:V:NTrees=150:NCuts=101:MaxDepth=1:UseFisherCuts:UseExclusiveVars:MinLinCorrForFisher=0." );
//   factory->BookMethod( TMVA::Types::kFisher, "FisherBS", fisher+":Boost_Num=100:Boost_Type=Bagging:Boost_Transform=step" );
   factory->BookMethod( TMVA::Types::kFisher, "FisherS", fisher+":Boost_Num=150:Boost_Type=AdaBoost:Boost_Transform=step" );


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
