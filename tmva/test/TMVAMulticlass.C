
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
#include "TPluginManager.h"

#include "TMVAGui.C"

#ifndef __CINT__
#include "TMVA/Tools.h"
#include "TMVA/Factory.h"
#endif

using namespace TMVA;

void TMVAMulticlass(){
   TString outfileName = "TMVAMulticlass.root";
   TFile* outputFile = TFile::Open( outfileName, "RECREATE" );
   TMVA::Factory *factory = new TMVA::Factory( "TMVAClassification", outputFile,
                                               "!V:!Silent:Color:DrawProgressBar:Transformations=I;D;P;G,D:AnalysisType=multiclass" );
   factory->AddVariable( "var0", 'F' );
   factory->AddVariable( "var1", 'F' );
   TFile *input(0);
   TString fname = "./data.root";
   if (!gSystem->AccessPathName( fname )) {
      // first we try to find data.root in the local directory
      std::cout << "--- TMVAMulticlass   : Accessing " << fname << std::endl;
      input = TFile::Open( fname );
   }
   else {
      gROOT->LoadMacro( "./createData.C");
      create_multiclassdata(20000);
      cout << " created data.root for tests of the multiclass features"<<endl;
      input = TFile::Open( fname );
   }
   if (!input) {
      std::cout << "ERROR: could not open data file" << std::endl;
      exit(1);
   }
   TTree *tree     = (TTree*)input->Get("TreeR");
   
   gROOT->cd( outfileName+TString(":/") );
   factory->AddTree    ( tree, "Signal1",    1. , "cls==0"   );
   factory->AddTree    ( tree, "Signal2",    1. , "cls==1"   );
   factory->AddTree    ( tree, "Background",    1., "cls==2" );
   factory->PrepareTrainingAndTestTree( "", "SplitMode=Random:NormMode=NumEvents:!V" );

   factory->BookMethod( TMVA::Types::kBDT, "BDT", "!H:!V:NTrees=1000:BoostType=Grad:Shrinkage=0.30:UseBaggedGrad:GradBaggingFraction=0.6:SeparationType=GiniIndex:nCuts=20:NNodesMax=5");
   factory->BookMethod( TMVA::Types::kMLP, "MLP", "!H:!V:NeuronType=tanh:VarTransform=N:NCycles=100:HiddenLayers=N+5,3:TestRate=5"); // testing vartransforms
   factory->BookMethod( TMVA::Types::kMLP, "MLP2", "!H:!V:NeuronType=tanh:NCycles=100:HiddenLayers=N+5,3:TestRate=5");
   factory->BookMethod( TMVA::Types::kFDA, "FDA_GA",
                        "H:!V:Formula=(0)+(1)*x0+(2)*x1+(3)*x0*x1:ParRanges=(-1,1);(-10,10);(-10,10);(-10,10):FitMethod=GA:PopSize=300:Cycles=3:Steps=20:Trim=True:SaveBestGen=1" );
   
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
