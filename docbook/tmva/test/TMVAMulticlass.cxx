/**********************************************************************************
 * Project   : TMVA - a Root-integrated toolkit for multivariate data analysis    *
 * Package   : TMVA                                                               *
 * Executable: TMVAMulticlass                                                     *
 *                                                                                *
 * This macro provides a simple example for the training and testing of the TMVA  *
 * multiclass classification                                                      *
 **********************************************************************************/

#include <cstdlib>
#include <iostream>
#include <map>
#include <string>

#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TSystem.h"
#include "TROOT.h"

#include "TMVAMultiClassGui.C"

#ifndef __CINT__
#include "TMVA/Tools.h"
#include "TMVA/Factory.h"
#endif

using namespace TMVA;

int main(int argc, char** argv )
{
   
   TMVA::Tools::Instance();
   
   //---------------------------------------------------------------
   // default MVA methods to be trained + tested
   std::map<std::string,int> Use;
   Use["MLP"]             = 1;
   Use["BDTG"]            = 1;
   Use["FDA_GA"]          = 0;
   //---------------------------------------------------------------
   
   std::cout << std::endl;
   std::cout << "==> Start TMVAMulticlass" << std::endl;
   
   if (argc>1) {
      for (std::map<std::string,int>::iterator it = Use.begin();
           it != Use.end(); it++) {
         it->second = 0;
      }
   }
   for (int i=1; i<argc; i++) {
      std::string regMethod(argv[i]);
      if (Use.find(regMethod) == Use.end()) {
         std::cout << "Method " << regMethod << " not known in TMVA under this name. Please try one of:" << std::endl;
         for (std::map<std::string,int>::iterator it = Use.begin(); it != Use.end(); it++) std::cout << it->first << " ";
         std::cout << std::endl;
         return 1;
      }
      Use[regMethod] = kTRUE;
   }

   // Create a new root output file.
   TString outfileName = "TMVAMulticlass.root";
   TFile* outputFile = TFile::Open( outfileName, "RECREATE" );
   
   TMVA::Factory *factory = new TMVA::Factory( "TMVAMulticlass", outputFile,
                                               "!V:!Silent:Color:DrawProgressBar:Transformations=I;D;P;G,D:AnalysisType=multiclass" );
   factory->AddVariable( "var1", 'F' );
   factory->AddVariable( "var2", "Variable 2", "", 'F' );
   factory->AddVariable( "var3", "Variable 3", "units", 'F' );
   factory->AddVariable( "var4", "Variable 4", "units", 'F' );

   TFile *input(0);
   TString fname = "./tmva_example_multiple_background.root";
   if (!gSystem->AccessPathName( fname )) {
      // first we try to find the file in the local directory
      std::cout << "--- TMVAMulticlass   : Accessing " << fname << std::endl;
      input = TFile::Open( fname );
   }
   else {
      cout << "Creating testdata...." << std::endl;
      gROOT->ProcessLine(".L createData.C+");
      gROOT->ProcessLine("create_MultipleBackground(2000)");
      cout << " created tmva_example_multiple_background.root for tests of the multiclass features"<<endl;
      input = TFile::Open( fname );
   }
   if (!input) {
      std::cout << "ERROR: could not open data file" << std::endl;
      exit(1);
   }

   TTree *signal      = (TTree*)input->Get("TreeS");
   TTree *background0 = (TTree*)input->Get("TreeB0");
   TTree *background1 = (TTree*)input->Get("TreeB1");
   TTree *background2 = (TTree*)input->Get("TreeB2");
   
   gROOT->cd( outfileName+TString(":/") );
   factory->AddTree    (signal,"Signal");
   factory->AddTree    (background0,"bg0");
   factory->AddTree    (background1,"bg1");
   factory->AddTree    (background2,"bg2");
   
   factory->PrepareTrainingAndTestTree( "", "SplitMode=Random:NormMode=NumEvents:!V" );

   if (Use["BDTG"]) // gradient boosted decision trees
      factory->BookMethod( TMVA::Types::kBDT, "BDTG", "!H:!V:NTrees=1000:BoostType=Grad:Shrinkage=0.10:UseBaggedGrad:GradBaggingFraction=0.50:nCuts=20:NNodesMax=8"); 
   if (Use["MLP"]) // neural network
      factory->BookMethod( TMVA::Types::kMLP, "MLP", "!H:!V:NeuronType=tanh:NCycles=300:HiddenLayers=N+5,5:TestRate=5:EstimatorType=MSE");   
   if (Use["FDA_GA"]) // functional discriminant with GA minimizer
      factory->BookMethod( TMVA::Types::kFDA, "FDA_GA", "H:!V:Formula=(0)+(1)*x0+(2)*x1+(3)*x2+(4)*x3:ParRanges=(-1,1);(-10,10);(-10,10);(-10,10);(-10,10):FitMethod=GA:PopSize=300:Cycles=3:Steps=20:Trim=True:SaveBestGen=1" );
   
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
   std::cout << "==> TMVAClassification is done!" << std::endl
             << std::endl
             << "==> To view the results, launch the GUI: \"root -l ./TMVAMultiClassGui.C\"" << std::endl
             << std::endl;
   delete factory;
   
}
