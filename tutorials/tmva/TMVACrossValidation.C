/// \file
/// \ingroup tutorial_tmva
/// \notebook -nodraw
/// This example explains how to use the cross-validation feature of TMVA. It is
/// validated the Fisher algorithm with a 5-fold cross-validation.
/// - Project   : TMVA - a Root-integrated toolkit for multivariate data analysis
/// - Package   : TMVA
/// - Exectuable: TMVACrossValidation
///
/// \macro_output
/// \macro_code
/// \author Stefan Wunsch

#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TSystem.h"

#include "TMVA/DataLoader.h"
#include "TMVA/CrossValidation.h"
#include "TMVA/Tools.h"

void TMVACrossValidation()
{
   // This loads the library
   TMVA::Tools::Instance();

   // Load data
   TFile *input(0);
   TString fname = "./tmva_class_example.root";
   if (!gSystem->AccessPathName( fname )) {
      input = TFile::Open( fname ); // check if file in local directory exists
   }
   else {
      TFile::SetCacheFileDir(".");
      input = TFile::Open("http://root.cern.ch/files/tmva_class_example.root", "CACHEREAD");
   }
   if (!input) {
      std::cout << "ERROR: could not open data file" << std::endl;
      exit(1);
   }

   TTree* signalTree = (TTree*)input->Get("TreeS");
   TTree* background = (TTree*)input->Get("TreeB");

   // Setup dataloader
   TMVA::DataLoader* dataloader = new TMVA::DataLoader("dataset");

   dataloader->AddSignalTree(signalTree);
   dataloader->AddBackgroundTree(background);

   dataloader->AddVariable("var1");
   dataloader->AddVariable("var2");
   dataloader->AddVariable("var3");
   dataloader->AddVariable("var4");

   dataloader->PrepareTrainingAndTestTree("", "SplitMode=Random:NormMode=NumEvents:!V");

   // Setup cross-validation with Fisher method
   TMVA::CrossValidation cv(dataloader);
   cv.BookMethod(TMVA::Types::kFisher, "Fisher", "!H:!V:Fisher");

   // Run cross-validation and print results
   cv.Evaluate();
   auto results = cv.GetResults();
   for (auto r : results)
      r.Print();
}

int main()
{
   TMVACrossValidation();
}
