/// \file
/// \ingroup tutorial_tmva
/// \notebook -nodraw
/// This macro provides an example of how to use Cross Evaluation.
///
/// As input data is used a toy-MC sample consisting of four Gaussian-distributed
/// and linearly correlated input variables.
///
/// The output file "TMVA.root" can be analysed with the use of dedicated
/// macros (simply say: root -l <macro.C>), which can be conveniently
/// invoked through a GUI that will appear at the end of the run of this macro.
/// Launch the GUI via the command:
///
///     root -l
///     TMVA::TMVAGui("TMVA.root")
///
/// - Project   : TMVA - a ROOT-integrated toolkit for multivariate data analysis
/// - Package   : TMVA
/// - Root Macro: TMVACrossEvaluation
///
/// \macro_output
/// \macro_code
/// \author Kim Albertsson (adapted from code originally by Andreas Hoecker)


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
#include "TMVA/DataLoader.h"
#include "TMVA/Tools.h"
#include "TMVA/TMVAGui.h"

int TMVACrossEvaluation()
{
   // This loads the library
   TMVA::Tools::Instance();

   // --------------------------------------------------------------------------------------------------

   // Here the preparation phase begins

   // Read training and test data
   // (it is also possible to use ASCII format as input -> see TMVA Users Guide)
   TFile *input(0);
   TString fname = "./tmva_class_example.root";
   if (!gSystem->AccessPathName( fname )) {
      input = TFile::Open( fname ); // check if file in local directory exists
   } else {
      TFile::SetCacheFileDir(".");
      input = TFile::Open("http://root.cern.ch/files/tmva_class_example.root", "CACHEREAD");
   }
   if (!input) {
      std::cout << "ERROR: could not open data file" << std::endl;
      exit(1);
   }
   std::cout << "--- TMVACrossEvaluation       : Using input file: " << input->GetName() << std::endl;

   // Register the training and test trees

   TTree *signalTree     = (TTree*)input->Get("TreeS");
   TTree *background     = (TTree*)input->Get("TreeB");

   // Create a ROOT output file where TMVA will store ntuples, histograms, etc.
   TString outfileName( "TMVA.root" );
   TFile* outputFile = TFile::Open( outfileName, "RECREATE" );

   TMVA::DataLoader * dataloader = new TMVA::DataLoader("dataset");
   
   dataloader->AddVariable( "myvar1 := var1+var2", 'F' );
   dataloader->AddVariable( "myvar2 := var1-var2", "Expression 2", "", 'F' );
   dataloader->AddVariable( "var3",                "Variable 3", "units", 'F' );
   dataloader->AddVariable( "var4",                "Variable 4", "units", 'F' );

   dataloader->AddSpectator( "spec1 := var1*2",  "Spectator 1", "units", 'F' );
   dataloader->AddSpectator( "spec2 := var1*3",  "Spectator 2", "units", 'F' );

   dataloader->AddSignalTree    ( signalTree, 1.0 );
   dataloader->AddBackgroundTree( background, 1.0 );
   dataloader->SetBackgroundWeightExpression( "weight" );
   dataloader->PrepareTrainingAndTestTree( "", "", "nTest_Signal=1"
                                                   ":nTest_Background=1"
                                                   ":SplitMode=Random"
                                                   ":NormMode=NumEvents"
                                                   ":!V");

   // Setting up the CrossEvaluation context (which wraps a TMVA::Factory 
   // internally).
   // New options for the CE context are
   //    - SplitExpr
   //    - NumFolds
   // NumFolds controls how many parts the input data is split into. These parts
   // are later reassembled in a leave-one-out fashion to make NumFolds 
   // different independent test sets.
   // Split is an expression that is evaluated per event and indicates what fold
   // that event should be 
   // to assign an event to a fold. The calculation is
   //    fold = spectatorValue % numFolds;
   // The idea here is that spectatorValue should be something like an event
   // number, that is integral, random and independent from actual data values.
   // This last property ensures that if a calibration is changed the same event
   // will still be assigned the same fold.
   
   UInt_t numFolds = 2;
   TString analysisType = "Classification";
   TString splitExpr = "int(fabs([spec1]))%int([NumFolds])";

   TString methodOptions = Form("!V"
                                ":!Silent"
                                ":ModelPersistence"
                                ":AnalysisType=%s"
                                ":NumFolds=%i"
                                ":SplitExpr=%s",
                                analysisType.Data(),
                                numFolds,
                                splitExpr.Data());

   TMVA::CrossEvaluation ce {"TMVACrossEvaluation", dataloader, outputFile, methodOptions};

   // ### Book MVA methods
   // Boosted Decision Trees
   ce.BookMethod( TMVA::Types::kBDT, "BDTG",
                        "!H:!V:NTrees=1000:MinNodeSize=2.5%:BoostType=Grad:Shrinkage=0.10:nCuts=20:MaxDepth=2" );
   
   // --------------------------------------------------------------------------------------------------

   // Now you can train, test and evaluate the performance of the booked method
   
   // ce.TrainAllMethods();    // Train MVAs using the set of training events
   // ce.TestAllMethods();     // Evaluate all MVAs using the set of test events
   // ce.EvaluateAllMethods(); // Evaluate and compare performance of all configured MVAs

   ce.Evaluate(); // Does all of the three above combined.

   // --------------------------------------------------------------

   // Save the output
   outputFile->Close();

   std::cout << "==> Wrote root file: " << outputFile->GetName() << std::endl;
   std::cout << "==> TMVACrossEvaluation is done!" << std::endl;

   // Launch the GUI for the root macros
   if (!gROOT->IsBatch()) TMVA::TMVAGui( outfileName );

   return 0;
}

int main( int argc, char** argv )
{
   TMVACrossEvaluation();
}
