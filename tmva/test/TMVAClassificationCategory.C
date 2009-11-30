// @(#)root/tmva $Id: TMVAClassificationCategory.C,v 1.36 2009-04-14 13:08:13 andreas.hoecker Exp $
/**********************************************************************************
 * Project   : TMVA - a Root-integrated toolkit for multivariate data analysis    *
 * Package   : TMVA                                                               *
 * Root Macro: TMVAClassificationCategory                                         *
 *                                                                                *
 * This macro provides examples for the training and testing of the               *
 * TMVA classifiers in categorisation mode.                                       *
 *                                                                                *
 * As input data is used a toy-MC sample consisting of four Gaussian-distributed  *
 * and linearly correlated input variables with category (eta) dependent          *
 * properties.                                                                    *
 *                                                                                * 
 * For this example, only Fisher and Likelihood are used. Run via:                *
 *                                                                                *
 *    root -l TMVAClassificationCategory.C                                        *
 *                                                                                *
 * The output file "TMVA.root" can be analysed with the use of dedicated          *
 * macros (simply say: root -l <macro.C>), which can be conveniently              *
 * invoked through a GUI that will appear at the end of the run of this macro.    *
 **********************************************************************************/

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

#if not defined(__CINT__) || defined(__MAKECINT__)
// needs to be included when makecint runs (ACLIC)
#include "TMVA/Factory.h"
#include "TMVA/Tools.h"
#endif

// two types of category methods are implemented
Bool_t UseOffsetMethod = kTRUE;

void TMVAClassificationCategory() 
{
   //---------------------------------------------------------------

   std::cout << std::endl << "==> Start TMVAClassificationCategory" << std::endl;

   bool batchMode(false);

   // Create a new root output file.
   TString outfileName( "TMVA.root" );
   TFile* outputFile = TFile::Open( outfileName, "RECREATE" );

   // Create the factory object. Later you can choose the methods
   // whose performance you'd like to investigate. The factory will
   // then run the performance analysis for you.
   //
   // The first argument is the base of the name of all the
   // weightfiles in the directory weight/ 
   //
   // The second argument is the output file for the training results
   // All TMVA output can be suppressed by removing the "!" (not) in 
   // front of the "Silent" argument in the option string
   std::string factoryOptions( "!V:!Silent:Transformations=I;D;P;G,D" );
   if (batchMode) factoryOptions += ":!Color:!DrawProgressBar";

   TMVA::Factory *factory = new TMVA::Factory( "TMVAClassificationCategory", outputFile, factoryOptions );

   // If you wish to modify default settings 
   // (please check "src/Config.h" to see all available global options)
   //    (TMVA::gConfig().GetVariablePlotting()).fTimesRMS = 8.0;
   //    (TMVA::gConfig().GetIONames()).fWeightFileDir = "myWeightDirectory";

   // Define the input variables that shall be used for the MVA training
   // note that you may also use variable expressions, such as: "3*var1/var2*abs(var3)"
   // [all types of expressions that can also be parsed by TTree::Draw( "expression" )]
   factory->AddVariable( "var1", 'F' );
   factory->AddVariable( "var2", 'F' );
   factory->AddVariable( "var3", 'F' );
   factory->AddVariable( "var4", 'F' );

   // You can add so-called "Spectator variables", which are not used in the MVA training, 
   // but will appear in the final "TestTree" produced by TMVA. This TestTree will contain the 
   // input variables, the response values of all trained MVAs, and the spectator variables
   factory->AddSpectator( "eta" );

   // load the signal and background event samples from ROOT trees
   TFile *input(0);
   TString fname( "" );
   if (UseOffsetMethod) fname = "../execs/data/toy_sigbkg_categ_offset.root";
   else                 fname = "../execs/data/toy_sigbkg_categ_varoff.root";
   if (!gSystem->AccessPathName( fname )) {
      // first we try to find tmva_example.root in the local directory
      std::cout << "--- TMVAClassificationCategory: Accessing " << fname << std::endl;
      input = TFile::Open( fname );
   } 

   if (!input) {
      std::cout << "ERROR: could not open data file: " << fname << std::endl;
      exit(1);
   }

   TTree *signal     = (TTree*)input->Get("TreeS");
   TTree *background = (TTree*)input->Get("TreeB");

   /// global event weights per tree (see below for setting event-wise weights)
   Double_t signalWeight     = 1.0;
   Double_t backgroundWeight = 1.0;
   
   /// you can add an arbitrary number of signal or background trees
   factory->AddSignalTree    ( signal,     signalWeight     );
   factory->AddBackgroundTree( background, backgroundWeight );
   
   // Apply additional cuts on the signal and background samples (can be different)
   TCut mycuts = ""; // for example: TCut mycuts = "abs(var1)<0.5 && abs(var2-0.5)<1";
   TCut mycutb = ""; // for example: TCut mycutb = "abs(var1)<0.5";

   // tell the factory to use all remaining events in the trees after training for testing:
   factory->PrepareTrainingAndTestTree( mycuts, mycutb,
                                        "nTrain_Signal=0:nTrain_Background=0:SplitMode=Random:NormMode=NumEvents:!V" );

   // Fisher discriminant   
   factory->BookMethod( TMVA::Types::kFisher, "Fisher", "!H:!V:Fisher" );

   // Likelihood
   factory->BookMethod( TMVA::Types::kLikelihood, "Likelihood", 
                        "!H:!V:TransformOutput:PDFInterpol=Spline2:NSmoothSig[0]=20:NSmoothBkg[0]=20:NSmoothBkg[1]=10:NSmooth=1:NAvEvtPerBin=50" ); 

   // Categorised classifier
   TMVA::MethodCategory* mcat = 0;
   
   // the variable sets
   TString theCat1Vars = "var1:var2:var3:var4";
   TString theCat2Vars = (UseOffsetMethod ? "var1:var2:var3:var4" : "var1:var2:var3");

   // the Fisher 
   TMVA::MethodBase* fiCat = factory->BookMethod( TMVA::Types::kCategory, "FisherCat","" );
   mcat = dynamic_cast<TMVA::MethodCategory*>(fiCat);
   mcat->AddMethod("abs(eta)<=1.3",theCat1Vars, TMVA::Types::kFisher,"Category_Fisher_1","!H:!V:Fisher");
   mcat->AddMethod("abs(eta)>1.3", theCat2Vars, TMVA::Types::kFisher,"Category_Fisher_2","!H:!V:Fisher");

   // the Likelihood
   TMVA::MethodBase* liCat = factory->BookMethod( TMVA::Types::kCategory, "LikelihoodCat","" );
   mcat = dynamic_cast<TMVA::MethodCategory*>(liCat);
   mcat->AddMethod("abs(eta)<=1.3",theCat1Vars, TMVA::Types::kLikelihood,"Category_Likelihood_1","!H:!V:TransformOutput:PDFInterpol=Spline2:NSmoothSig[0]=20:NSmoothBkg[0]=20:NSmoothBkg[1]=10:NSmooth=1:NAvEvtPerBin=50");
   mcat->AddMethod("abs(eta)>1.3", theCat2Vars, TMVA::Types::kLikelihood,"Category_Likelihood_2","!H:!V:TransformOutput:PDFInterpol=Spline2:NSmoothSig[0]=20:NSmoothBkg[0]=20:NSmoothBkg[1]=10:NSmooth=1:NAvEvtPerBin=50");

   // ---- Now you can tell the factory to train, test, and evaluate the MVAs

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
   std::cout << "==> TMVAClassificationCategory is done!" << std::endl;      

   // Clean up
   delete factory;

   // Launch the GUI for the root macros
   if (!gROOT->IsBatch()) TMVAGui( outfileName );
}

