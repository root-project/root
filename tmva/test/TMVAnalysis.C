// @(#)root/tmva $Id: TMVAnalysis.C,v 1.9 2007/04/04 06:54:30 brun Exp $
/**********************************************************************************
 * Project   : TMVA - a Root-integrated toolkit for multivariate data analysis    *
 * Package   : TMVA                                                               *
 * Root Macro: TMVAnalysis                                                        *
 *                                                                                *
 * This macro gives an example on training and testing of several                 *
 * Multivariate Analyser (MVA) methods.                                           *
 *                                                                                *
 * As input file we use a toy MC sample (you find it in TMVA/examples/data).      *
 *                                                                                *
 * The methods to be used can be switched on and off by means of booleans.        *
 *                                                                                *
 * The output file "TMVA.root" can be analysed with the use of dedicated          *
 * macros (simply say: root -l <macro.C>), which can be conveniently              *
 * invoked through a GUI that will appear at the end of the run of this macro.    *
 **********************************************************************************/

#include <iostream>

#include "TCut.h"
#include "TFile.h"
#include "TSystem.h"
#include "TTree.h"
// requires links
// #include "TMVA/Factory.h"
// #include "TMVA/Tools.h"
// #include "TMVA/Config.h"

#include "TMVAGui.C"

// ---------------------------------------------------------------
// choose MVA methods to be trained + tested
Bool_t Use_Cuts            = 0;
Bool_t Use_CutsD           = 0;
Bool_t Use_CutsGA          = 1;
Bool_t Use_Likelihood      = 1;
Bool_t Use_LikelihoodD     = 0; // the "D" extension indicates decorrelated input variables (see option strings)
Bool_t Use_LikelihoodPCA   = 1; // the "PCA" extension indicates PCA-transformed input variables (see option strings)
Bool_t Use_LikelihoodKDE   = 0;
Bool_t Use_PDERS           = 1;
Bool_t Use_PDERSD          = 0;
Bool_t Use_PDERSPCA        = 0;
Bool_t Use_HMatrix         = 1;
Bool_t Use_Fisher          = 1;
Bool_t Use_MLP             = 1; // this is the recommended ANN
Bool_t Use_CFMlpANN        = 0; 
Bool_t Use_TMlpANN         = 0; 
Bool_t Use_BDT             = 1;
Bool_t Use_BDTD            = 0;
Bool_t Use_RuleFit         = 1;
Bool_t Use_SVM_Gauss       = 1;
Bool_t Use_SVM_Poly        = 0;
Bool_t Use_SVM_Lin         = 0;
// ---------------------------------------------------------------

// read input data file with ascii format (otherwise ROOT) ?
Bool_t ReadDataFromAsciiIFormat = kFALSE;

void TMVAnalysis( TString myMethodList = "" ) 
{
   // explicit loading of the shared libTMVA is done in TMVAlogon.C, defined in .rootrc
   // if you use your private .rootrc, or run from a different directory, please copy the 
   // corresponding lines from .rootrc

   // methods to be processed can be given as an argument; use format:
   //
   // mylinux~> root -l TMVAnalysis.C\(\"myMethod1,myMethod2,myMethod3\"\)
   //
   TList* mlist = TMVA::Tools::ParseFormatLine( myMethodList, " :," );

   if (mlist->GetSize()>0) {
      Use_CutsGA = Use_CutsD = Use_Cuts
         = Use_LikelihoodKDE = Use_LikelihoodPCA = Use_LikelihoodD = Use_Likelihood
         = Use_PDERSPCA = Use_PDERSD = Use_PDERS = Use_MLP = Use_CFMlpANN = Use_TMlpANN
         = Use_HMatrix = Use_Fisher = Use_BDTD = Use_BDT = Use_RuleFit 
         = Use_SVM_Gauss = Use_SVM_Poly = Use_SVM_Lin 
         = 0;

      if (mlist->FindObject( "Cuts"          ) != 0) Use_Cuts          = 1; 
      if (mlist->FindObject( "CutsD"         ) != 0) Use_CutsD         = 1; 
      if (mlist->FindObject( "CutsGA"        ) != 0) Use_CutsGA        = 1; 
      if (mlist->FindObject( "Likelihood"    ) != 0) Use_Likelihood    = 1; 
      if (mlist->FindObject( "LikelihoodD"   ) != 0) Use_LikelihoodD   = 1; 
      if (mlist->FindObject( "LikelihoodPCA" ) != 0) Use_LikelihoodPCA = 1; 
      if (mlist->FindObject( "LikelihoodKDE" ) != 0) Use_LikelihoodKDE = 1; 
      if (mlist->FindObject( "PDERSPCA"      ) != 0) Use_PDERSPCA      = 1; 
      if (mlist->FindObject( "PDERSD"        ) != 0) Use_PDERSD        = 1; 
      if (mlist->FindObject( "PDERS"         ) != 0) Use_PDERS         = 1; 
      if (mlist->FindObject( "HMatrix"       ) != 0) Use_HMatrix       = 1; 
      if (mlist->FindObject( "Fisher"        ) != 0) Use_Fisher        = 1; 
      if (mlist->FindObject( "MLP"           ) != 0) Use_MLP           = 1; 
      if (mlist->FindObject( "CFMlpANN"      ) != 0) Use_CFMlpANN      = 1; 
      if (mlist->FindObject( "TMlpANN"       ) != 0) Use_TMlpANN       = 1; 
      if (mlist->FindObject( "BDTD"          ) != 0) Use_BDTD          = 1; 
      if (mlist->FindObject( "BDT"           ) != 0) Use_BDT           = 1; 
      if (mlist->FindObject( "RuleFit"       ) != 0) Use_RuleFit       = 1; 
      if (mlist->FindObject( "SVM_Gauss"     ) != 0) Use_SVM_Gauss     = 1; 
      if (mlist->FindObject( "SVM_Poly"      ) != 0) Use_SVM_Poly      = 1; 
      if (mlist->FindObject( "SVM_Lin"       ) != 0) Use_SVM_Lin       = 1; 

      delete mlist;
   }
  
   std::cout << "Start Test TMVAnalysis" << std::endl
        << "======================" << std::endl
        << std::endl;
   std::cout << "Testing all standard methods may take about 10 minutes of running..." << std::endl;

   // Create a new root output file.
   Char_t outfileName[80];
   sprintf( outfileName,"TMVA.root" );
   TFile* outputFile = TFile::Open( outfileName, "RECREATE" );

   // Create the factory object. Later you can choose the methods
   // whose performance you'd like to investigate. The factory will
   // then run the performance analysis for you.
   //
   // The first argument is the base of the name of all the
   // weightfiles in the directory weight/ 
   //
   // The second argument is the output file for the training results
   TMVA::Factory *factory = new TMVA::Factory( "MVAnalysis", outputFile, Form("!V:%sColor", gROOT->IsBatch()?"!":"") );

   // if you wish to modify default settings:
   // TMVA::gConfig().variablePlotting.timesRMS = 16.0;

   if (ReadDataFromAsciiIFormat) {
      // load the signal and background event samples from ascii files
      // format in file must be:
      // var1/F:var2/F:var3/F:var4/F
      // 0.04551   0.59923   0.32400   -0.19170
      // ...

      TString datFileS = "data/toy_sig_lincorr.dat";
      TString datFileB = "data/toy_bkg_lincorr.dat";
      if (!factory->SetInputTrees( datFileS, datFileB )) exit(1);
   }
   else {
      // load the signal and background event samples from ROOT trees
      TFile *input(0);
      TString fname = "./tmva_example.root";
      if (!gSystem->AccessPathName( fname )) {
         // first we try to find tmva_example.root in the local directory
         cout << "--- TMVAnalysis  : accessing " << fname << endl;
         input = TFile::Open( fname );
      } 
      else { 
         // second we try accessing the file via the web from
         // http://root.cern.ch/files/tmva_example.root
         cout << "--- TMVAnalysis  : accessing tmva_example.root file from http://root.cern.ch/files" << endl;
         cout << "--- TMVAnalysis  : for faster startup you may consider downloading it into you local directory" << endl;
         input = TFile::Open( "http://root.cern.ch/files/tmva_example.root" );
      }

      if (!input) {
         std::cout << "ERROR: could not open data file" << std::endl;
         exit(1);
      }

      TTree *signal     = (TTree*)input->Get("TreeS");
      TTree *background = (TTree*)input->Get("TreeB");

      // global event weights (see below for setting event-wise weights)
      Double_t signalWeight     = 1.0;
      Double_t backgroundWeight = 1.0;

      factory->AddSignalTree    ( signal,     signalWeight );
      factory->AddBackgroundTree( background, backgroundWeight );
   }
   
   // Define the input variables that shall be used for the MVA training
   // note that you may also use variable expressions, such as: "3*var1/var2*abs(var3)"
   // [all types of expressions that can also be parsed by TTree::Draw( "expression" )]
   factory->AddVariable("var1+var2", 'F');
   factory->AddVariable("var1-var2", 'F');
   factory->AddVariable("var3", 'F');
   factory->AddVariable("var4", 'F');

   // This would set individual event weights (the variables defined in the 
   // expression need to exist in the original TTree)
   // factory->SetWeightExpression("weight1*weight2");

   // Apply additional cuts on the signal and background sample. 
   TCut mycut = ""; // for example: TCut mycut = "abs(var1)<0.5 && abs(var2-0.5)<1";

   // tell the factory to use all remaining events in the trees after training for testing:
   factory->PrepareTrainingAndTestTree( mycut, "NSigTrain=3000:NBkgTrain=3000:SplitMode=Random:!V" );  

   // If no numbers of events are given, half of the events in the tree are used for training, and 
   // the other haof for testing:
   //   factory->PrepareTrainingAndTestTree( mycut, "SplitMode=random:!V" );  
   // To also specify the number of testing events, use:
   //   factory->PrepareTrainingAndTestTree( mycut, 
   //                                        "NSigTrain=3000:NBkgTrain=3000:NSigTest=3000:NBkgTest=3000:SplitMode=Random:!V" );  
   // an equivalent of writing this is:
   // the old-style call:
   //   factory->PrepareTrainingAndTestTree( mycut, Ntrain, Ntest );
   // is kept for backward compatibility, but depreciated

   // ---- book MVA methods
   //
   // please lookup the various method configuration options in the corresponding cxx files, eg:
   // src/MethoCuts.cxx, etc.

   // Cut optimisation
   if (Use_Cuts) 
     factory->BookMethod( TMVA::Types::kCuts, "Cuts", "!V:MC:EffSel:MC_NRandCuts=100000:MC_VarProp=FSmart" );

   if (Use_CutsD) 
     factory->BookMethod( TMVA::Types::kCuts, "CutsD", "!V:MC:EffSel:MC_NRandCuts=200000:MC_VarProp=FSmart:VarTransform=Decorrelate" );

   if (Use_CutsGA) 
   // alternatively, use the powerful cut optimisation with a Genetic Algorithm
     factory->BookMethod( TMVA::Types::kCuts, "CutsGA",
                         "!V:GA:EffSel:GA_nsteps=40:GA_cycles=3:GA_popSize=300:GA_SC_steps=10:GA_SC_rate=5:GA_SC_factor=0.95" );

   // Likelihood
   if (Use_Likelihood) 
      factory->BookMethod( TMVA::Types::kLikelihood, "Likelihood", "!V:!TransformOutput:Spline=2:NSmoothSig[0]=100:NSmoothBkg[0]=10:NSmoothBkg[1]=100:NSmooth=10:NAvEvtPerBin=50" ); 

   // test the decorrelated likelihood
   if (Use_LikelihoodD) 
      factory->BookMethod( TMVA::Types::kLikelihood, "LikelihoodD", "!V:!TransformOutput:Spline=2:NSmoothSig[0]=100:NSmoothBkg[0]=10:NSmooth=5:NAvEvtPerBin=50:VarTransform=Decorrelate" ); 

   if (Use_LikelihoodPCA) 
      factory->BookMethod( TMVA::Types::kLikelihood, "LikelihoodPCA", "!V:!TransformOutput:Spline=2:NSmoothSig[0]=100:NSmoothBkg[0]=10:NSmooth=5:NAvEvtPerBin=50:VarTransform=PCA" ); 
 
   // test the new kernel density estimator
   if (Use_LikelihoodKDE) 
      factory->BookMethod( TMVA::Types::kLikelihood, "LikelihoodKDE", "!V:!TransformOutput:UseKDE:KDEtype=Gauss:KDEiter=Nonadaptive:KDEborder=None:NAvEvtPerBin=50" ); 

   // Fisher:
   if (Use_Fisher)
      factory->BookMethod( TMVA::Types::kFisher, "Fisher", "!V:Fisher:CreateMVAPdfs:NbinsMVAPdf=50:NsmoothMVAPdf=1" );    

   // the new TMVA ANN: MLP (recommended ANN)
   if (Use_MLP)
      factory->BookMethod( TMVA::Types::kMLP, "MLP", "!V:NCycles=200:HiddenLayers=N+1,N:TestRate=5" );

   // CF(Clermont-Ferrand)ANN
   if (Use_CFMlpANN)
      factory->BookMethod( TMVA::Types::kCFMlpANN, "CFMlpANN", "!V:H:NCycles=500:HiddenLayers=N,N"  ); // n_cycles:#nodes:#nodes:...  
  
   // Tmlp(Root)ANN
   if (Use_TMlpANN)
      factory->BookMethod( TMVA::Types::kTMlpANN, "TMlpANN", "!V:NCycles=200:HiddenLayers=N+1,N"  ); // n_cycles:#nodes:#nodes:...
  
   // HMatrix
   if (Use_HMatrix)
      factory->BookMethod( TMVA::Types::kHMatrix, "HMatrix", "!V" ); // H-Matrix (chi2-squared) method
  
   // PDE - RS method
   if (Use_PDERS)
      factory->BookMethod( TMVA::Types::kPDERS, "PDERS", 
                           "!V:VolumeRangeMode=Adaptive:KernelEstimator=Gauss:GaussSigma=0.3:NEventsMin=400:NEventsMax=600:InitialScale=0.99" );
   
   if (Use_PDERSD) 
      factory->BookMethod( TMVA::Types::kPDERS, "PDERSD", 
                           "!V:VolumeRangeMode=Adaptive:KernelEstimator=Gauss:GaussSigma=0.3:NEventsMin=400:NEventsMax=600:InitialScale=0.99:VarTransform=Decorrelate" );

   if (Use_PDERSPCA) 
      factory->BookMethod( TMVA::Types::kPDERS, "PDERSPCA", 
                           "!V:VolumeRangeMode=Adaptive:KernelEstimator=Gauss:GaussSigma=0.3:NEventsMin=400:NEventsMax=600:InitialScale=0.99:VarTransform=PCA" );
  
   // Boosted Decision Trees
   if (Use_BDT)
      factory->BookMethod( TMVA::Types::kBDT, "BDT", 
                           "!V:NTrees=400:BoostType=AdaBoost:SeparationType=GiniIndex:nEventsMin=20:nCuts=20:PruneMethod=CostComplexity:PruneStrength=4.5" );
   if (Use_BDTD)
      factory->BookMethod( TMVA::Types::kBDT, "BDTD", 
                           "!V:NTrees=400:BoostType=AdaBoost:SeparationType=GiniIndex:nEventsMin=20:nCuts=20:PruneMethod=CostComplexity:PruneStrength=4.5:VarTransform=Decorrelate" );

   // Friedman's RuleFit method
   if (Use_RuleFit)
      factory->BookMethod( TMVA::Types::kRuleFit, "RuleFit",
                           "!V:NTrees=20:SampleFraction=-1:fEventsMin=0.1:nCuts=20:SeparationType=GiniIndex:Model=ModRuleLinear:GDTau=0.6:GDTauMin=0.0:GDTauMax=1.0:GDNTau=20:GDStep=0.01:GDNSteps=5000:GDErrScale=1.1:RuleMinDist=0.0001:MinImp=0.001" );

   // Support Vector Machines using three different Kernel types (Gauss, polynomial and linear)
   if (Use_SVM_Gauss)
      factory->BookMethod( TMVA::Types::kSVM, "SVM_Gauss",
                           "Sigma=2:C=1:Tol=0.001:Kernel=Gauss" );
   if (Use_SVM_Poly)
      factory->BookMethod( TMVA::Types::kSVM, "SVM_Poly",
                           "Order=4:Theta=1:C=0.1:Tol=0.001:Kernel=Polynomial" );
   if (Use_SVM_Lin)
      factory->BookMethod( TMVA::Types::kSVM, "SVM_Lin",
                           "!V:Kernel=Linear:C=1:Tol=0.001" );

   // ---- Now you can tell the factory to train, test, and evaluate the MVAs. 

   // Train MVAs using the set of training events
   factory->TrainAllMethods();

   // Evaluate all MVAs using the set of test events
   factory->TestAllMethods();

   // Evaluate and compare performance of all configured MVAs
   factory->EvaluateAllMethods();    
  
   // Save the output.
   outputFile->Close();

   std::cout << "==> wrote root file TMVA.root" << std::endl;
   std::cout << "==> TMVAnalysis is done!" << std::endl;      

   // clean up
   delete factory;

   // open the GUI for the root macros
   if (!gROOT->IsBatch()) TMVAGui( outfileName );
}
