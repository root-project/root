/**********************************************************************************
 * Project   : TMVA - a Root-integrated toolkit for multivariate data analysis    *
 * Package   : TMVA                                                               *
 * Root Macro: TMVAnalysis                                                        *
 *                                                                                *
 * This exectutable provides an example on training and testing of several        *
 * Multivariate Analyser (MVA) methods                                            *
 *                                                                                *
 * As input file we use a of a toy MC sample (you find it in .TMVA/examples/data) *
 *  or the standard Root example  Higgs analysis, which is part of the Root       *
 * tutorial (and hence present in any Root release)                               *
 *                                                                                *
 * The methods to be used can be switched on and off by means of the boolians     *
 * below                                                                          *
 *                                                                                *
 * The output file "TMVA.root" can be analysed with the use of dedicated          *
 * macros (simply say: root -l <macro.C>), which can also be called through       *
 * GUI that will appear at the end of the run of this macro.                      *
 **********************************************************************************/

#include "TMVAGui.C"

// ---------------------------------------------------------------
// choose MVA methods to be trained + tested
Bool_t Use_Cuts            = 1;
Bool_t Use_CutsD           = 1;
Bool_t Use_Likelihood      = 1;
Bool_t Use_LikelihoodD     = 1; // the "D" extension indicates decorrelated input variables (see option strings)
Bool_t Use_PDERS           = 1;
Bool_t Use_PDERSD          = 1;
Bool_t Use_HMatrix         = 1;
Bool_t Use_Fisher          = 1;
Bool_t Use_MLP             = 1; // this is the recommended ANN
Bool_t Use_CFMlpANN        = 0;
Bool_t Use_TMlpANN         = 0;
Bool_t Use_BDT             = 1;
Bool_t Use_RuleFit         = 1;

// read input data file with ascii format (otherwise ROOT) ?
Bool_t ReadDataFromAsciiIFormat = kFALSE;

void TMVAnalysis() 
{
   // explicit loading of the shared libTMVA is done in TMVAlogon.C, defined in TMVA/macros/.rootrc
   // if you use your private .rootrc, or run from a different directory, please copy the 
   // corresponding lines from TMVA/macros/.rootrc
  
   cout << "Start Test TMVAnalysis" << endl
        << "======================" << endl
        << endl;
   cout << "Testing all methods takes about 4 minutes. By excluding" << endl
        << "some of the computing expensive MVA methods the demonstration" << endl
        << "will finish much faster." << endl
        << "  1) Test all methods" << endl
        << "  2) Fast methods only" << endl
        << "Your choice (1:default): " << flush;
   Int_t selection=1;
   char selc='1';
   cin.get(selc);
   if (selc=='2')
      Use_Cuts = Use_CutsD = Use_PDERS = Use_CFMlpANN = Use_TMlpANN = Use_MLP = Use_BDT = kFALSE;

   // Create a new root output file.
   TFile* outputFile = TFile::Open( "TMVA.root", "RECREATE" );

   // Create the factory object. Later you can choose the methods whose performance 
   // you'd like to investigate. The factory will then run the performance analysis
   // for you.
   TMVA::Factory *factory = new TMVA::Factory( "MVAnalysis", outputFile, "" );

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
      TString fname = "../examples/data/toy_sigbkg.root";
      if (!gSystem->AccessPathName(fname)) {
         input = TFile::Open(fname);
      } 
      else {
         cout << "ERROR: could not access data file: " << fname << endl;
      }
      if (!input) {
         cout << "ERROR: could not open data file: " << fname << endl;
         exit(1);
      }

      TTree *signal     = (TTree*)input->Get("TreeS");
      TTree *background = (TTree*)input->Get("TreeB");

      // global event weights (see below for setting event-wise weights)
      Double_t signalWeight     = 1.0;
      Double_t backgroundWeight = 1.0;

      // sanity check
      if (!signal || !background) {
         cout << "ERROR: unknown tree(s)" << endl;
         exit(1);
      }
      if (!factory->SetInputTrees( signal, background, signalWeight, backgroundWeight)) exit(1);
   }
   
   // Define the input variables that shall be used for the MVA training
   // note that you may also use variable expressions, such as: "3*var1/var2*abs(var3)"
   // [all types of expressions that can also be parsed by TTree::Draw( "expression" )]
   factory->AddVariable("var1", 'F');
   factory->AddVariable("var2", 'F');
   factory->AddVariable("var3", 'F');
   factory->AddVariable("var4", 'F');

   // This would set individual event weights (the variables defined in the 
   // expression need to exist in the original TTree)
   // factory->SetWeightExpression("weight1*weight2");
   
   // Apply additional cuts on the signal and background sample. 
   // Assumptions on size of training and testing sample:
   //    a) equal number of signal and background events is used for training
   //    b) any numbers of signal and background events are used for testing
   //    c) an explicit syntax can violate a)
   // more Documentation with the Factory class
   TCut mycut = ""; // for example: TCut mycut = "abs(var1)<0.5 && abs(var2-0.5)<1";

   factory->PrepareTrainingAndTestTree( mycut, 2000, 4000 );  
  
   // ---- book MVA methods
   //
   // please lookup the various method configuration options in the corresponding cxx files, eg:
   // src/MethoCuts.cxx, etc.

   // Cut optimisation
   if (Use_Cuts) 
     factory->BookMethod( TMVA::Types::Cuts, "Cuts", "!V:MC:EffSel:MC_NRandCuts=100000:AllFSmart" );

   // alternatively, use the powerful cut optimisation with a Genetic Algorithm
   // factory->BookMethod( TMVA::Types::Cuts, "CutsGA",
   //                      "!V:GA:EffSel:GA_nsteps=40:GA_cycles=30:GA_popSize=100:GA_SC_steps=10:GA_SC_offsteps=5:GA_SC_factor=0.95" );

   if (Use_CutsD) 
     factory->BookMethod( TMVA::Types::Cuts, "CutsD", "!V:MC:EffSel:MC_NRandCuts=200000:AllFSmart:Preprocess=Decorrelate" );

   // Likelihood
   if (Use_Likelihood)
      factory->BookMethod( TMVA::Types::Likelihood, "Likelihood", "!V:!TransformOutput:Spline=2:NSmooth=5" ); 

   // test the decorrelated likelihood
   if (Use_LikelihoodD)
      factory->BookMethod( TMVA::Types::Likelihood, "LikelihoodD", "!V:!TransformOutput:Spline=2:NSmooth=5:Preprocess=Decorrelate"); 
 
   // Fisher:
   if (Use_Fisher) 
      factory->BookMethod( TMVA::Types::Fisher, "Fisher", "!V:Fisher" );    
  
   // the new TMVA ANN: MLP (recommended ANN)
   if (Use_MLP)
      factory->BookMethod( TMVA::Types::MLP, "MLP", "!V:NCycles=200:HiddenLayers=N+1,N:TestRate=5" );

   // CF(Clermont-Ferrand)ANN
   if (Use_CFMlpANN)
      factory->BookMethod( TMVA::Types::CFMlpANN, "CFMlpANN", "!V:H:NCycles=5000:HiddenLayers=N,N"  ); // n_cycles:#nodes:#nodes:...  
  
   // Tmlp(Root)ANN
   if (Use_TMlpANN)
      factory->BookMethod( TMVA::Types::TMlpANN, "TMlpANN", "!V:NCycles=200:HiddenLayers=N+1,N"  ); // n_cycles:#nodes:#nodes:...
  
   // HMatrix
   if (Use_HMatrix)
      factory->BookMethod( TMVA::Types::HMatrix, "HMatrix", "!V" ); // H-Matrix (chi2-squared) method
  
   // PDE - RS method
   if (Use_PDERS)
      factory->BookMethod( TMVA::Types::PDERS, "PDERS", 
                           "!V:VolumeRangeMode=RMS:KernelEstimator=Teepee:MaxVIterations=50:InitialScale=0.99" ) ;

   if (Use_PDERSD) 
      factory->BookMethod( TMVA::Types::PDERS, "PDERSD", 
                           "!V:VolumeRangeMode=RMS:KernelEstimator=Teepee:MaxVIterations=50:InitialScale=0.99:Preprocess=Decorrelate" ) ;
  
   // Boosted Decision Trees
   if (Use_BDT)
      factory->BookMethod( TMVA::Types::BDT, "BDT", 
                           "!V:NTrees=200:BoostType=AdaBoost:SeparationType=GiniIndex:nEventsMin=400:SignalFraction=0.:nCuts=20:PruneStrength=10" );
    
   // Friedman's RuleFit method
   if (Use_RuleFit)
      factory->BookMethod( TMVA::Types::RuleFit, "RuleFit", 
                           "!V:NTrees=20:SampleFraction=-1:nEventsMin=60:nCuts=20:MinImp=0.001:Model=ModLinear:GDTau=0.6:GDStep=0.01:GDNSteps=100000:SeparationType=GiniIndex:RuleMaxDist=0.00001" );
                                                                                                    

   // ---- Now you can tell the factory to train, test, and evaluate the MVAs. 

   // Train MVAs.
   factory->TrainAllMethods();

   // Test MVAs.
   factory->TestAllMethods();

   // Evaluate MVAs
   factory->EvaluateAllMethods();    
  
   // Save the output.
   outputFile->Close();

   cout << "==> wrote root file TMVA.root" << endl;
   cout << "==> TMVAnalysis is done!" << endl;      

   // clean up
   delete factory;

   // open the GUI for the root macros
   TMVAGui();
} 
