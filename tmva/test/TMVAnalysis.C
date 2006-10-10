/**********************************************************************************
 * Project   : TMVA - a Root-integrated toolkit for multivariate data analysis    *
 * Package   : TMVA                                                               *
 * Root Macro: TMVAnalysis                                                        *
 *                                                                                *
 * This exectutable provides an example on training and testing of several        *
 * Multivariate Analyser (MVA) methods                                            *
 *                                                                                *
 * As input file we use a standard Root example of a Higgs analysis, which        *
 * is part of the Root tutorial (and hence present in any Root release)           *
 *                                                                                *
 * The methods to be used can be switched on and off by means of the boolians     *
 * below                                                                          *
 *                                                                                *
 * The output file "TMVA.root" can be analysed with the use of dedicated          *
 * macros (simply say: root -l <macro.C>) :                                       *
 *                                                                                *
 *   - variables.C     ==> show us the MVA input variables for signal and backgr  *
 *   - correlations.C  ==> show us the correlations between the MVA input vars    *
 *   - mvas.C          ==> show the trained MVAs for the test events              *
 *   - efficiencies.C  ==> show us the background rejections versus signal effs   *
 *                         for all MVAs used                                      *
 *                                                                                *
 * TMVA allows to train and test multiple MVAs for different phase space          *
 * regions. This is however not realised in this simple example.                  *
 *                                                                                *
 **********************************************************************************/

void TMVAnalysis() {

   // initialisation
   gSystem->Load("libMLP");   // ROOT's Multilayer Perceptron library is needed
   gSystem->Load("libTMVA");  // and of course the TMVA library
   gROOT->ProcessLine(".L loader.C+");

   // ---------------------------------------------------------------
   // choose MVA methods to be trained + tested
   Bool_t Use_Cuts            = 1;
   Bool_t Use_Likelihood      = 1;
   Bool_t Use_LikelihoodD     = 1;
   Bool_t Use_PDERS           = 1;
   Bool_t Use_HMatrix         = 1;
   Bool_t Use_Fisher          = 1;
   Bool_t Use_CFMlpANN        = 1;
   Bool_t Use_TMlpANN         = 1;
   Bool_t Use_BDT_GiniIndex   = 1; // default BDT method
   Bool_t Use_BDT_CrossEntro  = 0;
   Bool_t Use_BDT_SdivStSpB   = 0;
   Bool_t Use_BDT_MisClass    = 0;
   Bool_t Use_BDT_Bagging_Gini= 0;
   // ---------------------------------------------------------------
   Bool_t EvaluateVariables   = 0; // perform evaluation for each input variable
   // ---------------------------------------------------------------
  
   cout << "Start Test TMVAnalysis" << endl
        << "======================" << endl
        << endl;
   cout << "Testing all methods takes about 10-20 minutes. By excluding" << endl
        << "some of the computing expensive analysis methods the demonstration" << endl
        << "will finish much faster." << endl
        << "  1) All methods (15 min)" << endl
        << "  2) Fast methods only (1 min)" << endl
        << "Your choice (1:default): " << flush;
   int selection=1;
   char selc='1';
   cin.get(selc);
   if(selc=='2')
      Use_Cuts = Use_Likelihood = Use_PDERS = Use_HMatrix = Use_CFMlpANN = Use_TMlpANN = Use_BDT_GiniIndex = 0;

   // Create a new root output file.
   TFile* outputFile = TFile::Open( "TMVA.root", "RECREATE" );

   // Create the factory object. Later you can choose the methods whose performance 
   // you'd like to investigate. The factory will then run the performance analysis
   // for you.
   TMVA::Factory *factory = new TMVA::Factory( "MVAnalysis", outputFile, "" ) ;

   // Define the signal and background event samples.
   TFile *input(0);
   const char *fname = "tmva_example.root";
   TFile *input = 0;
   if (!gSystem->AccessPathName(fname)) {
      input = TFile::Open(fname);
   } else {
      printf("accessing %s file from http://root.cern.ch/files\n",fname);
      input = TFile::Open(Form("http://root.cern.ch/files/%s",fname));
   }
   if (!input) return;

   TTree *signal     = (TTree*)input->Get("TreeS");
   TTree *background = (TTree*)input->Get("TreeB");
   if( ! factory->SetInputTrees( signal, background )) return; 


   // Define the input variables. These are used in the TMVA.
   vector<TString>* inputVars = new vector<TString>;
   inputVars->push_back("var1");
   inputVars->push_back("var2");
   inputVars->push_back("var3");
   inputVars->push_back("var4");
   factory->SetInputVariables( inputVars );    
  

   // Apply additional cuts on the signal and background sample. 
   // Assumptions on size of training and testing sample:
   //    a) equal number of signal and background events is used for training
   //    b) any numbers of signal and background events are used for testing
   //    c) an explicit syntax can violate a)
   // more Documentation with the Factory class
   TCut mycut = "";
   factory->PrepareTrainingAndTestTree( mycut, 2000, 4000 );  
  

   // Book the MVA methods you like to investigate.

   // MethodCuts:
   // format of option string: "OptMethod:EffMethod:Option_var1:...:Option_varn"
   // "OptMethod" can be:
   //     - "GA"    : Genetic Algorithm (recommended)
   //     - "MC"    : Monte-Carlo optimization 
   // "EffMethod" can be:
   //     - "EffSel": compute efficiency by event counting
   //     - "EffPDF": compute efficiency from PDFs
   // === For "GA" method ======
   // "Option_var1++" are (see GA for explanation of parameters):
   //     - fGa_nsteps        
   //     - fGa_preCalc        
   //     - fGa_SC_steps        
   //     - fGa_SC_offsteps 
   //     - fGa_SC_factor   
   // === For "MC" method ======
   // "Option_var1" is number of random samples
   // "Option_var2++" can be 
   //     - "FMax"  : ForceMax   (the max cut is fixed to maximum of variable i)
   //     - "FMin"  : ForceMin   (the min cut is fixed to minimum of variable i)
   //     - "FSmart": ForceSmart (the min or max cut is fixed to min/max, based on mean value)
   //     - Adding "All" to "option_vari", eg, "AllFSmart" will use this option for all variables
   //     - if "option_vari" is empty (== ""), no assumptions on cut min/max are made
   // ---------------------------------------------------------------------------------- 
   if (Use_Cuts) 
      factory->BookMethod( "MethodCuts",  "V:GA:EffSel:30:3:10:5:0.95" );
   // factory->BookMethod( "MethodCuts",  "V:MC:EffSel:10000:AllFSmart" );
  
   // MethodLikelihood options:
   // histogram_interpolation_method:nsmooth:nsmooth:n_aveEvents_per_bin:Decorrelation
   if (Use_Likelihood)
      factory->BookMethod( TMVA::Types::Likelihood, "Spline2:3"           ); 
   if (Use_LikelihoodD)
      factory->BookMethod( TMVA::Types::Likelihood, "Spline2:10:25:D"); 
 
   // MethodFisher:
   if (Use_Fisher)
      factory->BookMethod( TMVA::Types::Fisher, "Fisher" ); // Fisher method ("Fi" or "Ma")
  
   // Method CF(Clermont-Ferrand)ANN:
   if (Use_CFMlpANN)
      factory->BookMethod( TMVA::Types::CFMlpANN, "5000:N:N"  ); // n_cycles:#nodes:#nodes:...  
  
   // Method CF(Root)ANN:
   if (Use_TMlpANN)
      factory->BookMethod( TMVA::Types::TMlpANN, "200:N+1:N"  ); // n_cycles:#nodes:#nodes:...
  
   // MethodHMatrix:
   if (Use_HMatrix)
      factory->BookMethod( TMVA::Types::HMatrix ); // H-Matrix (chi2-squared) method
  
   // PDE - RS method
   // format and syntax of option string: "VolumeRangeMode:options"
   // where:
   //  VolumeRangeMode - all methods defined in private enum "VolumeRangeMode" 
   //  options         - deltaFrac in case of VolumeRangeMode=MinMax/RMS
   //                  - nEventsMin/Max, maxVIterations, scale for VolumeRangeMode=Adaptive
   if (Use_PDERS)
      factory->BookMethod( TMVA::Types::PDERS, "Adaptive:50:100:50:0.99" ); 
  
   // MethodBDT (Boosted Decision Trees) options:
   // format and syntax of option string: "nTrees:BoostType:SeparationType:
   //                                      nEventsMin:dummy:
   //                                      nCuts:SignalFraction"
   // nTrees:          number of trees in the forest to be created
   // BoostType:       the boosting type for the trees in the forest (AdaBoost e.t.c..)
   // SeparationType   the separation criterion applied in the node splitting
   // nEventsMin:      the minimum number of events in a node (leaf criteria, stop splitting)
   // dummy:           dummy option to keep backward compatible
   //                  continue splitting. !!
   //                  !!! Needs to be set to zero, as it doesn't work properly otherwise
   //                     ... it's strange though and not yet understood !!!
   // nCuts:  the number of steps in the optimisation of the cut for a node
   // SignalFraction:  scale parameter of the number of Bkg events  
   //                  applied to the training sample to simulate different initial purity
   //                  of your data sample. 
   //
   // known SeparationTypes are:
   //    - MisClassificationError
   //    - GiniIndex
   //    - CrossEntropy
   // known BoostTypes are:
   //    - AdaBoost
   //    - Bagging

   if (Use_BDT_GiniIndex)
      factory->BookMethod( TMVA::Types::BDT, "200:AdaBoost:GiniIndex:10:0.:20" );
   if (Use_BDT_CrossEntro)
      factory->BookMethod( TMVA::Types::BDT, "200:AdaBoost:CrossEntropy:10:0.:20" );
   if (Use_BDT_SdivStSpB)
      factory->BookMethod( TMVA::Types::BDT, "200:AdaBoost:SdivSqrtSplusB:10:0.:20" );
   if (Use_BDT_MisClass)
      factory->BookMethod( TMVA::Types::BDT, "200:AdaBoost:MisClassificationError:10:0.:20" );
   if (Use_BDT_Bagging_Gini)
      factory->BookMethod( TMVA::Types::BDT, "200:Bagging:GiniIndex:10:0.:20","bagging" );
  

   // Now you can tell the factory to train, test, and evaluate the MVAs. 

   // Train MVAs.
   factory->TrainAllMethods();


   // Test MVAs.
   factory->TestAllMethods();


   // Evaluate variables.
   if (EvaluateVariables) factory->EvaluateAllVariables();


   // Evaluate MVAs
   factory->EvaluateAllMethods();    

  
   // Save the output.
   outputFile->Close();


   cout << "==> wrote root file TMVA.root" << endl;
   cout << "==> TMVAnalysis is done!" << endl;      


   // clean up
   delete factory;
   delete inputVars;

   gROOT->Reset();
   gStyle->SetScreenFactor(1); //if you have a large screen, select 1,2 or 1.4
   bar = new TControlBar("vertical", "Checks",0,0);
   bar->AddButton("Input Variables",                             ".x variables.C",    "Plots all input variables (macro variables.C)");
   bar->AddButton("Variable Correlations",                       ".x correlations.C", "Plots correlations between all input variables (macro variables.C)");
   bar->AddButton("Output MVA Variables",                        ".x mvas.C",         "Plots the output variable of each method (macro mvas.C)");
   bar->AddButton("Background Rejection vs Signal Efficiencies", ".x efficiencies.C", "Plots background rejection vs signal efficiencies (macro efficiencies.C)");
   bar->AddButton("Quit",   ".q", "Quit");
   bar->Show();
   gROOT->SaveContext();

} 
