/**********************************************************************************
 * Project   : TMVA - a Root-integrated toolkit for multivariate data analysis    *
 * Package   : TMVA                                                               *
 * Root Macro: TMVAnalysis                                                        *
 *                                                                                *
 * This exectutable provides an example on training and testing of several 	  *
 * Multivariate Analyser (MVA) methods						  *
 * 										  *
 * As input file we use a standard Root example of a Higgs analysis, which 	  *
 * is part of the Root tutorial (and hence present in any Root release)		  *
 *										  *
 * The methods to be used can be switched on and off by means of the boolians	  *
 * below									  *
 *										  *
 * The output file "TMVA.root" can be analysed with the use of dedicated	  *
 * macros (simply say: root -l <macro.C>) :					  *
 *										  *
 *   - variables.C     ==> show us the MVA input variables for signal and backgr  *
 *   - correlations.C  ==> show us the correlations between the MVA input vars	  *
 *   - mvas.C          ==> show the trained MVAs for the test events		  *
 *   - efficiencies.C  ==> show us the background rejections versus signal effs   *
 *                         for all MVAs used					  *
 *										  *
 * TMVA allows to train and test multiple MVAs for different phase space 	  *
 * regions. This is however not realised in this simple example. For more	  *
 * information on this please have a look into the development executable 	  *
 * devTMVA.cpp									  *
 *                                                                                *
 **********************************************************************************/
{  
  // initialisation
  gROOT->Reset();
  gSystem->Load("libMLP");
  gSystem->Load("libTMVA");

  // ---------------------------------------------------------------
  // choose MVA methods to be trained + tested
  Bool_t Use_MethodCuts           = 0; //0
  Bool_t Use_MethodLikelihood     = 1;
  Bool_t Use_MethodLikelihoodD    = 1;
  Bool_t Use_MethodFisher         = 1;
  Bool_t Use_MethodCFMlpANN       = 0;  //0
  Bool_t Use_MethodTMlpANN        = 0;  //0
  Bool_t Use_MethodHMatrix        = 1;
  Bool_t Use_MethodPDERS          = 0; //0
  Bool_t Use_MethodBDT_GiniIndex  = 0; // default BDT method
  Bool_t Use_MethodBDT_CrossEntro = 0; //0
  Bool_t Use_MethodBDT_SdivStSpB  = 0; //0
  Bool_t Use_MethodBDT_MisClass   = 0; //0
  // ---------------------------------------------------------------
  Bool_t EvaluateVariables        = 0; // perform evaluation for each input variable
  // ---------------------------------------------------------------
  
  cout << "==> start TMVAnalysis" << endl;

  // the root output file
  TFile* target = TFile::Open( "TMVA.root", "RECREATE" );

  //
  // create the vactory object and claim which variance tool you 
  // would like to use:
  //
  TMVA_Factory *factory = new TMVA_Factory( "MVAnalysis", target ) ;

  // this is the variable vector, defining what's used in the TMVA
  vector<TString>* inputVars = new vector<TString>;
  
  cout << "==> perform 'Higgs' analysis" << endl;

  TFile *input      = new TFile("$ROOTSYS/tutorials/mlpHiggs.root");
  TTree *signal     = (TTree*)input->Get("sig_filtered");
  TTree *background = (TTree*)input->Get("bg_filtered");
  
  factory->SetInputTrees( signal, background );
      
  //
  // Definition of input variables 
  inputVars->push_back("msumf");
  inputVars->push_back("ptsumf");
  inputVars->push_back("acolin");
  factory->SetInputVariables( inputVars );    
  
  factory->PrepareTrainingAndTestTree( "msumf > 0", -1 );    

  // ---- book MVA methods
  //
  // MethodCut options
  // format of option string: Method:nbin_effBvsSHist:nRandCuts:Option_var1:...:Option_varn
  // "Method" can be:
  //     - "MC"    : Monte Carlo optimization (recommended)
  //     - "FitSel": Minuit Fit: "Fit_Migrad" or "Fit_Simplex"
  //     - "FitPDF": PDF-based: only useful for uncorrelated input variables
  // "option_vari" can be 
  //     - "FMax"  : ForceMax   (the max cut is fixed to maximum of variable i)
  //     - "FMin"  : ForceMin   (the min cut is fixed to minimum of variable i)
  //     - "FSmart": ForceSmart (the min or max cut is fixed to min/max, based on mean value)
  //     - Adding "All" to "option_vari", eg, "AllFSmart" will use this option for all variables
  //     - if "option_vari" is empty (== ""), no assumptions on cut min/max are made
  if (Use_MethodCuts) 
    factory->BookMethod( "MethodCuts",  "V:MC:30000:AllFSmart" );
  
  // MethodLikelihood options:
  // histogram_interpolation_method:nsmooth:nsmooth:n_aveEvents_per_bin:Decorrelation
  if (Use_MethodLikelihood)
    factory->BookMethod( "MethodLikelihood", "Spline2:3"           ); 
  if (Use_MethodLikelihoodD)
    factory->BookMethod( "MethodLikelihood", "Spline2:10:25:D", "D" ); 
 
  // MethodFisher:
  if (Use_MethodFisher)
    factory->BookMethod( "MethodFisher",     "Fisher" ); // Fisher method ("Fi" or "Ma")
  
  // Method CF(Clermont-Ferrand)ANN:
  if (Use_MethodCFMlpANN)
    factory->BookMethod( "MethodCFMlpANN", "10000:N:N"  ); // n_cycles:#nodes:#nodes:...  
  
  // Method CF(Root)ANN:
  if (Use_MethodTMlpANN)
    factory->BookMethod( "MethodTMlpANN",    "2000:N+1:N"  ); // n_cycles:#nodes:#nodes:...
  
  // MethodHMatrix:
  if (Use_MethodHMatrix)
    factory->BookMethod( "MethodHMatrix" ); // H-Matrix (chi2-squared) method
  
  // PDE - RS method
  // format and syntax of option string: "VolumeRangeMode:options"
  // where:
  //  VolumeRangeMode - all methods defined in private enum "VolumeRangeMode" 
  //  options         - deltaFrac in case of VolumeRangeMode=MinMax/RMS
  //                  - nEventsMin/Max, maxVIterations, scale for VolumeRangeMode=Adaptive
  if (Use_MethodPDERS)
    factory->BookMethod( "MethodPDERS", "Adaptive:50:100:50:0.99" ); 
  
  // MethodBDT (Boosted Decision Trees) options:
  // format and syntax of option string: "nTrees:SeparationType:BoostType:
  //                                      nEventsMin:minNodePurity:maxNodePurity:
  //                                      nCuts:IntervalCut?"
  // known SeparationTypes are
  //    MisClassificationError,
  //    GiniIndex, 
  //    CrossEntropy;
  // known BoostTypes are
  //    AdaBoost
  //    EpsilonBoost
  // nEventsMin: the minimum Number of events in a node (leaf criteria)
  // SeparationGain:  the minimum gain in separation required in order to
  //                  continue splitting. !! 
  //   !!! Needs to be set to zero, as it doesn't work... it's strange though!!!
  // nCuts:  the number of steps in the optimisation of the cut for a node
  // known SeparationTypes are:
  //    - MisClassificationError
  //    - GiniIndex
  //    - CrossEntropy
  // known BoostTypes are:
  //    - AdaBoost
  //    - EpsilonBoost
  if (Use_MethodBDT_GiniIndex)
    factory->BookMethod( "MethodBDT", "200:AdaBoost:GiniIndex:10:0.:20" );
  if (Use_MethodBDT_CrossEntro)
    factory->BookMethod( "MethodBDT", "200:AdaBoost:CrossEntropy:10:0.:20" );
  if (Use_MethodBDT_SdivStSpB)
    factory->BookMethod( "MethodBDT", "200:AdaBoost:SdivSqrtSplusB:10:0.:20" );
  if (Use_MethodBDT_MisClass)
    factory->BookMethod( "MethodBDT", "200:AdaBoost:MisClassificationError:10:0.:20" );

  // ---- train, test and evaluate the MVAs 

  // train MVAs
  factory->TrainAllMethods();

  // test MVAs
  factory->TestAllMethods();

  // evaluate variables
  if (EvaluateVariables) factory->EvaluateAllVariables();

  // evaluate MVAs
  factory->EvaluateAllMethods();    
  
  // ---- terminate macro

  target->Close();

  // clean up
  delete factory;
  delete inputVars;
  
  cout << "==> wrote root file TMVA.root" << endl;
  cout << "==> TMVAnalysis is done!" << endl;      
} 
