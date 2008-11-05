// @(#)root/tmva $Id$
/**********************************************************************************
 * Project   : TMVA - a Root-integrated toolkit for multivariate data analysis    *
 * Package   : TMVA                                                               *
 * Root Macro: TMVAnalysis                                                        *
 *                                                                                *
 * This macro provides examples for the training and testing of the               *
 * TMVA classifiers.                                                              *
 *                                                                                *
 * As input data is used a toy-MC sample consisting of four Gaussian-distributed  *
 * and linearly correlated input variables.                                       *
 *                                                                                *
 * The methods to be used can be switched on and off by means of booleans, or     *
 * via the prompt command, for example:                                           *
 *                                                                                *
 *    root -l TMVAnalysis.C\(\"Fisher,Likelihood\"\)                              *
 *                                                                                *
 * (note that the backslashes are mandatory)                                      *
 * If no method given, a default set is used.                                     *
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

#include "TMVAGui.C"
#include "TMVA/Tools.h"
#include "TMVA/Factory.h"
#include "TPluginManager.h"

// ---------------------------------------------------------------
// choose MVA methods to be trained + tested
Bool_t Use_Cuts            = 0;
Bool_t Use_CutsD           = 0;
Bool_t Use_CutsPCA         = 0;
Bool_t Use_CutsGA          = 1;
Bool_t Use_CutsSA          = 0;
// ---
Bool_t Use_Likelihood      = 1;
Bool_t Use_LikelihoodD     = 0; // the "D" extension indicates decorrelated input variables (see option strings)
Bool_t Use_LikelihoodPCA   = 1; // the "PCA" extension indicates PCA-transformed input variables (see option strings)
Bool_t Use_LikelihoodKDE   = 0;
Bool_t Use_LikelihoodMIX   = 0;
// ---
Bool_t Use_PDERS           = 1;
Bool_t Use_PDERSD          = 0;
Bool_t Use_PDERSPCA        = 0;
Bool_t Use_PDERSkNN        = 0; // depreciated until further notice
Bool_t Use_PDEFoam         = 0; // preparation for new TMVA version "reader". This method is not available in this version of TMVA
Bool_t Use_KNN             = 1;
// ---
Bool_t Use_HMatrix         = 1;
Bool_t Use_Fisher          = 1;
// ---
Bool_t Use_FDA_GA          = 0;
Bool_t Use_FDA_MC          = 0;
Bool_t Use_FDA_MT          = 1;
Bool_t Use_FDA_GAMT        = 0;
Bool_t Use_FDA_MCMT        = 0;
// ---
Bool_t Use_MLP             = 1; // this is the recommended ANN
Bool_t Use_CFMlpANN        = 0; 
Bool_t Use_TMlpANN         = 0; 
// ---
Bool_t Use_BDT             = 1;
Bool_t Use_BDTD            = 1;
// ---
Bool_t Use_RuleFit         = 1;
Bool_t Use_RuleFitJF       = 0;
// ---
Bool_t Use_SVM_Gauss       = 1;
Bool_t Use_SVM_Poly        = 0;
Bool_t Use_SVM_Lin         = 0;

Bool_t Use_Plugin          = 0;
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
   // if you like to use a method via the plugin mechanism, we recommend using
   // 
   // mylinux~> root -l TMVAnalysis.C\(\"P_myMethod\"\)
   // (an example is given for using the BDT as plugin (see below),
   // but of course the real application is when you write your own
   // method based)
   //
   TList* mlist = 0;

   if (myMethodList != "") {
      Use_CutsGA = Use_CutsSA = Use_CutsD = Use_CutsPCA = Use_Cuts
         = Use_LikelihoodKDE = Use_LikelihoodMIX = Use_LikelihoodPCA = Use_LikelihoodD = Use_Likelihood
         = Use_PDERSPCA = Use_PDERSD = Use_PDERS = Use_PDERSkNN
	 = Use_PDEFoam
         = Use_KNN
         = Use_MLP = Use_CFMlpANN = Use_TMlpANN
         = Use_HMatrix = Use_Fisher = Use_BDTD = Use_BDT
         = Use_RuleFit = Use_RuleFitJF
         = Use_SVM_Gauss = Use_SVM_Poly = Use_SVM_Lin 
         = Use_FDA_GA = Use_FDA_MC = Use_FDA_MT = Use_FDA_GAMT = Use_FDA_MCMT 
         = Use_Plugin
         = 0;

      mlist = TMVA::gTools().ParseFormatLine( myMethodList, " :," );

      if (mlist->FindObject( "Cuts"          ) != 0) Use_Cuts          = 1; 
      if (mlist->FindObject( "CutsD"         ) != 0) Use_CutsD         = 1; 
      if (mlist->FindObject( "CutsPCA"       ) != 0) Use_CutsPCA       = 1; 
      if (mlist->FindObject( "CutsGA"        ) != 0) Use_CutsGA        = 1; 
      if (mlist->FindObject( "CutsSA"        ) != 0) Use_CutsSA        = 1; 
      if (mlist->FindObject( "Likelihood"    ) != 0) Use_Likelihood    = 1; 
      if (mlist->FindObject( "LikelihoodD"   ) != 0) Use_LikelihoodD   = 1; 
      if (mlist->FindObject( "LikelihoodPCA" ) != 0) Use_LikelihoodPCA = 1; 
      if (mlist->FindObject( "LikelihoodKDE" ) != 0) Use_LikelihoodKDE = 1; 
      if (mlist->FindObject( "LikelihoodMIX" ) != 0) Use_LikelihoodMIX = 1; 
      if (mlist->FindObject( "PDERSPCA"      ) != 0) Use_PDERSPCA      = 1; 
      if (mlist->FindObject( "PDERSD"        ) != 0) Use_PDERSD        = 1; 
      if (mlist->FindObject( "PDERSkNN"      ) != 0) Use_PDERSkNN      = 1; 
      if (mlist->FindObject( "PDERS"         ) != 0) Use_PDERS         = 1; 
      if (mlist->FindObject( "PDEFoam"       ) != 0) Use_PDEFoam       = 1; 
      if (mlist->FindObject( "KNN"           ) != 0) Use_KNN           = 1; 
      if (mlist->FindObject( "HMatrix"       ) != 0) Use_HMatrix       = 1; 
      if (mlist->FindObject( "Fisher"        ) != 0) Use_Fisher        = 1; 
      if (mlist->FindObject( "MLP"           ) != 0) Use_MLP           = 1; 
      if (mlist->FindObject( "CFMlpANN"      ) != 0) Use_CFMlpANN      = 1; 
      if (mlist->FindObject( "TMlpANN"       ) != 0) Use_TMlpANN       = 1; 
      if (mlist->FindObject( "BDTD"          ) != 0) Use_BDTD          = 1; 
      if (mlist->FindObject( "BDT"           ) != 0) Use_BDT           = 1; 
      if (mlist->FindObject( "RuleFitJF"     ) != 0) Use_RuleFitJF     = 1; 
      if (mlist->FindObject( "RuleFit"       ) != 0) Use_RuleFit       = 1; 
      if (mlist->FindObject( "SVM_Gauss"     ) != 0) Use_SVM_Gauss     = 1; 
      if (mlist->FindObject( "SVM_Poly"      ) != 0) Use_SVM_Poly      = 1; 
      if (mlist->FindObject( "SVM_Lin"       ) != 0) Use_SVM_Lin       = 1; 
      if (mlist->FindObject( "FDA_MC"        ) != 0) Use_FDA_MC        = 1; 
      if (mlist->FindObject( "FDA_GA"        ) != 0) Use_FDA_GA        = 1; 
      if (mlist->FindObject( "FDA_MT"        ) != 0) Use_FDA_MT        = 1; 
      if (mlist->FindObject( "FDA_GAMT"      ) != 0) Use_FDA_GAMT      = 1; 
      if (mlist->FindObject( "FDA_MCMT"      ) != 0) Use_FDA_MCMT      = 1; 
      
      TListIter mIt(mlist);
      TObjString *os;
      while( (os = (TObjString*) mIt()) && !Use_Plugin ) {
         TString s(os->GetString());
         if(s.BeginsWith("P_")) Use_Plugin=1;
      }
   }
  
   std::cout << "Start TMVAnalysis" << std::endl
             << "=================" << std::endl
             << std::endl;
   std::cout << "Running all standard methods may take about 10 minutes of running..." << std::endl;

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
   TMVA::Factory *factory = new TMVA::Factory( "TMVAnalysis", outputFile, 
                                               Form("!V:!Silent:%sColor", gROOT->IsBatch()?"!":"") );

   // If you wish to modify default settings 
   // (please check "src/Config.h" to see all available global options)
   //    (TMVA::gConfig().GetVariablePlotting()).fTimesRMS = 8.0;
   //    (TMVA::gConfig().GetIONames()).fWeightFileDir = "myWeightDirectory";

   // Define the input variables that shall be used for the MVA training
   // note that you may also use variable expressions, such as: "3*var1/var2*abs(var3)"
   // [all types of expressions that can also be parsed by TTree::Draw( "expression" )]
//    factory->AddVariable( "var1+var2", 'F' );
//    factory->AddVariable( "var1-var2", 'F' );
   factory->AddVariable( "var3",      'I' );
   factory->AddVariable( "var4",      'I' );

   // read training and test data
   if (ReadDataFromAsciiIFormat) {
      // load the signal and background event samples from ascii files
      // format in file must be:
      // var1/F:var2/F:var3/F:var4/F
      // 0.04551   0.59923   0.32400   -0.19170
      // ...

      TString datFileS = "tmva_example_sig.dat";
      TString datFileB = "tmva_example_bkg.dat";
      if (!factory->SetInputTrees( datFileS, datFileB )) exit(1);
   }
   else {
      // load the signal and background event samples from ROOT trees
      TFile *input(0);
      TString fname = "./data.root";
      if (!gSystem->AccessPathName( fname )) {
         // first we try to find tmva_example.root in the local directory
         std::cout << "--- TMVAnalysis    : Accessing " << fname << std::endl;
         input = TFile::Open( fname );
      } 
      else { 
         // second we try accessing the file via the web from
         // http://root.cern.ch/files/tmva_example.root
         std::cout << "--- TMVAnalysis    : Accessing tmva_example.root file from http://root.cern.ch/files" << std::endl;
         std::cout << "--- TMVAnalysis    : For faster startup you may consider downloading it into you local directory" << std::endl;
         input = TFile::Open( "http://root.cern.ch/files/tmva_example.root" );
      }

      if (!input) {
         std::cout << "ERROR: could not open data file" << std::endl;
         exit(1);
      }

      TTree *signal     = (TTree*)input->Get("TreeS");
      TTree *background = (TTree*)input->Get("TreeB");

      // global event weights per tree (see below for setting event-wise weights)
      Double_t signalWeight     = 1.0;
      Double_t backgroundWeight = 1.0;

      // ====== register trees ====================================================
      //
      // the following method is the prefered one:
      // you can add an arbitrary number of signal or background trees
      factory->AddSignalTree    ( signal,     signalWeight     );
      factory->AddBackgroundTree( background, backgroundWeight );

      // To give different trees for training and testing, do as follows:
      //    factory->AddSignalTree( signalTrainingTree, signalTrainWeight, "Training" );
      //    factory->AddSignalTree( signalTestTree,     signalTestWeight,  "Test" );

      // Use the following code instead of the above two or four lines to add signal and background 
      // training and test events "by hand"
      // NOTE that in this case one should not give expressions (such as "var1+var2") in the input 
      //      variable definition, but simply compute the expression before adding the event
      // 
      //    // --- begin ----------------------------------------------------------
      //    std::vector<Double_t> vars( 4 ); // vector has size of number of input variables
      //    Float_t  treevars[4];
      //    for (Int_t ivar=0; ivar<4; ivar++) signal->SetBranchAddress( Form( "var%i", ivar+1 ), &(treevars[ivar]) );
      //    for (Int_t i=0; i<signal->GetEntries(); i++) {
      //       signal->GetEntry(i);
      //       for (Int_t ivar=0; ivar<4; ivar++) vars[ivar] = treevars[ivar];
      //       // add training and test events; here: first half is training, second is testing
      //       // note that the weight can also be event-wise	
      //       if (i < signal->GetEntries()/2) factory->AddSignalTrainingEvent( vars, signalWeight ); 
      //       else                            factory->AddSignalTestEvent    ( vars, signalWeight ); 
      //    }
      //
      //    for (Int_t ivar=0; ivar<4; ivar++) background->SetBranchAddress( Form( "var%i", ivar+1 ), &(treevars[ivar]) );
      //    for (Int_t i=0; i<background->GetEntries(); i++) {
      //       background->GetEntry(i); 
      //       for (Int_t ivar=0; ivar<4; ivar++) vars[ivar] = treevars[ivar];
      //       // add training and test events; here: first half is training, second is testing
      //       // note that the weight can also be event-wise	
      //       if (i < background->GetEntries()/2) factory->AddBackgroundTrainingEvent( vars, backgroundWeight ); 
      //       else                                factory->AddBackgroundTestEvent    ( vars, backgroundWeight ); 
      //    }
      //    // --- end ------------------------------------------------------------
      //
      // ====== end of register trees ==============================================
   }
   
   // This would set individual event weights (the variables defined in the 
   // expression need to exist in the original TTree)
   //    for signal    : factory->SetSignalWeightExpression("weight1*weight2");
   //    for background: factory->SetBackgroundWeightExpression("weight1*weight2");

   // Apply additional cuts on the signal and background samples (can be different)
   TCut mycuts = ""; // for example: TCut mycuts = "abs(var1)<0.5 && abs(var2-0.5)<1";
   TCut mycutb = ""; // for example: TCut mycutb = "abs(var1)<0.5";

   // tell the factory to use all remaining events in the trees after training for testing:
   factory->PrepareTrainingAndTestTree( mycuts, mycutb,
                                        "SplitMode=Random:NormMode=NumEvents:!V" );

   // If no numbers of events are given, half of the events in the tree are used for training, and 
   // the other half for testing:
   //    factory->PrepareTrainingAndTestTree( mycut, "SplitMode=random:!V" );  
   // To also specify the number of testing events, use:
   //    factory->PrepareTrainingAndTestTree( mycut, 
   //                                         "NSigTrain=3000:NBkgTrain=3000:NSigTest=3000:NBkgTest=3000:SplitMode=Random:!V" );  

   // ---- Book MVA methods
   //
   // please lookup the various method configuration options in the corresponding cxx files, eg:
   // src/MethoCuts.cxx, etc.
   // it is possible to preset ranges in the option string in which the cut optimisation should be done:
   // "...:CutRangeMin[2]=-1:CutRangeMax[2]=1"...", where [2] is the third input variable

   // Cut optimisation
   if (Use_Cuts) 
     factory->BookMethod( TMVA::Types::kCuts, "Cuts", 
                          "!H:!V:FitMethod=MC:EffSel:SampleSize=200000:VarProp=FSmart" );

   if (Use_CutsD) 
     factory->BookMethod( TMVA::Types::kCuts, "CutsD", 
                          "!H:!V:FitMethod=MC:EffSel:SampleSize=200000:VarProp=FSmart:VarTransform=Decorrelate" );

   if (Use_CutsPCA) 
     factory->BookMethod( TMVA::Types::kCuts, "CutsPCA", 
                          "!H:!V:FitMethod=MC:EffSel:SampleSize=200000:VarProp=FSmart:VarTransform=PCA" );

   if (Use_CutsGA)
     factory->BookMethod( TMVA::Types::kCuts, "CutsGA",
                         "H:!V:FitMethod=GA:EffSel:Steps=30:Cycles=3:PopSize=100:SC_steps=10:SC_rate=5:SC_factor=0.95:VarProp=FSmart" );
   
   if (Use_CutsSA)
      factory->BookMethod( TMVA::Types::kCuts, "CutsSA",
                           "H:!V:FitMethod=SA:EffSel:MaxCalls=150000:KernelTemperature=IncreasingAdaptive:InitialTemperature=1e+6:MinTemperature=1e-6:Eps=1e-10:UseDefaultScale" );
   
   // Likelihood
   if (Use_Likelihood) 
      factory->BookMethod( TMVA::Types::kLikelihood, "Likelihood", 
                           "!H:!V:!TransformOutput:PDFInterpol=Spline2:NSmoothSig[0]=10:NSmoothBkg[0]=10:NSmoothBkg[1]=10:NSmooth=10:NAvEvtPerBin=50" ); 

   // test the decorrelated likelihood
   if (Use_LikelihoodD) 
      factory->BookMethod( TMVA::Types::kLikelihood, "LikelihoodD", 
                           "!H:!V:!TransformOutput:PDFInterpol=Spline2:NSmoothSig[0]=10:NSmoothBkg[0]=10:NSmooth=5:NAvEvtPerBin=50:VarTransform=Decorrelate" ); 

   if (Use_LikelihoodPCA) 
      factory->BookMethod( TMVA::Types::kLikelihood, "LikelihoodPCA", 
                           "!H:!V:!TransformOutput:PDFInterpol=Spline2:NSmoothSig[0]=10:NSmoothBkg[0]=10:NSmooth=5:NAvEvtPerBin=50:VarTransform=PCA" ); 
 
   // test the new kernel density estimator
   if (Use_LikelihoodKDE) 
      factory->BookMethod( TMVA::Types::kLikelihood, "LikelihoodKDE", 
                           "!H:!V:!TransformOutput:PDFInterpol=KDE:KDEtype=Gauss:KDEiter=Nonadaptive:KDEborder=None:NAvEvtPerBin=50" ); 

   // test the mixed splines and kernel density estimator (depending on which variable)
   if (Use_LikelihoodMIX) 
      factory->BookMethod( TMVA::Types::kLikelihood, "LikelihoodMIX", 
                           "!H:!V:!TransformOutput:PDFInterpol[0]=KDE:PDFInterpol[1]=KDE:PDFInterpol[2]=Spline2:PDFInterpol[3]=Spline2:KDEtype=Gauss:KDEiter=Nonadaptive:KDEborder=None:NAvEvtPerBin=50" ); 

   // PDE - RS method
   if (Use_PDERS)
      factory->BookMethod( TMVA::Types::kPDERS, "PDERS", 
                           "!H:!V:NormTree=T:VolumeRangeMode=Adaptive:KernelEstimator=Gauss:GaussSigma=0.3:NEventsMin=400:NEventsMax=600" );
   // And the options strings for the MinMax and RMS methods, respectively:
   //      "!H:!V:VolumeRangeMode=MinMax:DeltaFrac=0.2:KernelEstimator=Gauss:GaussSigma=0.3" );   
   //      "!H:!V:VolumeRangeMode=RMS:DeltaFrac=3:KernelEstimator=Gauss:GaussSigma=0.3" );   

   if (Use_PDERSkNN) // depreciated until further notice
      factory->BookMethod( TMVA::Types::kPDERS, "PDERSkNN", 
                           "!H:!V:VolumeRangeMode=kNN:KernelEstimator=Gauss:GaussSigma=0.3:NEventsMin=400:NEventsMax=600" );

   if (Use_PDERSD) 
      factory->BookMethod( TMVA::Types::kPDERS, "PDERSD", 
                           "!H:!V:VolumeRangeMode=Adaptive:KernelEstimator=Gauss:GaussSigma=0.3:NEventsMin=400:NEventsMax=600:VarTransform=Decorrelate" );

   if (Use_PDERSPCA) 
      factory->BookMethod( TMVA::Types::kPDERS, "PDERSPCA", 
                           "!H:!V:VolumeRangeMode=Adaptive:KernelEstimator=Gauss:GaussSigma=0.3:NEventsMin=400:NEventsMax=600:VarTransform=PCA" );

   // this method (PDEFoam) is not (yet) available in this TMVA-version -------------------------------
   // // PDEFoam method
   // if (Use_PDEFoam)
   //     factory->BookMethod( TMVA::Types::kPDEFoam, "PDEFoam", 
   //                          "!H:!V:nCells=200:!SigBgSeparated:Frac=0.999:CreateMVAPdfs" );
   // -------------------------------------------------------------------------------------------------

   // K-Nearest Neighbour classifier (KNN)
   if (Use_KNN)
      factory->BookMethod( TMVA::Types::kKNN, "KNN", 
                           "nkNN=400:TreeOptDepth=6:ScaleFrac=0.8:!UseKernel:!Trim" );  

   // H-Matrix (chi2-squared) method
   if (Use_HMatrix)
      factory->BookMethod( TMVA::Types::kHMatrix, "HMatrix", "!H:!V" ); 

   // Fisher discriminant
   if (Use_Fisher)
      factory->BookMethod( TMVA::Types::kFisher, "Fisher", 
                           "H:!V:!Normalise:CreateMVAPdfs:Fisher:NbinsMVAPdf=50:NsmoothMVAPdf=1" );

   // Function discrimination analysis (FDA) -- test of various fitters - the recommended one is Minuit (or GA or SA)
   if (Use_FDA_MC) 
      factory->BookMethod( TMVA::Types::kFDA, "FDA_MC",
                          "H:!V:Formula=(0)+(1)*x0+(2)*x1+(3)*x2+(4)*x3:ParRanges=(-1,1);(-10,10);(-10,10);(-10,10);(-10,10):FitMethod=MC:SampleSize=100000:Sigma=0.1" );
   
   if (Use_FDA_GA) // can also use Simulated Annealing (SA) algorithm (see Cuts_SA options)
      factory->BookMethod( TMVA::Types::kFDA, "FDA_GA",
                           "H:!V:Formula=(0)+(1)*x0+(2)*x1+(3)*x2+(4)*x3:ParRanges=(-1,1);(-10,10);(-10,10);(-10,10);(-10,10):FitMethod=GA:PopSize=100:Cycles=3:Steps=20:Trim=True:SaveBestGen=0" );

   if (Use_FDA_MT) 
      factory->BookMethod( TMVA::Types::kFDA, "FDA_MT",
                           "H:!V:Formula=(0)+(1)*x0+(2)*x1+(3)*x2+(4)*x3:ParRanges=(-1,1);(-10,10);(-10,10);(-10,10);(-10,10):FitMethod=MINUIT:ErrorLevel=1:PrintLevel=-1:FitStrategy=2:UseImprove:UseMinos:SetBatch" );

   if (Use_FDA_GAMT) 
      factory->BookMethod( TMVA::Types::kFDA, "FDA_GAMT",
                           "H:!V:Formula=(0)+(1)*x0+(2)*x1+(3)*x2+(4)*x3:ParRanges=(-1,1);(-10,10);(-10,10);(-10,10);(-10,10):FitMethod=GA:Converger=MINUIT:ErrorLevel=1:PrintLevel=-1:FitStrategy=0:!UseImprove:!UseMinos:SetBatch:Cycles=1:PopSize=5:Steps=5:Trim" );

   if (Use_FDA_MCMT) 
      factory->BookMethod( TMVA::Types::kFDA, "FDA_MCMT",
                           "H:!V:Formula=(0)+(1)*x0+(2)*x1+(3)*x2+(4)*x3:ParRanges=(-1,1);(-10,10);(-10,10);(-10,10);(-10,10):FitMethod=MC:Converger=MINUIT:ErrorLevel=1:PrintLevel=-1:FitStrategy=0:!UseImprove:!UseMinos:SetBatch:SampleSize=20" );

   // TMVA ANN: MLP (recommended ANN) -- all ANNs in TMVA are Multilayer Perceptrons
   if (Use_MLP)
      factory->BookMethod( TMVA::Types::kMLP, "MLP", "H:!V:!Normalise:NeuronType=tanh:NCycles=200:HiddenLayers=N+1,N:TestRate=5" );

   // CF(Clermont-Ferrand)ANN
   if (Use_CFMlpANN)
      factory->BookMethod( TMVA::Types::kCFMlpANN, "CFMlpANN", "!H:!V:NCycles=500:HiddenLayers=N+1,N"  ); // n_cycles:#nodes:#nodes:...  
  
   // Tmlp(Root)ANN
   if (Use_TMlpANN)
      factory->BookMethod( TMVA::Types::kTMlpANN, "TMlpANN", "!H:!V:NCycles=200:HiddenLayers=N+1,N:LearningMethod=BFGS:ValidationFraction=0.3"  ); // n_cycles:#nodes:#nodes:...
  
   // Support Vector Machines using three different Kernel types (Gauss, polynomial and linear)
   if (Use_SVM_Gauss)
      factory->BookMethod( TMVA::Types::kSVM, "SVM_Gauss", "Sigma=2:C=1:Tol=0.001:Kernel=Gauss" );

   if (Use_SVM_Poly)
      factory->BookMethod( TMVA::Types::kSVM, "SVM_Poly", "Order=4:Theta=1:C=0.1:Tol=0.001:Kernel=Polynomial" );

   if (Use_SVM_Lin)
      factory->BookMethod( TMVA::Types::kSVM, "SVM_Lin", "!H:!V:Kernel=Linear:C=1:Tol=0.001" );  

   // Boosted Decision Trees (second one with decorrelation)
   if (Use_BDT)
      factory->BookMethod( TMVA::Types::kBDT, "BDT", 
                           "!H:!V:NTrees=400:BoostType=AdaBoost:SeparationType=GiniIndex:nCuts=20:PruneMethod=CostComplexity:PruneStrength=1.5" );
   if (Use_BDTD)
      factory->BookMethod( TMVA::Types::kBDT, "BDTD", 
                           "!H:!V:NTrees=400:BoostType=AdaBoost:SeparationType=GiniIndex:nCuts=20:PruneMethod=CostComplexity:PruneStrength=1.5:VarTransform=Decorrelate" );

   // RuleFit -- TMVA implementation of Friedman's method
   if (Use_RuleFit)
      factory->BookMethod( TMVA::Types::kRuleFit, "RuleFit",
                           "H:!V:RuleFitModule=RFTMVA:Model=ModRuleLinear:MinImp=0.001:RuleMinDist=0.001:NTrees=20:fEventsMin=0.01:fEventsMax=0.5:GDTau=-1.0:GDTauPrec=0.01:GDStep=0.01:GDNSteps=10000:GDErrScale=1.02" );

   // Friedman's RuleFit method, implementation by J. Friedman
   if (Use_RuleFitJF)
      factory->BookMethod( TMVA::Types::kRuleFit, "RuleFitJF",
                           "!V:RuleFitModule=RFFriedman:Model=ModRuleLinear:GDStep=0.01:GDNSteps=10000:GDErrScale=1.1:RFNendnodes=4" );
 
   // --------------------------------------------------------------------------------------------------

   // As an example how to use the ROOT plugin mechanism, book BDT via
   // plugin mechanism
   if (Use_Plugin && mlist != 0) {
      TListIter mIt(mlist);
      TObjString *os;
      while( (os = (TObjString*) mIt()) ) {
         TString s(os->GetString());
         if(! s.BeginsWith("P_")) continue;
         s = s(2,s.Length());
         //
         // first the plugin has to be defined, which can happen either through the following line in the local or global .rootrc:
         //
         // # plugin handler          plugin name(regexp) class to be instanciated library        constructor format
         // Plugin.TMVA@@MethodBase:  ^BDT                TMVA::MethodBDT          TMVA.1         "MethodBDT(TString,TString,DataSet&,TString)"
         // 
         // or by telling the global plugin manager directly
         gPluginMgr->AddHandler("TMVA@@MethodBase", "BDT", "TMVA::MethodBDT", "TMVA.1", "MethodBDT(TString,TString,DataSet&,TString)");
         if (s == "BDT")
            factory->BookMethod( TMVA::Types::kPlugins, "BDT",
                                 "!H:!V:NTrees=400:BoostType=AdaBoost:SeparationType=GiniIndex:nCuts=20:PruneMethod=CostComplexity:PruneStrength=4.5" );
      }
   }

   // --------------------------------------------------------------------------------------------------

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
   std::cout << "==> TMVAnalysis is done!" << std::endl;      

   delete mlist;
   delete factory;

   // Launch the GUI for the root macros
   if (!gROOT->IsBatch()) TMVAGui( outfileName );
}
