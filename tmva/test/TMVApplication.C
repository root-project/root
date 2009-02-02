/**********************************************************************************
 * Project   : TMVA - a Root-integrated toolkit for multivariate data analysis    *
 * Package   : TMVA                                                               *
 * Exectuable: TMVApplication                                                     *
 *                                                                                *
 * This macro provides a simple example on how to use the trained classifiers     *
 * within an analysis module                                                      *
 *                                                                                *
 * ------------------------------------------------------------------------------ *
 * see also the alternative (slightly faster) way to retrieve the MVA values in   *
 * examples/TMVApplicationAlternative.cxx                                         *
 * ------------------------------------------------------------------------------ *
 **********************************************************************************/

// ---------------------------------------------------------------
// choose MVA methods to be applied
Bool_t Use_Cuts            = 0;
Bool_t Use_CutsD           = 0;
Bool_t Use_CutsGA          = 1;
Bool_t Use_Likelihood      = 1;
Bool_t Use_LikelihoodD     = 0; // the "D" extension indicates decorrelated input variables (see option strings)
Bool_t Use_LikelihoodPCA   = 1; // the "PCA" extension indicates PCA-transformed input variables (see option strings)
Bool_t Use_PDERS           = 1;
Bool_t Use_PDERSD          = 0;
Bool_t Use_PDERSPCA        = 0;
Bool_t Use_KNN             = 1;
Bool_t Use_HMatrix         = 1;
Bool_t Use_Fisher          = 1;
Bool_t Use_FDA_GA          = 0;
Bool_t Use_FDA_MT          = 1;
Bool_t Use_MLP             = 1; // this is the recommended ANN
Bool_t Use_CFMlpANN        = 0; 
Bool_t Use_TMlpANN         = 0; 
Bool_t Use_SVM_Gauss       = 1;
Bool_t Use_SVM_Poly        = 0;
Bool_t Use_SVM_Lin         = 0;
Bool_t Use_BDT             = 1;
Bool_t Use_BDTD            = 1;
Bool_t Use_RuleFit         = 1;
Bool_t Use_Plugin          = 0;
// ---------------------------------------------------------------

#include <iostream>

#include "TFile.h"
#include "TSystem.h"
#include "TTree.h"
#include "TCut.h"
#include "TStopwatch.h"
#include "TPluginManager.h"

#ifndef __CINT__
#include "TMVA/Tools.h"
#include "TMVA/Reader.h"
#include "TMVA/MethodCuts.h"
#endif

void TMVApplication( TString myMethodList = "" ) 
{
   cout << endl;
   cout << "==> Start TMVApplication" << endl;
   
   TList* mlist = 0;

   if (myMethodList != "") {
      Use_CutsGA = Use_CutsD = Use_Cuts
         = Use_LikelihoodPCA = Use_LikelihoodD = Use_Likelihood
         = Use_PDERSPCA = Use_PDERSD = Use_PDERS 
         = Use_KNN
         = Use_MLP = Use_CFMlpANN = Use_TMlpANN
         = Use_HMatrix = Use_Fisher = Use_BDTD = Use_BDT = Use_RuleFit 
         = Use_SVM_Gauss = Use_SVM_Poly = Use_SVM_Lin
         = Use_FDA_GA = Use_FDA_MT
         = Use_Plugin
         = 0;

      mlist = TMVA::gTools().ParseFormatLine( myMethodList, " :," );

      if (mlist->FindObject( "Cuts"          ) != 0) Use_Cuts          = 1; 
      if (mlist->FindObject( "CutsD"         ) != 0) Use_CutsD         = 1; 
      if (mlist->FindObject( "CutsGA"        ) != 0) Use_CutsGA        = 1; 
      if (mlist->FindObject( "Likelihood"    ) != 0) Use_Likelihood    = 1; 
      if (mlist->FindObject( "LikelihoodD"   ) != 0) Use_LikelihoodD   = 1; 
      if (mlist->FindObject( "LikelihoodPCA" ) != 0) Use_LikelihoodPCA = 1; 
      if (mlist->FindObject( "PDERS"         ) != 0) Use_PDERS         = 1; 
      if (mlist->FindObject( "PDERSD"        ) != 0) Use_PDERSD        = 1; 
      if (mlist->FindObject( "PDERSPCA"      ) != 0) Use_PDERSPCA      = 1; 
      if (mlist->FindObject( "KNN"           ) != 0) Use_KNN           = 1; 
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
      if (mlist->FindObject( "FDA_MT"        ) != 0) Use_FDA_MT        = 1; 
      if (mlist->FindObject( "FDA_GA"        ) != 0) Use_FDA_GA        = 1; 
      if (mlist->FindObject( "Plugin"        ) != 0) Use_Plugin        = 1; 
   }

   //
   // create the Reader object
   //
   TMVA::Reader *reader = new TMVA::Reader( "!Color:!Silent" );    

   // create a set of variables and declare them to the reader
   // - the variable names must corresponds in name and type to 
   // those given in the weight file(s) that you use
   Float_t var1, var2;
   Float_t var3, var4;
   reader->AddVariable( "var1+var2", &var1 );
   reader->AddVariable( "var1-var2", &var2 );
   reader->AddVariable( "var3",      &var3 );
   reader->AddVariable( "var4",      &var4 );

   //
   // book the MVA methods
   //
   string dir    = "weights/";
   string prefix = "TMVAnalysis";
   string weightfile = "";
   TString methodName = "";


   if (Use_Cuts)          { methodName = "Cuts method";           weightfile = "_Cuts.weights.txt";   reader->BookMVA( methodName, dir+prefix+weightfile );}
   if (Use_CutsD)         { methodName = "CutsD method";          weightfile = "_CutsD.weights.txt";  reader->BookMVA( methodName, dir+prefix+weightfile );}
   if (Use_CutsGA)        { methodName = "CutsGA method";         weightfile = "_CutsGA.weights.txt"; reader->BookMVA( methodName, dir+prefix+weightfile );}
   if (Use_Likelihood)    { methodName = "Likelihood method";     weightfile = "_Likelihood.weights.txt";  reader->BookMVA( methodName, dir+prefix+weightfile );}
   if (Use_LikelihoodD)   { methodName = "LikelihoodD method";    weightfile = "_LikelihoodD.weights.txt";  reader->BookMVA( methodName, dir+prefix+weightfile );}
   if (Use_LikelihoodPCA) { methodName = "LikelihoodPCA method";  weightfile = "_LikelihoodPCA.weights.txt";  reader->BookMVA( methodName, dir+prefix+weightfile );}
   if (Use_PDERS)         { methodName = "PDERS method";          weightfile = "_PDERS.weights.txt"; reader->BookMVA( methodName, dir+prefix+weightfile ); }
   if (Use_PDERSD)        { methodName = "PDERSD method";         weightfile = "_PDERSD.weights.txt";  reader->BookMVA( methodName, dir+prefix+weightfile );}
   if (Use_PDERSPCA)      { methodName = "PDERSPCA method";       weightfile = "_PDERSPCA.weights.txt";  reader->BookMVA( methodName, dir+prefix+weightfile );}
   if (Use_KNN)           { methodName = "KNN method";            weightfile = "_KNN.weights.txt"; reader->BookMVA( methodName, dir+prefix+weightfile ); }
   if (Use_HMatrix)       { methodName = "HMatrix method";        weightfile = "_HMatrix.weights.txt";  reader->BookMVA( methodName, dir+prefix+weightfile );}
   if (Use_Fisher)        { methodName = "Fisher method";         weightfile = "_Fisher.weights.txt"; reader->BookMVA( methodName, dir+prefix+weightfile ); }
   if (Use_MLP)           { methodName = "MLP method";            weightfile = "_MLP.weights.txt";  reader->BookMVA( methodName, dir+prefix+weightfile );}
   if (Use_CFMlpANN)      { methodName = "CFMlpANN method";       weightfile = "_CFMlpANN.weights.txt";  reader->BookMVA( methodName, dir+prefix+weightfile );}
   if (Use_TMlpANN)       { methodName = "TMlpANN method";        weightfile = "_TMlpANN.weights.txt";  reader->BookMVA( methodName, dir+prefix+weightfile );}
   if (Use_BDT)           { methodName = "BDT method";            weightfile = "_BDT.weights.txt"; reader->BookMVA( methodName, dir+prefix+weightfile ); }
   if (Use_BDTD)          { methodName = "BDTD method";           weightfile = "_BDTD.weights.txt";  reader->BookMVA( methodName, dir+prefix+weightfile );}
   if (Use_RuleFit)       { methodName = "RuleFit method";        weightfile = "_RuleFit.weights.txt";  reader->BookMVA( methodName, dir+prefix+weightfile );}
   if (Use_SVM_Gauss)     { methodName = "SVM_Gauss method";      weightfile = "_SVM_Gauss.weights.txt"; reader->BookMVA( methodName, dir+prefix+weightfile ); }
   if (Use_SVM_Poly)      { methodName = "SVM_Poly method";       weightfile = "_SVM_Poly.weights.txt";  reader->BookMVA( methodName, dir+prefix+weightfile );}
   if (Use_SVM_Lin)       { methodName = "SVM_Lin method";        weightfile = "_SVM_Lin.weights.txt";  reader->BookMVA( methodName, dir+prefix+weightfile );}
   if (Use_FDA_MT)        { methodName = "FDA_MT method";         weightfile = "_FDA_MT.weights.txt";  reader->BookMVA( methodName, dir+prefix+weightfile );}
   if (Use_FDA_GA)        { methodName = "FDA_GA method";         weightfile = "_FDA_GA.weights.txt"; reader->BookMVA( methodName, dir+prefix+weightfile ); }


   // !!!!!! dont try this:
   //  reader->BookMVA( "Fisher method", "../macros/weights/TMVAnalysis_Fisher.weights.txt" );
   //
   //  this will cause CINT to fatally crash sometimes and funnily not in all directory-folders.
   //
   //  sometimes the characters of the first string are written over the characters of the second
   //  string or even some more important memory regions. 
   //
   //  Conclusion:
   //  Do it like shown above: prepare first a TString object and give this TString to the reader.
   //  CINT preferes it like that. 
   // 
   //  Example of how it can be done:
   //    std::string dir    = "../macros/weights/";
   //    std::string prefix = "TMVAnalysis";
   //    std::string name = "_Fisher.weights.txt";
   //    TString method = "Fisher method";
   //    reader->BookMVA( method, dir + prefix + name );

   // example how to use your own method as plugin
   if (Use_Plugin) {
      // the weight file contains a line 
      // Method         : MethodName::InstanceName

      // if MethodName is not a known TMVA method, it is assumed to be
      // a user implemented method which has to be loaded via the
      // plugin mechanism
      
      // for user implemented methods the line in the weight file can be
      // Method         : PluginName::InstanceName
      // where PluginName can be anything

      // before usage the plugin has to be defined, which can happen
      // either through the following line in .rootrc:
      // # plugin handler          plugin       class            library        constructor format
      // Plugin.TMVA@@MethodBase:  PluginName   MethodClassName  UserPackage    "MethodName(DataSet&,TString)"
      //  
      // or by telling the global plugin manager directly
      gPluginMgr->AddHandler("TMVA@@MethodBase", "PluginName", "MethodClassName", "UserPackage", "MethodName(DataSet&,TString)");
      // the class is then looked for in libUserPackage.so

      // now the method can be booked like any other
      reader->BookMVA( "User method",         dir + prefix + "_User.weights.txt" );
   }

   // book output histograms
   UInt_t nbin = 100;
   TH1F
      *histLk(0), *histLkD(0), *histLkPCA(0), *histPD(0), *histPDD(0), *histPDPCA(0), *histKNN(0), *histHm(0), *histFi(0),
      *histNn(0), *histNnC(0), *histNnT(0), *histBdt(0), *histBdtD(0), *histRf(0), *histSVMG(0), *histSVMP(0), *histSVML(0),
      *histFDAMT(0), *histFDAGA(0), *histPBdt(0);

   if (Use_Likelihood)    histLk    = new TH1F( "MVA_Likelihood",    "MVA_Likelihood",    nbin,  0, 1 );
   if (Use_LikelihoodD)   histLkD   = new TH1F( "MVA_LikelihoodD",   "MVA_LikelihoodD",   nbin,  0.000001, 0.9999 );
   if (Use_LikelihoodPCA) histLkPCA = new TH1F( "MVA_LikelihoodPCA", "MVA_LikelihoodPCA", nbin,  0, 1 );
   if (Use_PDERS)         histPD    = new TH1F( "MVA_PDERS",         "MVA_PDERS",         nbin,  0, 1 );
   if (Use_PDERSD)        histPDD   = new TH1F( "MVA_PDERSD",        "MVA_PDERSD",        nbin,  0, 1 );
   if (Use_PDERSPCA)      histPDPCA = new TH1F( "MVA_PDERSPCA",      "MVA_PDERSPCA",      nbin,  0, 1 );
   if (Use_KNN)           histKNN   = new TH1F( "MVA_KNN",           "MVA_KNN",           nbin,  0, 1 );
   if (Use_HMatrix)       histHm    = new TH1F( "MVA_HMatrix",       "MVA_HMatrix",       nbin, -0.95, 1.55 );
   if (Use_Fisher)        histFi    = new TH1F( "MVA_Fisher",        "MVA_Fisher",        nbin, -4, 4 );
   if (Use_MLP)           histNn    = new TH1F( "MVA_MLP",           "MVA_MLP",           nbin, -1.25, 1.5 );
   if (Use_CFMlpANN)      histNnC   = new TH1F( "MVA_CFMlpANN",      "MVA_CFMlpANN",      nbin,  0, 1 );
   if (Use_TMlpANN)       histNnT   = new TH1F( "MVA_TMlpANN",       "MVA_TMlpANN",       nbin, -1.3, 1.3 );
   if (Use_BDT)           histBdt   = new TH1F( "MVA_BDT",           "MVA_BDT",           nbin, -0.8, 0.8 );
   if (Use_BDTD)          histBdtD  = new TH1F( "MVA_BDTD",          "MVA_BDTD",          nbin, -0.8, 0.8 );
   if (Use_RuleFit)       histRf    = new TH1F( "MVA_RuleFit",       "MVA_RuleFit",       nbin, -2.0, 2.0 );
   if (Use_SVM_Gauss)     histSVMG  = new TH1F( "MVA_SVM_Gauss",     "MVA_SVM_Gauss",     nbin, 0.0, 1.0 );
   if (Use_SVM_Poly)      histSVMP  = new TH1F( "MVA_SVM_Poly",      "MVA_SVM_Poly",      nbin, 0.0, 1.0 );
   if (Use_SVM_Lin)       histSVML  = new TH1F( "MVA_SVM_Lin",       "MVA_SVM_Lin",       nbin, 0.0, 1.0 );
   if (Use_FDA_MT)        histFDAMT = new TH1F( "MVA_FDA_MT",        "MVA_FDA_MT",        nbin, -2.0, 3.0 );
   if (Use_FDA_GA)        histFDAGA = new TH1F( "MVA_FDA_GA",        "MVA_FDA_GA",        nbin, -2.0, 3.0 );
   if (Use_Plugin)        histPBdt  = new TH1F( "MVA_PBDT",          "MVA_BDT",           nbin, -0.8, 0.8 );

   // book example histogram for probability (the other methods are done similarly)
   TH1F *probHistFi(0), *rarityHistFi(0);
   if (Use_Fisher) {
      probHistFi   = new TH1F( "PROBA_MVA_Fisher",  "PROBA_MVA_Fisher",  nbin, 0, 1 );
      rarityHistFi = new TH1F( "RARITY_MVA_Fisher", "RARITY_MVA_Fisher", nbin, 0, 1 );
   }

   // Prepare input tree (this must be replaced by your data source)
   // in this example, there is a toy tree with signal and one with background events
   // we'll later on use only the "signal" events for the test in this example.
   //   
   TFile *input(0);
   TString fname = "./tmva_example.root";   
   if (!gSystem->AccessPathName( fname )) {
      // first we try to find tmva_example.root in the local directory
      cout << "--- Accessing data file: " << fname << endl;
      input = TFile::Open( fname );
   } 
   else { 
      // second we try accessing the file via the web from
      // http://root.cern.ch/files/tmva_example.root
      cout << "--- Accessing tmva_example.root file from http://root.cern.ch/files" << endl;
      cout << "--- for faster startup you may consider downloading it into you local directory" << endl;
      input = TFile::Open("http://root.cern.ch/files/tmva_example.root");
   }
   
   if (!input) {
      cout << "ERROR: could not open data file: " << fname << endl;
      exit(1);
   }

   //
   // prepare the tree
   // - here the variable names have to corresponds to your tree
   // - you can use the same variables as above which is slightly faster,
   //   but of course you can use different ones and copy the values inside the event loop
   //
   TTree* theTree = (TTree*)input->Get("TreeS");
   cout << "--- Select signal sample" << endl;
   Float_t userVar1, userVar2;
   theTree->SetBranchAddress( "var1", &userVar1 );
   theTree->SetBranchAddress( "var2", &userVar2 );
   theTree->SetBranchAddress( "var3", &var3 );
   theTree->SetBranchAddress( "var4", &var4 );

   // efficiency calculator for cut method
   Int_t    nSelCuts = 0, nSelCutsD = 0, nSelCutsGA = 0;
   Double_t effS     = 0.7;

   cout << "--- Processing: " << theTree->GetEntries() << " events" << endl;
   TStopwatch sw;
   sw.Start();
   for (Long64_t ievt=0; ievt<theTree->GetEntries();ievt++) {

      if (ievt%1000 == 0){
         cout << "--- ... Processing event: " << ievt << endl;
      }

      theTree->GetEntry(ievt);

      var1 = userVar1 + userVar2;
      var2 = userVar1 - userVar2;

      // 
      // return the MVAs and fill to histograms
      // 
      if (Use_Cuts) {
         // Cuts is a special case: give the desired signal efficienciy
         Bool_t passed = reader->EvaluateMVA( "Cuts method", effS );
         if (passed) nSelCuts++;
      }
      if (Use_CutsD) {
         // Cuts is a special case: give the desired signal efficienciy
         Bool_t passed = reader->EvaluateMVA( "CutsD method", effS );
         if (passed) nSelCutsD++;
      }
      if (Use_CutsGA) {
         // Cuts is a special case: give the desired signal efficienciy
         Bool_t passed = reader->EvaluateMVA( "CutsGA method", effS );
         if (passed) nSelCutsGA++;
      }

      if (Use_Likelihood   )   histLk    ->Fill( reader->EvaluateMVA( "Likelihood method"    ) );
      if (Use_LikelihoodD  )   histLkD   ->Fill( reader->EvaluateMVA( "LikelihoodD method"   ) );
      if (Use_LikelihoodPCA)   histLkPCA ->Fill( reader->EvaluateMVA( "LikelihoodPCA method" ) );
      if (Use_PDERS        )   histPD    ->Fill( reader->EvaluateMVA( "PDERS method"         ) );
      if (Use_PDERSD       )   histPDD   ->Fill( reader->EvaluateMVA( "PDERSD method"        ) );
      if (Use_PDERSPCA     )   histPDPCA ->Fill( reader->EvaluateMVA( "PDERSPCA method"      ) );
      if (Use_KNN          )   histKNN   ->Fill( reader->EvaluateMVA( "KNN method"           ) );
      if (Use_HMatrix      )   histHm    ->Fill( reader->EvaluateMVA( "HMatrix method"       ) );
      if (Use_Fisher       )   histFi    ->Fill( reader->EvaluateMVA( "Fisher method"        ) );
      if (Use_MLP          )   histNn    ->Fill( reader->EvaluateMVA( "MLP method"           ) );
      if (Use_CFMlpANN     )   histNnC   ->Fill( reader->EvaluateMVA( "CFMlpANN method"      ) );
      if (Use_TMlpANN      )   histNnT   ->Fill( reader->EvaluateMVA( "TMlpANN method"       ) );
      if (Use_BDT          )   histBdt   ->Fill( reader->EvaluateMVA( "BDT method"           ) );
      if (Use_BDTD         )   histBdtD  ->Fill( reader->EvaluateMVA( "BDTD method"          ) );
      if (Use_RuleFit      )   histRf    ->Fill( reader->EvaluateMVA( "RuleFit method"       ) );
      if (Use_SVM_Gauss    )   histSVMG  ->Fill( reader->EvaluateMVA( "SVM_Gauss method"     ) );
      if (Use_SVM_Poly     )   histSVMP  ->Fill( reader->EvaluateMVA( "SVM_Poly method"      ) );
      if (Use_SVM_Lin      )   histSVML  ->Fill( reader->EvaluateMVA( "SVM_Lin method"       ) );
      if (Use_FDA_MT       )   histFDAMT ->Fill( reader->EvaluateMVA( "FDA_MT method"        ) );
      if (Use_FDA_GA       )   histFDAGA ->Fill( reader->EvaluateMVA( "FDA_GA method"        ) );
      if (Use_Plugin       )   histPBdt  ->Fill( reader->EvaluateMVA( "P_BDT method"         ) );

      // retrieve probability instead of MVA output
      if (Use_Fisher       )   {
         probHistFi->Fill( reader->GetProba ( "Fisher method" ) );
         rarityHistFi->Fill( reader->GetRarity( "Fisher method" ) );
      }
   }
   sw.Stop();
   cout << "--- End of event loop: "; sw.Print();
   // get elapsed time
   if (Use_Cuts)   cout << "--- Efficiency for Cuts method  : " << double(nSelCuts)/theTree->GetEntries()
                        << " (for a required signal efficiency of " << effS << ")" << endl;
   if (Use_CutsD)  cout << "--- Efficiency for CutsD method : " << double(nSelCutsD)/theTree->GetEntries()
                        << " (for a required signal efficiency of " << effS << ")" << endl;
   if (Use_CutsGA) cout << "--- Efficiency for CutsGA method: " << double(nSelCutsGA)/theTree->GetEntries()
                        << " (for a required signal efficiency of " << effS << ")" << endl;

   if (Use_Cuts || Use_CutsD || Use_CutsGA ){
      // test: retrieve cuts for particular signal efficiency

      // workaround for CINT (dynamic_cast does not work)
      // mcuts = TMVA::MethodCuts::DynamicCast( reader->FindMVA( "CutsGA method" ) );
      TMVA::MethodCuts* mcuts = dynamic_cast<TMVA::MethodCuts*>( reader->FindMVA( "CutsGA method" ) );

      if (mcuts) {      
         std::vector<Double_t> cutsMin;
         std::vector<Double_t> cutsMax;
         mcuts->GetCuts( 0.7, cutsMin, cutsMax );
         cout << "--- -------------------------------------------------------------" << endl;
         cout << "--- Retrieve cut values for signal efficiency of 0.7 from Reader" << endl;
         for (UInt_t ivar=0; ivar<cutsMin.size(); ivar++) {
            cout << "... Cut: " 
                 << cutsMin[ivar] 
                 << " < \"" 
                 << mcuts->GetInputVar(ivar)
                 << "\" <= " 
                 << cutsMax[ivar] << endl;
         }
         cout << "--- -------------------------------------------------------------" << endl;
      }
   }

   //
   // write histograms
   //
   TFile *target  = new TFile( "TMVApp.root","RECREATE" );
   if (Use_Likelihood   )   histLk    ->Write();
   if (Use_LikelihoodD  )   histLkD   ->Write();
   if (Use_LikelihoodPCA)   histLkPCA ->Write();
   if (Use_PDERS        )   histPD    ->Write();
   if (Use_PDERSD       )   histPDD   ->Write();
   if (Use_PDERSPCA     )   histPDPCA ->Write();
   if (Use_KNN          )   histKNN   ->Write();
   if (Use_HMatrix      )   histHm    ->Write();
   if (Use_Fisher       )   histFi    ->Write();
   if (Use_MLP          )   histNn    ->Write();
   if (Use_CFMlpANN     )   histNnC   ->Write();
   if (Use_TMlpANN      )   histNnT   ->Write();
   if (Use_BDT          )   histBdt   ->Write();
   if (Use_BDTD         )   histBdtD  ->Write();
   if (Use_RuleFit      )   histRf    ->Write();
   if (Use_SVM_Gauss    )   histSVMG  ->Write();
   if (Use_SVM_Poly     )   histSVMP  ->Write();
   if (Use_SVM_Lin      )   histSVML  ->Write();
   if (Use_FDA_MT       )   histFDAMT ->Write();
   if (Use_FDA_GA       )   histFDAGA ->Write();
   if (Use_Plugin       )   histPBdt  ->Write();

   // write also probability hists
   if (Use_Fisher) { if( probHistFi != 0 ) probHistFi->Write(); if( rarityHistFi != 0 ) rarityHistFi->Write(); }
   target->Close();

   cout << "--- Created root file: \"TMVApp.root\" containing the MVA output histograms" << endl;
  
   delete mlist;
   delete reader;
    
   cout << "==> TMVApplication is done!" << endl << endl;
} 
