/**********************************************************************************
 * Project   : TMVA - a Root-integrated toolkit for multivariate data analysis    *
 * Package   : TMVA                                                               *
 * Exectuable: PlotDecisionBoundary                                               *
 *                                                                                *
 * Plots decision boundary for simple cases in 2 (3?) dimensions                  *
 **********************************************************************************/

#include <cstdlib>
#include <vector>
#include <iostream>
#include <map>
#include <string>

#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TSystem.h"
#include "TROOT.h"
#include "TStopwatch.h"
#include "TGraph.h"
#include "TH2F.h"
#include "TH3F.h"
#include "TMVAGui.C"

#if not defined(__CINT__) || defined(__MAKECINT__)
#include "TMVA/Tools.h"
#include "TMVA/Reader.h"
#include "TMVA/MethodCuts.h"
#endif

using namespace TMVA;


void plot(TH2D *sig, TH2D *bkg, TH2F *MVA, TString v0="var0", TString v1="var1"){

   TCanvas *c = new TCanvas(Form("DecisionBoundary%s",MVA->GetTitle()),MVA->GetTitle(),800,800);

   gStyle->SetPalette(1);
   MVA->SetXTitle(v0);
   MVA->SetYTitle(v1);
   MVA->SetStats(0);
   MVA->Draw("cont1");
   sig->SetMarkerColor(2);
   bkg->SetMarkerColor(4);
   sig->SetMarkerStyle(20);
   bkg->SetMarkerStyle(20);
   sig->SetMarkerSize(.5);
   bkg->SetMarkerSize(.5);
   sig->Draw("same");
   bkg->Draw("same");
}


void PlotDecisionBoundary( TString myMethodList = "",TString v0="var0", TString v1="var1", TString dataFileName = "/home/hvoss/TMVA/TMVA_data/data/data_3Bumps.root", TString weightFilePrefix="TMVA") 
{   
   //---------------------------------------------------------------
   // default MVA methods to be trained + tested

   // this loads the library
   TMVA::Tools::Instance();

   std::map<std::string,int> Use;

   Use["CutsGA"]          = 0; // other "Cuts" methods work identically
   // ---
   Use["Likelihood"]      = 0;
   Use["LikelihoodD"]     = 0; // the "D" extension indicates decorrelated input variables (see option strings)
   Use["LikelihoodPCA"]   = 0; // the "PCA" extension indicates PCA-transformed input variables (see option strings)
   Use["LikelihoodKDE"]   = 0;
   Use["LikelihoodMIX"]   = 0;
   // ---
   Use["PDERS"]           = 0;
   Use["PDERSD"]          = 0;
   Use["PDERSPCA"]        = 0;
   Use["PDERSkNN"]        = 0; // depreciated until further notice
   Use["PDEFoam"]         = 0;
   // --
   Use["KNN"]             = 0;
   // ---
   Use["HMatrix"]         = 0;
   Use["Fisher"]          = 0;
   Use["FisherG"]         = 0;
   Use["BoostedFisher"]   = 0;
   Use["LD"]              = 0;
   // ---
   Use["FDA_GA"]          = 0;
   Use["FDA_SA"]          = 0;
   Use["FDA_MC"]          = 0;
   Use["FDA_MT"]          = 0;
   Use["FDA_GAMT"]        = 0;
   Use["FDA_MCMT"]        = 0;
   // ---
   Use["MLP"]             = 0; // this is the recommended ANN
   Use["MLPBFGS"]         = 0; // recommended ANN with optional training method
   Use["CFMlpANN"]        = 0; // *** missing
   Use["TMlpANN"]         = 0; 
   // ---
   Use["SVM"]             = 0;
   // ---
   Use["BDT"]             = 0;
   Use["BDTD"]            = 0;
   Use["BDTG"]            = 0;
   Use["BDTB"]            = 0;
   // ---
   Use["RuleFit"]         = 0;
   // ---
   Use["Category"]        = 0;
   // ---
   Use["Plugin"]          = 0;
   // ---------------------------------------------------------------

   std::cout << std::endl;
   std::cout << "==> Start TMVAClassificationApplication" << std::endl;

   if (myMethodList != "") {
      for (std::map<std::string,int>::iterator it = Use.begin(); it != Use.end(); it++) it->second = 0;

      std::vector<TString> mlist = gTools().SplitString( myMethodList, ',' );
      for (UInt_t i=0; i<mlist.size(); i++) {
         std::string regMethod(mlist[i]);

         if (Use.find(regMethod) == Use.end()) {
            std::cout << "Method \"" << regMethod << "\" not known in TMVA under this name. Choose among the following:" << std::endl;
            for (std::map<std::string,int>::iterator it = Use.begin(); it != Use.end(); it++) std::cout << it->first << " ";
            std::cout << std::endl;
            return;
         }
         Use[regMethod] = 1;
      }
   }

   //
   // create the Reader object
   //
   TMVA::Reader *reader = new TMVA::Reader( "!Color:!Silent" );    

   // create a set of variables and declare them to the reader
   // - the variable names must corresponds in name and type to 
   // those given in the weight file(s) that you use
   Float_t var0, var1;
   reader->AddVariable( v0,                &var0 );
   reader->AddVariable( v1,                &var1 );

   //
   // book the MVA methods
   //
   TString dir    = "weights/";
   TString prefix = weightFilePrefix;

   // book method(s)
   for (std::map<std::string,int>::iterator it = Use.begin(); it != Use.end(); it++) {
      if (it->second) {
         TString methodName = it->first + " method";
         TString weightfile = dir + prefix + "_" + TString(it->first) + ".weights.xml";
         reader->BookMVA( methodName, weightfile ); 
      }
   }
   
   TFile *f = new TFile(dataFileName);
   TTree *signal     = (TTree*)f->Get("TreeS");
   TTree *background = (TTree*)f->Get("TreeB");


   

//Declaration of leaves types
   Float_t         svar0;
   Float_t         svar1;
   Float_t         bvar0;
   Float_t         bvar1;

   // Set branch addresses.
   signal->SetBranchAddress(v0,&svar0);
   signal->SetBranchAddress(v1,&svar1);
   background->SetBranchAddress(v0,&bvar0);
   background->SetBranchAddress(v1,&bvar1);


   UInt_t nbin = 50;
   Float_t xmax = signal->GetMaximum(v0.Data());
   Float_t xmin = signal->GetMinimum(v0.Data());
   Float_t ymax = signal->GetMaximum(v1.Data());
   Float_t ymin = signal->GetMinimum(v1.Data());

   TH2D *hs=new TH2D("hs","",nbin,xmin,xmax,nbin,ymin,ymax);   
   TH2D *hb=new TH2D("hb","",nbin,xmin,xmax,nbin,ymin,ymax);   


   Long64_t nentries;
   nentries = TreeS->GetEntries();
   for (Long64_t is=0; is<nentries;is++) {
      signal->GetEntry(is);
      hs->Fill(svar0,svar1);
   }
   nentries = TreeB->GetEntries();
   for (Long64_t ib=0; ib<nentries;ib++) {
      background->GetEntry(ib);
      hb->Fill(bvar0,bvar1);
   }


   hb->SetMarkerColor(4);
   hs->SetMarkerColor(2);


   // book output histograms
   TH2F *histLk(0), *histLkD(0), *histLkPCA(0), *histLkKDE(0), *histLkMIX(0), *histPD(0), *histPDD(0);
   TH2F *histPDPCA(0), *histPDEFoam(0), *histPDEFoamErr(0), *histPDEFoamSig(0), *histKNN(0), *histHm(0);
   TH2F *histFi(0), *histFiG(0), *histFiB(0), *histLD(0), *histNn(0), *histNnC(0), *histNnT(0), *histBdt(0), *histBdtG(0), *histBdtD(0);
   TH2F *histRf(0), *histSVMG(0), *histSVMP(0), *histSVML(0), *histFDAMT(0), *histFDAGA(0), *histCat(0), *histPBdt(0);

   if (Use["Likelihood"])    histLk      = new TH2F( "MVA_Likelihood",    "MVA_Likelihood",    nbin,xmin,xmax,nbin,ymin,ymax);
   if (Use["LikelihoodD"])   histLkD     = new TH2F( "MVA_LikelihoodD",   "MVA_LikelihoodD",   nbin,xmin,xmax,nbin,ymin,ymax);
   if (Use["LikelihoodPCA"]) histLkPCA   = new TH2F( "MVA_LikelihoodPCA", "MVA_LikelihoodPCA", nbin,xmin,xmax,nbin,ymin,ymax);
   if (Use["LikelihoodKDE"]) histLkKDE   = new TH2F( "MVA_LikelihoodKDE", "MVA_LikelihoodKDE", nbin,xmin,xmax,nbin,ymin,ymax);
   if (Use["LikelihoodMIX"]) histLkMIX   = new TH2F( "MVA_LikelihoodMIX", "MVA_LikelihoodMIX", nbin,xmin,xmax,nbin,ymin,ymax);
   if (Use["PDERS"])         histPD      = new TH2F( "MVA_PDERS",         "MVA_PDERS",         nbin,xmin,xmax,nbin,ymin,ymax);
   if (Use["PDERSD"])        histPDD     = new TH2F( "MVA_PDERSD",        "MVA_PDERSD",        nbin,xmin,xmax,nbin,ymin,ymax);
   if (Use["PDERSPCA"])      histPDPCA   = new TH2F( "MVA_PDERSPCA",      "MVA_PDERSPCA",      nbin,xmin,xmax,nbin,ymin,ymax);
   if (Use["KNN"])           histKNN     = new TH2F( "MVA_KNN",           "MVA_KNN",           nbin,xmin,xmax,nbin,ymin,ymax);
   if (Use["HMatrix"])       histHm      = new TH2F( "MVA_HMatrix",       "MVA_HMatrix",       nbin,xmin,xmax,nbin,ymin,ymax);
   if (Use["Fisher"])        histFi      = new TH2F( "MVA_Fisher",        "MVA_Fisher",        nbin,xmin,xmax,nbin,ymin,ymax);
   if (Use["FisherG"])        histFiG    = new TH2F( "MVA_FisherG",       "MVA_FisherG",       nbin,xmin,xmax,nbin,ymin,ymax);
   if (Use["BoostedFisher"])  histFiB    = new TH2F( "MVA_BoostedFisher", "MVA_BoostedFisher", nbin,xmin,xmax,nbin,ymin,ymax);
   if (Use["LD"])            histLD      = new TH2F( "MVA_LD",            "MVA_LD",            nbin,xmin,xmax,nbin,ymin,ymax);
   if (Use["MLP"])           histNn      = new TH2F( "MVA_MLP",           "MVA_MLP",           nbin,xmin,xmax,nbin,ymin,ymax);
   if (Use["CFMlpANN"])      histNnC     = new TH2F( "MVA_CFMlpANN",      "MVA_CFMlpANN",      nbin,xmin,xmax,nbin,ymin,ymax);
   if (Use["TMlpANN"])       histNnT     = new TH2F( "MVA_TMlpANN",       "MVA_TMlpANN",       nbin,xmin,xmax,nbin,ymin,ymax);
   if (Use["BDT"])           histBdt     = new TH2F( "MVA_BDT",           "MVA_BDT",           nbin,xmin,xmax,nbin,ymin,ymax);
   if (Use["BDTD"])          histBdtD    = new TH2F( "MVA_BDTD",          "MVA_BDTD",          nbin,xmin,xmax,nbin,ymin,ymax);
   if (Use["BDTG"])          histBdtG    = new TH2F( "MVA_BDTG",          "MVA_BDTG",          nbin,xmin,xmax,nbin,ymin,ymax);
   if (Use["RuleFit"])       histRf      = new TH2F( "MVA_RuleFit",       "MVA_RuleFit",       nbin,xmin,xmax,nbin,ymin,ymax);
   if (Use["SVM_Gauss"])     histSVMG    = new TH2F( "MVA_SVM_Gauss",     "MVA_SVM_Gauss",     nbin,xmin,xmax,nbin,ymin,ymax);
   if (Use["SVM_Poly"])      histSVMP    = new TH2F( "MVA_SVM_Poly",      "MVA_SVM_Poly",      nbin,xmin,xmax,nbin,ymin,ymax);
   if (Use["SVM_Lin"])       histSVML    = new TH2F( "MVA_SVM_Lin",       "MVA_SVM_Lin",       nbin,xmin,xmax,nbin,ymin,ymax);
   if (Use["FDA_MT"])        histFDAMT   = new TH2F( "MVA_FDA_MT",        "MVA_FDA_MT",        nbin,xmin,xmax,nbin,ymin,ymax);
   if (Use["FDA_GA"])        histFDAGA   = new TH2F( "MVA_FDA_GA",        "MVA_FDA_GA",        nbin,xmin,xmax,nbin,ymin,ymax);
   if (Use["Category"])      histCat     = new TH2F( "MVA_Category",      "MVA_Category",      nbin,xmin,xmax,nbin,ymin,ymax);
   if (Use["Plugin"])        histPBdt    = new TH2F( "MVA_PBDT",          "MVA_BDT",           nbin,xmin,xmax,nbin,ymin,ymax); 



   // Prepare input tree (this must be replaced by your data source)
   // in this example, there is a toy tree with signal and one with background events
   // we'll later on use only the "signal" events for the test in this example.



   for (Int_t ibin=1; ibin<nbin+1; ibin++){
      for (Int_t jbin=1; jbin<nbin+1; jbin++){
         var0 = hs->GetXaxis()->GetBinCenter(ibin);
         var1 = hs->GetYaxis()->GetBinCenter(jbin);
         
         
         if (Use["Likelihood"   ])   histLk     ->SetBinContent(ibin,jbin, reader->EvaluateMVA( "Likelihood method"    ) );
         if (Use["LikelihoodD"  ])   histLkD    ->SetBinContent(ibin,jbin, reader->EvaluateMVA( "LikelihoodD method"   ) );
         if (Use["LikelihoodPCA"])   histLkPCA  ->SetBinContent(ibin,jbin, reader->EvaluateMVA( "LikelihoodPCA method" ) );
         if (Use["LikelihoodKDE"])   histLkKDE  ->SetBinContent(ibin,jbin, reader->EvaluateMVA( "LikelihoodKDE method" ) );
         if (Use["LikelihoodMIX"])   histLkMIX  ->SetBinContent(ibin,jbin, reader->EvaluateMVA( "LikelihoodMIX method" ) );
         if (Use["PDERS"        ])   histPD     ->SetBinContent(ibin,jbin, reader->EvaluateMVA( "PDERS method"         ) );
         if (Use["PDERSD"       ])   histPDD    ->SetBinContent(ibin,jbin, reader->EvaluateMVA( "PDERSD method"        ) );
         if (Use["PDERSPCA"     ])   histPDPCA  ->SetBinContent(ibin,jbin, reader->EvaluateMVA( "PDERSPCA method"      ) );
         if (Use["KNN"          ])   histKNN    ->SetBinContent(ibin,jbin, reader->EvaluateMVA( "KNN method"           ) );
         if (Use["HMatrix"      ])   histHm     ->SetBinContent(ibin,jbin, reader->EvaluateMVA( "HMatrix method"       ) );
         if (Use["Fisher"       ])   histFi     ->SetBinContent(ibin,jbin, reader->EvaluateMVA( "Fisher method"        ) );
         if (Use["FisherG"      ])   histFiG    ->SetBinContent(ibin,jbin, reader->EvaluateMVA( "FisherG method"        ) );
         if (Use["BoostedFisher"])   histFiB    ->SetBinContent(ibin,jbin, reader->EvaluateMVA( "BoostedFisher method"        ) );
         if (Use["LD"           ])   histLD     ->SetBinContent(ibin,jbin, reader->EvaluateMVA( "LD method"            ) );
         if (Use["MLP"          ])   histNn     ->SetBinContent(ibin,jbin, reader->EvaluateMVA( "MLP method"           ) );
         if (Use["CFMlpANN"     ])   histNnC    ->SetBinContent(ibin,jbin, reader->EvaluateMVA( "CFMlpANN method"      ) );
         if (Use["TMlpANN"      ])   histNnT    ->SetBinContent(ibin,jbin, reader->EvaluateMVA( "TMlpANN method"       ) );
         if (Use["BDT"          ])   histBdt    ->SetBinContent(ibin,jbin, reader->EvaluateMVA( "BDT method"           ) );
         if (Use["BDTD"         ])   histBdtD   ->SetBinContent(ibin,jbin, reader->EvaluateMVA( "BDTD method"          ) );
         if (Use["BDTG"         ])   histBdtG   ->SetBinContent(ibin,jbin, reader->EvaluateMVA( "BDTG method"          ) );
         if (Use["RuleFit"      ])   histRf     ->SetBinContent(ibin,jbin, reader->EvaluateMVA( "RuleFit method"       ) );
         if (Use["SVM_Gauss"    ])   histSVMG   ->SetBinContent(ibin,jbin, reader->EvaluateMVA( "SVM_Gauss method"     ) );
         if (Use["SVM_Poly"     ])   histSVMP   ->SetBinContent(ibin,jbin, reader->EvaluateMVA( "SVM_Poly method"      ) );
         if (Use["SVM_Lin"      ])   histSVML   ->SetBinContent(ibin,jbin, reader->EvaluateMVA( "SVM_Lin method"       ) );
         if (Use["FDA_MT"       ])   histFDAMT  ->SetBinContent(ibin,jbin, reader->EvaluateMVA( "FDA_MT method"        ) );
         if (Use["FDA_GA"       ])   histFDAGA  ->SetBinContent(ibin,jbin, reader->EvaluateMVA( "FDA_GA method"        ) );
         if (Use["Category"     ])   histCat    ->SetBinContent(ibin,jbin, reader->EvaluateMVA( "Category method"         ) );
         if (Use["Plugin"       ])   histPBdt   ->SetBinContent(ibin,jbin, reader->EvaluateMVA( "P_BDT method"         ) );
      }
   }


   std::cout << "--- Created root file: \"TMVApp.root\" containing the MVA output histograms" << std::endl;
  
   delete reader;



    
   std::cout << "==> TMVAClassificationApplication is done!" << endl << std::endl;



   gStyle->SetPalette(1);

   if (Use["Likelihood"   ]) plot(hs,hb,histLk     ,v0,v1);
   if (Use["LikelihoodD"  ]) plot(hs,hb,histLkD    ,v0,v1);
   if (Use["LikelihoodPCA"]) plot(hs,hb,histLkPCA  ,v0,v1);
   if (Use["LikelihoodKDE"]) plot(hs,hb,histLkKDE  ,v0,v1);
   if (Use["LikelihoodMIX"]) plot(hs,hb,histLkMIX  ,v0,v1);
   if (Use["PDERS"        ]) plot(hs,hb,histPD     ,v0,v1);
   if (Use["PDERSD"       ]) plot(hs,hb,histPDD    ,v0,v1);
   if (Use["PDERSPCA"     ]) plot(hs,hb,histPDPCA  ,v0,v1);
   if (Use["KNN"          ]) plot(hs,hb,histKNN    ,v0,v1);
   if (Use["HMatrix"      ]) plot(hs,hb,histHm     ,v0,v1);
   if (Use["Fisher"       ]) plot(hs,hb,histFi     ,v0,v1);
   if (Use["FisherG"      ]) plot(hs,hb,histFiG    ,v0,v1);
   if (Use["BoostedFisher"]) plot(hs,hb,histFiB    ,v0,v1);
   if (Use["LD"           ]) plot(hs,hb,histLD     ,v0,v1);
   if (Use["MLP"          ]) plot(hs,hb,histNn     ,v0,v1);
   if (Use["CFMlpANN"     ]) plot(hs,hb,histNnC    ,v0,v1);
   if (Use["TMlpANN"      ]) plot(hs,hb,histNnT    ,v0,v1);
   if (Use["BDT"          ]) plot(hs,hb,histBdt    ,v0,v1);
   if (Use["BDTD"         ]) plot(hs,hb,histBdtD   ,v0,v1);
   if (Use["BDTG"         ]) plot(hs,hb,histBdtG   ,v0,v1); 
   if (Use["RuleFit"      ]) plot(hs,hb,histRf     ,v0,v1);
   if (Use["SVM_Gauss"    ]) plot(hs,hb,histSVMG   ,v0,v1);
   if (Use["SVM_Poly"     ]) plot(hs,hb,histSVMP   ,v0,v1);
   if (Use["SVM_Lin"      ]) plot(hs,hb,histSVML   ,v0,v1);
   if (Use["FDA_MT"       ]) plot(hs,hb,histFDAMT  ,v0,v1);
   if (Use["FDA_GA"       ]) plot(hs,hb,histFDAGA  ,v0,v1);
   if (Use["Category"     ]) plot(hs,hb,histCat    ,v0,v1);
   if (Use["Plugin"       ]) plot(hs,hb,histPBdt   ,v0,v1);

   
   //
   // write histograms
   //
   TFile *target  = new TFile( "TMVApp.root","RECREATE" );

   hs->Write();
   hb->Write();

   if (Use["Likelihood"   ])   histLk     ->Write();
   if (Use["LikelihoodD"  ])   histLkD    ->Write();
   if (Use["LikelihoodPCA"])   histLkPCA  ->Write();
   if (Use["LikelihoodKDE"])   histLkKDE  ->Write();
   if (Use["LikelihoodMIX"])   histLkMIX  ->Write();
   if (Use["PDERS"        ])   histPD     ->Write();
   if (Use["PDERSD"       ])   histPDD    ->Write();
   if (Use["PDERSPCA"     ])   histPDPCA  ->Write();
   if (Use["KNN"          ])   histKNN    ->Write();
   if (Use["HMatrix"      ])   histHm     ->Write();
   if (Use["Fisher"       ])   histFi     ->Write();
   if (Use["FisherG"      ])   histFiG    ->Write();
   if (Use["BoostedFisher"])   histFiB    ->Write();
   if (Use["LD"           ])   histLD     ->Write();
   if (Use["MLP"          ])   histNn     ->Write();
   if (Use["CFMlpANN"     ])   histNnC    ->Write();
   if (Use["TMlpANN"      ])   histNnT    ->Write();
   if (Use["BDT"          ])   histBdt    ->Write();
   if (Use["BDTD"         ])   histBdtD   ->Write();
   if (Use["BDTG"         ])   histBdtG   ->Write(); 
   if (Use["RuleFit"      ])   histRf     ->Write();
   if (Use["SVM_Gauss"    ])   histSVMG   ->Write();
   if (Use["SVM_Poly"     ])   histSVMP   ->Write();
   if (Use["SVM_Lin"      ])   histSVML   ->Write();
   if (Use["FDA_MT"       ])   histFDAMT  ->Write();
   if (Use["FDA_GA"       ])   histFDAGA  ->Write();
   if (Use["Category"     ])   histCat    ->Write();
   if (Use["Plugin"       ])   histPBdt   ->Write();

   target->Close();

} 

