#include "TH2.h"
#include "TF1.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TLine.h"
#include "TEventList.h"

const Double_t dxbin = (0.17-0.13)/40;   // Bin-width  
const Double_t sigma = 0.0012;
TEventList *elist = 0;
Bool_t useList, fillList;
TH1  *hdmd;
TH2F *h2;

//_____________________________________________________________________
Double_t fdm5(Double_t *xx, Double_t *par)
{
   Double_t x = xx[0];
   if (x <= 0.13957) return 0;
   Double_t xp3 = (x-par[3])*(x-par[3]);
   Double_t res = dxbin*(par[0]*TMath::Power(x-0.13957, par[1])    
       + par[2] / 2.5066/par[4]*TMath::Exp(-xp3/2/par[4]/par[4]));   
   return res;
}

//_____________________________________________________________________
Double_t fdm2(Double_t *xx, Double_t *par)
{
   Double_t x = xx[0];
   if (x <= 0.13957) return 0;
   Double_t xp3 = (x-0.1454)*(x-0.1454);
   Double_t res = dxbin*(par[0]*TMath::Power(x-0.13957, 0.25)    
       + par[1] / 2.5066/sigma*TMath::Exp(-xp3/2/sigma/sigma));   
   return res;
}

int h1analysis_Begin(TTree *tree, TH1* selhtemp, TSelector *sel) {
   
// function called before starting the event loop
//  -it performs some cleanup
//  -it creates histograms
//  -it sets some initialisation for the event list

   //print the option specified in the Process function.
   TString option = sel->GetOption();
   printf("Starting h1analysis with process option: %s\n",option.Data());
     
   //some cleanup in case this function had already been executed
   //delete any previously generated histograms or functions
   gDirectory->Delete("hdmd");
   gDirectory->Delete("h2*");
   delete gROOT->GetFunction("f5");
   delete gROOT->GetFunction("f2");
   
   //create histograms
   hdmd = selhtemp;// sel->GetObject(); // new TH1F("hdmd","dm_d",40,0.13,0.17);
   hdmd->SetName("hdmd");
   hdmd->SetTitle("dm_d");
   hdmd->SetBins(40,0.13,0.17);
   hdmd->ResetBit(kCanDelete);
   
   h2   = new TH2F("h2","ptD0 vs dm_d",30,0.135,0.165,30,-3,6);

   //process cases with event list
   fillList = kFALSE;
   useList  = kFALSE;
   tree->SetEventList(0);
   delete gDirectory->GetList()->FindObject("elist");

   // case when one creates/fills the event list
   if (option.Contains("fillList")) {
      fillList = kTRUE;
      elist = new TEventList("elist","selection from Cut",5000);
   }

   // case when one uses the event list generated in a previous call
   if (option.Contains("useList")) {
      useList  = kTRUE;
      TFile f("elist.root");
      elist = (TEventList*)f.Get("elist");
      if (elist) elist->SetDirectory(0); //otherwise the file destructor will delete elist
      tree->SetEventList(elist);
   }

   return 0;
}

//_____________________________________________________________________
void h1analysis_Terminate()
{
   // function called at the end of the event loop
   
   //create the canvas for the h1analysis fit
   gStyle->SetOptFit();

   TCanvas *c1 = new TCanvas("c1","h1analysis analysis",10,10,800,600);
   c1->SetBottomMargin(0.15);
   hdmd->GetXaxis()->SetTitle("m_{K#pi#pi} - m_{K#pi}[GeV/c^{2}]");
   hdmd->GetXaxis()->SetTitleOffset(1.4);
   
   //fit histogram hdmd with function f5 using the loglikelihood option
   TF1 *f5 = new TF1("f5",fdm5,0.139,0.17,5); 
   f5->SetParameters(1000000, .25, 2000, .1454, .001);
   hdmd->Fit("f5","lr");
   
   //create the canvas for tau d0
   gStyle->SetOptFit(0);
   gStyle->SetOptStat(1100);
   TCanvas *c2 = new TCanvas("c2","tauD0",100,100,800,600);
   c2->SetGrid();
   c2->SetBottomMargin(0.15);

   // Project slices of 2-d histogram h2 along X , then fit each slice
   // with function f2 and make a histogram for each fit parameter
   // Note that the generated histograms are added to the list of objects
   // in the current directory.
   TF1 *f2 = new TF1("f2",fdm2,0.139,0.17,2);
   f2->SetParameters(10000, 10);
   h2->FitSlicesX(f2,0,0,1,"qln");
   TH1D *h2_1 = (TH1D*)gDirectory->Get("h2_1");
   h2_1->GetXaxis()->SetTitle("#tau[ps]");
   h2_1->SetMarkerStyle(21);
   h2_1->Draw();
   c2->Update();
   TLine *line = new TLine(0,0,0,c2->GetUymax());
   line->Draw();
   
   //save the event list to a Root file if one was produced
   if (fillList) {
      TFile efile("elist.root","recreate");
      elist->Write();
   }
}

#ifdef __MAKECINT__
#pragma link C++ function h1analysis_Begin;
#pragma link C++ function fdm5;
#pragma link C++ function fdm2;
#pragma link C++ function h1analysis_Terminate;
#endif
