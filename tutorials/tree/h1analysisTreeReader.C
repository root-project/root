#include "h1analysisTreeReader.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TPaveStats.h"
#include "TLine.h"
#include "TMath.h"
#include "TFile.h"
#include "TROOT.h"


const Double_t dxbin = (0.17-0.13)/40;   // Bin-width
const Double_t sigma = 0.0012;

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
//_____________________________________________________________________
Bool_t h1analysisTreeReader::Process(Long64_t entry){
// entry is the entry number in the current Tree
// Selection function to select D* and D0.
   myTreeReader.SetLocalEntry(entry);
   fProcessed++;
   //in case one entry list is given in input, the selection has already been done.
   if (!useList) {
      // Return as soon as a bad entry is detected
      if (TMath::Abs(*fMd0_d-1.8646) >= 0.04) return kFALSE;
      if (*fPtds_d <= 2.5) return kFALSE;
      if (TMath::Abs(*fEtads_d) >= 1.5) return kFALSE;
      (*fIk)--; //original fIk used f77 convention starting at 1
      (*fIpi)--;


      if (fNhitrp.At(*fIk)* fNhitrp.At(*fIpi) <= 1) return kFALSE;


      if (fRend.At(*fIk) -fRstart.At(*fIk)  <= 22) return kFALSE;
      if (fRend.At(*fIpi)-fRstart.At(*fIpi) <= 22) return kFALSE;
      if (fNlhk.At(*fIk) <= 0.1)    return kFALSE;
      if (fNlhpi.At(*fIpi) <= 0.1)  return kFALSE;
      (*fIpis)--; if (fNlhpi.At(*fIpis) <= 0.1) return kFALSE;
      if (*fNjets < 1)          return kFALSE;
   }
   // if option fillList, fill the entry list
   if (fillList) elist->Enter(entry);

   //fill some histograms
   hdmd->Fill(*fDm_d);
   h2->Fill(*fDm_d,*fRpd0_t/0.029979*1.8646/ *fPtd0_d);

   return kTRUE;
}

void h1analysisTreeReader::Begin(TTree* /*myTree*/) {
// function called before starting the event loop
//  -it performs some cleanup
//  -it creates histograms
//  -it sets some initialisation for the entry list

   Reset();

   //print the option specified in the Process function.
   TString option = GetOption();
   Info("Begin", "starting h1analysis with process option: %s", option.Data());

   delete gDirectory->GetList()->FindObject("elist");

   // case when one creates/fills the entry list
   if (option.Contains("fillList")) {
      fillList = kTRUE;
      elist = new TEntryList("elist", "H1 selection from Cut");
      // Add to the input list for processing in PROOF, if needed
      if (fInput) {
         fInput->Add(new TNamed("fillList",""));
         // We send a clone to avoid double deletes when importing the result
         fInput->Add(elist);
         // This is needed to avoid warnings from output-to-members mapping
         elist = 0;
      }
   }
   if (fillList) Info("Begin", "creating an entry-list");
   // case when one uses the entry list generated in a previous call
   if (option.Contains("useList")) {
      useList  = kTRUE;
      if (fInput) {
         // Option "useList" not supported in PROOF directly
         Warning("Begin", "option 'useList' not supported in PROOF - ignoring");
         Warning("Begin", "the entry list must be set on the chain *before* calling Process");
      } else {
         TFile f("elist.root");
         elist = (TEntryList*)f.Get("elist");
         if (elist) elist->SetDirectory(0); //otherwise the file destructor will delete elist
      }
   }
}

void h1analysisTreeReader::SlaveBegin(TTree *myTree){

// function called before starting the event loop
//  -it performs some cleanup
//  -it creates histograms
//  -it sets some initialisation for the entry list

   Init(myTree);

   //print the option specified in the Process function.
   TString option = GetOption();
   Info("SlaveBegin",
        "starting h1analysis with process option: %s (tree: %p)", option.Data(), myTree);

   //create histograms
   hdmd = new TH1F("hdmd","Dm_d",40,0.13,0.17);
   h2   = new TH2F("h2","ptD0 vs Dm_d",30,0.135,0.165,30,-3,6);

   fOutput->Add(hdmd);
   fOutput->Add(h2);

   // Entry list stuff (re-parse option because on PROOF only SlaveBegin is called)
   if (option.Contains("fillList")) {
      fillList = kTRUE;
      // Get the list
      if (fInput) {
         if ((elist = (TEntryList *) fInput->FindObject("elist")))
            // Need to clone to avoid problems when destroying the selector
            elist = (TEntryList *) elist->Clone();
         if (elist)
            fOutput->Add(elist);
         else
            fillList = kFALSE;
      }
   }
   if (fillList) Info("SlaveBegin", "creating an entry-list");
}

void h1analysisTreeReader::Terminate() {
   // function called at the end of the event loop

   hdmd = dynamic_cast<TH1F*>(fOutput->FindObject("hdmd"));
   h2 = dynamic_cast<TH2F*>(fOutput->FindObject("h2"));

   if (hdmd == 0 || h2 == 0) {
      Error("Terminate", "hdmd = %p , h2 = %p", hdmd, h2);
      return;
   }

   //create the canvas for the h1analysis fit
   gStyle->SetOptFit();
   TCanvas *c1 = new TCanvas("c1","h1analysis analysis",10,10,800,600);
   c1->SetBottomMargin(0.15);
   hdmd->GetXaxis()->SetTitle("m_{K#pi#pi} - m_{K#pi}[GeV/c^{2}]");
   hdmd->GetXaxis()->SetTitleOffset(1.4);

   //fit histogram hdmd with function f5 using the loglfIkelihood option
   if (gROOT->GetListOfFunctions()->FindObject("f5"))
      delete gROOT->GetFunction("f5");
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
   if (gROOT->GetListOfFunctions()->FindObject("f2"))
      delete gROOT->GetFunction("f2");
   TF1 *f2 = new TF1("f2",fdm2,0.139,0.17,2);
   f2->SetParameters(10000, 10);
   h2->FitSlicesX(f2,0,-1,1,"qln");
   TH1D *h2_1 = (TH1D*)gDirectory->Get("h2_1");
   h2_1->GetXaxis()->SetTitle("#tau[ps]");
   h2_1->SetMarkerStyle(21);
   h2_1->Draw();
   c2->Update();
   TLine *line = new TLine(0,0,0,c2->GetUymax());
   line->Draw();

   // Have the number of entries on the first histogram (to cross check when running
   // with entry lists)
   TPaveStats *psdmd = (TPaveStats *)hdmd->GetListOfFunctions()->FindObject("stats");
   psdmd->SetOptStat(1110);
   c1->Modified();

   //save the entry list to a Root file if one was produced
   if (fillList) {
      if (!elist)
         elist = dynamic_cast<TEntryList*>(fOutput->FindObject("elist"));
      if (elist) {
         Printf("Entry list 'elist' created:");
         elist->Print();
         TFile efile("elist.root","recreate");
         elist->Write();
      } else {
         Error("Terminate", "entry list requested but not found in output");
      }
   }
   // Notify the amount of processed events
   if (!fInput) Info("Terminate", "processed %lld events", fProcessed);
}

void h1analysisTreeReader::SlaveTerminate(){

}

Bool_t h1analysisTreeReader::Notify() {
//   called when loading a new file
//   get branch pointers

   Info("Notify","processing file: %s",myTreeReader.GetTree()->GetCurrentFile()->GetName());

   if (elist && myTreeReader.GetTree()) {
      if (fillList) {
         elist->SetTree(myTreeReader.GetTree());
      } else if (useList) {
         myTreeReader.GetTree()->SetEntryList(elist);
      }
   }
   return kTRUE;
}
