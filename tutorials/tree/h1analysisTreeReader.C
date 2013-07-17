#include "h1analysisTreeReader.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"
#include "TTreeReaderArray.h"
#include "TH2.h"
#include "TF1.h"
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

Bool_t h1analysisTreeReader::Process(Long64_t entry){
// entry is the entry number in the current Tree
// Selection function to select D* and D0.
   myTreeReader->SetEntry(fChainOffset + entry);
   fProcessed++;
   //in case one entry list is given in input, the selection has already been done.
   if (!useList) {
      // Read only the necessary branches to select entries.
      // return as soon as a bad entry is detected
      // to read complete event, call fChain->GetTree()->GetEntry(entry)
      if (TMath::Abs(**md0_d-1.8646) >= 0.04) return kFALSE;
      if (**ptds_d <= 2.5) return kFALSE;
      if (TMath::Abs(**etads_d) >= 1.5) return kFALSE;
      (**ik)--; //original ik used f77 convention starting at 1
      (**ipi)--;
      
      
      if (nhitrp->At(**ik)* nhitrp->At(**ipi) <= 1) return kFALSE;
      
      
      if (rend->At(**ik) -rstart->At(**ik)  <= 22) return kFALSE;
      if (rend->At(**ipi)-rstart->At(**ipi) <= 22) return kFALSE;
      if (nlhk->At(**ik) <= 0.1)    return kFALSE;
      if (nlhpi->At(**ipi) <= 0.1)  return kFALSE;
      (**ipis)--; if (nlhpi->At(**ipis) <= 0.1) return kFALSE;
      if (**njets < 1)          return kFALSE;
   }
   // if option fillList, fill the entry list
   if (fillList) elist->Enter(entry);

   // to read complete event, call fChain->GetTree()->GetEntry(entry)
   // read branches not processed in ProcessCut
   //read branch holding dm_d
   //read branch holding rpd0_t
   //read branch holding ptd0_d

   //fill some histograms
   hdmd->Fill(**dm_d);
   h2->Fill(**dm_d,**rpd0_t/0.029979*1.8646/ **ptd0_d);

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

   //process cases with entry list
   if (fChain) fChain->SetEntryList(0);
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

   myTreeReader = new TTreeReader(myTree);

   nrun       = new TTreeReaderValue<Int_t>    (*myTreeReader, "nrun"    );
   nevent     = new TTreeReaderValue<Int_t>    (*myTreeReader, "nevent"  );
   nentry     = new TTreeReaderValue<Int_t>    (*myTreeReader, "nentry"  );
   trelem     = new TTreeReaderArray<UChar_t>  (*myTreeReader, "trelem"  );
   subtr      = new TTreeReaderArray<UChar_t>  (*myTreeReader, "subtr"   );
   rawtr      = new TTreeReaderArray<UChar_t>  (*myTreeReader, "rawtr"   );
   L4subtr    = new TTreeReaderArray<UChar_t>  (*myTreeReader, "L4subtr" );
   L5class    = new TTreeReaderArray<UChar_t>  (*myTreeReader, "L5class" );
   E33        = new TTreeReaderValue<Float_t>  (*myTreeReader, "E33"     );
   de33       = new TTreeReaderValue<Float_t>  (*myTreeReader, "de33"    );
   x33        = new TTreeReaderValue<Float_t>  (*myTreeReader, "x33"     );
   dx33       = new TTreeReaderValue<Float_t>  (*myTreeReader, "dx33"    );
   y33        = new TTreeReaderValue<Float_t>  (*myTreeReader, "y33"     );
   dy33       = new TTreeReaderValue<Float_t>  (*myTreeReader, "dy33"    );
   E44        = new TTreeReaderValue<Float_t>  (*myTreeReader, "E44"     );
   de44       = new TTreeReaderValue<Float_t>  (*myTreeReader, "de44"    );
   x44        = new TTreeReaderValue<Float_t>  (*myTreeReader, "x44"     );
   dx44       = new TTreeReaderValue<Float_t>  (*myTreeReader, "dx44"    );
   y44        = new TTreeReaderValue<Float_t>  (*myTreeReader, "y44"     );
   dy44       = new TTreeReaderValue<Float_t>  (*myTreeReader, "dy44"    );
   Ept        = new TTreeReaderValue<Float_t>  (*myTreeReader, "Ept"     );
   dept       = new TTreeReaderValue<Float_t>  (*myTreeReader, "dept"    );
   xpt        = new TTreeReaderValue<Float_t>  (*myTreeReader, "xpt"     );
   dxpt       = new TTreeReaderValue<Float_t>  (*myTreeReader, "dxpt"    );
   ypt        = new TTreeReaderValue<Float_t>  (*myTreeReader, "ypt"     );
   dypt       = new TTreeReaderValue<Float_t>  (*myTreeReader, "dypt"    );
   pelec      = new TTreeReaderArray<Float_t>  (*myTreeReader, "pelec"   );
   flagelec   = new TTreeReaderValue<Int_t>    (*myTreeReader, "flagelec");
   xeelec     = new TTreeReaderValue<Float_t>  (*myTreeReader, "xeelec"  );
   yeelec     = new TTreeReaderValue<Float_t>  (*myTreeReader, "yeelec"  );
   Q2eelec    = new TTreeReaderValue<Float_t>  (*myTreeReader, "Q2eelec" );
   nelec      = new TTreeReaderValue<Int_t>    (*myTreeReader, "nelec"   );
   Eelec      = new TTreeReaderArray<Float_t>  (*myTreeReader, "Eelec"   );
   thetelec   = new TTreeReaderArray<Float_t>  (*myTreeReader, "thetelec");
   phielec    = new TTreeReaderArray<Float_t>  (*myTreeReader, "phielec" );
   xelec      = new TTreeReaderArray<Float_t>  (*myTreeReader, "xelec"   );
   Q2elec     = new TTreeReaderArray<Float_t>  (*myTreeReader, "Q2elec"  );
   xsigma     = new TTreeReaderArray<Float_t>  (*myTreeReader, "xsigma"  );
   Q2sigma    = new TTreeReaderArray<Float_t>  (*myTreeReader, "Q2sigma" );
   sumc       = new TTreeReaderArray<Float_t>  (*myTreeReader, "sumc"    );
   sumetc     = new TTreeReaderValue<Float_t>  (*myTreeReader, "sumetc"  );
   yjbc       = new TTreeReaderValue<Float_t>  (*myTreeReader, "yjbc"    );
   Q2jbc      = new TTreeReaderValue<Float_t>  (*myTreeReader, "Q2jbc"   );
   sumct      = new TTreeReaderArray<Float_t>  (*myTreeReader, "sumct"   );
   sumetct    = new TTreeReaderValue<Float_t>  (*myTreeReader, "sumetct" );
   yjbct      = new TTreeReaderValue<Float_t>  (*myTreeReader, "yjbct"   );
   Q2jbct     = new TTreeReaderValue<Float_t>  (*myTreeReader, "Q2jbct"  );
   Ebeamel    = new TTreeReaderValue<Float_t>  (*myTreeReader, "Ebeamel" );
   Ebeampr    = new TTreeReaderValue<Float_t>  (*myTreeReader, "Ebeampr" );
   pvtx_d     = new TTreeReaderArray<Float_t>  (*myTreeReader, "pvtx_d"  );
   cpvtx_d    = new TTreeReaderArray<Float_t>  (*myTreeReader, "cpvtx_d" );
   pvtx_t     = new TTreeReaderArray<Float_t>  (*myTreeReader, "pvtx_t"  );
   cpvtx_t    = new TTreeReaderArray<Float_t>  (*myTreeReader, "cpvtx_t" );
   ntrkxy_t   = new TTreeReaderValue<Int_t>    (*myTreeReader, "ntrkxy_t");
   prbxy_t    = new TTreeReaderValue<Float_t>  (*myTreeReader, "prbxy_t" );
   ntrkz_t    = new TTreeReaderValue<Int_t>    (*myTreeReader, "ntrkz_t" );
   prbz_t     = new TTreeReaderValue<Float_t>  (*myTreeReader, "prbz_t"  );
   nds        = new TTreeReaderValue<Int_t>    (*myTreeReader, "nds"     );
   rankds     = new TTreeReaderValue<Int_t>    (*myTreeReader, "rankds"  );
   qds        = new TTreeReaderValue<Int_t>    (*myTreeReader, "qds"     );
   pds_d      = new TTreeReaderArray<Float_t>  (*myTreeReader, "pds_d"   );
   ptds_d     = new TTreeReaderValue<Float_t>  (*myTreeReader, "ptds_d"  );
   etads_d    = new TTreeReaderValue<Float_t>  (*myTreeReader, "etads_d" );
   dm_d       = new TTreeReaderValue<Float_t>  (*myTreeReader, "dm_d"    );
   ddm_d      = new TTreeReaderValue<Float_t>  (*myTreeReader, "ddm_d"   );
   pds_t      = new TTreeReaderArray<Float_t>  (*myTreeReader, "pds_t"   );
   dm_t       = new TTreeReaderValue<Float_t>  (*myTreeReader, "dm_t"    );
   ddm_t      = new TTreeReaderValue<Float_t>  (*myTreeReader, "ddm_t"   );
   ik         = new TTreeReaderValue<Int_t>    (*myTreeReader, "ik"      );
   ipi        = new TTreeReaderValue<Int_t>    (*myTreeReader, "ipi"     );
   ipis       = new TTreeReaderValue<Int_t>    (*myTreeReader, "ipis"    );
   pd0_d      = new TTreeReaderArray<Float_t>  (*myTreeReader, "pd0_d"   );
   ptd0_d     = new TTreeReaderValue<Float_t>  (*myTreeReader, "ptd0_d"  );
   etad0_d    = new TTreeReaderValue<Float_t>  (*myTreeReader, "etad0_d" );
   md0_d      = new TTreeReaderValue<Float_t>  (*myTreeReader, "md0_d"   );
   dmd0_d     = new TTreeReaderValue<Float_t>  (*myTreeReader, "dmd0_d"  );
   pd0_t      = new TTreeReaderArray<Float_t>  (*myTreeReader, "pd0_t"   );
   md0_t      = new TTreeReaderValue<Float_t>  (*myTreeReader, "md0_t"   );
   dmd0_t     = new TTreeReaderValue<Float_t>  (*myTreeReader, "dmd0_t"  );
   pk_r       = new TTreeReaderArray<Float_t>  (*myTreeReader, "pk_r"    );
   ppi_r      = new TTreeReaderArray<Float_t>  (*myTreeReader, "ppi_r"   );
   pd0_r      = new TTreeReaderArray<Float_t>  (*myTreeReader, "pd0_r"   );
   md0_r      = new TTreeReaderValue<Float_t>  (*myTreeReader, "md0_r"   );
   Vtxd0_r    = new TTreeReaderArray<Float_t>  (*myTreeReader, "Vtxd0_r" );
   cvtxd0_r   = new TTreeReaderArray<Float_t>  (*myTreeReader, "cvtxd0_r");
   dxy_r      = new TTreeReaderValue<Float_t>  (*myTreeReader, "dxy_r"   );
   dz_r       = new TTreeReaderValue<Float_t>  (*myTreeReader, "dz_r"    );
   psi_r      = new TTreeReaderValue<Float_t>  (*myTreeReader, "psi_r"   );
   rd0_d      = new TTreeReaderValue<Float_t>  (*myTreeReader, "rd0_d"   );
   drd0_d     = new TTreeReaderValue<Float_t>  (*myTreeReader, "drd0_d"  );
   rpd0_d     = new TTreeReaderValue<Float_t>  (*myTreeReader, "rpd0_d"  );
   drpd0_d    = new TTreeReaderValue<Float_t>  (*myTreeReader, "drpd0_d" );
   rd0_t      = new TTreeReaderValue<Float_t>  (*myTreeReader, "rd0_t"   );
   drd0_t     = new TTreeReaderValue<Float_t>  (*myTreeReader, "drd0_t"  );
   rpd0_t     = new TTreeReaderValue<Float_t>  (*myTreeReader, "rpd0_t"  );
   drpd0_t    = new TTreeReaderValue<Float_t>  (*myTreeReader, "drpd0_t" );
   rd0_dt     = new TTreeReaderValue<Float_t>  (*myTreeReader, "rd0_dt"  );
   drd0_dt    = new TTreeReaderValue<Float_t>  (*myTreeReader, "drd0_dt" );
   prbr_dt    = new TTreeReaderValue<Float_t>  (*myTreeReader, "prbr_dt" );
   prbz_dt    = new TTreeReaderValue<Float_t>  (*myTreeReader, "prbz_dt" );
   rd0_tt     = new TTreeReaderValue<Float_t>  (*myTreeReader, "rd0_tt"  );
   drd0_tt    = new TTreeReaderValue<Float_t>  (*myTreeReader, "drd0_tt" );
   prbr_tt    = new TTreeReaderValue<Float_t>  (*myTreeReader, "prbr_tt" );
   prbz_tt    = new TTreeReaderValue<Float_t>  (*myTreeReader, "prbz_tt" );
   ijetd0     = new TTreeReaderValue<Int_t>    (*myTreeReader, "ijetd0"  );
   ptr3d0_j   = new TTreeReaderValue<Float_t>  (*myTreeReader, "ptr3d0_j");
   ptr2d0_j   = new TTreeReaderValue<Float_t>  (*myTreeReader, "ptr2d0_j");
   ptr3d0_3   = new TTreeReaderValue<Float_t>  (*myTreeReader, "ptr3d0_3");
   ptr2d0_3   = new TTreeReaderValue<Float_t>  (*myTreeReader, "ptr2d0_3");
   ptr2d0_2   = new TTreeReaderValue<Float_t>  (*myTreeReader, "ptr2d0_2");
   Mimpds_r   = new TTreeReaderValue<Float_t>  (*myTreeReader, "Mimpds_r");
   Mimpbk_r   = new TTreeReaderValue<Float_t>  (*myTreeReader, "Mimpbk_r");
   ntracks    = new TTreeReaderValue<Int_t>    (*myTreeReader, "ntracks" );
   pt         = new TTreeReaderArray<Float_t>  (*myTreeReader, "pt"      );
   kappa      = new TTreeReaderArray<Float_t>  (*myTreeReader, "kappa"   );
   phi        = new TTreeReaderArray<Float_t>  (*myTreeReader, "phi"     );
   theta      = new TTreeReaderArray<Float_t>  (*myTreeReader, "theta"   );
   dca        = new TTreeReaderArray<Float_t>  (*myTreeReader, "dca"     );
   z0         = new TTreeReaderArray<Float_t>  (*myTreeReader, "z0"      );
   covar      = new TTreeReaderArray<Float_t>  (*myTreeReader, "covar"   );
   nhitrp     = new TTreeReaderArray<Int_t>    (*myTreeReader, "nhitrp"  );
   prbrp      = new TTreeReaderArray<Float_t>  (*myTreeReader, "prbrp"   );
   nhitz      = new TTreeReaderArray<Int_t>    (*myTreeReader, "nhitz"   );
   prbz       = new TTreeReaderArray<Float_t>  (*myTreeReader, "prbz"    );
   rstart     = new TTreeReaderArray<Float_t>  (*myTreeReader, "rstart"  );
   rend       = new TTreeReaderArray<Float_t>  (*myTreeReader, "rend"    );
   lhk        = new TTreeReaderArray<Float_t>  (*myTreeReader, "lhk"     );
   lhpi       = new TTreeReaderArray<Float_t>  (*myTreeReader, "lhpi"    );
   nlhk       = new TTreeReaderArray<Float_t>  (*myTreeReader, "nlhk"    );
   nlhpi      = new TTreeReaderArray<Float_t>  (*myTreeReader, "nlhpi"   );
   dca_d      = new TTreeReaderArray<Float_t>  (*myTreeReader, "dca_d"   );
   ddca_d     = new TTreeReaderArray<Float_t>  (*myTreeReader, "ddca_d"  );
   dca_t      = new TTreeReaderArray<Float_t>  (*myTreeReader, "dca_t"   );
   ddca_t     = new TTreeReaderArray<Float_t>  (*myTreeReader, "ddca_t"  );
   muqual     = new TTreeReaderArray<Int_t>    (*myTreeReader, "muqual"  );
   imu        = new TTreeReaderValue<Int_t>    (*myTreeReader, "imu"     );
   imufe      = new TTreeReaderValue<Int_t>    (*myTreeReader, "imufe"   );
   njets      = new TTreeReaderValue<Int_t>    (*myTreeReader, "njets"   );
   E_j        = new TTreeReaderArray<Float_t>  (*myTreeReader, "E_j"     );
   pt_j       = new TTreeReaderArray<Float_t>  (*myTreeReader, "pt_j"    );
   theta_j    = new TTreeReaderArray<Float_t>  (*myTreeReader, "theta_j" );
   eta_j      = new TTreeReaderArray<Float_t>  (*myTreeReader, "eta_j"   );
   phi_j      = new TTreeReaderArray<Float_t>  (*myTreeReader, "phi_j"   );
   m_j        = new TTreeReaderArray<Float_t>  (*myTreeReader, "m_j"     );
   thrust     = new TTreeReaderValue<Float_t>  (*myTreeReader, "thrust"  );
   pthrust    = new TTreeReaderArray<Float_t>  (*myTreeReader, "pthrust" );
   thrust2    = new TTreeReaderValue<Float_t>  (*myTreeReader, "thrust2" );
   pthrust2   = new TTreeReaderArray<Float_t>  (*myTreeReader, "pthrust2");
   spher      = new TTreeReaderValue<Float_t>  (*myTreeReader, "spher"   );
   aplan      = new TTreeReaderValue<Float_t>  (*myTreeReader, "aplan"   );
   plan       = new TTreeReaderValue<Float_t>  (*myTreeReader, "plan"    );
   nnout      = new TTreeReaderArray<Float_t>  (*myTreeReader, "nnout"   );

   fChainOffset = 0;

   //print the option specified in the Process function.
   TString option = GetOption();
   Info("SlaveBegin",
        "starting h1analysis with process option: %s (tree: %p)", option.Data(), myTree);

   //create histograms
   hdmd = new TH1F("hdmd","dm_d",40,0.13,0.17);
   h2   = new TH2F("h2","ptD0 vs dm_d",30,0.135,0.165,30,-3,6);

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

   //fit histogram hdmd with function f5 using the loglikelihood option
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
   delete myTreeReader;

   delete  nrun;
   delete  nevent;
   delete  nentry;
   delete  trelem;
   delete  subtr;
   delete  rawtr;
   delete  L4subtr;
   delete  L5class;
   delete  E33;
   delete  de33;
   delete  x33;
   delete  dx33;
   delete  y33;
   delete  dy33;
   delete  E44;
   delete  de44;
   delete  x44;
   delete  dx44;
   delete  y44;
   delete  dy44;
   delete  Ept;
   delete  dept;
   delete  xpt;
   delete  dxpt;
   delete  ypt;
   delete  dypt;
   delete  pelec;
   delete  flagelec;
   delete  xeelec;
   delete  yeelec;
   delete  Q2eelec;
   delete  nelec;
   delete  Eelec;
   delete  thetelec;
   delete  phielec;
   delete  xelec;
   delete  Q2elec;
   delete  xsigma;
   delete  Q2sigma;
   delete  sumc;
   delete  sumetc;
   delete  yjbc;
   delete  Q2jbc;
   delete  sumct;
   delete  sumetct;
   delete  yjbct;
   delete  Q2jbct;
   delete  Ebeamel;
   delete  Ebeampr;
   delete  pvtx_d;
   delete  cpvtx_d;
   delete  pvtx_t;
   delete  cpvtx_t;
   delete  ntrkxy_t;
   delete  prbxy_t;
   delete  ntrkz_t;
   delete  prbz_t;
   delete  nds;
   delete  rankds;
   delete  qds;
   delete  pds_d;
   delete  ptds_d;
   delete  etads_d;
   delete  dm_d;
   delete  ddm_d;
   delete  pds_t;
   delete  dm_t;
   delete  ddm_t;
   delete  ik;
   delete  ipi;
   delete  ipis;
   delete  pd0_d;
   delete  ptd0_d;
   delete  etad0_d;
   delete  md0_d;
   delete  dmd0_d;
   delete  pd0_t;
   delete  md0_t;
   delete  dmd0_t;
   delete  pk_r;
   delete  ppi_r;
   delete  pd0_r;
   delete  md0_r;
   delete  Vtxd0_r;
   delete  cvtxd0_r;
   delete  dxy_r;
   delete  dz_r;
   delete  psi_r;
   delete  rd0_d;
   delete  drd0_d;
   delete  rpd0_d;
   delete  drpd0_d;
   delete  rd0_t;
   delete  drd0_t;
   delete  rpd0_t;
   delete  drpd0_t;
   delete  rd0_dt;
   delete  drd0_dt;
   delete  prbr_dt;
   delete  prbz_dt;
   delete  rd0_tt;
   delete  drd0_tt;
   delete  prbr_tt;
   delete  prbz_tt;
   delete  ijetd0;
   delete  ptr3d0_j;
   delete  ptr2d0_j;
   delete  ptr3d0_3;
   delete  ptr2d0_3;
   delete  ptr2d0_2;
   delete  Mimpds_r;
   delete  Mimpbk_r;
   delete  ntracks;
   delete  pt;
   delete  kappa;
   delete  phi;
   delete  theta;
   delete  dca;
   delete  z0;
   delete  covar;
   delete  nhitrp;
   delete  prbrp;
   delete  nhitz;
   delete  prbz;
   delete  rstart;
   delete  rend;
   delete  lhk;
   delete  lhpi;
   delete  nlhk;
   delete  nlhpi;
   delete  dca_d;
   delete  ddca_d;
   delete  dca_t;
   delete  ddca_t;
   delete  muqual;
   delete  imu;
   delete  imufe;
   delete  njets;
   delete  E_j;
   delete  pt_j;
   delete  theta_j;
   delete  eta_j;
   delete  phi_j;
   delete  m_j;
   delete  thrust;
   delete  pthrust;
   delete  thrust2;
   delete  pthrust2;
   delete  spher;
   delete  aplan;
   delete  plan;
   delete  nnout;
}

Bool_t h1analysisTreeReader::Notify() {
//   called when loading a new file
//   get branch pointers

   Info("Notify","processing file: %s",myTreeReader->GetTree()->GetCurrentFile()->GetName());
   fChainOffset = myTreeReader->GetTree()->GetChainOffset();

   if (elist && fChain) {
      if (fillList) {
         elist->SetTree(fChain);
      } else if (useList) {
         fChain->SetEntryList(elist);
      }
   }
   return kTRUE;
}
