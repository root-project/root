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
#include "TMVA/SeparationBase.h"
#include "TMVA/GiniIndex.h"
#include "TMVA/MisClassificationErrror.h"
#endif

using namespace TMVA;


void plot(TH2D *sig, TH2D *bkg, TH2F *MVA, TString v0="var0", TString v1="var1",Float_t mvaCut){

   TCanvas *c = new TCanvas(Form("DecisionBoundary%s",MVA->GetTitle()),MVA->GetTitle(),800,800);

   cout <<  "MVACut = "<<mvaCut << endl;
   gStyle->SetPalette(1);
   MVA->SetXTitle(v0);
   MVA->SetYTitle(v1);
   MVA->SetStats(0);
   Double_t contours[1];
   contours[0]=mvaCut;
   MVA->SetLineWidth(7);
   MVA->SetLineStyle(1);
   MVA->SetMarkerColor(1);
   MVA->SetLineColor(1);
   MVA->SetContour(1, contours);
   sig->SetMarkerColor(4);
   bkg->SetMarkerColor(2);
   sig->SetMarkerStyle(20);
   bkg->SetMarkerStyle(20);
   sig->SetMarkerSize(.2);
   bkg->SetMarkerSize(.2);
   sig->Draw();
   bkg->Draw("same");
   MVA->Draw("CONT2 same");
}


void PlotDecisionBoundary( TString weightFile = "weights/TMVAClassification_BDT.weights.xml",TString v0="var0", TString v1="var1", TString dataFileName = "/home/hvoss/TMVA/TMVA_data/data/data_circ.root") 
{   
   //---------------------------------------------------------------
   // default MVA methods to be trained + tested

   // this loads the library
   TMVA::Tools::Instance();

   std::cout << std::endl;
   std::cout << "==> Start TMVAClassificationApplication" << std::endl;


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
   // book the MVA method
   //
   reader->BookMVA( "M1", weightFile ); 
   
   TFile *f = new TFile(dataFileName);
   TTree *signal     = (TTree*)f->Get("TreeS");
   TTree *background = (TTree*)f->Get("TreeB");


//Declaration of leaves types
   Float_t         svar0;
   Float_t         svar1;
   Float_t         bvar0;
   Float_t         bvar1;
   Float_t         sWeight=1.0; // just in case you have weight defined, also set these branchaddresses
   Float_t         bWeight=1.0*signal->GetEntries()/background->GetEntries(); // just in case you have weight defined, also set these branchaddresses

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
 
   xmax = TMath::Max(xmax,background->GetMaximum(v0.Data()));  
   xmin = TMath::Min(xmin,background->GetMinimum(v0.Data()));
   ymax = TMath::Max(ymax,background->GetMaximum(v1.Data()));
   ymin = TMath::Min(ymin,background->GetMinimum(v1.Data()));


   TH2D *hs=new TH2D("hs","",nbin,xmin,xmax,nbin,ymin,ymax);   
   TH2D *hb=new TH2D("hb","",nbin,xmin,xmax,nbin,ymin,ymax);   
   hs->SetXTitle(v0);
   hs->SetYTitle(v1);
   hb->SetXTitle(v0);
   hb->SetYTitle(v1);
   hs->SetMarkerColor(4);
   hb->SetMarkerColor(2);


   TH2F * hist = new TH2F( "MVA",    "MVA",    nbin,xmin,xmax,nbin,ymin,ymax);

   // Prepare input tree (this must be replaced by your data source)
   // in this example, there is a toy tree with signal and one with background events
   // we'll later on use only the "signal" events for the test in this example.

   Float_t MinMVA=10000, MaxMVA=-100000;
   for (Int_t ibin=1; ibin<nbin+1; ibin++){
      for (Int_t jbin=1; jbin<nbin+1; jbin++){
         var0 = hs->GetXaxis()->GetBinCenter(ibin);
         var1 = hs->GetYaxis()->GetBinCenter(jbin);
         Float_t mvaVal=reader->EvaluateMVA( "M1" ) ;
         if (MinMVA>mvaVal) MinMVA=mvaVal;
         if (MaxMVA<mvaVal) MaxMVA=mvaVal;
         hist->SetBinContent(ibin,jbin, mvaVal);
      }
   }

   // creating a fine histograms containing the error rate
   const Int_t nValBins=100;
   Double_t    sum = 0.;

   TH1F *mvaS= new TH1F("mvaS","",nValBins,MinMVA,MaxMVA);
   TH1F *mvaB= new TH1F("mvaB","",nValBins,MinMVA,MaxMVA);
   TH1F *mvaSC= new TH1F("mvaSC","",nValBins,MinMVA,MaxMVA);
   TH1F *mvaBC= new TH1F("mvaBC","",nValBins,MinMVA,MaxMVA);

   Long64_t nentries;
   nentries = TreeS->GetEntries();
   for (Long64_t is=0; is<nentries;is++) {
      signal->GetEntry(is);
      sum +=sWeight;
      var0 = svar0;
      var1 = svar1;
      Float_t mvaVal=reader->EvaluateMVA( "M1" ) ;
      hs->Fill(svar0,svar1);
      mvaS->Fill(mvaVal,sWeight);
   }
   nentries = TreeB->GetEntries();
   for (Long64_t ib=0; ib<nentries;ib++) {
      background->GetEntry(ib);
      sum +=bWeight;
      var0 = bvar0;
      var1 = bvar1;
      Float_t mvaVal=reader->EvaluateMVA( "M1" ) ;
      hb->Fill(bvar0,bvar1);
      mvaB->Fill(mvaVal,bWeight);
   }

   //SeparationBase *sepGain = new MisClassificationError();
   //SeparationBase *sepGain = new GiniIndex();
   SeparationBase *sepGain = new CrossEntropy();

   Double_t sTot = mvaS->GetSum();
   Double_t bTot = mvaB->GetSum();

   mvaSC->SetBinContent(1,mvaS->GetBinContent(1));
   mvaBC->SetBinContent(1,mvaB->GetBinContent(1));
   Double_t sSel=mvaSC->GetBinContent(1);
   Double_t bSel=mvaBC->GetBinContent(1);
   Double_t separationGain=sepGain->GetSeparationGain(sSel,bSel,sTot,bTot);
   Double_t mvaCut=mvaSC->GetBinCenter(1);
   Double_t mvaCutOrientation=1; // 1 if mva > mvaCut --> Signal and -1 if mva < mvaCut (i.e. mva*-1 > mvaCut*-1) --> Signal
   for (Int_t ibin=2;ibin<nValBins;ibin++){ 
      mvaSC->SetBinContent(ibin,mvaS->GetBinContent(ibin)+mvaSC->GetBinContent(ibin-1));
      mvaBC->SetBinContent(ibin,mvaB->GetBinContent(ibin)+mvaBC->GetBinContent(ibin-1));
    
      sSel=mvaSC->GetBinContent(ibin);
      bSel=mvaBC->GetBinContent(ibin);

      if (separationGain < sepGain->GetSeparationGain(sSel,bSel,sTot,bTot) && mvaSC->GetBinCenter(ibin)<0){
         separationGain = sepGain->GetSeparationGain(sSel,bSel,sTot,bTot);
         mvaCut=mvaSC->GetBinCenter(ibin);
         if (sSel/bSel > (sTot-sSel)/(bTot-bSel)) mvaCutOrientation=-1;
         else                                     mvaCutOrientation=1;
     }
   }
   

   cout << "Min="<<MinMVA << " Max=" << MaxMVA 
        << " sTot=" << sTot
        << " bTot=" << bTot
        << " sepGain="<<separationGain
        << " cut=" << mvaCut
        << " cutOrientation="<<mvaCutOrientation
        << endl;


   delete reader;

   gStyle->SetPalette(1);

  
   plot(hs,hb,hist     ,v0,v1,mvaCut);


   TCanvas *cm=new TCanvas ("cm","",900,1200);
   cm->cd();
   cm->Divide(1,2);
   cm->cd(1);
   mvaS->SetLineColor(4);
   mvaB->SetLineColor(2);
   mvaS->Draw();
   mvaB->Draw("same");

   cm->cd(2);
   mvaSC->SetLineColor(4);
   mvaBC->SetLineColor(2);
   mvaBC->Draw();
   mvaSC->Draw("same");

   // TH1F *add=(TH1F*)mvaBC->Clone("add");
   // add->Add(mvaSC);

   // add->Draw();

   // errh->Draw("same");

   //
   // write histograms
   //
   TFile *target  = new TFile( "TMVAPlotDecisionBoundary.root","RECREATE" );

   hs->Write();
   hb->Write();

   hist->Write();

   target->Close();

} 

