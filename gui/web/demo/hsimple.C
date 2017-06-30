/// \file
/// \ingroup Tutorials
/// \notebook
///  This program creates :
///    - a one dimensional histogram
///    - a two dimensional histogram
///    - a profile histogram
///    - a memory-resident ntuple
///
///  These objects are filled with some random numbers and saved on a file.
///  If get=1 the macro returns a pointer to the TFile of "hsimple.root"
///          if this file exists, otherwise it is created.
///  The file "hsimple.root" is created in $ROOTSYS/tutorials if the caller has
///  write access to this directory, otherwise the file is created in $PWD
///
/// \macro_image
/// \macro_output
/// \macro_code
///
/// \author Rene Brun

#include <TFile.h>
#include <TNtuple.h>
#include <TH2.h>
#include <TProfile.h>
#include <TCanvas.h>
#include <TFrame.h>
#include <TROOT.h>
#include <TMath.h>
#include <TSystem.h>
#include <TRandom3.h>
#include <TBenchmark.h>
#include <TInterpreter.h>
#include <TBox.h>
#include <TLine.h>
#include <TMarker.h>
#include <TText.h>
#include <TExec.h>


void DrawTextOverHisto()
{
   TH1 *hpx = (TH1 *)gPad->GetListOfPrimitives()->FindObject("hpx");
   if (!hpx) return;

   Int_t bin = hpx->GetMaximumBin();
   Double_t y = hpx->GetBinContent(bin);

   // latex does not support NDC, why?
   // TLatex *l = new TLatex(0.5,y*0.9,Form("%4.2f",y));

   TText *l = new TText(0.5,0.8,Form("%4.2f",y));
   l->SetNDC(kTRUE); // normal coordinates not yet working
   l->SetTextSize(0.025);
   l->SetTextFont(42);
   l->SetTextAlign(21);

   l->Paint();
   delete l;
}

TFile *hsimple(Int_t getFile=0)
{
   TString filename = "hsimple.root";
   TString dir = gROOT->GetTutorialsDir();
   dir.ReplaceAll("/./","/");
   TFile *hfile = 0;
   if (getFile) {
      // if the argument getFile =1 return the file "hsimple.root"
      // if the file does not exist, it is created
      TString fullPath = dir+"hsimple.root";
      if (!gSystem->AccessPathName(fullPath,kFileExists)) {
         hfile = TFile::Open(fullPath); //in $ROOTSYS/tutorials
         if (hfile) return hfile;
      }
      //otherwise try $PWD/hsimple.root
      if (!gSystem->AccessPathName("hsimple.root",kFileExists)) {
         hfile = TFile::Open("hsimple.root"); //in current dir
         if (hfile) return hfile;
      }
   }
   //no hsimple.root file found. Must generate it !
   //generate hsimple.root in current directory if we have write access
   if (gSystem->AccessPathName(".",kWritePermission)) {
      printf("you must run the script in a directory with write access\n");
      return 0;
   }
   hfile = (TFile*)gROOT->FindObject(filename); if (hfile) hfile->Close();
   hfile = new TFile(filename,"RECREATE","Demo ROOT file with histograms");

   // Create some histograms, a profile histogram and an ntuple
   TH1F *hpx = new TH1F("hpx","This is the px distribution",100,-4,4);
   hpx->SetFillColor(48);
   TH2F *hpxpy = new TH2F("hpxpy","py vs px",40,-4,4,40,-4,4);
   TProfile *hprof = new TProfile("hprof","Profile of pz versus px",100,-4,4,0,20);
   //TNtuple *ntuple = new TNtuple("ntuple","Demo ntuple","px:py:pz:random:i");

   gBenchmark->Start("hsimple");

   printf("Before canvas %d\n", gROOT->IsLineProcessing());

   // Create a new canvas.
   TCanvas *c1 = new TCanvas("c1","Dynamic Filling Example",200,10,700,500);
   if (c1) {
      c1->SetFillColor(42);
      c1->GetFrame()->SetFillColor(21);
      c1->GetFrame()->SetBorderSize(6);
      c1->GetFrame()->SetBorderMode(-1);
   }

   hpx->Draw();

   TBox *box = new TBox(0,0,1,1);
   box->SetFillColor(4);
   box->Draw();

   TLine* l1 = new TLine(0.2, 0.2, 0.5, 0.5);
   l1->SetLineColor(3);
   l1->SetNDC(kTRUE);
   l1->Draw();

   TMarker* m1 = new TMarker(0.5, 0.5, 12);
   m1->SetMarkerColor(5);
   // m1->SetNDC(kTRUE);
   m1->Draw();

   TText *t1 = new TText(0.5, 0.5, "text");
   t1->SetTextColor(7);
   t1->SetNDC(kTRUE);
   t1->Draw();

   TExec *exec1 = new TExec("exec1","DrawTextOverHisto()");
   exec1->Draw();


   // normal coordinates not yet working
   //TLine* l2 = new TLine(0, 0, 1, 1);
   //l2->SetLineColor(4);
   //l2->SetNDC(kFALSE);
   //l2->Draw();

   printf("First time islineproc %d\n", gROOT->IsLineProcessing());

   // Fill histograms randomly
   TRandom3 randomNum;
   Float_t px, py, pz;
   const Int_t kUPDATE = 1000;
   for (Int_t i = 0; i < 2e9; i++) {
      randomNum.Rannor(px,py);
      pz = px*px + py*py;
      Float_t rnd = randomNum.Rndm();
      hpx->Fill(px);
      hpxpy->Fill(px,py);
      hprof->Fill(px,pz);
      //ntuple->Fill(px,py,pz,rnd,i);
      if (i && (i%kUPDATE) == 0) {
         if (c1) {

            Double_t xx = 2*TMath::Cos(i/1.7e6);
            Double_t yy = (0.6 + 0.2*TMath::Sin(i/1e6)) * hpx->GetBinContent(hpx->GetMaximumBin());
            box->SetX1(xx); box->SetX2(xx+1);
            box->SetY1(yy); box->SetY2(yy*1.2);
            box->SetFillColor((Int_t)( 2 + TMath::Abs(TMath::Floor(5*TMath::Cos(i/1e7)))));

            m1->SetX(-xx); m1->SetY(0.5*yy);
            m1->SetMarkerStyle(27 + (Int_t)TMath::Floor(7*TMath::Cos(i/1e7)));

            //l2->SetX1(xx+1); l2->SetX2(xx+2);
            //l2->SetY1(yy*0.6); l2->SetY2(yy*0.7);

            xx = 0.3 + 0.2*TMath::Cos(i/2e6);
            yy = 0.3 + 0.2*TMath::Sin(i/3e6);
            l1->SetX1(xx); l1->SetX2(xx+0.2*TMath::Cos(i/7e6));
            l1->SetY1(yy); l1->SetY2(yy+0.2*TMath::Sin(i/7e6));

            t1->SetText(1-xx, 1-yy, TString::Format("Loop %d", i));

            c1->Modified();
            c1->Update();

            // if (i == kUPDATE) c1->SaveAs("canvas.svg");

            // if ((i % (kUPDATE*20000)) == 0) c1->SaveAs(TString::Format("canvas%d.gif", i/kUPDATE/20000));
         }
         if (gSystem->ProcessEvents())
            break;
      }
   }
   gBenchmark->Show("hsimple");


   // while (gSystem->ProcessEvents()) c1->Update();


   // Save all objects in this file
   hpx->SetFillColor(0);
   hfile->Write();
   hpx->SetFillColor(48);
   c1->Modified();
   return hfile;

   // Note that the file is automatically close when application terminates
   // or when the file destructor is called.
}
