#include "TCanvas.h"
#include "TFrame.h"
#include "TBenchmark.h"
#include "TString.h"
#include "TF1.h"
#include "TH1.h"
#include "TFile.h"
#include "TROOT.h"
#include "TError.h"
#include "TInterpreter.h"
#include "TSystem.h"
#include "TPaveText.h"

void fit1() {
   //Simple fitting example (1-d histogram with an interpreted function)
   //To see the output of this macro, click begin_html <a href="gif/fit1.gif">here</a>. end_html
   //Author: Rene Brun

   TCanvas *c1 = new TCanvas("c1_fit1","The Fit Canvas",200,10,700,500);
   c1->SetGridx();
   c1->SetGridy();
   c1->GetFrame()->SetFillColor(21);
   c1->GetFrame()->SetBorderMode(-1);
   c1->GetFrame()->SetBorderSize(5);

   gBenchmark->Start("fit1");
   //
   // We connect the ROOT file generated in a previous tutorial
   // (see begin_html <a href="fillrandom.C.html">Filling histograms with random numbers from a function</a>) end_html
   //
   TString dir = gSystem->UnixPathName(__FILE__);
   dir.ReplaceAll("fit1.C","");
   dir.ReplaceAll("/./","/");
   TFile *file = TFile::Open("fillrandom.root");
   if (!file) {
      gROOT->ProcessLine(Form(".x %s../hist/fillrandom.C",dir.Data()));
      file = TFile::Open("fillrandom.root");
      if (!file) return;
   }

   //
   // The function "ls()" lists the directory contents of this file
   //
   file->ls();

   //
   // Get object "sqroot" from the file. Undefined objects are searched
   // for using gROOT->FindObject("xxx"), e.g.:
   // TF1 *sqroot = (TF1*) gROOT.FindObject("sqroot")
   //
   TF1 * sqroot = 0;
   file->GetObject("sqroot",sqroot);
   if (!sqroot){
      Error("fit1.C","Cannot find object sqroot of type TF1\n");
      return;
   }
   sqroot->Print();

   //
   // Now get and fit histogram h1f with the function sqroot
   //
   TH1F* h1f = 0;
   file->GetObject("h1f",h1f);
   if (!h1f){
      Error("fit1.C","Cannot find object h1f of type TH1F\n");
      return;
   }
   h1f->SetFillColor(45);
   h1f->Fit("sqroot");

   // We now annotate the picture by creating a PaveText object
   // and displaying the list of commands in this macro
   //
   TPaveText * fitlabel = new TPaveText(0.6,0.3,0.9,0.80,"NDC");
   fitlabel->SetTextAlign(12);
   fitlabel->SetFillColor(42);
   fitlabel->ReadFile(Form("%sfit1_C.txt",dir.Data()));
   fitlabel->Draw();
   c1->Update();
   gBenchmark->Show("fit1");
}
