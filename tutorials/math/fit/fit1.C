/// \file
/// \ingroup tutorial_fit
/// \notebook
/// Simple fitting example (1-d histogram with an interpreted function)
///
/// \macro_image
/// \macro_output
/// \macro_code
///
/// \author Rene Brun

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
   TCanvas *c1 = new TCanvas("c1_fit1","The Fit Canvas",200,10,700,500);
   c1->SetGridx();
   c1->SetGridy();
   c1->GetFrame()->SetFillColor(21);
   c1->GetFrame()->SetBorderMode(-1);
   c1->GetFrame()->SetBorderSize(5);

   // (for more details, see 
   // <a href="hist001_TH1_fillrandom_userfunc.C.nbconvert.ipynb">filling histograms with random numbers from a function</a>)
   TFormula *form1 = new TFormula("form1", "abs(sin(x)/x)");
   TF1 *sqroot = new TF1("sqroot", "x*gaus(0) + [3]*form1", 0.0, 10.0);
   sqroot->SetLineColor(4);
   sqroot->SetLineWidth(6);
   // Set parameters to the functions "gaus" and "form1".
   sqroot->SetParameters(10.0, 4.0, 1.0, 20.0);
   sqroot->Print();
   
   TH1D *h1d = new TH1D("h1d", "Test random numbers", 200, 0.0, 10.0);
   h1d->FillRandom("sqroot", 10000);

   //
   // Now fit histogram h1d with the function sqroot
   //
   h1d->SetFillColor(45);
   h1d->Fit("sqroot");

   // We now annotate the picture by creating a PaveText object
   // and displaying the list of commands in this macro
   //
   TString dir = gROOT->GetTutorialDir();
   dir.Append("/math/fit/");
   TPaveText * fitlabel = new TPaveText(0.6,0.4,0.9,0.75,"NDC");
   fitlabel->SetTextAlign(12);
   fitlabel->SetFillColor(42);
   fitlabel->ReadFile(Form("%sfit1_C.txt", dir.Data()));
   fitlabel->Draw();
   c1->Update();
}
