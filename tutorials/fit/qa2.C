/// \file
/// \ingroup tutorial_fit
/// \notebook -js
/// Test generation of random numbers distributed according to a function defined by the user
///
/// \macro_image
/// \macro_output
/// \macro_code
///
/// \author Rene Brun

#include <TBenchmark.h>
#include <TCanvas.h>
#include <TF1.h>
#include <TFormula.h>
#include <TH1F.h>
#include <TPaveLabel.h>

void qa2() {
   //Fill a 1-D histogram from a parametric function
   TCanvas *c1 = new TCanvas("c1","The FillRandom example",0,0,700,500);

   gBenchmark->Start("fillrandom");
   //
   // A function (any dimension) or a formula may reference
   // an already defined formula
   //
   TFormula *form1 = new TFormula("form1","abs(sin(x)/x)");
   TF1 *sqroot = new TF1("sqroot","x*gaus(0) + [3]*form1",0,10);
   sqroot->SetParameters(10,4,1,20);

   //
   // Create a one dimensional histogram (one float per bin)
   // and fill it following the distribution in function sqroot.
   //
   TH1F *h1f = new TH1F("h1f","Test random numbers",200,0,10);
   h1f->SetFillColor(45);
   h1f->FillRandom("sqroot",100000);
   h1f->Draw();
   TPaveLabel *lfunction = new TPaveLabel(5,39,9.8,46,"The sqroot function");
   lfunction->SetFillColor(41);

   c1->SetGridx();
   c1->SetGridy();

   h1f->SetDirectory(0);

   c1->Update();

   sqroot->SetParameters(200,4,1,20);
}
