/// \file
/// \ingroup tutorial_math
/// \notebook
/// Example macro showcasing some special mathematical functions.
/// 
/// To execute the macro type in:
///
/// ~~~{.cpp}
/// root[0] .x mathcoreSpecFunc.C
/// ~~~
///
/// It will create a canvas with the representation of the tgamma, lgamma, erf and erfc functions.
///
/// \macro_image
/// \macro_code
///
/// \author Andras Zsenei

#include "TF1.h"
#include "TSystem.h"
#include "TCanvas.h"

void mathcoreSpecFunc() {

   TF1 *f1a = new TF1("f1a","ROOT::Math::tgamma(x)",0,20);
   TF1 *f2a = new TF1("f2a","ROOT::Math::lgamma(x)",0,100);
   TF1 *f3a = new TF1("f3a","ROOT::Math::erf(x)",0,5);
   TF1 *f4a = new TF1("f4a","ROOT::Math::erfc(x)",0,5);

   TCanvas *c1 = new TCanvas("c1","c1",800,600);

   f1a->SetLineColor(kBlue);
   f2a->SetLineColor(kBlue);
   f3a->SetLineColor(kBlue);
   f4a->SetLineColor(kBlue);

   c1->Divide(2,2);

   c1->cd(1);
   f1a->Draw();
   c1->cd(2);
   f2a->Draw();
   c1->cd(3);
   f3a->Draw();
   c1->cd(4);
   f4a->Draw();

}
