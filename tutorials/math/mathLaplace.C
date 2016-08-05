/// \file
/// \ingroup tutorial_math
/// \notebook
/// Test the TMath::LaplaceDist and TMath::LaplaceDistI functions
///
/// \macro_image
/// \macro_code
///
/// \author Anna Kreshuk

#include "TMath.h"
#include "TCanvas.h"
#include "TF1.h"
#include "TLegend.h"

void mathLaplace(){
   TCanvas *c1=new TCanvas("c1", "TMath::LaplaceDist",600,800);
   c1->Divide(1, 2);
   TVirtualPad *pad1 = c1->cd(1);
   pad1->SetGrid();
   TF1 *flaplace = new TF1("flaplace", "TMath::LaplaceDist(x, [0], [1])", -10, 10);
   flaplace->SetParameters(0, 1);
   TF1 *f1 = flaplace->DrawCopy();
   f1->SetLineColor(kRed);
   f1->SetLineWidth(1);
   flaplace->SetParameters(0, 2);
   TF1 *f2 = flaplace->DrawCopy("same");
   f2->SetLineColor(kGreen);
   f2->SetLineWidth(1);
   flaplace->SetParameters(2, 1);
   TF1 *f3 = flaplace->DrawCopy("same");
   f3->SetLineColor(kBlue);
   f3->SetLineWidth(1);
   flaplace->SetParameters(2, 2);
   TF1 *f4 = flaplace->DrawCopy("same");
   f4->SetLineColor(kMagenta);
   f4->SetLineWidth(1);
   TLegend *legend1 = new TLegend(.7,.7,.9,.9);
   legend1->AddEntry(f1,"alpha=0 beta=1","l");
   legend1->AddEntry(f2,"alpha=0 beta=2","l");
   legend1->AddEntry(f3,"alpha=2 beta=1","l");
   legend1->AddEntry(f4,"alpha=2 beta=2","l");
   legend1->Draw();

   TVirtualPad *pad2 = c1->cd(2);
   pad2->SetGrid();
   TF1 *flaplacei=new TF1("flaplacei", "TMath::LaplaceDistI(x, [0], [1])", -10, 10);
   flaplacei->SetParameters(0, 1);
   TF1 *g1=flaplacei->DrawCopy();
   g1->SetLineColor(kRed);
   g1->SetLineWidth(1);
   flaplacei->SetParameters(0, 2);
   TF1 *g2=flaplacei->DrawCopy("same");
   g2->SetLineColor(kGreen);
   g2->SetLineWidth(1);
   flaplacei->SetParameters(2, 1);
   TF1 *g3=flaplacei->DrawCopy("same");
   g3->SetLineColor(kBlue);
   g3->SetLineWidth(1);
   flaplacei->SetParameters(2, 2);
   TF1 *g4=flaplacei->DrawCopy("same");
   g4->SetLineColor(kMagenta);
   g4->SetLineWidth(1);

   TLegend *legend2 = new TLegend(.7,.15,0.9,.35);
   legend2->AddEntry(f1,"alpha=0 beta=1","l");
   legend2->AddEntry(f2,"alpha=0 beta=2","l");
   legend2->AddEntry(f3,"alpha=2 beta=1","l");
   legend2->AddEntry(f4,"alpha=2 beta=2","l");
   legend2->Draw();
   c1->cd();
}
