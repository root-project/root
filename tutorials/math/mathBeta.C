// Test the TMath::BetaDist and TMath::BetaDistI functions
// author: Anna Kreshuk

#include "TMath.h"
#include "TCanvas.h"
#include "TF1.h"
#include "TLegend.h"

void mathBeta() {
  TCanvas *c1=new TCanvas("c1", "TMath::BetaDist",600,800);
  c1->Divide(1, 2);
  TVirtualPad *pad1 = c1->cd(1);
  pad1->SetGrid();
  TF1 *fbeta = new TF1("fbeta", "TMath::BetaDist(x, [0], [1])", 0, 1);
  fbeta->SetParameters(0.5, 0.5);
  TF1 *f1 = fbeta->DrawCopy();
  f1->SetLineColor(kRed);
  f1->SetLineWidth(1);
  fbeta->SetParameters(0.5, 2);
  TF1 *f2 = fbeta->DrawCopy("same");
  f2->SetLineColor(kGreen);
  f2->SetLineWidth(1);
  fbeta->SetParameters(2, 0.5);
  TF1 *f3 = fbeta->DrawCopy("same");
  f3->SetLineColor(kBlue);
  f3->SetLineWidth(1);
  fbeta->SetParameters(2, 2);
  TF1 *f4 = fbeta->DrawCopy("same");
  f4->SetLineColor(kMagenta);
  f4->SetLineWidth(1);
  TLegend *legend1 = new TLegend(.5,.7,.8,.9);
  legend1->AddEntry(f1,"p=0.5  q=0.5","l");
  legend1->AddEntry(f2,"p=0.5  q=2","l");
  legend1->AddEntry(f3,"p=2    q=0.5","l");
  legend1->AddEntry(f4,"p=2    q=2","l");
  legend1->Draw();

  TVirtualPad *pad2 = c1->cd(2);
  pad2->SetGrid();
  TF1 *fbetai=new TF1("fbetai", "TMath::BetaDistI(x, [0], [1])", 0, 1);
  fbetai->SetParameters(0.5, 0.5);
  TF1 *g1=fbetai->DrawCopy();
  g1->SetLineColor(kRed);
  g1->SetLineWidth(1);
  fbetai->SetParameters(0.5, 2);
  TF1 *g2=fbetai->DrawCopy("same");
  g2->SetLineColor(kGreen);
  g2->SetLineWidth(1);
  fbetai->SetParameters(2, 0.5);
  TF1 *g3=fbetai->DrawCopy("same");
  g3->SetLineColor(kBlue);
  g3->SetLineWidth(1);
  fbetai->SetParameters(2, 2);
  TF1 *g4=fbetai->DrawCopy("same");
  g4->SetLineColor(kMagenta);
  g4->SetLineWidth(1);

  TLegend *legend2 = new TLegend(.7,.15,0.9,.35);
  legend2->AddEntry(f1,"p=0.5  q=0.5","l");
  legend2->AddEntry(f2,"p=0.5  q=2","l");
  legend2->AddEntry(f3,"p=2    q=0.5","l");
  legend2->AddEntry(f4,"p=2    q=2","l");
  legend2->Draw();
  c1->cd();
}
