/// \file
/// \ingroup tutorial_hist
/// This tutorial shows highlight mode for histogram 4
///
/// \macro_code
///
/// \date March 2018
/// \author Jan Musinsky

#include <TCanvas.h>
#include <TF1.h>
#include <TH1.h>
#include <TText.h>
#include <TROOT.h>
#include <TStyle.h>

void HighlightZoom(TVirtualPad *pad, TObject *obj, Int_t xhb, Int_t yhb);

void hlHisto4()
{
   TCanvas *c1 = new TCanvas("c1", "", 0, 0, 600, 400);
   TF1 *f1 = new TF1("f1", "x*gaus(0) + [3]*abs(sin(x)/x)", -50.0, 50.0);
   f1->SetParameters(20.0, 4.0, 1.0, 20.0);
   TH1F *h1 = new TH1F("h1", "Test random numbers", 200, -50.0, 50.0);
   h1->FillRandom("f1", 100000);
   h1->Draw();
   h1->Fit(f1, "Q");
   gStyle->SetGridColor(kGray);
   c1->SetGrid();

   TText *info = new TText(0.0, h1->GetMaximum()*0.7, "please move the mouse over the frame");
   info->SetTextSize(0.04);
   info->SetTextAlign(22);
   info->SetTextColor(kRed-1);
   info->SetBit(kCannotPick);
   info->Draw();
   c1->Update();

   h1->SetHighlight();
   c1->HighlightConnect("HighlightZoom(TVirtualPad*,TObject*,Int_t,Int_t)");
}

void HighlightZoom(TVirtualPad *pad, TObject *obj, Int_t xhb, Int_t yhb)
{
   TH1F *h = (TH1F *)obj;
   if(!h) return;

   TCanvas *c2 = (TCanvas *)gROOT->GetListOfCanvases()->FindObject("c2");
   static TH1 *hz = 0;
   if (!h->IsHighlight()) { // after highlight disabled
      if (c2) delete c2;
      if (hz) { delete hz; hz = 0; }
      return;
   }

   if (!c2) {
      c2 = new TCanvas("c2", "c2", 605, 0, 400, 400);
      c2->SetGrid();
      if (hz) hz->Draw(); // after reopen this canvas
   }
   if (!hz) {
      hz = (TH1 *)h->Clone("hz");
      hz->SetTitle(TString::Format("%s (zoomed)", hz->GetTitle()));
      hz->SetStats(kFALSE);
      hz->Draw();
      c2->Update();
      hz->SetHighlight(kFALSE);
   }

   Int_t zf = hz->GetNbinsX()*0.05; // zoom factor
   hz->GetXaxis()->SetRange(xhb-zf, xhb+zf);

   c2->Modified();
   c2->Update();
}
