/// \file
/// \ingroup tutorial_graphs
/// This tutorial shows highlight mode for graph 1
///
/// \macro_code
///
/// \date March 2018
/// \author Jan Musinsky

#include <TCanvas.h>
#include <TH1.h>
#include <TF1.h>
#include <TGraph.h>
#include <TText.h>

TList *l = 0;

void HighlightHisto(TVirtualPad *pad, TObject *obj, Int_t ihp, Int_t y);

void hlGraph1()
{
   TCanvas *ch = new TCanvas("ch", "ch", 0, 0, 700, 500);
   const Int_t n = 500;
   Double_t x[n], y[n];
   TH1F *h;
   l = new TList();

   for (Int_t i = 0; i < n; i++) {
      h = new TH1F(TString::Format("h_%03d", i), "", 100, -3.0, 3.0);
      h->FillRandom("gaus", 1000);
      h->Fit("gaus", "Q");
      h->SetMaximum(250.0); // for n > 200
      l->Add(h);
      x[i] = i;
      y[i] = h->GetFunction("gaus")->GetParameter(2);
   }

   TGraph *g = new TGraph(n, x, y);
   g->SetMarkerStyle(6);
   g->Draw("AP");

   TPad *ph = new TPad("ph", "ph", 0.3, 0.4, 1.0, 1.0);
   ph->SetFillColor(kBlue-10);
   ph->Draw();
   ph->cd();
   TText *info = new TText(0.5, 0.5, "please move the mouse over the graph");
   info->SetTextAlign(22);
   info->Draw();
   ch->cd();

   g->SetHighlight();
   ch->HighlightConnect("HighlightHisto(TVirtualPad*,TObject*,Int_t,Int_t)");
}

void HighlightHisto(TVirtualPad *pad, TObject *obj, Int_t ihp, Int_t y)
{
   TVirtualPad *ph = (TVirtualPad *)pad->FindObject("ph");
   if (!ph) return;

   if (ihp == -1) { // after highlight disabled
      ph->Clear();
      return;
   }

   ph->cd();
   l->At(ihp)->Draw();
   gPad->Update();
}
