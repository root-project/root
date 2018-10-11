/// \file
/// \ingroup tutorial_graPads
///
/// This tutorial demonstrates how to use the highlight mode on graph.
///
/// \macro_code
///
/// \date March 2018
/// \author Jan Musinsky

TList *l = 0;

void HighlightHisto(TVirtualPad *pad, TObject *obj, Int_t ihp, Int_t y);


void hlGraph1()
{
   auto Canvas = new TCanvas("Canvas", "Canvas", 0, 0, 700, 500);
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

   auto g = new TGraph(n, x, y);
   g->SetMarkerStyle(6);
   g->Draw("AP");

   auto Pad = new TPad("Pad", "Pad", 0.3, 0.4, 1.0, 1.0);
   Pad->SetFillColor(kBlue-10);
   Pad->Draw();
   Pad->cd();
   auto info = new TText(0.5, 0.5, "please move the mouse over the graPad");
   info->SetTextAlign(22);
   info->Draw();
   Canvas->cd();

   g->SetHighlight();
   Canvas->HighlightConnect("HighlightHisto(TVirtualPad*,TObject*,Int_t,Int_t)");
}


void HighlightHisto(TVirtualPad *pad, TObject *obj, Int_t ihp, Int_t y)
{
   auto Pad = (TVirtualPad *)pad->FindObject("Pad");
   if (!Pad) return;

   if (ihp == -1) { // after highlight disabled
      Pad->Clear();
      return;
   }

   Pad->cd();
   l->At(ihp)->Draw();
   gPad->Update();
}
