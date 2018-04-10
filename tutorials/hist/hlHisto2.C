/// \file
/// \ingroup tutorial_hist
/// This tutorial shows highlight mode for histogram 2
///
/// \macro_code
///
/// \date March 2018
/// \author Jan Musinsky

#include <TCanvas.h>
#include <TH2.h>
#include <TRandom.h>
#include <TROOT.h>
#include <TText.h>

void Highlight2(TVirtualPad *pad, TObject *obj, Int_t xhb, Int_t yhb);

void hlHisto2()
{
   TCanvas *c1 = new TCanvas("c1", "c1", 0, 0, 500, 500);
   TH2F *h2 = new TH2F("h2", "", 50, -5.0, 5.0, 50, -5.0, 5.0);
   for (Int_t i = 0; i < 10000; i++) h2->Fill(gRandom->Gaus(), gRandom->Gaus());
   h2->Draw("col");

   TText *info = new TText(0.0, -4.0, "please move the mouse over the frame");
   info->SetTextAlign(22);
   info->SetTextSize(0.04);
   info->SetTextColor(kRed+1);
   info->SetBit(kCannotPick);
   info->Draw();
   c1->Update();

   h2->SetHighlight();
   c1->HighlightConnect("Highlight2(TVirtualPad*,TObject*,Int_t,Int_t)");
}

void Highlight2(TVirtualPad *pad, TObject *obj, Int_t xhb, Int_t yhb)
{
   TH2F *h2 = (TH2F *)obj;
   if(!h2) return;
   TCanvas *c2 = (TCanvas *)gROOT->GetListOfCanvases()->FindObject("c2");
   if (!h2->IsHighlight()) { // after highlight disabled
      if (c2) delete c2;
      return;
   }

   TH1 *px = h2->ProjectionX("_px", yhb, yhb);
   TH1 *py = h2->ProjectionY("_py", xhb, xhb);
   px->SetTitle(TString::Format("ProjectionX of biny[%02d]", yhb));
   py->SetTitle(TString::Format("ProjectionY of binx[%02d]", xhb));

   if (!c2) {
      c2 = new TCanvas("c2", "c2", 505, 0, 600, 600);
      c2->Divide(1, 2);
      c2->cd(1);
      px->Draw();
      c2->cd(2);
      py->Draw();
   }

   c2->GetPad(1)->Modified();
   c2->GetPad(2)->Modified();
   c2->Update();
}
