/// \file
/// \ingroup tutorial_hist
/// This tutorial shows highlight mode for histogram 1
///
/// \macro_code
///
/// \date March 2018
/// \author Jan Musinsky

#include <TCanvas.h>
#include <TH2.h>
#include <TRandom.h>
#include <TText.h>

void HighlightTitle(TVirtualPad *pad, TObject *obj, Int_t xhb, Int_t yhb);

void hlHisto1()
{
   TCanvas *c1 = new TCanvas();
   TH2F *h2 = new TH2F("h2", "", 50, -5.0, 5.0, 50, -5.0, 5.0);
   for (Int_t i = 0; i < 10000; i++) h2->Fill(gRandom->Gaus(), gRandom->Gaus());
   h2->Draw();

   TText *info = new TText(0.0, -4.0, "please move the mouse over the frame");
   info->SetTextAlign(22);
   info->SetTextColor(kRed+1);
   info->SetBit(kCannotPick);
   info->Draw();
   c1->Update();

   h2->SetHighlight();
   c1->HighlightConnect("HighlightTitle(TVirtualPad*,TObject*,Int_t,Int_t)");
}

void HighlightTitle(TVirtualPad *pad, TObject *obj, Int_t xhb, Int_t yhb)
{
   TH2F *h2 = (TH2F *)obj;
   if (!h2) return;
   if (!h2->IsHighlight()) { // after highlight disabled
      h2->SetTitle("");
      return;
   }

   TString t;
   t.Form("bin[%02d, %02d] (%5.2f, %5.2f) content %g", xhb, yhb,
          h2->GetXaxis()->GetBinCenter(xhb), h2->GetYaxis()->GetBinCenter(yhb),
          h2->GetBinContent(xhb, yhb));
   h2->SetTitle(t.Data());
   pad->Update();
}
