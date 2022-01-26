/// \file
/// \ingroup tutorial_hist
///
/// This tutorial demonstrates how the highlight mechanism can be used on an histogram.
/// A 2D histogram is booked an filled with a random gaussian distribution.
/// Then an highlight method is connected to the histogram. Moving the mouse
/// on the histogram will update the histogram title in real time according to
/// the highlighted bin.
///
/// \macro_code
///
/// \date March 2018
/// \author Jan Musinsky

TText *info = nullptr;

void HighlightTitle(TVirtualPad *pad, TObject *obj, Int_t xhb, Int_t yhb)
{
   auto h2 = dynamic_cast<TH2F*>(obj);
   if (!h2) return;
   if (!h2->IsHighlight()) { // after highlight disabled
      h2->SetTitle("Disable highlight");
      return;
   }
   if (info) info->SetTitle("");
   TString t;
   t.Form("bin[%02d, %02d] (%5.2f, %5.2f) content %g", xhb, yhb,
          h2->GetXaxis()->GetBinCenter(xhb), h2->GetYaxis()->GetBinCenter(yhb),
          h2->GetBinContent(xhb, yhb));
   h2->SetTitle(t.Data());
   pad->Update();
}

void hlHisto1()
{
   auto c1 = new TCanvas();
   c1->HighlightConnect("HighlightTitle(TVirtualPad*,TObject*,Int_t,Int_t)");

   auto h2 = new TH2F("h2", "", 50, -5.0, 5.0, 50, -5.0, 5.0);
   for (Int_t i = 0; i < 10000; i++) h2->Fill(gRandom->Gaus(), gRandom->Gaus());
   h2->Draw();

   info = new TText(0.0, -4.0, "please move the mouse over the frame");
   info->SetTextAlign(22);
   info->SetTextColor(kRed+1);
   info->SetBit(kCannotPick);
   info->Draw();
   c1->Update();

   // call after update to apply changes in the histogram painter
   h2->SetHighlight();
}


