/// \file
/// \ingroup tutorial_hist
///
/// This tutorial demonstrates how the highlight mechanism can be used on an histogram.
/// A 1D histogram is created.
/// Then an highlight method is connected to the histogram. Moving the mouse
/// on the histogram will open a new canvas showing in real time a zoom around
/// the highlighted bin.
///
/// \macro_code
///
/// \date March 2018
/// \author Jan Musinsky

TText *info = nullptr;
TH1 *hz = nullptr;

void HighlightZoom(TVirtualPad *pad, TObject *obj, Int_t xhb, Int_t yhb)
{
   auto h = dynamic_cast<TH1F *>(obj);
   if(!h) return;

   auto Canvas2 = (TCanvas *)gROOT->GetListOfCanvases()->FindObject("Canvas2");
   if (!h->IsHighlight()) { // after highlight disabled
      if (Canvas2) delete Canvas2;
      if (hz) { delete hz; hz = nullptr; }
      return;
   }

   if (info) info->SetTitle("");

   if (!Canvas2) {
      Canvas2 = new TCanvas("Canvas2", "Canvas2", 605, 0, 400, 400);
      Canvas2->SetGrid();
      if (hz) hz->Draw(); // after reopen this canvas
   }
   if (!hz) {
      hz = (TH1 *)h->Clone("hz");
      hz->SetTitle(TString::Format("%s (zoomed)", hz->GetTitle()));
      hz->SetStats(kFALSE);
      hz->Draw();
      Canvas2->Update();
      hz->SetHighlight(kFALSE);
   }

   Int_t zf = hz->GetNbinsX()*0.05; // zoom factor
   hz->GetXaxis()->SetRange(xhb-zf, xhb+zf);

   Canvas2->Modified();
   Canvas2->Update();
}

void hlHisto4()
{
   auto Canvas1 = new TCanvas("Canvas1", "", 0, 0, 600, 400);
   Canvas1->HighlightConnect("HighlightZoom(TVirtualPad*,TObject*,Int_t,Int_t)");

   auto f1 = new TF1("f1", "x*gaus(0) + [3]*abs(sin(x)/x)", -50.0, 50.0);
   f1->SetParameters(20.0, 4.0, 1.0, 20.0);
   auto h1 = new TH1F("h1", "Test random numbers", 200, -50.0, 50.0);
   h1->FillRandom("f1", 100000);
   h1->Draw();
   h1->Fit(f1, "Q");
   gStyle->SetGridColor(kGray);
   Canvas1->SetGrid();

   info = new TText(0.0, h1->GetMaximum()*0.7, "please move the mouse over the frame");
   info->SetTextSize(0.04);
   info->SetTextAlign(22);
   info->SetTextColor(kRed-1);
   info->SetBit(kCannotPick);
   info->Draw();

   Canvas1->Update();

   // configure highlight at the end when histogram is already painted
   h1->SetHighlight();
}
