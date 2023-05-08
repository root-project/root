/// \file
/// \ingroup tutorial_hist
///
/// This tutorial demonstrates how the highlight mechanism can be used on an histogram.
/// A 2D histogram is booked an filled with a random gaussian distribution and
/// drawn with the "col" option.
/// Then an highlight method is connected to the histogram. Moving the mouse
/// on the histogram open a new canvas displaying the two X and Y projections
/// at the highlighted bin.
///
/// \macro_code
///
/// \date March 2018
/// \author Jan Musinsky

TText *info = nullptr;

void Highlight2(TVirtualPad *pad, TObject *obj, Int_t xhb, Int_t yhb)
{
   auto h2 = dynamic_cast<TH2F *>(obj);
   if(!h2) return;
   auto CanvasProj = (TCanvas *) gROOT->GetListOfCanvases()->FindObject("CanvasProj");
   if (!h2->IsHighlight()) { // after highlight disabled
      if (CanvasProj) delete CanvasProj;
      h2->SetTitle("Disable highlight");
      return;
   }

   if (info) info->SetTitle("");

   auto px = h2->ProjectionX("_px", yhb, yhb);
   auto py = h2->ProjectionY("_py", xhb, xhb);
   px->SetTitle(TString::Format("ProjectionX of biny[%02d]", yhb));
   py->SetTitle(TString::Format("ProjectionY of binx[%02d]", xhb));

   if (!CanvasProj) {
      CanvasProj = new TCanvas("CanvasProj", "CanvasProj", 505, 0, 600, 600);
      CanvasProj->Divide(1, 2);
      CanvasProj->cd(1);
      px->Draw();
      CanvasProj->cd(2);
      py->Draw();
   }

   h2->SetTitle(TString::Format("Highlight bin [%02d, %02d]", xhb, yhb).Data());
   pad->Modified();
   pad->Update();

   CanvasProj->GetPad(1)->Modified();
   CanvasProj->GetPad(2)->Modified();
   CanvasProj->Update();
}

void hlHisto2()
{
   auto c1 = new TCanvas("Canvas", "Canvas", 0, 0, 500, 500);
   c1->HighlightConnect("Highlight2(TVirtualPad*,TObject*,Int_t,Int_t)");

   auto h2 = new TH2F("h2", "", 50, -5.0, 5.0, 50, -5.0, 5.0);
   for (Int_t i = 0; i < 10000; i++) h2->Fill(gRandom->Gaus(), gRandom->Gaus());
   h2->Draw("col");

   info = new TText(0.0, -4.0, "please move the mouse over the frame");
   info->SetTextAlign(22);
   info->SetTextSize(0.04);
   info->SetTextColor(kRed+1);
   info->SetBit(kCannotPick);
   info->Draw();
   c1->Update();

   h2->SetHighlight();
}

