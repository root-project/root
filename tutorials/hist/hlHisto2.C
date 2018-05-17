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

void Highlight2(TVirtualPad *pad, TObject *obj, Int_t xhb, Int_t yhb);

TText *info;


void hlHisto2()
{
   auto Canvas = new TCanvas("Canvas", "Canvas", 0, 0, 500, 500);
   auto h2 = new TH2F("h2", "", 50, -5.0, 5.0, 50, -5.0, 5.0);
   for (Int_t i = 0; i < 10000; i++) h2->Fill(gRandom->Gaus(), gRandom->Gaus());
   h2->Draw("col");

   info = new TText(0.0, -4.0, "please move the mouse over the frame");
   info->SetTextAlign(22);
   info->SetTextSize(0.04);
   info->SetTextColor(kRed+1);
   info->SetBit(kCannotPick);
   info->Draw();
   Canvas->Update();

   h2->SetHighlight();
   Canvas->HighlightConnect("Highlight2(TVirtualPad*,TObject*,Int_t,Int_t)");
}


void Highlight2(TVirtualPad *pad, TObject *obj, Int_t xhb, Int_t yhb)
{
   auto h2 = (TH2F *)obj;
   if(!h2) return;
   auto CanvasProj = (TCanvas *) gROOT->GetListOfCanvases()->FindObject("CanvasProj");
   if (!h2->IsHighlight()) { // after highlight disabled
      if (CanvasProj) delete CanvasProj;
      return;
   }

   info->SetTitle("");

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

   CanvasProj->GetPad(1)->Modified();
   CanvasProj->GetPad(2)->Modified();
   CanvasProj->Update();
}
