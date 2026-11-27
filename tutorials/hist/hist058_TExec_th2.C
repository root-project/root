/// \file
/// \ingroup tutorial_hist
/// Echo object at mouse position and show a graphics line.
/// Example of macro called when a mouse event occurs in a pad.
///
/// Example:
/// ~~~{.cpp}
///   TFile::Open("hsimple.root");
///   hpxpy->Draw("colz");
///   gPad->AddExec("ex2", ".x hist058_TExec_th2.C");
/// ~~~
/// When moving the mouse in the canvas, a second canvas shows the
/// projection along X of the bin corresponding to the Y position
/// of the mouse. The resulting histogram is fitted with a gaussian.
/// A "dynamic" line shows the current bin position in Y.
/// This more elaborated example can be used as a starting point
/// to develop more powerful interactive applications exploiting CLING
/// as a development engine.
///
/// \macro_code
///
/// \date February 2023
/// \authors Rene Brun, Sergey Linev

void hist058_TExec_th2()
{
   if (!gPad) {
      Error("hist058_TExec_th2", "gPad is null, you are not supposed to run this macro");
      return;
   }

   TH2 *h = dynamic_cast<TH2 *>(gPad->GetSelected());
   if (!h)
      return;

   Double_t py = gPad->AbsPixeltoY(gPad->GetEventY());
   Double_t xmin = gPad->GetUxmin();
   Double_t xmax = gPad->GetUxmax();
   Double_t ymin = gPad->GetUymin();
   Double_t ymax = gPad->GetUymax();
   Int_t biny = h->GetYaxis()->FindBin(py);

   if ((py >= ymin) && (py <= ymax)) {
      ymin = h->GetYaxis()->GetBinLowEdge(biny);
      ymax = h->GetYaxis()->GetBinUpEdge(biny);
   } else {
      xmax = xmin; // paint dummy rect to remove previous one
      ymax = ymin;
      biny = -1;
   }

   gPad->PaintBox(xmin, ymin, xmax, ymax, "ilhist058");

   gPad->Update();

   if (biny < 0)
      return;

   auto c2 = static_cast<TCanvas *>(gROOT->GetListOfCanvases()->FindObject("c2"));

   // remember active pad and restore it when leave function
   TVirtualPad::TContext ctxt;

   // create or set the new canvas c2
   if (c2)
      delete c2->GetPrimitive("Projection");
   else
      c2 = new TCanvas("c2", "Projection Canvas", 710, 10, 700, 500);
   c2->SetGrid();
   c2->cd();

   // draw slice corresponding to mouse position
   TH1D *hp = h->ProjectionX("", biny, biny);
   hp->SetName("Projection");
   hp->SetTitle(TString::Format("Projection of biny=%d", biny));
   hp->Fit("gaus", "ql");
   c2->Update();
}
