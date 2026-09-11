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

   if (!gPad->FeedbackMode(kTRUE)) {
      Error("hist058_TExec_th2", "Feedback mode is not supported");
      return;
   }

   int px = gPad->GetEventX();
   int py = gPad->GetEventY();
   TObject *select = gPad->GetSelected();
   float uxmin = gPad->GetUxmin();
   float uxmax = gPad->GetUxmax();
   auto c2 = static_cast<TCanvas *>(gROOT->GetListOfCanvases()->FindObject("c2"));

   TH2 *h = dynamic_cast<TH2 *>(select);
   if (!h)
      return;

   // misuse of pad uniqueid for last paint position
   int pyold = gPad->GetUniqueID();
   gPad->SetUniqueID(0);

   if (pyold && c2) {
      // erase line at old position
      Float_t upyold = gPad->AbsPixeltoY(pyold);
      gPad->PaintLine(uxmin, upyold, uxmax, upyold);
   }

   Float_t upy = gPad->AbsPixeltoY(py);
   Float_t y = gPad->PadtoY(upy);

   // paint a line at current position
   gPad->PaintLine(uxmin, upy, uxmax, upy);

   // remember last paint position, misuse of pad uniqueid
   gPad->SetUniqueID(py);

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
   Int_t biny = h->GetYaxis()->FindBin(y);
   TH1D *hp = h->ProjectionX("", biny, biny);
   hp->SetName("Projection");
   hp->SetTitle(TString::Format("Projection of biny=%d", biny));
   hp->Fit("gaus", "ql");
   c2->Update();
}
