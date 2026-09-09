/// \file
/// \ingroup tutorial_hist
/// \notebook -js
/// \preview Show the slice of a TH2 following the mouse position.
///
/// \macro_image
/// \macro_code
///
/// \date November 2022
/// \authors Rene Brun, Sergey Linev

void hist105_TExec_dynamic_slice()
{
   // Create a new canvas.
   TCanvas *c1 = new TCanvas("c1", "Dynamic Slice Example", 10, 10, 700, 500);

   // create a 2-d histogram, fill and draw it
   TH2F *hpxpy = new TH2F("hpxpy", "py vs px", 40, -4, 4, 40, -4, 4);
   hpxpy->SetStats(0);
   Double_t px, py;
   for (Int_t i = 0; i < 50000; i++) {
      gRandom->Rannor(px, py);
      hpxpy->Fill(px, py);
   }
   hpxpy->Draw("col");

   // Add a TExec object to the canvas
   c1->AddExec("dynamic", "DynamicExec()");
}

void DynamicExec()
{
   // Example of function called when a mouse event occurs in a pad.
   // When moving the mouse in the canvas, a second canvas shows the
   // projection along X of the bin corresponding to the Y position
   // of the mouse. The resulting histogram is fitted with a gaussian.
   // A "dynamic" line shows the current bin position in Y.
   // This more elaborated example can be used as a starting point
   // to develop more powerful interactive applications exploiting Cling
   // as a development engine.

   static int pyold = 0;

   if (!gPad->FeedbackMode(kTRUE)) {
      Error("DynamicExec", "Feedback mode is not supported");
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

   if (pyold) {
      // erase line at old position
      Float_t upyold = gPad->AbsPixeltoY(pyold);
      gPad->PaintLine(uxmin, upyold, uxmax, upyold);
      pyold = 0;
   }

   Float_t upy = gPad->AbsPixeltoY(py);
   Float_t y = gPad->PadtoY(upy);

   // draw a line at current position
   gPad->GetPainter()->DrawLine(uxmin, upy, uxmax, upy);

   // remember position of last painted line
   pyold = py;

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
   hp->SetFillColor(38);
   hp->SetName("Projection");
   hp->SetTitle(TString::Format("Projection of biny=%d", biny));
   hp->Fit("gaus", "ql");
   hp->GetFunction("gaus")->SetLineColor(kRed);
   hp->GetFunction("gaus")->SetLineWidth(6);
   c2->Update();
}
