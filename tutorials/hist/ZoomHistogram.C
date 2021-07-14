/// \file
/// \ingroup tutorial_hist
/// \notebook -js
/// Changing the Range on the X-Axis of a Histogram
///
/// Image produced by `.x ZoomHistogram.C`
///
/// This demonstrates how to zoom into a histogram by
/// changing the range on one of the axes (or both).
///
/// \macro_image
/// \macro_code
///
/// \author Advait Dhingra

void ZoomHistogram() {

   TH1F *norm = new TH1F("Normal Histogram", "Normal Histogram", 100, 0, 100);

   for (int i = 0; i < 100; ++i) {
     Double_t x = gRandom->Gaus(50, 10);
     norm->Fill(x);
   }

   TH1F *zoom = (TH1F*) norm->Clone("zoom");
   zoom->SetTitle("Zoomed-in Histogram");
   zoom->GetXaxis()->SetRangeUser(50, 100);

   TCanvas *c1 = new TCanvas("c1", "Histogram", 1500, 700);
   c1->Divide(2, 1);

   c1->cd(1);
   norm->Draw();
   c1->cd(2);
   zoom->Draw();
}
