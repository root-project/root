/// \file
/// \ingroup tutorial_hist
/// \notebook -js
/// Changing the Range on the X-Axis of a Histogram
///
/// Image produced by `.x RangeChange.C`
///
/// This demonstrates how to zoom into a Histogram by
/// changing the range on one of the axes (or both).
///
/// \macro_image
/// \macro_code
///
/// \author Advait Dhingra

void RangeChange() {

   TH1F *norm = new TH1F("Normal Histogram", "Normal Histogram", 100, 0, 100);
   TH1F *twozoom = new TH1F("2x Zoom in Histogram", "2x Zoom in Histogram", 100, 0, 100);
   twozoom->GetXaxis()->SetRangeUser(25, 75);

   TRandom2 rand;

   for (int i = 0; i < 100; ++i) {
     Double_t j = rand.Gaus(50, 10);
     norm->Fill(j);
     twozoom->Fill(j);
   }

   TCanvas *c1 = new TCanvas("c1", "Histogram", 1500, 700);
   c1->Divide(2, 1);

   c1->cd(1);
   norm->Draw();
   c1->cd(2);
   twozoom->Draw();
}
