/// \file
/// \ingroup tutorial_hist
/// \notebook -js
/// This tutorial illustrates how to create an histogram with hexagonal
/// bins (TH2Poly). The method TH2Poly::Honeycomb allows to build automatically
/// an honeycomb binning.
///
/// \macro_code
/// \macro_image
///
/// \date August 2023
/// \author  Olivier Couet

void hist038_TH2Poly_honeycomb()
{
   TCanvas *C = new TCanvas("C", "C", 1200, 600);
   C->Divide(2, 1);

   TH2Poly *hc1 = new TH2Poly();
   hc1->Honeycomb(0, 0, .1, 5, 5);
   hc1->SetTitle("Option V (default)");
   hc1->SetStats(0);
   hc1->Fill(.1, .1, 15.);
   hc1->Fill(.4, .4, 10.);
   hc1->Fill(.5, .5, 20.);

   TH2Poly *hc2 = new TH2Poly();
   hc2->Honeycomb(0, 0, .1, 5, 5, "h");
   hc2->SetTitle("Option H");
   hc2->SetStats(0);
   hc2->Fill(.1, .1, 15.);
   hc2->Fill(.4, .4, 10.);
   hc2->Fill(.5, .5, 20.);

   C->cd(1)->SetGrid();
   hc1->Draw("colz L");
   C->cd(2)->SetGrid();
   hc2->Draw("colz L");
}
