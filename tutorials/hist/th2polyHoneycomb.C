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
/// \author  Olivier Couet

void th2polyHoneycomb(){
   TH2Poly *hc = new TH2Poly();
   hc->SetTitle("Honeycomb example");
   hc->Honeycomb(0,0,.1,25,25);

   TRandom ran;
   for (int i = 0; i<30000; i++) {
      hc->Fill(ran.Gaus(2.,1), ran.Gaus(2.,1));
   }

   hc->Draw("colz");
}
