/// \file
/// \ingroup tutorial_graphics
/// \notebook
/// This example shows how to use the style "Auto"
///
/// \macro_image
/// \macro_code
///
/// \author Olivier Couet

void styleAuto() {
   gROOT->SetStyle("Auto");

   auto h1 = new TH1D("h1","histogram 1 with L option" , 100, -1, 1);
   h1->SetBit(TH1::kNoStats);
   auto h2 = new TH1D("h2","histogram 2 with P option" , 100, -1, 1);
   auto h3 = new TH1D("h3","histogram 3 with LP option", 100, -1, 1);

   h1->FillRandom("gaus",100000);
   h2->FillRandom("gaus",100000);
   h3->FillRandom("gaus",100000);

   h1->Draw("Pl");
//   h2->Draw("P SAMES");
//h3->Draw("LP SAMES");
 //  gPad->BuildLegend();
}


/*
- h3scat painted with option SCAT (not default anymore)
- avoid superimposed stat box in case of SAMES
- Automatic coloring in case of negative color indices
*/