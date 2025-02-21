/// \file
/// \ingroup tutorial_hist
/// \notebook
/// \preview Palette coloring for 2D histograms' stack is activated thanks to the option PFC
/// (Palette Fill Color).
/// When this option is given to `THStack::Draw` the histograms  in the
/// `THStack` get their color from the current color palette defined by
/// `gStyle->SetPalette(...)`. The color is determined according to the number of
/// histograms.
///
/// In this example four 2D histograms are displayed with palette coloring.
/// The color of each graph is picked inside the palette number 1.
///
/// \macro_image
/// \macro_code
///
/// \date November 2022
/// \author Olivier Couet

void hist025_THStack_2d_palette_color()
{
   gStyle->SetPalette(1);
   auto h1 = new TH2F("h1", "h1", 20, 0, 6, 20, -4, 4);
   auto h2 = new TH2F("h2", "h2", 20, 0, 6, 20, -4, 4);
   auto h3 = new TH2F("h3", "h3", 20, 0, 6, 20, -4, 4);
   auto h4 = new TH2F("h4", "h4", 20, 0, 6, 20, -4, 4);
   auto h5 = new TH2F("h5", "h5", 20, 0, 6, 20, -4, 4);
   h2->Fill(2., 0., 5);
   h3->Fill(3., 0., 10);
   h4->Fill(4., 0., 15);
   h5->Fill(5., 0., 20);
   auto hs = new THStack("hs", "Test of palette colored lego stack");
   hs->Add(h1);
   hs->Add(h2);
   hs->Add(h3);
   hs->Add(h4);
   hs->Add(h5);
   hs->Draw("0lego1 PFC");
}
