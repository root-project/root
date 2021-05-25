/// \file
/// \ingroup tutorial_hist
/// \notebook
/// Palette coloring for histograms' stack is activated thanks to the options `PFC`
/// (Palette Fill Color), `PLC` (Palette Line Color) and `AMC` (Palette Marker Color).
/// When one of these options is given to `THStack::Draw` the histograms  in the
/// `THStack` get their color from the current color palette defined by
/// `gStyle->SetPalette(...)`. The color is determined according to the number of
/// histograms.
///
/// In this example four histograms are displayed with palette coloring.
/// The color of each histogram is picked inside the palette `kOcean`.
///
/// \macro_image
/// \macro_code
///
/// \author Olivier Couet

void thstackpalettecolor()
{
   auto hs = new THStack("hs","Stacked 1D histograms colored using kOcean palette");

   gStyle->SetPalette(kOcean);

   // Create three 1-d histograms  and add them in the stack
   auto h1st = new TH1F("h1st","test hstack",100,-4,4);
   h1st->FillRandom("gaus",20000);
   hs->Add(h1st);

   auto h2st = new TH1F("h2st","test hstack",100,-4,4);
   h2st->FillRandom("gaus",15000);
   hs->Add(h2st);

   auto h3st = new TH1F("h3st","test hstack",100,-4,4);
   h3st->FillRandom("gaus",10000);
   hs->Add(h3st);

   // draw the stack
   hs->Draw("pfc nostack");
}
