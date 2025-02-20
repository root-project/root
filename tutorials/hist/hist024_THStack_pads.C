/// \file
/// \ingroup tutorial_hist
/// \notebook
/// \preview Drawing stack histograms on subpads.
///
/// In this example three histograms are displayed on separate pads.
/// If canvas divided in advance - provided subpads will be used by the THStack.
///
/// \macro_image
/// \macro_code
///
/// \date June 2024
/// \author Sergey Linev

void hist024_THStack_pads()
{
   auto hs = new THStack("hs", "Stacked 1D histograms");

   // Create three 1-d histograms  and add them in the stack
   auto h1st = new TH1F("h1st", "test hstack 1", 100, -4, 4);
   h1st->FillRandom("gaus", 20000);
   hs->Add(h1st);

   auto h2st = new TH1F("h2st", "test hstack 2", 100, -4, 4);
   h2st->FillRandom("gaus", 15000);
   hs->Add(h2st);

   auto h3st = new TH1F("h3st", "test hstack 3", 100, -4, 4);
   h3st->FillRandom("gaus", 10000);
   hs->Add(h3st);

   auto c1 = new TCanvas("c1", "THStack drawing on pads", 800, 800);

   // prepare subpads for drawing of histograms
   c1->Divide(1, 3);

   // draw thstack on canvas with "pads" draw option
   c1->Add(hs, "pads");
}
