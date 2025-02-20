/// \file
/// \ingroup tutorial_hist
/// \notebook
/// \preview This example demonstrates how to use the accessible color schemes with THStack.
/// In this example, the color scheme with six colors is used.
/// It also shows that the grayscale version is an acceptable alternative.
///
/// \macro_image
/// \macro_code
///
/// \date October 2024
/// \author Olivier Couet

void hist026_THStack_color_scheme()
{
   auto c1 = new TCanvas();
   auto hs = new THStack("hs", "Stacked 1D histograms colored using 6-colors scheme");

   // Create six 1-d histograms  and add them in the stack
   auto h1st = new TH1F("h1st", "A", 100, -4, 4);
   h1st->FillRandom("gaus", 20000);
   h1st->SetFillColor(kP6Blue);
   hs->Add(h1st);

   auto h2st = new TH1F("h2st", "B", 100, -4, 4);
   h2st->FillRandom("gaus", 15000);
   h2st->SetFillColor(kP6Yellow);
   hs->Add(h2st);

   auto h3st = new TH1F("h3st", "C", 100, -4, 4);
   h3st->FillRandom("gaus", 10000);
   h3st->SetFillColor(kP6Red);
   hs->Add(h3st);

   auto h4st = new TH1F("h4st", "D", 100, -4, 4);
   h4st->FillRandom("gaus", 10000);
   h4st->SetFillColor(kP6Grape);
   hs->Add(h4st);

   auto h5st = new TH1F("h5st", "E", 100, -4, 4);
   h5st->FillRandom("gaus", 10000);
   h5st->SetFillColor(kP6Gray);
   hs->Add(h5st);

   auto h6st = new TH1F("h6st", "F", 100, -4, 4);
   h6st->FillRandom("gaus", 10000);
   h6st->SetFillColor(kP6Violet);
   hs->Add(h6st);

   // draw the stack with colors
   hs->Draw();
   TLegend *l = gPad->BuildLegend(.8, .55, 1., .9, "", "F");
   l->SetLineWidth(0);
   l->SetFillStyle(0);

   // draw the stack using gray-scale
   auto c2 = new TCanvas();
   c2->SetGrayscale();
   hs->Draw();
   l->Draw();
}
