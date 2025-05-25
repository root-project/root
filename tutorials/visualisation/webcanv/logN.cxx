/// \file
/// \ingroup tutorial_webcanv
/// \notebook -js
/// Logarithmic scales support in web canvas.
///
/// Shows support of log2, ln, log8 and log25 scales
/// Any integer base for logarithm can be specified as well
///
/// Functionality available only in web-based graphics
///
/// \macro_image (tcanvas_js)
/// \macro_code
///
/// \author Sergey Linev

void logN()
{
   auto h1 = new TH1I("hist", "Random data", 100, -5, 5);
   h1->FillRandom("gaus", 10000);

   auto c1 = new TCanvas("c1", "Logarithmic scales", 1200, 800);

   if (!gROOT->IsBatch() && !c1->IsWeb())
      ::Warning("logN.cxx", "macro will not work without enabling web-based canvas");

   c1->Divide(2, 2);

   c1->GetPad(1)->SetLogy(2);  // configure log2
   c1->GetPad(1)->Add(h1, ""); // draw with default draw option

   c1->GetPad(2)->SetLogy(3);   // configure ln - 3 is special case
   c1->GetPad(2)->Add(h1, "l"); // draw histogram as line

   c1->GetPad(3)->SetLogy(8);   // configure log8
   c1->GetPad(3)->Add(h1, "c"); // draw histogram as curve

   c1->GetPad(4)->SetLogy(25);  // configure log25
   c1->GetPad(4)->Add(h1, "E"); // draw histogram as errors
}
