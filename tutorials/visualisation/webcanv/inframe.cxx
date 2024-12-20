/// \file
/// \ingroup tutorial_webcanv
/// \notebook -js
/// Drawing primitives inside and outside of the frame.
///
/// In normal ROOT graphics all objects drawn on the pad and therefore
/// requires special treatment to be able drawn only inside frame borders.
/// In web-based graphics objects automatically clipped by frame border - if drawn inside frame.
/// Macro demonstrates usage of "frame" draw option for TLine, TBox, TMarker and TLatex classes.
/// If user interactively change zooming range "in-frame" objects automatically clipped.
///
/// Functionality available only in web-based graphics
///
/// \macro_image (tcanvas_js)
/// \macro_code
///
/// \author Sergey Linev

void inframe()
{
   auto c1 = new TCanvas("c1", "Drawing inside frame", 1200, 800);

   if (!gROOT->IsBatch() && !c1->IsWeb())
      ::Warning("inframe.cxx", "macro may not work without enabling web-based canvas");

   c1->DrawFrame(0., 0., 10., 10., "Usage of \"frame\" draw options");

   auto latex = new TLatex(3., 8., "Text in the frame");
   latex->SetTextColor(kCyan);
   latex->SetTextSize(0.08);
   latex->SetTextAlign(22);
   c1->Add(latex, "frame");

   // draw line and label on the pad
   auto l1 = new TLine(-0.5, 5, 10.5, 5);
   l1->SetLineColor(kBlue);
   c1->Add(l1);

   auto tl1 = new TLatex(0.5, 5, "line outside");
   tl1->SetTextColor(kBlue);
   tl1->SetTextAlign(13);
   c1->Add(tl1);

   // draw line and label in the frame
   auto l2 = new TLine(-0.5, 5.2, 10.5, 5.2);
   l2->SetLineColor(kGreen);
   c1->Add(l2, "frame");

   auto tl2 = new TLatex(0.5, 5.3, "line inside");
   tl2->SetTextColor(kGreen);
   tl2->SetTextAlign(11);
   c1->Add(tl2, "frame");

   // draw box and label on the pad
   auto b1 = new TBox(-0.5, 1, 4, 3);
   b1->SetFillColor(kBlue);
   c1->Add(b1);

   auto tb1 = new TLatex(0.5, 3.1, "box outside");
   tb1->SetTextColor(kBlue);
   tb1->SetTextAlign(11);
   c1->Add(tb1);

   // draw box and label in the frame
   auto b2 = new TBox(6, 1, 10.5, 3);
   b2->SetFillColor(kGreen);
   c1->Add(b2, "frame");

   auto b2_dash = new TBox(6, 1, 10.5, 3);
   b2_dash->SetFillStyle(0);
   b2_dash->SetLineColor(kRed);
   b2_dash->SetLineStyle(kDotted);
   b2_dash->SetLineWidth(3);
   c1->Add(b2_dash); // show clipped

   auto tb2 = new TLatex(6.5, 3.1, "box inside");
   tb2->SetTextColor(kGreen);
   tb2->SetTextAlign(11);
   c1->Add(tb2, "frame");

   // draw marker and label on the pad
   auto m1 = new TMarker(9.5, 7., 29);
   m1->SetMarkerColor(kBlue);
   m1->SetMarkerSize(3);
   c1->Add(m1);

   auto tm1 = new TLatex(9.3, 7., "outside");
   tm1->SetTextColor(kBlue);
   tm1->SetTextAlign(32);
   c1->Add(tm1);

   // draw marker and label in the frame
   auto m2 = new TMarker(9.5, 8., 34);
   m2->SetMarkerColor(kGreen);
   m2->SetMarkerSize(3);
   c1->Add(m2, "frame");

   auto tm2 = new TLatex(9.3, 8., "inside");
   tm2->SetTextColor(kGreen);
   tm2->SetTextAlign(32);
   c1->Add(tm2, "frame");
}
