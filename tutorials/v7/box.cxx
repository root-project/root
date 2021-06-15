/// \file
/// \ingroup tutorial_v7
///
/// This ROOT 7 example demonstrates how to create a ROOT 7 canvas (RCanvas) and
/// draw ROOT 7 boxes in it (RBox). It generates a set of boxes using the
/// "normal" coordinates' system.
///
/// \macro_image (rcanvas_js)
/// \macro_code
///
/// \date 2018-10-10
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!
/// \author Olivier Couet

#include "ROOT/RCanvas.hxx"
#include "ROOT/RColor.hxx"
#include "ROOT/RBox.hxx"
#include "ROOT/RPadPos.hxx"

void box()
{
   using namespace ROOT::Experimental;

   // Create a canvas to be displayed.
   auto canvas = RCanvas::Create("RBox drawing");

   auto box1 = canvas->Draw<RBox>(RPadPos(0.1_normal, 0.3_normal), RPadPos(0.3_normal,0.6_normal));
   box1->AttrBorder().SetColor(RColor::kBlue).SetWidth(5);
   box1->AttrFill().SetColor(RColor(0, 255, 0, 127)); // 50% opaque

   auto box2 = canvas->Draw<RBox>(RPadPos(0.4_normal, 0.2_normal), RPadPos(0.6_normal,0.7_normal));
   box2->AttrBorder().SetColor(RColor::kRed).SetStyle(2).SetWidth(10);
   box2->AttrFill().SetColor(RColor(0, 0, 255, 179)); // 70% opaque

   auto box3 = canvas->Draw<RBox>(RPadPos(0.7_normal, 0.4_normal), RPadPos(0.9_normal,0.6_normal));
   box3->AttrBorder().SetWidth(3);
   box3->AttrFill().SetColor(RColor::kBlue);

   auto box4 = canvas->Draw<RBox>(RPadPos(0.7_normal, 0.7_normal), RPadPos(0.9_normal,0.9_normal));
   box4->AttrBorder().SetWidth(4);

   auto box5 = canvas->Draw<RBox>(RPadPos(0.7_normal, 0.1_normal), RPadPos(0.9_normal,0.3_normal));
   box5->AttrBorder().SetRx(10).SetRy(10).SetWidth(2);

   canvas->Show();
}
