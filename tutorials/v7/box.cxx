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
#include <ROOT/RPadPos.hxx>

void box()
{
   using namespace ROOT::Experimental;

   // Create a canvas to be displayed.
   auto canvas = RCanvas::Create("Canvas Title");

   auto Box1 = canvas->Draw<RBox>(RPadPos(0.1_normal, 0.3_normal), RPadPos(0.3_normal,0.6_normal));
   RColor Color1(0, 255, 0, 0.5); // 50% opaque
   RColor Color2(0, 0, 255, 0.7); // 70% opaque

   Box1->SetLineColor(Color1).SetLineWidth(5);
   Box1->SetFillColor(RColor::kRed);

   auto Box2 = canvas->Draw<RBox>(RPadPos(0.4_normal, 0.2_normal), RPadPos(0.6_normal,0.7_normal));
   Box2->SetLineColor(Color2).SetLineStyle(2).SetLineWidth(10);
   Box2->SetFillColor(RColor::kGreen);

   auto Box3 = canvas->Draw<RBox>(RPadPos(0.7_normal, 0.4_normal), RPadPos(0.9_normal,0.6_normal));
   Box3->SetLineWidth(3);
   Box3->SetFillColor(RColor::kBlue);

   auto Box4 = canvas->Draw<RBox>(RPadPos(0.7_normal, 0.7_normal), RPadPos(0.9_normal,0.9_normal));
   Box4->SetLineWidth(3);

   auto Box5 = canvas->Draw<RBox>(RPadPos(0.7_normal, 0.1_normal), RPadPos(0.9_normal,0.3_normal));
   Box5->SetLineWidth(3);

   canvas->Show();
}
