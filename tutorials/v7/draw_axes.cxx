/// \file
/// \ingroup tutorial_v7
///
/// This ROOT7 example demonstrates how to create a RCanvas and
/// draw several RAxis objects with different options.
///
/// \macro_image (axes.png)
/// \macro_code
///
/// \date 2020-11-03
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!
/// \authors Sergey Linev <S.Linev@gsi.de>

#include <ROOT/RCanvas.hxx>
#include <ROOT/RAxisDrawable.hxx>

// macro must be here while cling is not capable to load
// library automatically for outlined function see ROOT-10336
R__LOAD_LIBRARY(libROOTHistDraw)

void draw_axes()
{
   using namespace ROOT::Experimental;

   // Create a canvas to be displayed.
   auto canvas = RCanvas::Create("RAxis examples");

   RAxisEquidistant axis1("regular", 10, 0., 100.);
   auto draw1 = std::make_shared<RAxisDrawable<RAxisEquidistant>>(axis1);
   draw1->SetP1({0.1_normal,0.9_normal}).SetP2({0.4_normal,0.9_normal});
   canvas->Draw(draw1);

   RAxisIrregular axis2("irregular", {0., 1., 2., 3., 10.});
   auto draw2 = std::make_shared<RAxisDrawable<RAxisIrregular>>(axis2);
   draw2->SetP1({0.1_normal,0.7_normal}).SetP2({0.4_normal,0.7_normal});
   canvas->Draw(draw2);

   std::vector<std::string> labels = {"first", "second", "third", "forth", "fifth"};
   RAxisLabels axis3("labels", labels);
   auto draw3 = std::make_shared<RAxisDrawable<RAxisLabels>>(axis3);
   draw3->SetP1({0.1_normal,0.5_normal}).SetP2({0.4_normal,0.5_normal});
   canvas->Draw(draw3);

   canvas->SetSize(1000, 800);

   // requires Chrome browser, runs in headless mode
   // canvas->SaveAs("axes.png");

   canvas->Show();
}
