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

   RAxisEquidistant axis0("vertical", 10, 0., 10.);
   auto draw0 = std::make_shared<RAxisDrawable<RAxisEquidistant>>(axis0);
   draw0->SetP1({0.05_normal,0.1_normal}).SetP2({0.05_normal,0.9_normal});
   draw0->AttrAxis().SetTitle("vertical");
   canvas->Draw(draw0);

   RAxisEquidistant axis1("regular", 10, 0., 100.);
   auto draw1 = std::make_shared<RAxisDrawable<RAxisEquidistant>>(axis1);
   draw1->SetP1({0.1_normal,0.9_normal}).SetP2({0.4_normal,0.9_normal});
   draw1->AttrAxis().SetTitle("regular").SetTitleCenter();
   canvas->Draw(draw1);

   RAxisIrregular axis2("irregular", {0., 1., 2., 3., 10.});
   auto draw2 = std::make_shared<RAxisDrawable<RAxisIrregular>>(axis2);
   draw2->SetP1({0.1_normal,0.7_normal}).SetP2({0.4_normal,0.7_normal});
   draw2->AttrAxis().SetTitle("irregular").SetTitleCenter();
   canvas->Draw(draw2);

   std::vector<std::string> labels = {"first", "second", "third", "forth", "fifth"};
   RAxisLabels axis3("labels", labels);
   auto draw3 = std::make_shared<RAxisDrawable<RAxisLabels>>(axis3);
   draw3->SetP1({0.1_normal,0.5_normal}).SetP2({0.4_normal,0.5_normal});
   draw3->AttrAxis().SetTitle("labels").SetTitleCenter();
   canvas->Draw(draw3);

   RAxisEquidistant axis4("log10", 10, 0.1, 100.);
   auto draw4 = std::make_shared<RAxisDrawable<RAxisEquidistant>>(axis4);
   draw4->SetP1({0.1_normal,0.3_normal}).SetP2({0.4_normal,0.3_normal});
   draw4->AttrAxis().SetLog(10).SetTitle("log10 scale").SetTitleCenter();
   canvas->Draw(draw4);

   RAxisEquidistant axis5("log2", 10, 0.125, 128.001);
   auto draw5 = std::make_shared<RAxisDrawable<RAxisEquidistant>>(axis5);
   draw5->SetP1({0.1_normal,0.1_normal}).SetP2({0.4_normal,0.1_normal});
   draw5->AttrAxis().SetLog(2).SetTitle("log2 scale").SetTitleCenter();
   canvas->Draw(draw5);

   RAxisEquidistant axis6("reverse", 10, 0., 10.);
   auto draw6 = std::make_shared<RAxisDrawable<RAxisEquidistant>>(axis6);
   draw6->SetP1({0.5_normal,0.9_normal}).SetP2({0.5_normal,0.1_normal});
   draw6->AttrAxis().SetTitle("reverse");
   canvas->Draw(draw6);

   canvas->SetSize(1000, 800);

   // requires Chrome browser, runs in headless mode
   // canvas->SaveAs("axes.png");

   canvas->Show();
}
