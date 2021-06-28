/// \file
/// \ingroup tutorial_v7
///
/// This ROOT7 example demonstrates how to create a RCanvas and
/// draw several RAxis objects with different options.
///
/// \macro_image (rcanvas_js)
/// \macro_code
///
/// \date 2020-11-03
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!
/// \authors Sergey Linev <S.Linev@gsi.de>

#include <ROOT/RCanvas.hxx>
#include <ROOT/RAxisDrawable.hxx>

#include "TDatime.h"

// macro must be here while cling is not capable to load
// library automatically for outlined function see ROOT-10336
R__LOAD_LIBRARY(libROOTGpadv7)


void draw_axes()
{
   using namespace ROOT::Experimental;

   // Create a canvas to be displayed.
   auto canvas = RCanvas::Create("RAxis examples");

   auto x1 = 0.08_normal, w1 = 0.36_normal, x2 = 0.57_normal, w2 = 0.36_normal;

   auto draw0 = canvas->Draw<RAxisDrawable>(RPadPos(0.03_normal,0.1_normal), true, 0.8_normal);
   draw0->axis.ticks.SetInvert();
   draw0->axis.ticks.size = 0.02_normal;
   draw0->axis.title = "vertical";

   auto draw1 = canvas->Draw<RAxisDrawable>(RPadPos(x1, 0.9_normal), false, w1);
   draw1->axis.title = "horizontal";
   draw1->axis.title.SetCenter();

   auto draw2 = canvas->Draw<RAxisDrawable>(RPadPos(x1, 0.7_normal), false, w1);
   draw2->axis.ticks.SetBoth();
   draw2->axis.ticks.size = 0.02_normal;
   draw2->axis.title = "both side ticks";
   draw2->axis.title.SetCenter();

   auto draw3 = canvas->Draw<RAxisDrawable>(RPadPos(x1, 0.5_normal), false, w1);
   draw3->axis.title = "center labels";
   draw3->axis.labels.center = true;

   auto draw4 = canvas->Draw<RAxisDrawable>(RPadPos(x1, 0.3_normal), false, w1);
   draw4->SetMinMax(TDatime(2020,11,12,9,0,0).Convert(), TDatime(2020,11,12,12,0,0).Convert());
   draw4->axis.SetTimeDisplay("%d/%m/%y %H:%M");
   draw4->axis.title = "time display";
   draw4->axis.labels.size = 0.01;
   draw4->axis.labels.color = RColor::kRed;

   std::vector<std::string> labels = {"first", "second", "third", "forth", "fifth"};
   auto draw5 = canvas->Draw<RAxisLabelsDrawable>(RPadPos(x1, 0.1_normal), false, w1);
   draw5->SetLabels(labels);
   draw5->axis.ticks.SetInvert();
   draw5->axis.title = "labels, swap ticks side";
   draw5->axis.title.SetLeft();

   auto draw6 = canvas->Draw<RAxisDrawable>(RPadPos(0.5_normal,0.9_normal), true, -0.8_normal);
   draw6->SetMinMax(0, 10);
   draw6->axis.ending.SetArrow();
   draw6->axis.title = "vertical negative length";

   auto draw7 = canvas->Draw<RAxisDrawable>(RPadPos(x2, 0.9_normal), false, w2);
   draw7->SetMinMax(1, 100);
   draw7->axis.log = 10;
   draw7->axis.title = "log10 scale";
   draw7->axis.title.SetCenter();
   draw7->axis.title.font = 12;
   draw7->axis.title.color = RColor::kGreen;
   draw7->axis.ending.SetCircle();

   auto draw8 = canvas->Draw<RAxisDrawable>(RPadPos(x2, 0.7_normal), false, w2);
   draw8->SetMinMax(0.125, 128);
   draw8->axis.log = 2;
   draw8->axis.title = "log2 scale";
   draw8->axis.title.SetCenter();

   auto draw9 = canvas->Draw<RAxisDrawable>(RPadPos(x2, 0.5_normal), false, w2);
   draw9->SetMinMax(1, 100);
   draw9->axis.log = 2.7182;
   draw9->axis.title = "ln scale";
   draw9->axis.title.SetCenter();

   auto draw10 = canvas->Draw<RAxisDrawable>(RPadPos(x2+w2, 0.3_normal), false, -w2);
   draw10->axis.ending.SetArrow();
   draw10->axis.title = "horizontal negative length";
   draw10->axis.title.SetCenter();

   auto draw11 = canvas->Draw<RAxisDrawable>(RPadPos(x2, 0.1_normal), false, w2);
   draw11->axis.ending.SetArrow();
   draw11->axis.reverse = true;
   draw11->axis.title = "horizontal reverse";
   draw11->axis.title.SetCenter();

   auto draw12 = canvas->Draw<RAxisDrawable>(RPadPos(0.97_normal, 0.1_normal), true, 0.8_normal);
   draw12->axis.ending.SetArrow();
   draw12->axis.ending.size = 0.01_normal;
   draw12->axis.ticks.SetBoth();
   draw12->axis.ticks.color = RColor::kBlue;
   draw12->axis.ticks.size = 0.01_normal;
   draw12->axis.title = "vertical axis with arrow";
   draw12->axis.title.SetCenter();

   canvas->SetSize(1000, 800);

   // requires Chrome browser, runs in headless mode
   // canvas->SaveAs("axes.png");

   canvas->Show();
}
