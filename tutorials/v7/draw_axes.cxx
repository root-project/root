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
   draw0->AttrAxis().SetTicksInvert().SetTicksSize(0.02_normal);
   draw0->AttrAxis().title = "vertical";

   auto draw1 = canvas->Draw<RAxisDrawable>(RPadPos(x1, 0.9_normal), false, w1);
   draw1->AttrAxis().title = "horizontal";
   draw1->AttrAxis().title.SetCenter();

   auto draw2 = canvas->Draw<RAxisDrawable>(RPadPos(x1, 0.7_normal), false, w1);
   draw2->AttrAxis().SetTicksBoth().SetTicksSize(0.02_normal);
   draw2->AttrAxis().title = "both side ticks";
   draw2->AttrAxis().title.SetCenter();

   auto draw3 = canvas->Draw<RAxisDrawable>(RPadPos(x1, 0.5_normal), false, w1);
   draw3->AttrAxis().title = "center labels";
   draw3->AttrAxis().labels.center = true;

   auto draw4 = canvas->Draw<RAxisDrawable>(RPadPos(x1, 0.3_normal), false, w1);
   draw4->SetMinMax(TDatime(2020,11,12,9,0,0).Convert(), TDatime(2020,11,12,12,0,0).Convert())
          .AttrAxis().SetTimeDisplay("%d/%m/%y %H:%M");
   draw4->AttrAxis().title = "time display";
   draw4->AttrAxis().labels.size = 0.01;
   draw4->AttrAxis().labels.color = RColor::kRed;

   std::vector<std::string> labels = {"first", "second", "third", "forth", "fifth"};
   auto draw5 = canvas->Draw<RAxisLabelsDrawable>(RPadPos(x1, 0.1_normal), false, w1);
   draw5->SetLabels(labels);
   draw5->AttrAxis().SetTicksInvert();
   draw5->AttrAxis().title = "labels, swap ticks side";
   draw5->AttrAxis().title.SetLeft();

   auto draw6 = canvas->Draw<RAxisDrawable>(RPadPos(0.5_normal,0.9_normal), true, -0.8_normal);
   draw6->SetMinMax(0, 10).AttrAxis().SetEndingArrow();
   draw6->AttrAxis().title = "vertical negative length";

   auto draw7 = canvas->Draw<RAxisDrawable>(RPadPos(x2, 0.9_normal), false, w2);
   draw7->SetMinMax(1, 100).AttrAxis().SetLog(10);
   draw7->AttrAxis().title = "log10 scale";
   draw7->AttrAxis().title.SetCenter();
   draw7->AttrAxis().title.SetFont(12);
   draw7->AttrAxis().title.color = RColor::kGreen;

   auto draw8 = canvas->Draw<RAxisDrawable>(RPadPos(x2, 0.7_normal), false, w2);
   draw8->SetMinMax(0.125, 128).AttrAxis().SetLog(2);
   draw8->AttrAxis().title = "log2 scale";
   draw8->AttrAxis().title.SetCenter();

   auto draw9 = canvas->Draw<RAxisDrawable>(RPadPos(x2, 0.5_normal), false, w2);
   draw9->SetMinMax(1, 100).AttrAxis().SetLog(2.7182);
   draw9->AttrAxis().title = "ln scale";
   draw9->AttrAxis().title.SetCenter();

   auto draw10 = canvas->Draw<RAxisDrawable>(RPadPos(x2+w2, 0.3_normal), false, -w2);
   draw10->AttrAxis().SetEndingArrow();
   draw10->AttrAxis().title = "horizontal negative length";
   draw10->AttrAxis().title.SetCenter();

   auto draw11 = canvas->Draw<RAxisDrawable>(RPadPos(x2, 0.1_normal), false, w2);
   draw11->AttrAxis().SetReverse().SetEndingArrow();
   draw11->AttrAxis().title = "horizontal reverse";
   draw11->AttrAxis().title.SetCenter();

   auto draw12 = canvas->Draw<RAxisDrawable>(RPadPos(0.97_normal, 0.1_normal), true, 0.8_normal);
   draw12->AttrAxis().SetTicksBoth().SetTicksSize(0.01_normal).SetTicksColor(RColor::kBlue)
                     .SetEndingArrow().SetEndingSize(0.01_normal);
   draw12->AttrAxis().title = "vertical axis with arrow";
   draw12->AttrAxis().title.SetCenter();

   canvas->SetSize(1000, 800);

   // requires Chrome browser, runs in headless mode
   // canvas->SaveAs("axes.png");

   canvas->Show();
}
