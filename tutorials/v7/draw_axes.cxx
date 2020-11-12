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
R__LOAD_LIBRARY(libROOTGpadv7)


void draw_axes()
{
   using namespace ROOT::Experimental;

   // Create a canvas to be displayed.
   auto canvas = RCanvas::Create("RAxis examples");

   auto draw0 = canvas->Draw<RAxisDrawable>(RPadPos(0.05_normal,0.1_normal), true, 0.8_normal);
   draw0->AttrAxis().SetMin(0).SetMax(100).SetTitle("vertical");

   auto draw1 = canvas->Draw<RAxisDrawable>(RPadPos(0.1_normal,0.9_normal), false, 0.3_normal);
   draw1->AttrAxis().SetMin(0).SetMax(100).SetTitle("regular").SetTitleCenter();

/*   RAxisIrregular axis2("irregular", {0., 1., 2., 3., 10.});
   auto draw2 = std::make_shared<RAxisDrawable<RAxisIrregular>>(axis2);
   draw2->SetPos({0.1_normal,0.7_normal}).SetLength(0.3_normal);
   draw2->AttrAxis().SetTitle("irregular").SetTitleCenter().LabelsAttr().SetSize(0.04).SetColor(RColor::kRed);
   canvas->Draw(draw2);
*/

   std::vector<std::string> labels = {"first", "second", "third", "forth", "fifth"};
   auto draw3 = canvas->Draw<RAxisLabelsDrawable>(RPadPos(0.1_normal,0.5_normal), false, 0.3_normal, labels);
   draw3->AttrAxis().SetTitle("labels");

   auto draw4 = canvas->Draw<RAxisDrawable>(RPadPos(0.1_normal,0.3_normal), false, 0.3_normal);
   draw4->AttrAxis().SetMin(1).SetMax(100).SetLog(10).SetTitle("log10 scale").SetTitleCenter().TitleAttr().SetFont(12).SetColor(RColor::kGreen);

   auto draw5 = canvas->Draw<RAxisDrawable>(RPadPos(0.1_normal,0.1_normal), false, 0.3_normal);
   draw5->AttrAxis().SetMin(0.125).SetMax(128.001).SetLog(2).SetTitle("log2 scale").SetTitleCenter();

   auto draw6 = canvas->Draw<RAxisDrawable>(RPadPos(0.5_normal,0.9_normal), true, -0.8_normal);
   draw6->AttrAxis().SetMin(0).SetMax(10).SetTitle("reverse").SetEndingArrow();

   auto draw7 = canvas->Draw<RAxisDrawable>(RPadPos(0.6_normal,0.9_normal), false, 0.3_normal);
   draw7->AttrAxis().SetMin(0).SetMax(100).SetTicksBoth().SetTicksSize(0.02_normal)
                    .SetTitle("both side ticks").SetTitleCenter();

   auto draw8 = canvas->Draw<RAxisDrawable>(RPadPos(0.6_normal,0.7_normal), false, 0.3_normal);
   draw8->AttrAxis().SetMin(0).SetMax(100).SetTitle("center labels").SetLabelsCenter();

   auto draw11 = canvas->Draw<RAxisDrawable>(RPadPos(0.95_normal,0.1_normal), true, 0.8_normal);
   draw11->AttrAxis().SetMin(0).SetMax(100).SetTitle("vertical axis with arrow").SetTitleCenter()
                     .SetTicksBoth().SetTicksSize(0.01_normal).SetTicksColor(RColor::kBlue)
                     .SetEndingArrow().SetEndingSize(0.01_normal);

   canvas->SetSize(1000, 800);

   // requires Chrome browser, runs in headless mode
   // canvas->SaveAs("axes.png");

   canvas->Show();
}
