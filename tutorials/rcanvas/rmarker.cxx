/// \file
/// \ingroup tutorial_rcanvas
///
/// This ROOT 7 example shows the various marker styles.
///
/// \macro_image (rcanvas_js)
/// \macro_code
///
/// \date 2018-03-18
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!
/// \author Iliana Betsou

#include "ROOT/RCanvas.hxx"
#include "ROOT/RText.hxx"
#include "ROOT/RMarker.hxx"
#include <string>

void rmarker()
{
   using namespace ROOT::Experimental;

   auto canvas = RCanvas::Create("RMarker styles");
   double num = 0.3;

   double x = 0;
   double dx = 1/16.0;
   for (int i = 1; i < 16; i++) {
      x += dx;
      for (int row=0;row<3;++row) {
         int style = i;

         if (row == 1)
            style += 19;
         else if (row==2)
            style += 34;

         RPadPos pt(RPadLength::Normal(x), .17_normal + 0.3_normal*row);
         auto text = canvas->Draw<RText>(pt, std::to_string(style));
         text->text.font = RAttrFont::kVerdana;
         text->text.size = 0.05;
         text->text.align = RAttrText::kCenterTop;
         text->text.color = RColor::kGreen;

         RPadPos pm(RPadLength::Normal(x), .25_normal + 0.3_normal*row);
         auto draw = canvas->Draw<RMarker>(pm);
         draw->marker.style = (RAttrMarker::EStyle) style;
         draw->marker.color = RColor::kBlue;
         draw->marker.size = 2.5;
      }
   }

   canvas->Show();
}
