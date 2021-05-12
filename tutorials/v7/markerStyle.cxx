/// \file
/// \ingroup tutorial_v7
///
/// This ROOT 7 example shows the various marker style.
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

void markerStyle()
{
   using namespace ROOT::Experimental;

   auto canvas = RCanvas::Create("Canvas Title");
   double num = 0.3;

   double x = 0;
   double dx = 1/16.0;
   for (int i=1;i<16;i++) {
      x += dx;
      for (int row=0;row<3;++row) {
         int style = i;

         if (row==1) style+=19; else if (row==2) style+=34;

         RPadPos pt(RPadLength::Normal(x), .12_normal + 0.3_normal*row);
         canvas->Draw<RText>(pt, std::to_string(style));

         RPadPos pm(RPadLength::Normal(x), .25_normal + 0.3_normal*row);
         canvas->Draw<RMarker>(pm)->AttrMarker().SetStyle(style).SetSize(2.5);
      }
   }

   canvas->Show();
}
