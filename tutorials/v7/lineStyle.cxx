/// \file
/// \ingroup tutorial_v7
///
/// This ROOT 7 example shows the various line styles.
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
#include "ROOT/RLine.hxx"

void lineStyle()
{
   using namespace ROOT::Experimental;

   auto canvas = RCanvas::Create("Canvas Title");
   double num = 0.3;

   for (int i=10; i>0; i--){
      num = num + 0.05;

      canvas->Draw<RText>(std::to_string(i))->SetPos({.3_normal, 1_normal*num}).SetTextSize(13).SetTextAlign(32).SetFont(52);

      canvas->Draw<RLine>()->SetP1({.32_normal,1_normal*num}).SetP2({.8_normal, 1_normal*num}).SetLineStyle(i);
   }

   canvas->Show();
}
