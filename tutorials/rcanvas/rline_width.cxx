/// \file
/// \ingroup tutorial_rcanvas
///
/// This ROOT 7 example shows the various line widths.
///
/// \macro_image (rcanvas_js)
/// \macro_code
///
/// \date 2018-03-18
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!
/// \authors Iliana Betsou, Sergey Linev

#include "ROOT/RCanvas.hxx"
#include "ROOT/RText.hxx"
#include "ROOT/RLine.hxx"

void rline_width()
{
   using namespace ROOT::Experimental;

   auto canvas = RCanvas::Create("Different line widths");
   double num = 0.3;

   for (int i = 10; i > 0; i--){
      num = num + 0.05;

      auto text = canvas->Add<RText>(RPadPos(.3_normal, 1_normal*num), std::to_string(i));
      text->text.size = 13;
      text->text.align = 32;
      text->text.font = 5;

      auto draw = canvas->Add<RLine>(RPadPos(.32_normal, 1_normal*num), RPadPos(.8_normal, 1_normal*num));
      draw->line.width = i;
   }

   canvas->Show();
}
