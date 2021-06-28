/// \file
/// \ingroup tutorial_v7
///
/// This ROOT 7 example shows the various line widths.
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

void lineWidth()
{
   using namespace ROOT::Experimental;

   auto canvas = RCanvas::Create("Canvas Title");
   double num = 0.3;

   for (int i=10; i>0; i--){
      num = num + 0.05;

      // one can create object ourself
      canvas->Draw(std::make_shared<RText>(RPadPos(.3_normal, 1_normal*num), std::to_string(i)))->AttrText().SetSize(13).SetAlign(32).SetFont(52);

      // or let it create by templated Draw<T> method
      auto draw = canvas->Draw<RLine>(RPadPos(.32_normal, 1_normal*num), RPadPos(.8_normal , 1_normal*num));
      draw->line.width = i;
   }

   canvas->Show();
}
