/// \file
/// \ingroup tutorial_v7
///
/// \macro_code
///
/// \date 2018-03-18
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!
/// \author Iliana Betsou

#include "ROOT/RCanvas.hxx"
#include "ROOT/RColor.hxx"
#include "ROOT/RText.hxx"
#include "ROOT/RLine.hxx"
#include "ROOT/RPadPos.hxx"
#include <string>

void lineWidth()
{
   using namespace ROOT::Experimental;

   auto canvas = RCanvas::Create("Canvas Title");
   double num = 0.3;

   for (int i=10; i>0; i--){
      num = num + 0.05;

      canvas->Draw<RText>(RPadPos(.3_normal, 1_normal*num), std::to_string(i))->AttrText().SetSize(13).SetAlign(32).SetFont(52);

      canvas->Draw<RLine>(RPadPos(.32_normal, 1_normal*num), RPadPos(.8_normal , 1_normal*num))->AttrLine().SetWidth(i);
   }

   canvas->Show();
}
