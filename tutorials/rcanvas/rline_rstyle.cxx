/// \file
/// \ingroup tutorial_v7
///
/// This ROOT 7 example demonstrates how to customize RLine object using RStyle
/// "normal" coordinates' system.
///
/// \macro_image (rcanvas_js)
/// \macro_code
///
/// \date 2019-10-04
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!
/// \author Iliana Betsou

#include "ROOT/RCanvas.hxx"
#include "ROOT/RText.hxx"
#include "ROOT/RLine.hxx"
#include "ROOT/RDirectory.hxx"
#include <string>

using namespace ROOT::Experimental;

void lineRStyle()
{
   auto canvas = RCanvas::Create("Canvas Title");
   double num = 0.3;

   for (int i = 10; i > 0; i--){
      num = num + 0.05;

      auto text = canvas->Add<RText>(RPadPos{.3_normal, 1_normal*num}, std::to_string(i));
      text->text.size = 13;
      text->text.align = 32;
      text->text.font = 5;

      auto line = canvas->Add<RLine>(RPadPos(.32_normal,1_normal*num), RPadPos(.8_normal, 1_normal*num));
      line->SetId(std::string("obj") + std::to_string(i));
      line->SetCssClass(std::string("user_class_") + std::to_string(i % 3));
   }

   auto style = std::make_shared<RStyle>();

   style->AddBlock(".user_class_1").AddInt("line_style", 4); // all lines with user_class_1 should get style 1
   style->AddBlock(".user_class_2").AddDouble("line_width", 5.); // all lines with user_class_2 should get line width 5
   style->AddBlock("#obj7").AddString("line_color", "#0000FF"); // line with id obj7 should be blue

   style->AddBlock("line").AddString("line_color", "red"); // all lines should get red color

   canvas->UseStyle(style);

   canvas->Show();

   // after leaving function scope style will be destroyed, one has to preserve it
   RDirectory::Heap().Add("lineRStyle", style);
}
