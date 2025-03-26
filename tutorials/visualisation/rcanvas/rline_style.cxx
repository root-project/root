/// \file
/// \ingroup tutorial_rcanvas
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

void rline_style()
{
   using namespace ROOT::Experimental;

   auto canvas = RCanvas::Create("Different RLine styles");
   auto posy = 0.9_normal;

   for (int i = 1; i <= 11; i++) {
      std::string lbl;
      if (i < 11)
         lbl = std::to_string(i);
      else
         lbl = "Custom";

      auto text = canvas->Add<RText>(RPadPos{.3_normal, posy}, lbl);
      text->text.size = 0.04;
      text->text.align = RAttrText::kRightCenter;
      text->text.font = RAttrFont::kArialOblique;

      auto draw = canvas->Add<RLine>(RPadPos{.32_normal, posy}, RPadPos{.8_normal, posy});
      if (i < 11)
         draw->line.style = (RAttrLine::EStyle) i;
      else
         draw->line.pattern = "10,2,8,2,10,6,1,6";

      posy -= 0.05_normal;
   }

   canvas->Show();
}
