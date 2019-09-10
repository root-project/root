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

void lineWidthNew()
{
   using namespace ROOT::Experimental;

   auto canvas = RCanvas::Create("Canvas Title");
   double num = 0.3;

   auto style = std::make_shared<RStyleNew>();

   style->AddBlock("line").AddInt("line_style", 2).AddString("line_color", "yellow");

   for (int i=10; i>0; i--){
      num = num + 0.05;

      // current way to draw primitives - it always returns drawing options
      RPadPos pt(.3_normal, RPadLength::Normal(num));
      auto optts = canvas->Draw(RText(pt, std::to_string(i)));
      optts->SetSize(13).SetAlign(32).SetFont(52);

      RPadPos pl1(.32_normal, RPadLength::Normal(num));
      RPadPos pl2(.8_normal , RPadLength::Normal(num));


      // new way of drawing - return drawable itself, which has all kind attributes
      auto line = canvas->DrawNew<RLineNew>(pl1, pl2);

      // access only line attributes - create so called "visitor" for RDrawable attributes
      auto attr = line->AttrLine();

      // change attributes
      attr.SetWidth(i);

      // evaluate attributes using value stored in drawable itself
      printf("Before applying RStyle: width %f style %d color %s\n", attr.GetWidth(), attr.GetStyle(), attr.GetColor().c_str());

      // assign style object to the attributes - now "Visitor" can use style as well
      attr.UseStyle(style);

      printf("Afetr applying RStyle: width %f style %d color %s\n", attr.GetWidth(), attr.GetStyle(), attr.GetColor().c_str());
   }

   // drawing of RLineNew not implemented, but attributes can be seen in JSON like:
   // of course, idea to have it much compact in JSON. For instance, I could "equipt"

/*
  "fAttr" : {
        "_typename" : "ROOT::Experimental::RDrawableAttributes",
        "fContIO" : {
          "_typename" : "ROOT::Experimental::RDrawableAttributes::Record_t",
          "user_class" : "",
          "map" : {"_typename": "unordered_map<string,ROOT::Experimental::RDrawableAttributes::Value_t*>", "line_width": {
            "_typename" : "ROOT::Experimental::RDrawableAttributes::DoubleValue_t",
            "v" : 2
          }}
        }
      }
 */

   canvas->Show();
}
