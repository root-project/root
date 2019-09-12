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

   style->AddBlock("line").AddInt("line_style", 2).AddString("line_color_rgb", "4,5,6");


   for (int i=10; i>0; i--){
      num = num + 0.05;

      // current way to draw primitives - it always returns drawing options
      RPadPosNew pt(RPadLengthNew::Normal(0.32), RPadLengthNew::Normal(num));
      auto text = canvas->DrawNew<RTextNew>(pt, std::to_string(i));
      text->AttrText().SetSize(13).SetAlign(32).SetFont(52);

      RPadPosNew pl1(RPadLengthNew::Normal(0.32), RPadLengthNew::Normal(num));
      RPadPosNew pl2(RPadLengthNew::Normal(0.8), RPadLengthNew::Normal(num));

      printf("pl1  horiz %f vert %f\n", pl1.Horiz().GetNormal(),  pl1.Vert().GetNormal());

      // new way of drawing - return drawable itself, which has all kind attributes
      auto line = canvas->DrawNew<RLineNew>(pl1, pl2);

      printf("line.P1  horiz %f vert %f\n", line->GetP1().Horiz().GetNormal(),  line->GetP1().Vert().GetNormal());

      // change line width
      line->AttrLine().SetWidth(i);

      printf("Current line width %f\n", line->AttrLine().GetWidth());

      // copy attributes
      auto attr = line->AttrLine();

      printf("Attribute line width %f\n", attr.GetWidth());

      // change attributes
      attr.SetWidth(i+1);

      line->AttrLine().Copy(attr);

      printf("Now line width %f\n", line->AttrLine().GetWidth());

      // assign style object to the attribute - now "Visitor" can use style as well
      line->AttrLine().UseStyle(style);

      printf("Afetr applying RStyle: width %f style %d color %s\n", line->AttrLine().GetWidth(), line->AttrLine().GetStyle(), line->AttrLine().Color().AsSVG().c_str());

      line->AttrLine().Color() = RColorNew(i+5, i+15, i+25);

      printf("Afetr setting color color %s\n", line->AttrLine().Color().AsSVG().c_str());

      auto col = line->AttrLine().Color();

      printf("Extracting color %s has alfa %s\n", col.AsSVG().c_str(), col.HasAlfa() ? "true" : "false");

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
