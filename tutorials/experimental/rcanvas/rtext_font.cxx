/// \file
/// \ingroup tutorial_rcanvas
///
/// This macro demonstrate usage of existing ROOT fonts for RText.
/// Also load of custom font is shown
///
/// \macro_image (rcanvas_js)
/// \macro_code
///
/// \date 2021-07-07
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!
/// \author Sergey Linev <s.linev@gsi.de>

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RCanvas.hxx"
#include "ROOT/RText.hxx"
#include "ROOT/RPadPos.hxx"

using namespace ROOT::Experimental;

void rtext_font()
{
   auto canvas = RCanvas::Create("RText fonts example");

   double posy = 0.93;

   auto drawText = [&canvas, &posy](RAttrFont::EFont font, bool is_comic = false) {
      auto text = canvas->Add<RText>(RPadPos(0.35, posy), "ABCDEFGH abcdefgh 0123456789 @#$");
      text->text.size = 0.04;
      text->text.align = RAttrText::kLeftCenter;
      if (is_comic)
         text->text.font.family = "Comic";
      else
         text->text.font = font;

      auto name = canvas->Add<RText>(RPadPos(0.33, posy), text->text.font.GetFullName());
      name->text.size = 0.03;
      name->text.align = RAttrText::kRightCenter;

      posy -= 0.05;
   };

   drawText(RAttrFont::kTimes);
   drawText(RAttrFont::kTimesItalic);
   drawText(RAttrFont::kTimesBold);
   drawText(RAttrFont::kTimesBoldItalic);

   drawText(RAttrFont::kArial);
   drawText(RAttrFont::kArialOblique);
   drawText(RAttrFont::kArialBold);
   drawText(RAttrFont::kArialBoldOblique);

   drawText(RAttrFont::kCourier);
   drawText(RAttrFont::kCourierOblique);
   drawText(RAttrFont::kCourierBold);
   drawText(RAttrFont::kCourierBoldOblique);

   drawText(RAttrFont::kVerdana);
   drawText(RAttrFont::kVerdanaItalic);
   drawText(RAttrFont::kVerdanaBold);
   drawText(RAttrFont::kVerdanaBoldItalic);

   // now draw text with custom font

   posy -= 0.03;
   std::string fname = __FILE__;
   auto pos = fname.find("rtext_font.cxx");
   if (pos > 0) { fname.resize(pos); fname.append("comic.woff2"); }
           else fname = "comic.woff2";
   canvas->Draw<RFont>("Comic", fname);
   drawText(RAttrFont::kTimes, true);

   canvas->Show();
}
