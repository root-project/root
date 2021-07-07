/// \file
/// \ingroup tutorial_rcanvas
///
/// This macro demonstrate the text align attribute for RText.
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
#include "ROOT/RColor.hxx"
#include "ROOT/RText.hxx"
#include "ROOT/RBox.hxx"
#include "ROOT/RPadPos.hxx"

using namespace ROOT::Experimental;

void rtext_align()
{
   auto canvas = RCanvas::Create("RText align example");

   auto box = canvas->Add<RBox>(RPadPos(0.1_normal, 0.1_normal), RPadPos(0.9_normal, 0.9_normal));
   box->border.style = 6;

   auto drawText = [&canvas](double x, double y, RAttrText::EAlign align, const std::string &lbl) {
      auto dbox = canvas->Add<RBox>(RPadPos(x-0.003, y-0.003), RPadPos(x+0.003, y+0.003));
      dbox->fill.color = RColor::kRed;
      dbox->fill.style = RAttrFill::kSolid;

      auto text = canvas->Add<RText>(RPadPos(x, y), lbl);
      text->text.size = 0.07;
      text->text.align = align;
   };

   drawText(0.1, 0.9, RAttrText::kLeftTop, "kLeftTop");

   drawText(0.1, 0.5, RAttrText::kLeftCenter, "kLeftCenter");

   drawText(0.1, 0.1, RAttrText::kLeftBottom, "kLeftBottom");

   drawText(0.9, 0.9, RAttrText::kRightTop, "kRightTop");

   drawText(0.9, 0.5, RAttrText::kRightCenter, "kRightCenter");

   drawText(0.9, 0.1, RAttrText::kRightBottom, "kRightBottom");

   drawText(0.5, 0.5, RAttrText::kCenter, "kCenter");

   drawText(0.5, 0.9, RAttrText::kCenterTop, "kCenterTop");

   drawText(0.5, 0.1, RAttrText::kCenterBottom, "kCenterBottom");

   canvas->Show();
}
