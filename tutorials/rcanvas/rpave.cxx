/// \file
/// \ingroup tutorial_rcanvas
///
/// This macro draw different variants of RPave on the RCanvas.
/// Also usage of custom font is demonstrated.
///
/// \macro_image (rcanvas_js)
/// \macro_code
///
/// \date 2020-06-18
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
/// \author Sergey Linev <s.linev@gsi.de>

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RCanvas.hxx"
#include "ROOT/RPave.hxx"
#include "ROOT/RPaveText.hxx"
#include "ROOT/RFont.hxx"

using namespace ROOT::Experimental;

void rpave()
{
   // Create a canvas to be displayed.
   auto canvas = RCanvas::Create("RPave example");

   // RFrame will be automatically created as well
   auto pave = canvas->Draw<RPave>();
   pave->fill.color = RColor::kBlue;
   pave->border.color = RColor::kGreen;
   pave->border.width = 3;
   pave->cornerY = -0.03_normal;
   pave->height = 0.2_normal;

   auto text = canvas->Draw<RPaveText>();
   text->AddLine("This is RPaveText");
   text->AddLine("It can have several lines");
   text->AddLine("It should be positioned below RPave");
   text->fill.color = RColor::kYellow;
   text->cornerY = 0.25_normal;
   text->height = 0.3_normal;

   std::string fname = __FILE__;
   auto pos = fname.find("draw_pave.cxx");
   if (pos > 0) { fname.resize(pos); fname.append("comic.woff2"); }
           else fname = "comic.woff2";
   canvas->Draw<RFont>("CustomFont", fname);

   auto text2 = canvas->Draw<RPaveText>();
   text2->AddLine("Example of custom font");
   text2->AddLine("It loaded from comic.woff2 file");
   text2->AddLine("One also can provide valid URL");
   text2->fill.color = RColor::kGreen;
   text2->cornerX = -0.4_normal;
   text2->cornerY = -0.03_normal;
   text2->height = 0.2_normal;
   text2->text.font.family = "CustomFont";

   canvas->SetSize(1000, 700);
   canvas->Show();
}
