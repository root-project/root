/// \file
/// \ingroup tutorial_rcanvas
///
/// This macro demonstrate the text attributes for RText. Angle, size and color are
/// changed in a loop. The text alignment and the text font are fixed.
///
/// \macro_image (rcanvas_js)
/// \macro_code
///
/// \date 2017-10-17
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!
/// \author Olivier Couet <Olivier.Couet@cern.ch>

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RCanvas.hxx"
#include "ROOT/RColor.hxx"
#include "ROOT/RText.hxx"
#include "ROOT/RPadPos.hxx"

using namespace ROOT::Experimental;

void rtext_angle()
{
   // Create a canvas to be displayed.
   auto canvas = RCanvas::Create("RText angle example");

   for (double angle = 0; angle <= 360; angle += 10) {
      auto draw = canvas->Draw<RText>(RPadPos(0.5_normal, 0.6_normal), "____  Hello World");

      draw->text.color = RColor((int) (0.38*angle), (int) (0.64*angle), (int) (0.76*angle));
      draw->text.size = 0.01 + angle/5000.;
      draw->text.angle = angle;
      draw->text.align = RAttrText::kLeftTop;
      draw->text.font = RAttrFont::kArial;
   }

   canvas->Show();
}
