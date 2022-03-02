/// \file
/// \ingroup tutorial_rcanvas
///
/// This ROOT 7 example shows how to create a frame.
///
/// \macro_image (rcanvas_js)
/// \macro_code
///
/// \date 2020-02-20
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!
/// \author Sergey Linev <s.linev@gsi.de>

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RCanvas.hxx"
#include "ROOT/RColor.hxx"
#include "ROOT/RFrameTitle.hxx"
#include "ROOT/RFrame.hxx"
#include "ROOT/RStyle.hxx"
#include "ROOT/RLine.hxx"
#include "ROOT/RText.hxx"
#include "ROOT/RBox.hxx"

using namespace ROOT::Experimental;

auto rframe_style = RStyle::Parse("frame { x_ticks_width: 3; y_ticks_width: 3; }"
                                 "title { margin: 0.02; height: 0.1; text_color: blue; text_size: 0.07; }"
                                 "line { line_width: 2; }"
                                 "text { text_align: 13; text_size: 0.03; }");

void rframe()
{
   // Create a canvas to be displayed.
   auto canvas = RCanvas::Create("RFrame with drawAxes enabled");

   // configure RFrame with direct API calls
   auto frame = canvas->AddFrame();
   // frame->fill.color = RColor::kBlue;
   // frame->fill.style = RAttrFill::kSolid;
   frame->border.color = RColor::kBlue;
   frame->border.width = 3;
   frame->margins = 0.2_normal; // set all margins first
   frame->margins.top = 0.25_normal;

   // let frame draw axes without need of any histogram
   frame->drawAxes = true;

   frame->x.min = 0;
   frame->x.max = 100;
   // frame->x.log = 2.;
   frame->x.zoomMin = 5.;
   frame->x.zoomMax = 95.;
   frame->x.line.color = RColor::kGreen; // or in CSS "x_line_color: green;"

   frame->y.min = 0;
   frame->y.max = 100;
   frame->y.zoomMin = 5;
   frame->y.zoomMax = 95;
   frame->y.line.color = RColor::kBlue; // or in CSS "y_line_color: blue;"

   auto title = canvas->Draw<RFrameTitle>("Frame title");
   title->margin = 0.01_normal;
   title->height = 0.1_normal;

   // draw line over the frame
   auto line0 = canvas->Draw<RLine>(RPadPos(100_px, .9_normal), RPadPos(900_px , .9_normal));
   auto text0 = canvas->Draw<RText>(RPadPos(100_px, .9_normal + 5_px), "Line drawn on pad, fix pixel length");
   text0->text.align = RAttrText::kLeftBottom;

   // draw line under the frame
   auto line1 = canvas->Draw<RLine>(RPadPos(.1_normal, .1_normal), RPadPos(.9_normal, .1_normal));
   auto text1 = canvas->Draw<RText>(RPadPos(.1_normal, .1_normal), "Line drawn on pad, normalized coordinates");

   // draw on left size of frame, but bound to frame, moved with frame
   auto line2 = canvas->Draw<RLine>(RPadPos(-.2_normal, -.1_normal), RPadPos(-.2_normal , 1.1_normal));
   line2->onFrame = true; // or via CSS "onFrame: true;"
   line2->line.color = RColor::kRed;

   auto text2 = canvas->Draw<RText>(RPadPos(-.2_normal - 5_px, -.1_normal), "Line drawn on frame, normalized coordinates");
   text2->onFrame = true; // or via CSS "onFrame: true;"
   text2->text.angle = 90;
   text2->text.align = RAttrText::kLeftBottom;

   // draw on right size of frame, user coordiante, moved and zoomed with frame
   auto line3 = canvas->Draw<RLine>(RPadPos(110_user, -.1_normal), RPadPos(110_user, 1.1_normal));
   line3->onFrame = true; // or via CSS "onFrame: true;"
   line3->line.color = RColor::kRed;

   auto text3 = canvas->Draw<RText>(RPadPos(110_user, -.1_normal), "Line drawn on frame, user coordinates");
   text3->onFrame = true; // or via CSS "onFrame: true;"
   text3->text.angle = 90;

   // draw box before line at same position as line ending with 40x40 px size and clipping on
   auto box4 = canvas->Draw<RBox>(RPadPos(80_user - 20_px, 80_user - 20_px), RPadPos(80_user + 20_px, 80_user + 20_px));
   box4->fill.color = RColor::kBlue;
   box4->fill.style = RAttrFill::kSolid;
   box4->clipping = true; // or via CSS "clipping: true;"
   box4->onFrame = true; // or via CSS "onFrame: true;"

   // draw line in the frame, allowed to set user coordinate
   auto line4 = canvas->Draw<RLine>(RPadPos(20_user, 20_user), RPadPos(80_user, 80_user));
   line4->clipping = true; // or via CSS "clipping: true;"
   line4->onFrame = true; // or via CSS "onFrame: true;"

   auto text4 = canvas->Draw<RText>(RPadPos(20_user, 20_user), "clipping on");
   text4->onFrame = true; // or via CSS "onFrame: true;"
   text4->clipping = true; // or via CSS "clipping: true;"

   // draw box before line at same position as line ending with 40x40 px size
   auto box5 = canvas->Draw<RBox>(RPadPos(80_user - 20_px, 20_user - 20_px), RPadPos(80_user + 20_px, 20_user + 20_px));
   box5->fill.color = RColor::kYellow;
   box5->fill.style = RAttrFill::kSolid;
   box5->onFrame = true; // or via CSS "onFrame: true;"

   // draw line in the frame, but disable default cutting by the frame borders
   auto line5 = canvas->Draw<RLine>(RPadPos(20_user, 80_user), RPadPos(80_user, 20_user));
   line5->onFrame = true; // or via CSS "onFrame: true;"

   auto text5 = canvas->Draw<RText>(RPadPos(20_user, 80_user), "clipping off");
   text5->onFrame = true; // or via CSS "onFrame: true;"
   text5->text.align = RAttrText::kLeftBottom;

   canvas->UseStyle(rframe_style);

   canvas->Show();
}
