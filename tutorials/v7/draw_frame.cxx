/// \file
/// \ingroup tutorial_v7
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

auto frame_style = RStyle::Parse("frame { x_ticks_width: 3; y_ticks_width: 3; }"
                                 "title { margin: 0.02; height: 0.1; text_color: blue; text_size: 0.07; }"
                                 "line { line_width: 2; }"
                                 "text { text_align: 13; text_size: 0.03; }");

void draw_frame()
{
   // Create a canvas to be displayed.
   auto canvas = RCanvas::Create("Canvas Title");

   // configure RFrame with direct API calls
   auto frame = canvas->AddFrame();
   // frame->AttrFill().SetColor(RColor::kBlue);
   frame->AttrBorder().SetColor(RColor::kBlue);
   frame->AttrBorder().SetWidth(3);
   frame->AttrMargins().SetTop(0.25_normal);
   frame->AttrMargins().SetAll(0.2_normal);

   // let frame draw axes without need of any histogram
   frame->SetDrawAxes(true);

   frame->AttrX().SetMinMax(0,100);
   // frame->AttrX().SetLog(2.);
   frame->AttrX().SetZoom(5.,95.);
   frame->AttrX().AttrLine().SetColor(RColor::kGreen); // or in CSS "x_line_color: green;"

   frame->AttrY().SetMinMax(0,100);
   frame->AttrY().SetZoom(5,95);
   frame->AttrY().AttrLine().SetColor(RColor::kGreen); // or in CSS "x_line_color: blue;"

   canvas->Draw<RFrameTitle>("Frame title")->SetMargin(0.01_normal).SetHeight(0.1_normal);

   // draw line over the frame
   auto line0 = canvas->Draw<RLine>(RPadPos(100_px, .9_normal), RPadPos(900_px , .9_normal));
   auto text0 = canvas->Draw<RText>(RPadPos(100_px, .9_normal + 5_px), "Line drawn on pad, fix pixel length");
   text0->AttrText().SetAlign(11);

   // draw line under the frame
   auto line1 = canvas->Draw<RLine>(RPadPos(.1_normal, .1_normal), RPadPos(.9_normal, .1_normal));
   auto text1 = canvas->Draw<RText>(RPadPos(.1_normal, .1_normal), "Line drawn on pad, normalized coordinates");

   // draw on left size of frame, but bound to frame, moved with frame
   auto line2 = canvas->Draw<RLine>(RPadPos(-.2_normal, -.1_normal), RPadPos(-.2_normal , 1.1_normal));
   line2->SetOnFrame(true); // or via CSS "onframe: true;"
   line2->AttrLine().SetColor(RColor::kRed);

   auto text2 = canvas->Draw<RText>(RPadPos(-.2_normal - 5_px, -.1_normal), "Line drawn on frame, normalized coordinates");
   text2->SetOnFrame(true); // or via CSS "onframe: true;"
   text2->AttrText().SetAngle(90).SetAlign(11);

   // draw on right size of frame, user coordiante, moved and zoomed with frame
   auto line3 = canvas->Draw<RLine>(RPadPos(110_user, -.1_normal), RPadPos(110_user, 1.1_normal));
   line3->SetOnFrame(true); // or via CSS "onframe: true;"
   line3->AttrLine().SetColor(RColor::kRed);

   auto text3 = canvas->Draw<RText>(RPadPos(110_user, -.1_normal), "Line drawn on frame, user coordinates");
   text3->SetOnFrame(true); // or via CSS "onframe: true;"
   text3->AttrText().SetAngle(90);

   // draw box before line at same position as line ending with 40x40 px size and clipping on
   auto box4 = canvas->Draw<RBox>(RPadPos(80_user - 20_px, 80_user - 20_px), RPadPos(80_user + 20_px, 80_user + 20_px));
   box4->AttrFill().SetColor(RColor::kBlue);
   box4->SetClipping(true); // or via CSS "clipping: true;"
   box4->SetOnFrame(true); // or via CSS "onframe: true;"

   // draw line in the frame, allowed to set user coordinate
   auto line4 = canvas->Draw<RLine>(RPadPos(20_user, 20_user), RPadPos(80_user, 80_user));
   line4->SetClipping(true); // or via CSS "clipping: true;"
   line4->SetOnFrame(true); // or via CSS "onframe: true;"

   auto text4 = canvas->Draw<RText>(RPadPos(20_user, 20_user), "clipping on");
   text4->SetOnFrame(true); // or via CSS "onframe: true;"
   text4->SetClipping(true); // or via CSS "clipping: true;"

   // draw box before line at same position as line ending with 40x40 px size
   auto box5 = canvas->Draw<RBox>(RPadPos(80_user - 20_px, 20_user - 20_px), RPadPos(80_user + 20_px, 20_user + 20_px));
   box5->AttrFill().SetColor(RColor::kYellow);
   box5->SetOnFrame(true); // or via CSS "onframe: true;"

   // draw line in the frame, but disable default cutting by the frame borders
   auto line5 = canvas->Draw<RLine>(RPadPos(20_user, 80_user), RPadPos(80_user, 20_user));
   line5->SetOnFrame(true); // or via CSS "onframe: true;"

   auto text5 = canvas->Draw<RText>(RPadPos(20_user, 80_user), "clipping off");
   text5->SetOnFrame(true); // or via CSS "onframe: true;"
   text5->AttrText().SetAlign(11);

   canvas->UseStyle(frame_style);

   canvas->Show();
}
