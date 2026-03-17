// @(#)root/graf:$Id$
// Author: Rene Brun   12/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <iostream>
#include "TROOT.h"
#include "Strlen.h"
#include "TWbox.h"
#include "TColor.h"
#include "TStyle.h"
#include "TVirtualPad.h"
#include "TVirtualPadPainter.h"


/** \class TWbox
\ingroup BasicGraphics

A TBox with a bordersize and a bordermode.
Example:
Begin_Macro(source)
{
   TWbox *twb = new TWbox(.1,.1,.9,.9,kRed+2,5,1);
   twb->Draw();
}
End_Macro
*/

////////////////////////////////////////////////////////////////////////////////
/// wbox normal constructor.
///
/// a WBOX is a box with a bordersize and a bordermode
/// the bordersize is in pixels
///  - bordermode = -1 box looks as it is behind the screen
///  - bordermode = 0  no special effects
///  - bordermode = 1  box looks as it is in front of the screen

TWbox::TWbox(Double_t x1, Double_t y1,Double_t x2, Double_t  y2,
             Color_t color ,Short_t bordersize ,Short_t bordermode)
       :TBox(x1,y1,x2,y2)
{
   fBorderSize  = bordersize;
   fBorderMode  = bordermode;
   SetFillColor(color);
   SetFillStyle(1001);
}

////////////////////////////////////////////////////////////////////////////////
/// wbox copy constructor.

TWbox::TWbox(const TWbox &wbox) : TBox(wbox)
{
   wbox.TWbox::Copy(*this);
}

////////////////////////////////////////////////////////////////////////////////
/// assignment operator

TWbox &TWbox::operator=(const TWbox &src)
{
   src.TWbox::Copy(*this);
   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Copy this wbox to wbox.

void TWbox::Copy(TObject &obj) const
{
   TBox::Copy(obj);
   ((TWbox&)obj).fBorderSize  = fBorderSize;
   ((TWbox&)obj).fBorderMode  = fBorderMode;
}

////////////////////////////////////////////////////////////////////////////////
/// Draw this wbox with its current attributes.

void TWbox::Draw(Option_t *option)
{
   AppendPad(option);
}

////////////////////////////////////////////////////////////////////////////////
/// Draw this wbox with new coordinates.

TWbox *TWbox::DrawWbox(Double_t x1, Double_t y1,Double_t x2, Double_t  y2,
                       Color_t color ,Short_t bordersize ,Short_t bordermode)
{
   TWbox *newwbox = new TWbox(x1,y1,x2,y2,color,bordersize,bordermode);
   newwbox->SetBit(kCanDelete);
   newwbox->AppendPad();
   return newwbox;
}

////////////////////////////////////////////////////////////////////////////////
/// Execute action corresponding to one event.
///
///  This member function is called when a WBOX object is clicked.

void TWbox::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
   TBox::ExecuteEvent(event, px, py);
}

////////////////////////////////////////////////////////////////////////////////
/// Paint this wbox with its current attributes.

void TWbox::Paint(Option_t *)
{
   PaintWbox(fX1, fY1, fX2, fY2, GetFillColor(), fBorderSize, fBorderMode);
}

////////////////////////////////////////////////////////////////////////////////
/// Draw this wbox with new coordinates.

void TWbox::PaintWbox(Double_t x1, Double_t y1, Double_t x2, Double_t  y2,
                      Color_t color, Short_t bordersize, Short_t bordermode)
{
   // Draw first wbox as a normal filled box
   TBox::PaintBox(x1, y1, x2, y2);

   // then paint 3d frame (depending on bordermode)
   if (!IsTransparent())
      PaintFrame(x1, y1, x2, y2, color, bordersize, bordermode, kTRUE);
}

////////////////////////////////////////////////////////////////////////////////
/// Paint a 3D frame around a box.

void TWbox::PaintFrame(Double_t x1, Double_t y1,Double_t x2, Double_t  y2,
                       Color_t color, Short_t bordersize, Short_t bordermode,
                       Bool_t /* tops */)
{
   if (bordermode == 0)
      return;

   auto oldcolor = GetFillColor();
   SetFillColor(color);

   PaintBorderOn(gPad, x1, y1, x2, y2, bordersize, bordermode);

   SetFillColor(oldcolor);
}

////////////////////////////////////////////////////////////////////////////////
/// Paint a 3D border around a box.
/// Used also by the pad painter

void TWbox::PaintBorderOn(TVirtualPad *pad,
                          Double_t x1, Double_t y1,Double_t x2 ,Double_t y2,
                          Short_t bordersize, Short_t bordermode, Bool_t with_selection)
{
   if (bordermode == 0)
      return;
   if (bordersize <= 0)
      bordersize = 2;

   auto pp = pad->GetPainter();
   if (!pp)
      return;

   Double_t ww = pad->GetWw(), wh = pad->GetWh();

   if (pp->GetPS()) {
      // SL: need to calculate page size to get real coordiantes for border
      // TODO: Code can be removed if border not need to be exact pixel size
      Float_t xsize = 20, ysize = 26;
      gStyle->GetPaperSize(xsize, ysize);
      Double_t ratio = wh/ww;
      if (xsize * ratio > ysize)
         xsize = ysize/ratio;
      else
         ysize = xsize*ratio;
      ww = 72 / 2.54 * xsize;
      wh = 72 / 2.54 * ysize;
   }

   const Double_t realBsX = bordersize / (pad->GetAbsWNDC() * ww) * (pad->GetX2() - pad->GetX1());
   const Double_t realBsY = bordersize / (pad->GetAbsHNDC() * wh) * (pad->GetY2() - pad->GetY1());

   // GetColorDark() and GetColorBright() use GetFillColor()
   Color_t fillcolor = GetFillColor();
   Color_t light = !fillcolor ? 0 : GetLightColor();
   Color_t dark = !fillcolor ? 0 : GetDarkColor();

   Double_t xl, xt, yl, yt;

   // Compute real left bottom & top right of the box in pixels
   if (pad->XtoPixel(x1) < pad->XtoPixel(x2)) {
      xl = x1;
      xt = x2;
   } else {
      xl = x2;
      xt = x1;
   }
   if (pad->YtoPixel(y1) > pad->YtoPixel(y2)) {
      yl = y1;
      yt = y2;
   } else {
      yl = y2;
      yt = y1;
   }

   Double_t frameXs[7] = {}, frameYs[7] = {};

   // Draw top&left part of the box
   frameXs[0] = xl;           frameYs[0] = yl;
   frameXs[1] = xl + realBsX; frameYs[1] = yl + realBsY;
   frameXs[2] = frameXs[1];   frameYs[2] = yt - realBsY;
   frameXs[3] = xt - realBsX; frameYs[3] = frameYs[2];
   frameXs[4] = xt;           frameYs[4] = yt;
   frameXs[5] = xl;           frameYs[5] = yt;
   frameXs[6] = xl;           frameYs[6] = yl;

   SetFillColor(bordermode == -1 ? dark : light);
   pp->SetAttFill(*this);
   pp->DrawFillArea(7, frameXs, frameYs);

   // Draw bottom&right part of the box
   frameXs[0] = xl;              frameYs[0] = yl;
   frameXs[1] = xl + realBsX;    frameYs[1] = yl + realBsY;
   frameXs[2] = xt - realBsX;    frameYs[2] = frameYs[1];
   frameXs[3] = frameXs[2];      frameYs[3] = yt - realBsY;
   frameXs[4] = xt;              frameYs[4] = yt;
   frameXs[5] = xt;              frameYs[5] = yl;
   frameXs[6] = xl;              frameYs[6] = yl;

   SetFillColor(bordermode == -1 ? light : dark);
   pp->SetAttFill(*this);
   pp->DrawFillArea(7, frameXs, frameYs);

   SetFillColor(fillcolor);

   if (with_selection) {
      Color_t oldlinecolor = GetLineColor();
      SetLineColor(GetFillColor() != 2 ? 2 : 4);
      pp->SetAttLine(*this);
      pp->DrawBox(xl + realBsX, yl + realBsY, xt - realBsX, yt - realBsY, TVirtualPadPainter::kHollow);
      SetLineColor(oldlinecolor);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Save primitive as a C++ statement(s) on output stream out

void TWbox::SavePrimitive(std::ostream &out, Option_t *option)
{
   SavePrimitiveConstructor(out, Class(), "wbox", TString::Format("%g, %g, %g, %g", fX1, fY1, fX2, fY2), kFALSE);

   SaveFillAttributes(out, "wbox", -1, -1);
   SaveLineAttributes(out, "wbox", 1, 1, 1);

   SavePrimitiveDraw(out, "wbox", option);
}
