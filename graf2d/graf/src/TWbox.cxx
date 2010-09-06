// @(#)root/graf:$Id$
// Author: Rene Brun   12/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "Riostream.h"
#include "TROOT.h"
#include "Strlen.h"
#include "TWbox.h"
#include "TColor.h"
#include "TVirtualPad.h"
#include "TVirtualX.h"
#include "TPoint.h"


ClassImp(TWbox)


//______________________________________________________________________________
//
// a TWbox is a TBox with a bordersize and a bordermode.
//
//Begin_Html
/*
<img src="gif/wbox.gif">
*/
//End_Html


//______________________________________________________________________________
TWbox::TWbox(): TBox()
{
   // wbox default constructor.

   fBorderSize  = 0;
   fBorderMode  = 0;
}


//______________________________________________________________________________
TWbox::TWbox(Double_t x1, Double_t y1,Double_t x2, Double_t  y2,
             Color_t color ,Short_t bordersize ,Short_t bordermode)
       :TBox(x1,y1,x2,y2)
{
   // wbox normal constructor.
   //
   // a WBOX is a box with a bordersize and a bordermode
   // the bordersize is in pixels
   // bordermode = -1 box looks as it is behind the screen
   // bordermode = 0  no special effects
   // bordermode = 1  box looks as it is in front of the screen

   fBorderSize  = bordersize;
   fBorderMode  = bordermode;
   SetFillColor(color);
   SetFillStyle(1001);
}


//______________________________________________________________________________
TWbox::~TWbox()
{
   // wbox default destructor.
}


//______________________________________________________________________________
TWbox::TWbox(const TWbox &wbox) : TBox(wbox)
{
   // wbox copy constructor.

   fBorderSize  = 0;
   fBorderMode  = 0;
   ((TWbox&)wbox).Copy(*this);
}


//______________________________________________________________________________
void TWbox::Copy(TObject &obj) const
{
   // Copy this wbox to wbox.

   TBox::Copy(obj);
   ((TWbox&)obj).fBorderSize  = fBorderSize;
   ((TWbox&)obj).fBorderMode  = fBorderMode;
}


//______________________________________________________________________________
void TWbox::Draw(Option_t *option)
{
   // Draw this wbox with its current attributes.

   AppendPad(option);
}


//______________________________________________________________________________
void TWbox::DrawWbox(Double_t x1, Double_t y1,Double_t x2, Double_t  y2,
                     Color_t color ,Short_t bordersize ,Short_t bordermode)
{
   // Draw this wbox with new coordinates.

   TWbox *newwbox = new TWbox(x1,y1,x2,y2,color,bordersize,bordermode);
   newwbox->SetBit(kCanDelete);
   newwbox->AppendPad();
}


//______________________________________________________________________________
void TWbox::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
   // Execute action corresponding to one event.
   //
   //  This member function is called when a WBOX object is clicked.

   TBox::ExecuteEvent(event, px, py);
}


//______________________________________________________________________________
void TWbox::Paint(Option_t *)
{
   // Paint this wbox with its current attributes.

   PaintWbox(fX1, fY1, fX2, fY2, GetFillColor(), fBorderSize, fBorderMode);
}


//______________________________________________________________________________
void TWbox::PaintWbox(Double_t x1, Double_t y1, Double_t x2, Double_t  y2,
                      Color_t color, Short_t bordersize, Short_t bordermode)
{
   // Draw this wbox with new coordinates.

   // Draw first wbox as a normal filled box
   TBox::PaintBox(x1, y1, x2, y2);

   // then paint 3d frame (depending on bordermode)
   if (!IsTransparent())
      PaintFrame(x1, y1, x2, y2, color, bordersize, bordermode, kTRUE);
}


//______________________________________________________________________________
void TWbox::PaintFrame(Double_t x1, Double_t y1,Double_t x2, Double_t  y2,
                       Color_t color, Short_t bordersize, Short_t bordermode,
                       Bool_t tops)
{
   // Paint a 3D frame around a box.

   if (bordermode == 0) return;
   if (bordersize <= 0) bordersize = 2;

   Short_t pxl,pyl,pxt,pyt,px1,py1,px2,py2;
   Double_t xl, xt, yl, yt;

   // Compute real left bottom & top right of the box in pixels
   px1 = gPad->XtoPixel(x1);   py1 = gPad->YtoPixel(y1);
   px2 = gPad->XtoPixel(x2);   py2 = gPad->YtoPixel(y2);
   if (px1 < px2) {pxl = px1; pxt = px2; xl = x1; xt = x2; }
   else           {pxl = px2; pxt = px1; xl = x2; xt = x1;}
   if (py1 > py2) {pyl = py1; pyt = py2; yl = y1; yt = y2;}
   else           {pyl = py2; pyt = py1; yl = y2; yt = y1;}

   if (!gPad->IsBatch()) {
      TPoint frame[7];

      // GetDarkColor() and GetLightColor() use GetFillColor()
      Color_t oldcolor = GetFillColor();
      SetFillColor(color);
      TAttFill::Modify();

      // Draw top&left part of the box
      frame[0].fX = pxl;                 frame[0].fY = pyl;
      frame[1].fX = pxl + bordersize;    frame[1].fY = pyl - bordersize;
      frame[2].fX = frame[1].fX;         frame[2].fY = pyt + bordersize;
      frame[3].fX = pxt - bordersize;    frame[3].fY = frame[2].fY;
      frame[4].fX = pxt;                 frame[4].fY = pyt;
      frame[5].fX = pxl;                 frame[5].fY = pyt;
      frame[6].fX = pxl;                 frame[6].fY = pyl;

      if (bordermode == -1) gVirtualX->SetFillColor(GetDarkColor());
      else                  gVirtualX->SetFillColor(GetLightColor());
      gVirtualX->DrawFillArea(7, frame);

      // Draw bottom&right part of the box
      frame[0].fX = pxl;                 frame[0].fY = pyl;
      frame[1].fX = pxl + bordersize;    frame[1].fY = pyl - bordersize;
      frame[2].fX = pxt - bordersize;    frame[2].fY = frame[1].fY;
      frame[3].fX = frame[2].fX;         frame[3].fY = pyt + bordersize;
      frame[4].fX = pxt;                 frame[4].fY = pyt;
      frame[5].fX = pxt;                 frame[5].fY = pyl;
      frame[6].fX = pxl;                 frame[6].fY = pyl;

      if (bordermode == -1) gVirtualX->SetFillColor(TColor::GetColorBright(GetFillColor()));
      else                  gVirtualX->SetFillColor(TColor::GetColorDark(GetFillColor()));
      gVirtualX->DrawFillArea(7, frame);

      gVirtualX->SetFillColor(-1);
      SetFillColor(oldcolor);
   }

   if (!tops) return;

   // same for PostScript
   // Double_t dx   = (xt - xl) *Double_t(bordersize)/Double_t(pxt - pxl);
   // Int_t border = gVirtualPS->XtoPS(xt) - gVirtualPS->XtoPS(xt-dx);

   gPad->PaintBorderPS(xl, yl, xt, yt, bordermode, bordersize,
                         GetDarkColor(), GetLightColor());
}


//______________________________________________________________________________
void TWbox::SavePrimitive(ostream &out, Option_t * /*= ""*/)
{
   // Save primitive as a C++ statement(s) on output stream out

   if (gROOT->ClassSaved(TWbox::Class())) {
      out<<"   ";
   } else {
      out<<"   TWbox *";
   }
   out<<"wbox = new TWbox("<<fX1<<","<<fY1<<","<<fX2<<","<<fY2<<");"<<endl;

   SaveFillAttributes(out,"wbox",0,1001);
   SaveLineAttributes(out,"wbox",1,1,1);

   out<<"   wbox->Draw();"<<endl;
}
