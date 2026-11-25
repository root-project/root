// @(#)root/graf2d:$Id$
// Author: Olivier Couet, Timur Pocheptsov 23/01/2012


/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//#define NDEBUG

#include <stdexcept>
#include <iostream>
#include <cstring>
#include <cassert>
#include <limits>

#include <Cocoa/Cocoa.h>

#include "QuartzFillArea.h"
#include "TColorGradient.h"
#include "QuartzMarker.h"
#include "CocoaPrivate.h"
#include "QuartzWindow.h"
#include "QuartzPixmap.h"
#include "QuartzUtils.h"
#include "QuartzDrawable.h"
#include "X11Drawable.h"
#include "QuartzText.h"
#include "QuartzLine.h"
#include "CocoaUtils.h"
#include "TTFhandle.h"
#include "TGQuartz.h"
#include "TString.h"
#include "TPoint.h"
#include "TColor.h"
#include "TStyle.h"
#include "TROOT.h"
#include "TMath.h"
#include "TEnv.h"

namespace X11 = ROOT::MacOSX::X11;
namespace Quartz = ROOT::Quartz;
namespace Util = ROOT::MacOSX::Util;

//______________________________________________________________________________
TGQuartz::TGQuartz() : TGCocoa()
{
   //Default ctor.
   if (!TTFhandle::Init())
      Error("TGQuartz", "TTFhandle::Init() failed");

   SetAA();
}


//______________________________________________________________________________
TGQuartz::TGQuartz(const char *name, const char *title)
            : TGCocoa(name, title)
{
   //Constructor.
   if (!TTFhandle::Init())
      Error("TGQuartz", "TTFhandle::Init() failed");

   SetAA();
}

//______________________________________________________________________________
void TGQuartz::DrawBoxW(WinContext_t wctxt, Int_t x1, Int_t y1, Int_t x2, Int_t y2, EBoxMode mode)
{
   if (auto ctxt = (QuartzDrawable *) wctxt)
      ctxt->DrawBox(x1, y1, x2, y2, mode);
}

//______________________________________________________________________________
void TGQuartz::DrawBox(Int_t x1, Int_t y1, Int_t x2, Int_t y2, EBoxMode mode)
{
   if (auto &ctxt = GetSelectedDrawable())
      ctxt->DrawBox(x1, y1, x2, y2, mode);
}

//______________________________________________________________________________
void TGQuartz::DrawFillAreaW(WinContext_t wctxt, Int_t n, TPoint *xy)
{
   if (auto ctxt = (QuartzDrawable *) wctxt)
      ctxt->DrawFillArea(n, xy);
}


//______________________________________________________________________________
void TGQuartz::DrawFillArea(Int_t n, TPoint *xy)
{
   if (auto &ctxt = GetSelectedDrawable())
      ctxt->DrawFillArea(n, xy);
}


//______________________________________________________________________________
void TGQuartz::DrawCellArray(Int_t /*x1*/, Int_t /*y1*/, Int_t /*x2*/, Int_t /*y2*/,
                             Int_t /*nx*/, Int_t /*ny*/, Int_t */*ic*/)
{
   //Noop.
}

//______________________________________________________________________________
void TGQuartz::DrawLineW(WinContext_t wctxt, Int_t x1, Int_t y1, Int_t x2, Int_t y2)
{
   if (auto ctxt = (QuartzDrawable *) wctxt)
      ctxt->DrawLine(x1, y1, x2, y2);
}

//______________________________________________________________________________
void TGQuartz::DrawLine(Int_t x1, Int_t y1, Int_t x2, Int_t y2)
{
   // Draw a line.
   // x1,y1        : begin of line
   // x2,y2        : end of line

   assert(fSelectedDrawable > fPimpl->GetRootWindowID() && "DrawLine, bad drawable is selected");

   if (auto &ctxt = GetSelectedDrawable())
      ctxt->DrawLine(x1, y1, x2, y2);
}


//______________________________________________________________________________
void TGQuartz::DrawPolyLineW(WinContext_t wctxt, Int_t n, TPoint *xy)
{
   if (auto ctxt = (QuartzDrawable *) wctxt)
      ctxt->DrawPolyLine(n, xy);
}

//______________________________________________________________________________
void TGQuartz::DrawPolyLine(Int_t n, TPoint *xy)
{
   //Comment from TVirtualX:
   // Draw a line through all points.
   // n         : number of points
   // xy        : list of points
   //End of comment.

   assert(fSelectedDrawable > fPimpl->GetRootWindowID() && "DrawPolyLine, bad drawable is selected");

   if (auto &ctxt = GetSelectedDrawable())
      ctxt->DrawPolyLine(n, xy);
}

//______________________________________________________________________________
void TGQuartz::DrawLinesSegmentsW(WinContext_t wctxt, Int_t n, TPoint *xy)
{
   if (auto ctxt = (QuartzDrawable *) wctxt)
      ctxt->DrawLinesSegments(n, xy);
}

//______________________________________________________________________________
void TGQuartz::DrawPolyMarkerW(WinContext_t wctxt, Int_t n, TPoint *xy)
{
   if (auto ctxt = (QuartzDrawable *) wctxt)
      ctxt->DrawPolyMarker(n, xy);
}

//______________________________________________________________________________
void TGQuartz::DrawPolyMarker(Int_t n, TPoint *xy)
{
   //Comment from TVirtualX:
   // Draw PolyMarker
   // n         : number of points
   // xy        : list of points
   //End of comment.

   if (auto &ctxt = GetSelectedDrawable())
      ctxt->DrawPolyMarker(n, xy);
}


//______________________________________________________________________________
void TGQuartz::DrawTextW(WinContext_t wctxt, Int_t x, Int_t y, Float_t angle, Float_t mgn,
                         const char *text, ETextMode mode)
{
   if (auto ctxt = (QuartzDrawable *) wctxt)
      ctxt->DrawText(x, y, angle, mgn, text, mode);
}

//______________________________________________________________________________
void TGQuartz::DrawText(Int_t x, Int_t y, Float_t angle, Float_t mgn,
                        const char *text, ETextMode mode)
{
   if (auto &ctxt = GetSelectedDrawable())
      ctxt->DrawText(x, y, angle, mgn, text, mode);
}

//______________________________________________________________________________
void TGQuartz::DrawTextW(WinContext_t wctxt, Int_t x, Int_t y, Float_t angle, Float_t mgn,
                         const wchar_t *text, ETextMode mode)
{
   if (auto ctxt = (QuartzDrawable *) wctxt)
      ctxt->DrawText(x, y, angle, mgn, text, mode);
}

//______________________________________________________________________________
void TGQuartz::DrawTTFglyphsW(WinContext_t wctxt, Int_t x1, Int_t y1, TTFhandle &ttf, ETextMode mode)
{
   if (auto ctxt = (QuartzDrawable *) wctxt)
      ctxt->DrawTTFglyphs(x1, y1, ttf, mode);
}

//______________________________________________________________________________
void TGQuartz::DrawText(Int_t x, Int_t y, Float_t angle, Float_t mgn,
                        const wchar_t *text, ETextMode mode)
{
   if (auto &ctxt = GetSelectedDrawable())
      ctxt->DrawText(x, y, angle, mgn, text, mode);
}


//______________________________________________________________________________
Float_t TGQuartz::GetTextMagnitude()
{
   // Returns the current font magnification factor
   return 0;
}

//______________________________________________________________________________
void TGQuartz::SetLineColor(Color_t cindex)
{
   // Set color index "cindex" for drawing lines.
   TAttLine::SetLineColor(cindex);

   if (auto &ctxt = GetSelectedDrawable())
      ctxt->SetAttLine(*this);
}


//______________________________________________________________________________
void TGQuartz::SetLineStyle(Style_t lstyle)
{
   // Set line style.
   TAttLine::SetLineStyle(lstyle);

   if (auto &ctxt = GetSelectedDrawable())
      ctxt->SetAttLine(*this);
}


//______________________________________________________________________________
void TGQuartz::SetLineWidth(Width_t width)
{
   // Set the line width.

   TAttLine::SetLineWidth(width);
   if (auto &ctxt = GetSelectedDrawable())
      ctxt->SetAttLine(*this);
}


//______________________________________________________________________________
void TGQuartz::SetFillColor(Color_t cindex)
{
   // Set color index "cindex" for fill areas.

   TAttFill::SetFillColor(cindex);
   if (auto &ctxt = GetSelectedDrawable())
      ctxt->SetAttFill(*this);
}


//______________________________________________________________________________
void TGQuartz::SetFillStyle(Style_t style)
{
   // Set fill area style.
   TAttFill::SetFillStyle(style);

   if (auto &ctxt = GetSelectedDrawable())
      ctxt->SetAttFill(*this);
}


//______________________________________________________________________________
void TGQuartz::SetMarkerColor(Color_t cindex)
{
   // Set color index "cindex" for markers.
   TAttMarker::SetMarkerColor(cindex);

   if (auto &ctxt = GetSelectedDrawable())
      ctxt->SetAttMarker(*this);
}


//______________________________________________________________________________
void TGQuartz::SetMarkerSize(Float_t markersize)
{
   // Set marker size index.
   //
   // markersize - the marker scale factor
   TAttMarker::SetMarkerSize(markersize);

   if (auto &ctxt = GetSelectedDrawable())
      ctxt->SetAttMarker(*this);
}


//______________________________________________________________________________
void TGQuartz::SetMarkerStyle(Style_t markerstyle)
{
   // Set marker style.

   TAttMarker::SetMarkerStyle(markerstyle);

   if (auto &ctxt = GetSelectedDrawable())
      ctxt->SetAttMarker(*this);
}


//______________________________________________________________________________
void TGQuartz::SetTextAlign(Short_t talign)
{
   // Set the text alignment.

   TAttText::SetTextAlign(talign);

   if (auto &ctxt = GetSelectedDrawable())
      ctxt->SetAttText(*this);
}

//______________________________________________________________________________
void TGQuartz::SetTextColor(Color_t cindex)
{
   // Set the color index "cindex" for text.

   TAttText::SetTextColor(cindex);

   if (auto &ctxt = GetSelectedDrawable())
      ctxt->SetAttText(*this);
}


//______________________________________________________________________________
void TGQuartz::SetTextFont(Font_t fontNumber)
{
   // Set the current text font number.

   TAttText::SetTextFont(fontNumber);

   if (auto &ctxt = GetSelectedDrawable())
      ctxt->SetAttText(*this);
}

//______________________________________________________________________________
Int_t TGQuartz::SetTextFont(char * /* fontName */, ETextSetMode /* mode */)
{
   Error("SetTextFont", "Direct TTF font setting not supported");
   return 1;
}

//______________________________________________________________________________
void TGQuartz::SetTextSize(Float_t textsize)
{
   // Set the current text size to "textsize"

   TAttText::SetTextSize(textsize);

   if (auto &ctxt = GetSelectedDrawable())
      ctxt->SetAttText(*this);
}

//______________________________________________________________________________
void TGQuartz::SetOpacity(Int_t /*percent*/)
{
   // Set opacity of the current window. This image manipulation routine
   // works by adding to a percent amount of neutral to each pixels RGB.
   // Since it requires quite some additional color map entries is it
   // only supported on displays with more than > 8 color planes (> 256
   // colors).
}

//______________________________________________________________________________
void TGQuartz::SetOpacityW(WinContext_t /* wctxt */, Int_t /* percent */)
{
}

//______________________________________________________________________________
void TGQuartz::SetAttFill(WinContext_t wctxt, const TAttFill &att)
{
   if (auto ctxt = (QuartzDrawable *)wctxt)
      ctxt->SetAttFill(att);

   // TODO: remove this after transition done
   TAttFill::SetFillColor(att.GetFillColor());
   TAttFill::SetFillStyle(att.GetFillStyle());
}

//______________________________________________________________________________
void TGQuartz::SetAttLine(WinContext_t wctxt, const TAttLine &att)
{
   if (auto ctxt = (QuartzDrawable *)wctxt)
      ctxt->SetAttLine(att);

   // TODO: remove this after transition done
   TAttLine::SetLineColor(att.GetLineColor());
   TAttLine::SetLineStyle(att.GetLineStyle());
   TAttLine::SetLineWidth(att.GetLineWidth());
}

//______________________________________________________________________________
void TGQuartz::SetAttMarker(WinContext_t wctxt, const TAttMarker &att)
{
   if (auto ctxt = (QuartzDrawable *)wctxt)
      ctxt->SetAttMarker(att);

   // TODO: remove this after transition done
   TAttMarker::SetMarkerColor(att.GetMarkerColor());
   TAttMarker::SetMarkerSize(att.GetMarkerSize());
   TAttMarker::SetMarkerStyle(att.GetMarkerStyle());
}

//______________________________________________________________________________
void TGQuartz::SetAttText(WinContext_t wctxt, const TAttText &att)
{
   if (auto ctxt = (QuartzDrawable *)wctxt)
      ctxt->SetAttText(att);

   // TODO: remove this after transition done
   TAttText::SetTextAlign(att.GetTextAlign());
   TAttText::SetTextAngle(att.GetTextAngle());
   TAttText::SetTextColor(att.GetTextColor());
   TAttText::SetTextSize(att.GetTextSize());
   TAttText::SetTextFont(att.GetTextFont());
}

//TTF related part.


//Aux. functions.

//______________________________________________________________________________
void TGQuartz::SetAA()
{
   if (gEnv) {
      const TString value(TString(gEnv->GetValue("Cocoa.EnableAntiAliasing", "auto")).Strip());
      if (value == "auto") {
         [[NSScreen mainScreen] backingScaleFactor] > 1. ? fUseAA = true : fUseAA = false;
      } else if (value == "no")
         fUseAA = false;
      else {
         assert(value == "yes" && "SetAA, value must be 'yes', 'no' or 'auto'");
         fUseAA = true;
      }
      const TString valuefa(TString(gEnv->GetValue("Cocoa.EnableFillAreaAntiAliasing", "auto")).Strip());
      if (valuefa == "auto") {
         [[NSScreen mainScreen] backingScaleFactor] > 1. ? fUseFAAA = true : fUseFAAA = false;
      } else if (valuefa == "no")
         fUseFAAA = false;
      else {
         assert(valuefa == "yes" && "SetAA, value must be 'yes', 'no' or 'auto'");
         fUseFAAA = true;
      }
   }
}

//______________________________________________________________________________
TAttFill &TGQuartz::GetAttFill(WinContext_t wctxt)
{
   // attributes stored in direct drawable (view) and not in underlying pixmap
   auto drawable = (NSObject<X11Drawable> *) wctxt;
   if (!drawable || !drawable.attFill)
      return *this;
   return *drawable.attFill;
}

//______________________________________________________________________________
TAttLine &TGQuartz::GetAttLine(WinContext_t wctxt)
{
   // attributes stored in direct drawable (view) and not in underlying pixmap
   auto drawable = (NSObject<X11Drawable> *) wctxt;
   if (!drawable || !drawable.attLine)
      return *this;
   return *drawable.attLine;
}

//______________________________________________________________________________
TAttMarker &TGQuartz::GetAttMarker(WinContext_t wctxt)
{
   // attributes stored in direct drawable (view) and not in underlying pixmap
   auto drawable = (NSObject<X11Drawable> *) wctxt;
   if (!drawable || !drawable.attMarker)
      return *this;
   return *drawable.attMarker;
}

//______________________________________________________________________________
TAttText &TGQuartz::GetAttText(WinContext_t wctxt)
{
   // attributes stored in direct drawable (view) and not in underlying pixmap
   auto drawable = (NSObject<X11Drawable> *) wctxt;
   if (!drawable || !drawable.attText)
      return *this;
   return *drawable.attText;
}

//______________________________________________________________________________
void *TGQuartz::GetPixmapDrawable(void *drawable0, const char *calledFrom) const
{
   assert(calledFrom != 0 && "GetDrawableChecked, calledFrom parameter is null");

   if (!drawable0)
      return nullptr;

   auto drawable = (NSObject<X11Drawable> *) drawable0;
   if (!drawable.fIsPixmap) {
      //TPad/TCanvas ALWAYS draw only into a pixmap.
      if ([drawable isKindOfClass : [QuartzView class]]) {
         QuartzView *view = (QuartzView *)drawable;
         if (!view.fBackBuffer) {
            Error(calledFrom, "Selected window is not double buffered");
            return nullptr;
         }

         drawable = view.fBackBuffer;
      } else {
         Error(calledFrom, "Selected drawable is neither a pixmap, nor a double buffered window");
         return nullptr;
      }
   }

   if (!drawable.fContext) {
      Error(calledFrom, "Context is null");
      return nullptr;
   }

   return drawable;
}
