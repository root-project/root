// @(#)root/win32gdk:$Id$
// Author: Sergey Linev   23/11/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_Win32Drawable
#define ROOT_Win32Drawable

#include "TVirtualX.h"

#include "Windows4Root.h"
#include "gdk/gdk.h"
#include "gdk/win32/gdkwin32.h"
#include <wchar.h>
#include "xatom.h"
#include <winuser.h>

class TGWin32;

const int kMAXGC = 7,
          kGCline = 0, kGCmark = 1, kGCfill = 2,
          kGCtext = 3, kGCinvt = 4, kGCdash = 5, kGCpxmp = 6;


//////////// internal classes & structures (very private) ////////////////

class Win32Drawable : public TVirtualX::XDrawable {
private:
   template <typename CharType>
   void DrawTextHelper(Int_t x, Int_t y, Float_t angle, Float_t mgn,
                       const CharType *text, TVirtualX::ETextMode mode);
   GdkImage *GetBackground(Int_t x, Int_t y, UInt_t w, UInt_t h);

public:
   TGWin32  &fWin32;              ///< reference to TGWin32 instance
   Int_t    fOpen = 0;             ///< 1 if the window is open, 0 if not
   Int_t    shared = 0;           ///< 1 if Qt window
   Int_t    double_buffer = 0;    ///< 1 if the double buffer is on, 0 if not
   Int_t    ispixmap = 0;         ///< 1 if pixmap, 0 if not
   GdkDrawable *drawing = nullptr;///< drawing area, equal to window or buffer
   GdkDrawable *window = nullptr; ///< win32 window
   GdkDrawable *buffer = nullptr; ///< pixmap used for double buffer
   UInt_t   width = 0;            ///< width of the window
   UInt_t   height = 0;           ///< height of the window
   Int_t    clip = 0;             ///< 1 if the clipping is on
   Int_t    xclip = 0;            ///< x coordinate of the clipping rectangle
   Int_t    yclip = 0;            ///< y coordinate of the clipping rectangle
   UInt_t   wclip = 0;            ///< width of the clipping rectangle
   UInt_t   hclip = 0;            ///< height of the clipping rectangle
   std::vector<ULong_t> new_colors; ///< new image colors for transparency (after processing)
   GdkGC   *fGClist[kMAXGC];      ///< array of GC objects for concrete window
   TVirtualX::EDrawMode drawMode = TVirtualX::kCopy;    ///< current draw mode
   TAttLine  fAttLine = {-1, -1, -1};  ///< current line attributes
   GdkLineStyle lineStyle = GDK_LINE_SOLID; ///< current line style
   Int_t lineWidth = 0;            ///< current line width
   std::vector<gint8> dashList;    ///< Gtk array for dashes
   Int_t     dashLength = 0;       ///< total length of dashes
   Int_t     dashOffset = 0;       ///< current dash offset
   TAttFill  fAttFill = {-1, -1};  ///< current fill attributes
   Int_t     fillHollow = 0;       ///< Flag if fill style is hollow
   Int_t      fillFasi = -1;         ///< fasi parameter for fill pattern
   GdkPixmap *fillPattern = nullptr; ///< current fill pattern
   TAttMarker fAttMarker = { -1, -1, -1 }; ///< current marker attribute
   TAttMarker::EMarkerShape markerType = TAttMarker::kShapeDot; ///< 5 differen kinds of marker
   Int_t     markerSize = 0;          ///< size of simple markers
   std::vector<TPoint> markerShape;   ///< marker shape points
   Int_t markerLineWidth = 0;         ///< line width used for marker
   TAttText fAttText;                 ///< current text attributes

   Win32Drawable(Int_t id, TGWin32 &win32);
   ~Win32Drawable() override;

   Bool_t    IsNull() const override;

   void      SetAttFill(const TAttFill &) override;
   void      SetAttLine(const TAttLine &) override;
   void      SetAttMarker(const TAttMarker &) override;
   void      SetAttText(const TAttText &) override;
   void      SetDrawMode(TVirtualX::EDrawMode) override;
   TVirtualX::EDrawMode GetDrawMode() override { return drawMode; }
   void      ClearWindow() override;
   void      CloseWindow() override;
   void      UpdateWindow(Int_t) override;
   void      SetOpacity(Int_t) override;
   void      CopyPixmap(Int_t, Int_t, Int_t) override;

   void      DrawBox(Int_t, Int_t, Int_t, Int_t, TVirtualX::EBoxMode) override;
   void      DrawFillArea(Int_t, TPoint *) override;
   void      DrawLine(Int_t, Int_t, Int_t, Int_t) override;
   void      DrawPolyLine(Int_t, TPoint *) override;
   void      DrawLinesSegments(Int_t, TPoint *) override;
   void      DrawPolyMarker(Int_t, TPoint *) override;
   void      DrawText(Int_t, Int_t, Float_t, Float_t, const char *, TVirtualX::ETextMode) override;
   void      DrawText(Int_t, Int_t, Float_t, Float_t, const wchar_t *, TVirtualX::ETextMode) override;
   void      DrawTTFglyphs(Int_t, Int_t, TTFhandle &, TVirtualX::ETextMode) override;
   Int_t     WriteGIF(const char *) override;
};


#endif