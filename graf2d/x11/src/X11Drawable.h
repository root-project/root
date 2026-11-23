// @(#)root/x11:$Id$
// Author: Sergey Linev   23/11/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT_X11Drawable
#define ROOT_X11Drawable

#include "TVirtualX.h"

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xatom.h>
#include <X11/cursorfont.h>
#include <X11/keysym.h>
#include <X11/xpm.h>

//
// Primitives Graphic Contexts global for all windows
//
const int kMAXGC = 7,
          kGCline = 0, kGCmark = 1, kGCfill = 2,
          kGCtext = 3, kGCinvt = 4, kGCdash = 5, kGCpxmp = 6;

const int kAlignNone = 0,
          kTLeft = 1,
          kTCenter = 2,
          kTRight = 3,
          kMLeft = 4,
          kMCenter = 5,
          kMRight = 6,
          kBLeft = 7,
          kBCenter = 8,
          kBRight = 9;

class TGX11;


/// Description of a X11 window.
class X11Drawable : public TVirtualX::XDrawable {
public:
   TGX11    &fX11;                    ///< reference to TGX11 instance
   Int_t    fOpen = 0;                ///< 1 if the window is open, 0 if not
   Int_t    fDoubleBuffer = 0;        ///< 1 if the double buffer is on, 0 if not
   Int_t    fIsPixmap = 0;            ///< 1 if pixmap, 0 if not
   Drawable fDrawing = 0;             ///< drawing area, equal to window or buffer
   Drawable fWindow = 0;              ///< X11 window
   Drawable fBuffer = 0;              ///< pixmap used for double buffer
   UInt_t   fWidth = 0;               ///< width of the window
   UInt_t   fHeight = 0;              ///< height of the window
   Int_t    fClip = 0;                ///< 1 if the clipping is on
   Int_t    fXclip = 0;               ///< x coordinate of the clipping rectangle
   Int_t    fYclip = 0;               ///< y coordinate of the clipping rectangle
   UInt_t   fWclip = 0;               ///< width of the clipping rectangle
   UInt_t   fHclip = 0;               ///< height of the clipping rectangle
   std::vector<ULong_t> fNewColors;   ///< extra image colors created for transparency (after processing)
   Bool_t   fShared = 0;              ///< notify when window is shared
   GC       fGClist[kMAXGC];          ///< list of GC object, individual for each window
   TVirtualX::EDrawMode drawMode = TVirtualX::kCopy;    ///< current draw mode
   TAttLine  fAttLine = {-1, -1, -1};  ///< current line attributes
   Int_t     lineWidth = 0;           ///< X11 line width
   Int_t     lineStyle = LineSolid;   ///< X11 line style
   std::vector<char> dashList;        ///< X11 array for dashes
   Int_t     dashLength = 0;          ///< total length of dashes
   Int_t     dashOffset = 0;          ///< current dash offset
   TAttFill  fAttFill = {-1, -1};     ///< current fill attributes
   Int_t     fillHollow = 0;          ///< X11 fill method
   Int_t     fillFasi = 0;            ///< selected fasi pattern
   Pixmap    fillPattern = 0;         ///< current initialized fill pattern
   TAttMarker fAttMarker = { -1, -1, -1 }; ///< current marker attribute
   TAttMarker::EMarkerShape markerType = TAttMarker::kShapeDot; ///< 5 differen kinds of marker
   Int_t     markerSize = 0;          ///< size of simple markers
   std::vector<TPoint> markerShape;   ///< marker shape points
   Int_t     markerLineWidth = 0;     ///< line width used for marker
   TAttText  fAttText;                ///< current text attribute
   Int_t     textAlign = kAlignNone;  ///< selected text align
   XFontStruct *textFont = nullptr;   ///< selected text font

   X11Drawable(Int_t id, TGX11  &x11);
   ~X11Drawable() override;

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
   void      DrawText(Int_t, Int_t, Float_t, Float_t, const wchar_t *, TVirtualX::ETextMode) override {}
   void      DrawTTFglyphs(Int_t, Int_t, TTFhandle &, TVirtualX::ETextMode) override {}
   Int_t     WriteGIF(const char *) override;

};


#endif