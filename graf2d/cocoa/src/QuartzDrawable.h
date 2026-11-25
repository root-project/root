// @(#)root/cocoa:$Id$
// Author: Sergey Linev   25/11/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_QuartzDrawable
#define ROOT_QuartzDrawable

#include "TVirtualX.h"

#include "X11Drawable.h"

class TGCocoa;

/// Description of a Quratz window or pixmap.

class QuartzDrawable : public TVirtualX::XDrawable {
friend class TGCocoa;
protected:
   NSObject<X11Drawable> *GetPixmapDrawable(const char *calledFrom) const;

   template<typename CharType>
   void DrawTextHelper(Int_t x, Int_t y, Float_t angle, Float_t mgn, const CharType *text, TVirtualX::ETextMode mode);

   void ClearDrawable();

public:
   TGCocoa  &fCocoa;                  ///< reference to TGCocoa instance
   NSObject<X11Drawable> *fDrawable = nullptr; ///< pointer on existing drawable class
   TVirtualX::EDrawMode fDrawMode = TVirtualX::kCopy;
   TAttLine  fAttLine = {-1, -1, -1};  ///< current line attributes
   TAttFill  fAttFill = {-1, -1};     ///< current fill attributes
   TAttMarker fAttMarker = { -1, -1, -1 }; ///< current marker attribute
   TAttText fAttText;                 ///< current text attribute

   QuartzDrawable(Int_t id, TGCocoa &cocoa, NSObject<X11Drawable> *drawable);
   ~QuartzDrawable() override;

   Bool_t    IsNull() const override;

   void      SetAttFill(const TAttFill &) override;
   void      SetAttLine(const TAttLine &) override;
   void      SetAttMarker(const TAttMarker &) override;
   void      SetAttText(const TAttText &) override;
   void      SetDrawMode(TVirtualX::EDrawMode) override;
   TVirtualX::EDrawMode GetDrawMode() override;
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