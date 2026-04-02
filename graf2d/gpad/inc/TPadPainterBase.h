// @(#)root/gpad:$Id$
// Author: Sergey Linev   1/04/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPadPainterBase
#define ROOT_TPadPainterBase

#include "TVirtualPadPainter.h"
#include "TAttFill.h"
#include "TAttLine.h"
#include "TAttMarker.h"
#include "TAttText.h"

class TPadPainterBase : public TVirtualPadPainter {
protected:
   TAttFill   fAttFill;   ///< current fill attributes
   TAttLine   fAttLine;   ///< current line attributes
   TAttMarker fAttMarker; ///< current marker attributes
   TAttText   fAttText;   ///< current text attributes

public:

   /// _____________________________________________________________________
   /// old methods only for backward compatibility

   //Line attributes to be set up in TPad.
   Color_t  GetLineColor() const override { return fAttLine.GetLineColor(); }
   Style_t  GetLineStyle() const override { return fAttLine.GetLineStyle(); }
   Width_t  GetLineWidth() const override { return fAttLine.GetLineWidth(); }

   void     SetLineColor(Color_t lcolor) override { fAttLine.SetLineColor(lcolor); SetAttLine(fAttLine); }
   void     SetLineStyle(Style_t lstyle) override { fAttLine.SetLineStyle(lstyle); SetAttLine(fAttLine); }
   void     SetLineWidth(Width_t lwidth) override { fAttLine.SetLineWidth(lwidth); SetAttLine(fAttLine); }

   //Fill attributes to be set up in TPad.
   Color_t  GetFillColor() const  override { return fAttFill.GetFillColor(); }
   Style_t  GetFillStyle() const override { return fAttFill.GetFillStyle(); }
   Bool_t   IsTransparent() const override { return fAttFill.IsTransparent(); }

   void     SetFillColor(Color_t fcolor) override { fAttFill.SetFillColor(fcolor); SetAttFill(fAttFill); }
   void     SetFillStyle(Style_t fstyle) override { fAttFill.SetFillStyle(fstyle); SetAttFill(fAttFill); }

   //Text attributes.
   Short_t  GetTextAlign() const override { return fAttText.GetTextAlign(); }
   Float_t  GetTextAngle() const override { return fAttText.GetTextAngle(); }
   Color_t  GetTextColor() const override { return fAttText.GetTextColor(); }
   Font_t   GetTextFont() const override { return fAttText.GetTextFont(); }
   Float_t  GetTextSize() const override { return fAttText.GetTextSize(); }
   Float_t  GetTextMagnitude() const override { return 1.; }

   void     SetTextAlign(Short_t align) override { fAttText.SetTextAlign(align); SetAttText(fAttText); }
   void     SetTextAngle(Float_t tangle) override { fAttText.SetTextAngle(tangle); SetAttText(fAttText); }
   void     SetTextColor(Color_t tcolor) override { fAttText.SetTextColor(tcolor); SetAttText(fAttText); }
   void     SetTextFont(Font_t tfont) override { fAttText.SetTextFont(tfont); SetAttText(fAttText); }
   void     SetTextSize(Float_t tsize) override { fAttText.SetTextSize(tsize); SetAttText(fAttText); }
   void     SetTextSizePixels(Int_t npixels) override { fAttText.SetTextSizePixels(npixels); SetAttText(fAttText); }

   //Marker attributes
   Color_t  GetMarkerColor() const override { return fAttMarker.GetMarkerColor(); }
   Style_t  GetMarkerStyle() const override { return fAttMarker.GetMarkerStyle(); }
   Size_t   GetMarkerSize()  const override { return fAttMarker.GetMarkerSize(); }

   void     SetMarkerColor(Color_t mcolor) override { fAttMarker.SetMarkerColor(mcolor); SetAttMarker(fAttMarker); }
   void     SetMarkerStyle(Style_t mstyle) override { fAttMarker.SetMarkerStyle(mstyle); SetAttMarker(fAttMarker); }
   void     SetMarkerSize(Size_t msize) override { fAttMarker.SetMarkerSize(msize); SetAttMarker(fAttMarker); }

  /// _____________________________________________________________________

   const TAttFill    &GetAttFill() const override { return fAttFill; }
   const TAttLine    &GetAttLine() const override { return fAttLine; }
   const TAttMarker  &GetAttMarker()const override { return fAttMarker; }
   const TAttText    &GetAttText() const override { return fAttText; }

   void SetAttFill(const TAttFill &att) override
   {
      if (&att != &fAttFill)
         att.Copy(fAttFill);
   }

   void SetAttLine(const TAttLine &att) override
   {
      if (&att != &fAttLine)
         att.Copy(fAttLine);
   }

   void SetAttMarker(const TAttMarker &att) override
   {
      if (&att != &fAttMarker)
         att.Copy(fAttMarker);
   }

   void SetAttText(const TAttText &att) override
   {
      if (&att != &fAttText)
         att.Copy(fAttText);
   }

   ClassDefOverride(TPadPainterBase, 0)//Pad painter with attributes handling
};

#endif
