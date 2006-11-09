// @(#)root/gui:$Name:  $:$Id: TGLabel.cxx,v 1.29 2006/07/26 13:36:43 rdm Exp $
// Author: Fons Rademakers   06/01/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
/**************************************************************************

    This source is based on Xclass95, a Win95-looking GUI toolkit.
    Copyright (C) 1996, 1997 David Barth, Ricky Ralston, Hector Peraza.

    Xclass95 is free software; you can redistribute it and/or
    modify it under the terms of the GNU Library General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

**************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLabel                                                              //
//                                                                      //
// This class handles GUI labels.                                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGLabel.h"
#include "TGWidget.h"
#include "TGString.h"
#include "TGResourcePool.h"
#include "Riostream.h"
#include "TColor.h"
#include "TClass.h"


const TGFont *TGLabel::fgDefaultFont = 0;
const TGGC   *TGLabel::fgDefaultGC = 0;

ClassImp(TGLabel)

//______________________________________________________________________________
TGLabel::TGLabel(const TGWindow *p, TGString *text, GContext_t norm,
                 FontStruct_t font, UInt_t options, ULong_t back) :
    TGFrame(p, 1, 1, options, back)
{
   // Create a label GUI object. TGLabel will become the owner of the
   // text and will delete it in its dtor.

   fText        = text;
   fTMode       = kTextCenterX | kTextCenterY;
   fTextChanged = kTRUE;
   fFontStruct  = font;
   fNormGC      = norm;
   fHasOwnFont  = kFALSE;
   fDisabled    = kFALSE;

   int max_ascent, max_descent;
   fTWidth  = gVirtualX->TextWidth(fFontStruct, fText->GetString(), fText->GetLength());
   gVirtualX->GetFontProperties(fFontStruct, max_ascent, max_descent);
   fTHeight = max_ascent + max_descent;
   Resize(fTWidth, fTHeight + 1);
   SetWindowName();
}

//______________________________________________________________________________
TGLabel::TGLabel(const TGWindow *p, const char *text, GContext_t norm,
                 FontStruct_t font, UInt_t options, ULong_t back) :
    TGFrame(p, 1, 1, options, back)
{
   // Create a label GUI object.

   fText        = new TGString(!text && !p ? GetName() : text);
   fTMode       = kTextCenterX | kTextCenterY;
   fTextChanged = kTRUE;
   fFontStruct  = font;
   fNormGC      = norm;
   fHasOwnFont  = kFALSE;
   fDisabled    = kFALSE;

   int max_ascent, max_descent;
   fTWidth  = gVirtualX->TextWidth(fFontStruct, fText->GetString(), fText->GetLength());
   gVirtualX->GetFontProperties(fFontStruct, max_ascent, max_descent);
   fTHeight = max_ascent + max_descent;
   Resize(fTWidth, fTHeight + 1);
   SetWindowName();
}

//______________________________________________________________________________
TGLabel::~TGLabel()
{
   // Delete label.

   if (fText) delete fText;

   if (fHasOwnFont) {
      TGGCPool *pool = fClient->GetGCPool();
      TGGC *gc = pool->FindGC(fNormGC);
      pool->FreeGC(gc);
   }
}

//______________________________________________________________________________
void TGLabel::SetText(TGString *new_text)
{
   // Set new text in label. After calling this method one needs to call
   // the parents frame's Layout() method to force updating of the label size.
   // The new_text is adopted by the TGLabel and will be properly deleted.

   if (fText) delete fText;
   fText        = new_text;
   fTextChanged = kTRUE;

   int max_ascent, max_descent;

   fTWidth = gVirtualX->TextWidth(fFontStruct, fText->GetString(), fText->GetLength());
   gVirtualX->GetFontProperties(fFontStruct, max_ascent, max_descent);
   fTHeight = max_ascent + max_descent;

   // Resize is done when parent's is Layout() is called
   //Resize(fTWidth, fTHeight + 1);
   fClient->NeedRedraw(this);
}

//______________________________________________________________________________
void TGLabel::DoRedraw()
{
   // Redraw label widget.

   int x, y;

   TGFrame::DoRedraw();

   if (fTextChanged) {
      fTextChanged = kFALSE;
   }

   if (fTMode & kTextLeft)
      x = 0;
   else if (fTMode & kTextRight)
      x = fWidth - fTWidth;
   else
      x = (fWidth - fTWidth) >> 1;

   if (fTMode & kTextTop)
      y = 0;
   else if (fTMode & kTextBottom)
      y = fHeight - fTHeight;
   else
      y = (fHeight - fTHeight) >> 1;

   int max_ascent, max_descent;
   gVirtualX->GetFontProperties(fFontStruct, max_ascent, max_descent);

   if (!fDisabled) {
      fText->Draw(fId, fNormGC, x, y + max_ascent);
   } else {
      FontH_t fontH;
      if (GetDefaultFontStruct() != fFontStruct) {
         fontH = gVirtualX->GetFontHandle(fFontStruct);
      } else {
         fontH = gVirtualX->GetFontHandle(GetDefaultFontStruct());
      }
      static TGGC *gc1 = 0;
      static TGGC *gc2 = 0;

      if (!gc1) {
         gc1 = fClient->GetResourcePool()->GetGCPool()->FindGC(GetHilightGC()());
         gc1 = new TGGC(*gc1); // copy
      }
      gc1->SetFont(fontH);
      fText->Draw(fId, gc1->GetGC(), x + 1, y + 1 + max_ascent);

      if (!gc2) {
         gc2 = fClient->GetResourcePool()->GetGCPool()->FindGC(GetShadowGC()());
         gc2 = new TGGC(*gc2); // copy
      }
      gc2->SetFont(fontH);
      fText->Draw(fId, gc2->GetGC(), x, y + max_ascent);
   }
}

//______________________________________________________________________________
void TGLabel::SetTextFont(FontStruct_t font, Bool_t global)
{
   // Changes text font.
   // If global is true font is changed globally - otherwise locally.

   FontH_t v = gVirtualX->GetFontHandle(font);
   if (!v) return;

   fTextChanged = kTRUE;

   fFontStruct = font;

   TGGCPool *pool =  fClient->GetResourcePool()->GetGCPool();
   TGGC *gc = pool->FindGC(fNormGC);

   if (!global) {
      gc = pool->GetGC((GCValues_t*)gc->GetAttributes(), kTRUE); // copy
      fHasOwnFont = kTRUE;
   }

   gc->SetFont(v);
   fNormGC = gc->GetGC();

   int max_ascent, max_descent;

   fTWidth  = gVirtualX->TextWidth(fFontStruct, fText->GetString(), fText->GetLength());
   gVirtualX->GetFontProperties(fFontStruct, max_ascent, max_descent);
   fTHeight = max_ascent + max_descent;

   // Resize is done when parent's is Layout() is called
   //Resize(fTWidth, fTHeight + 1);
   fClient->NeedRedraw(this);
}

//______________________________________________________________________________
void TGLabel::SetTextFont(const char *fontName, Bool_t global)
{
   // Changes text font specified by name.
   // If global is true font is changed globally - otherwise locally.

   TGFont *font = fClient->GetFont(fontName);

   if (font) {
      SetTextFont(font->GetFontStruct(), global);
   }
}

//______________________________________________________________________________
void TGLabel::SetTextFont(TGFont *font, Bool_t global)
{
   // Changes text font specified by pointer to TGFont object.
   // If global is true font is changed globally - otherwise locally.

   if (font) {
      SetTextFont(font->GetFontStruct(), global);
   }
}

//______________________________________________________________________________
void TGLabel::SetTextColor(Pixel_t color, Bool_t global)
{
   // Changes text color.
   // If global is true color is changed globally - otherwise locally.

   TGGCPool *pool =  fClient->GetResourcePool()->GetGCPool();
   TGGC *gc = pool->FindGC(fNormGC);

   if (!global) {
      gc = pool->GetGC((GCValues_t*)gc->GetAttributes(), kTRUE); // copy
      fHasOwnFont = kTRUE;
   }

   gc->SetForeground(color);
   fNormGC = gc->GetGC();

   fClient->NeedRedraw(this);
}

//______________________________________________________________________________
void TGLabel::SetTextColor(TColor *color, Bool_t global)
{
   // Changes text color.
   // If global is true color is changed globally - otherwise locally.

   if (color) {
      SetTextColor(color->GetPixel(), global);
   }
}

//______________________________________________________________________________
void TGLabel::SetTextJustify(Int_t mode)
{
   // Set text justification. Mode is an OR of the bits:
   // kTextTop, kTextBottom, kTextLeft, kTextRight, kTextCenterX and
   // kTextCenterY.

   fTextChanged = kTRUE;
   fTMode = mode;

   fClient->NeedRedraw(this);
}

//______________________________________________________________________________
Bool_t TGLabel::HasOwnFont() const
{
   // Returns kTRUE if text attributes are unique.
   // Returns kFALSE if text attributes are shared (global).

   return fHasOwnFont;
}

//______________________________________________________________________________
void TGLabel::SavePrimitive(ostream &out, Option_t *option /*= ""*/)
{
   // Save a label widget as a C++ statement(s) on output stream out.

   char quote = '"';

   // font + GC
   option = GetName()+5;         // unique digit id of the name
   char parGC[50], parFont[50];
   sprintf(parFont,"%s::GetDefaultFontStruct()",IsA()->GetName());
   sprintf(parGC,"%s::GetDefaultGC()()",IsA()->GetName());
   if ((GetDefaultFontStruct() != fFontStruct) || (GetDefaultGC()() != fNormGC)) {
      TGFont *ufont = fClient->GetResourcePool()->GetFontPool()->FindFont(fFontStruct);
      if (ufont) {
         ufont->SavePrimitive(out, option);
         sprintf(parFont,"ufont->GetFontStruct()");
      }

      TGGC *userGC = fClient->GetResourcePool()->GetGCPool()->FindGC(fNormGC);
      if (userGC) {
         userGC->SavePrimitive(out, option);
         sprintf(parGC,"uGC->GetGC()");
      }
   }

   if (fBackground != GetDefaultFrameBackground()) SaveUserColor(out, option);

   TString label = GetText()->GetString();
   label.ReplaceAll("\"","\\\"");

   out << "   TGLabel *";
   out << GetName() << " = new TGLabel("<< fParent->GetName()
       << "," << quote << label << quote;
   if (fBackground == GetDefaultFrameBackground()) {
      if (!GetOptions()) {
         if (fFontStruct == GetDefaultFontStruct()) {
            if (fNormGC == GetDefaultGC()()) {
               out <<");" << endl;
            } else {
               out << "," << parGC << ");" << endl;
            }
         } else {
            out << "," << parGC << "," << parFont << ");" << endl;
         }
      } else {
         out << "," << parGC << "," << parFont << "," << GetOptionString() <<");" << endl;
      }
   } else {
      out << "," << parGC << "," << parFont << "," << GetOptionString() << ",ucolor);" << endl;
   }

   if (fDisabled)
      out << "   " << GetName() << "->Disable();" << endl;

   out << "   " << GetName() << "->SetTextJustify(" <<  GetTextJustify() << ");" << endl;
}

//______________________________________________________________________________
FontStruct_t TGLabel::GetDefaultFontStruct()
{
   // Static returning label default font struct.

   if (!fgDefaultFont)
      fgDefaultFont = gClient->GetResourcePool()->GetDefaultFont();
   return fgDefaultFont->GetFontStruct();
}

//______________________________________________________________________________
const TGGC &TGLabel::GetDefaultGC()
{
   // Static returning label default graphics context.

   if (!fgDefaultGC)
      fgDefaultGC = gClient->GetResourcePool()->GetFrameGC();
   return *fgDefaultGC;
}
