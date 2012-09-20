// @(#)root/gui:$Id$
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
#include "TGString.h"
#include "TGWidget.h"
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
   fHasOwnFont  = kFALSE;
   fDisabled    = kFALSE;
   f3DStyle     = 0;
   fWrapLength  = -1;
   fTFlags      = 0;
   fMLeft = fMRight = fMTop = fMBottom = 0;

   if (!norm) {
      norm = GetDefaultGC().GetGC();
   }
   fNormGC = norm;

   if (!font) {
      font = fgDefaultFont->GetFontStruct();
   } 

   fFont = fClient->GetFontPool()->GetFont(font);
   fTLayout = fFont->ComputeTextLayout(fText->GetString(), fText->GetLength(),
                                        fWrapLength, kTextLeft, fTFlags,
                                        &fTWidth, &fTHeight);

   SetWindowAttributes_t wattr;
   wattr.fMask = kWAWinGravity | kWABitGravity;
   wattr.fBitGravity = 5; // center
   wattr.fWinGravity = 1;
   gVirtualX->ChangeWindowAttributes(fId, &wattr);

   Resize();
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
   fHasOwnFont  = kFALSE;
   fDisabled    = kFALSE;
   f3DStyle     = 0;
   fWrapLength  = -1;
   fTFlags      = 0;
   fMLeft = fMRight = fMTop = fMBottom = 0;

   if (!norm) {
      norm = GetDefaultGC().GetGC();
   }
   fNormGC = norm;

   if (!font) {
      font = fgDefaultFont->GetFontStruct();
   }

   fFont = fClient->GetFontPool()->GetFont(font);
   fTLayout = fFont->ComputeTextLayout(fText->GetString(), fText->GetLength(),
                                       fWrapLength, kTextLeft, fTFlags,
                                       &fTWidth, &fTHeight);

   SetWindowAttributes_t wattr;
   wattr.fMask = kWAWinGravity | kWABitGravity;
   wattr.fBitGravity = 5; // center
   wattr.fWinGravity = 1;
   gVirtualX->ChangeWindowAttributes(fId, &wattr);

   Resize();
   SetWindowName();
}

//______________________________________________________________________________
TGLabel::~TGLabel()
{
   // Delete label.

   if (fText) {
      delete fText;
   }

   if (fHasOwnFont) {
      TGGCPool *pool = fClient->GetGCPool();
      TGGC *gc = pool->FindGC(fNormGC);
      if (gc) pool->FreeGC(gc);
   }

   if (fFont != fgDefaultFont) {
      fClient->GetFontPool()->FreeFont(fFont);
   }

   delete fTLayout;
}

//______________________________________________________________________________
void TGLabel::Layout()
{
   // Layout label.

   delete fTLayout;
   fTLayout = fFont->ComputeTextLayout(fText->GetString(), fText->GetLength(),
                                       fWrapLength, kTextLeft, fTFlags,
                                       &fTWidth, &fTHeight);
   fClient->NeedRedraw(this);
}

//______________________________________________________________________________
TGDimension TGLabel::GetDefaultSize() const
{
   // Return default size.

   UInt_t w = GetOptions() & kFixedWidth ? fWidth : fTWidth + fMLeft + fMRight;
   UInt_t h = GetOptions() & kFixedHeight ? fHeight : fTHeight + fMTop + fMBottom + 1;
   return TGDimension(w, h);
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

   Layout();
}

//______________________________________________________________________________
void TGLabel::DrawText(GContext_t gc, Int_t x, Int_t y)
{
   // Draw text at position (x, y).

   fTLayout->DrawText(fId, gc, x, y, 0, -1);
   //fTLayout->UnderlineChar(fId, gc, x, y, fText->GetHotPos());
}

//______________________________________________________________________________
void TGLabel::DoRedraw()
{
   // Redraw label widget.

   int x, y;

   TGFrame::DoRedraw();
   fTextChanged = kFALSE;

   if (fTMode & kTextLeft) {
      x = fMLeft;
   } else if (fTMode & kTextRight) {
      x = fWidth - fTWidth - fMRight;
   } else {
      x = (fWidth - fTWidth + fMLeft - fMRight) >> 1;
   }

   if (fTMode & kTextTop) {
      y = 0;
   } else if (fTMode & kTextBottom) {
      y = fHeight - fTHeight;
   } else {
      y = (fHeight - fTHeight) >> 1;
   }


   if (!fDisabled) {
      TGGCPool *pool = fClient->GetResourcePool()->GetGCPool();
      TGGC *gc = pool->FindGC(fNormGC);

      if (!gc) {
         fNormGC = GetDefaultGC().GetGC();
         gc = pool->FindGC(fNormGC);
      }
      if (!gc) return;

      switch (f3DStyle) {
         case kRaisedFrame:
         case kSunkenFrame:
            {
               Pixel_t forecolor = gc->GetForeground();
               Pixel_t hi = TGFrame::GetWhitePixel();
               Pixel_t sh = forecolor;

               if (f3DStyle == kRaisedFrame) {
                  Pixel_t t = hi;
                  hi = sh; 
                  sh = t;
               }

               gc->SetForeground(hi);
               DrawText(gc->GetGC(), x+1, y+1);
               gc->SetForeground(sh);
               DrawText(gc->GetGC(), x, y);
               gc->SetForeground(forecolor);
            }
            break;

         default:
            DrawText(fNormGC, x, y);
            break;
      }
   } else { // disabled
      FontH_t fontH;

      if (GetDefaultFontStruct() != fFont->GetFontStruct()) {
         fontH = gVirtualX->GetFontHandle(fFont->GetFontStruct());
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
      DrawText(gc1->GetGC(), x + 1, y + 1);

      if (!gc2) {
         gc2 = fClient->GetResourcePool()->GetGCPool()->FindGC(GetShadowGC()());
         gc2 = new TGGC(*gc2); // copy
      }
      gc2->SetFont(fontH);
      DrawText(gc2->GetGC(), x, y);
   }
}

//______________________________________________________________________________
void TGLabel::SetTextFont(FontStruct_t fontStruct, Bool_t global)
{
   // Changes text font.
   // If global is true font is changed globally - otherwise locally.

   TGFont *font = fClient->GetFontPool()->GetFont(fontStruct);

   if (!font) {
      //error
      return;
   }

   SetTextFont(font, global);
}

//______________________________________________________________________________
void TGLabel::SetTextFont(const char *fontName, Bool_t global)
{
   // Changes text font specified by name.
   // If global is true font is changed globally - otherwise locally.

   TGFont *font = fClient->GetFont(fontName);

   if (!font) {
      // error
      return;
   }

   SetTextFont(font, global);
}

//______________________________________________________________________________
void TGLabel::SetTextFont(TGFont *font, Bool_t global)
{
   // Changes text font specified by pointer to TGFont object.
   // If global is true font is changed globally - otherwise locally.

   if (!font) {
      //error
      return;
   }

   TGFont *oldfont = fFont;
   fFont = fClient->GetFont(font);  // increase usage count
   if (!fFont) {
      fFont = oldfont;
      return;
   }

   TGGCPool *pool = fClient->GetResourcePool()->GetGCPool();
   TGGC *gc = pool->FindGC(fNormGC);

   if (!global) {
      if (gc == &GetDefaultGC() ) { // create new GC
         gc = pool->GetGC((GCValues_t*)gc->GetAttributes(), kTRUE); // copy ctor.
      }
      fHasOwnFont = kTRUE;
   }
   if (oldfont != fgDefaultFont) {
      fClient->FreeFont(oldfont);
   }
   if (gc) {
      gc->SetFont(fFont->GetFontHandle());
      fNormGC = gc->GetGC();
   }
   fTextChanged = kTRUE;
   Layout();
}

//______________________________________________________________________________
void TGLabel::SetTextColor(Pixel_t color, Bool_t global)
{
   // Changes text color.
   // If global is true color is changed globally - otherwise locally.

   TGGCPool *pool = fClient->GetResourcePool()->GetGCPool();
   TGGC *gc = pool->FindGC(fNormGC);

   if (!global) {
      if (gc == &GetDefaultGC() ) {
         gc = pool->GetGC((GCValues_t*)gc->GetAttributes(), kTRUE); // copy
      }
      fHasOwnFont = kTRUE;
   }
   if (gc) {
      gc->SetForeground(color);
      fNormGC = gc->GetGC();
   }
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

   SetWindowAttributes_t wattr;
   wattr.fMask = kWAWinGravity | kWABitGravity;
   wattr.fWinGravity = 1;

   switch (mode) {
      case kTextTop | kTextLeft:
         wattr.fBitGravity = 1; //NorthWestGravity
         break;
      case kTextTop | kTextCenterX:
      case kTextTop:
         wattr.fBitGravity = 2; //NorthGravity
         break;
      case kTextTop | kTextRight:
         wattr.fBitGravity = 3; //NorthEastGravity
         break;
      case kTextLeft | kTextCenterY:
      case kTextLeft:
         wattr.fBitGravity = 4; //WestGravity
         break;
      case kTextCenterY | kTextCenterX:
         wattr.fBitGravity = 5; //CenterGravity
         break;
      case kTextRight | kTextCenterY:
      case kTextRight:
         wattr.fBitGravity = 6; //EastGravity
         break;
      case kTextBottom | kTextLeft:
         wattr.fBitGravity = 7; //SouthWestGravity
         break;
      case kTextBottom | kTextCenterX:
      case kTextBottom:
         wattr.fBitGravity = 8; //SouthGravity
         break;
      case kTextBottom | kTextRight:
         wattr.fBitGravity = 9; //SouthEastGravity
         break;
      default:
         wattr.fBitGravity = 5; //CenterGravity
         break;
   }

   gVirtualX->ChangeWindowAttributes(fId, &wattr);

   Layout();
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
   TString parGC, parFont;
   parFont.Form("%s::GetDefaultFontStruct()",IsA()->GetName());
   parGC.Form("%s::GetDefaultGC()()",IsA()->GetName());

   if ((GetDefaultFontStruct() != fFont->GetFontStruct()) || (GetDefaultGC()() != fNormGC)) {
      TGFont *ufont = fClient->GetResourcePool()->GetFontPool()->FindFont(fFont->GetFontStruct());
      if (ufont) {
         ufont->SavePrimitive(out, option);
         parFont.Form("ufont->GetFontStruct()");
      }

      TGGC *userGC = fClient->GetResourcePool()->GetGCPool()->FindGC(fNormGC);
      if (userGC) {
         userGC->SavePrimitive(out, option);
         parGC.Form("uGC->GetGC()");
      }
   }

   if (fBackground != GetDefaultFrameBackground()) SaveUserColor(out, option);

   TString label = GetText()->GetString();
   label.ReplaceAll("\"","\\\"");
   label.ReplaceAll("\n","\\n");

   out << "   TGLabel *";
   out << GetName() << " = new TGLabel("<< fParent->GetName()
       << "," << quote << label << quote;
   if (fBackground == GetDefaultFrameBackground()) {
      if (!GetOptions()) {
         if (fFont->GetFontStruct() == GetDefaultFontStruct()) {
            if (fNormGC == GetDefaultGC()()) {
               out <<");" << endl;
            } else {
               out << "," << parGC.Data() << ");" << endl;
            }
         } else {
            out << "," << parGC.Data() << "," << parFont.Data() << ");" << endl;
         }
      } else {
         out << "," << parGC.Data() << "," << parFont.Data() << "," << GetOptionString() <<");" << endl;
      }
   } else {
      out << "," << parGC.Data() << "," << parFont.Data() << "," << GetOptionString() << ",ucolor);" << endl;
   }
   if (option && strstr(option, "keep_names"))
      out << "   " << GetName() << "->SetName(\"" << GetName() << "\");" << endl;

   if (fDisabled)
      out << "   " << GetName() << "->Disable();" << endl;

   out << "   " << GetName() << "->SetTextJustify(" <<  GetTextJustify() << ");" << endl;
   out << "   " << GetName() << "->SetMargins(" << fMLeft << "," << fMRight << ",";
   out << fMTop << "," << fMBottom << ");" << endl;
   out << "   " << GetName() << "->SetWrapLength(" << fWrapLength << ");" << endl;

}

//______________________________________________________________________________
FontStruct_t TGLabel::GetDefaultFontStruct()
{
   // Static returning label default font struct.

   if (!fgDefaultFont) {
      fgDefaultFont = gClient->GetResourcePool()->GetDefaultFont();
   }
   return fgDefaultFont->GetFontStruct();
}

//______________________________________________________________________________
const TGGC &TGLabel::GetDefaultGC()
{
   // Static returning label default graphics context.

   if (!fgDefaultGC) {
      fgDefaultGC = gClient->GetResourcePool()->GetFrameGC();
   }
   return *fgDefaultGC;
}
