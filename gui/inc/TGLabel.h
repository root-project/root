// @(#)root/gui:$Id$
// Author: Fons Rademakers   06/01/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLabel
#define ROOT_TGLabel


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLabel                                                              //
//                                                                      //
// This class handles GUI labels.                                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif
#ifndef ROOT_TGDimension
#include "TGDimension.h"
#endif
#ifndef ROOT_TGString
#include "TGString.h"
#endif

class TColor;
class TGTextLayout;
class TGFont;

class TGLabel : public TGFrame {

protected:
   TGString      *fText;         // label text
   UInt_t         fTWidth;       // text width
   UInt_t         fTHeight;      // text height
   Int_t          fMLeft;        // margin left
   Int_t          fMRight;       // margin right
   Int_t          fMTop;         // margin top
   Int_t          fMBottom;      // margin bottom
   Int_t          fTMode;        // text alignment
   Int_t          f3DStyle;      // 3D style (0 - normal, kRaisedFrame - raised, kSunkenFrame - sunken)
   Int_t          fWrapLength;   // wrap length
   Int_t          fTFlags;       // text flags (see TGFont.h  ETextLayoutFlags)
   Bool_t         fTextChanged;  // has text changed
   GContext_t     fNormGC;       // graphics context used for drawing label
   TGFont        *fFont;         // font to draw label
   TGTextLayout  *fTLayout;      // text layout   
   Bool_t         fHasOwnFont;   // kTRUE - font defined locally,  kFALSE - globally
   Bool_t         fDisabled;     // if kTRUE label looks disabled (shaded text)

   virtual void DoRedraw();
   virtual void DrawText(GContext_t gc, Int_t x, Int_t y);

   static const TGFont  *fgDefaultFont;
   static const TGGC    *fgDefaultGC;

private:
   TGLabel(const TGLabel&);             // not implemented
   TGLabel& operator=(const TGLabel&);  // not implemented

public:
   static FontStruct_t  GetDefaultFontStruct();
   static const TGGC   &GetDefaultGC();

   TGLabel(const TGWindow *p, TGString *text,
           GContext_t norm = GetDefaultGC()(),
           FontStruct_t font = GetDefaultFontStruct(),
           UInt_t options = kChildFrame,
           Pixel_t back = GetDefaultFrameBackground());
   TGLabel(const TGWindow *p = 0, const char *text = 0,
           GContext_t norm = GetDefaultGC()(),
           FontStruct_t font = GetDefaultFontStruct(),
           UInt_t options = kChildFrame,
           Pixel_t back = GetDefaultFrameBackground());

   virtual ~TGLabel();

   virtual TGDimension GetDefaultSize() const; 

   const TGString *GetText() const { return fText; }
   virtual const char *GetTitle() const { return fText->Data(); }
   virtual void SetText(TGString *newText);
   void SetText(const char *newText) { SetText(new TGString(newText)); }
   virtual void ChangeText(const char *newText) { SetText(newText); } //*MENU*icon=bld_rename.png*
   virtual void SetTitle(const char *label) { SetText(label); }
   void  SetText(Int_t number) { SetText(new TGString(number)); }
   void  SetTextJustify(Int_t tmode);
   Int_t GetTextJustify() const { return fTMode; }
   virtual void SetTextFont(TGFont *font, Bool_t global = kFALSE);
   virtual void SetTextFont(FontStruct_t font, Bool_t global = kFALSE);
   virtual void SetTextFont(const char *fontName, Bool_t global = kFALSE);
   virtual void SetTextColor(Pixel_t color, Bool_t global = kFALSE);
   virtual void SetTextColor(TColor *color, Bool_t global = kFALSE);
   virtual void SetForegroundColor(Pixel_t fore) { SetTextColor(fore); }
   virtual void Disable(Bool_t on = kTRUE)
               { fDisabled = on; fClient->NeedRedraw(this); } //*TOGGLE* *GETTER=IsDisabled
   virtual void Enable() { fDisabled = kFALSE; fClient->NeedRedraw(this); }
   Bool_t IsDisabled() const { return fDisabled; }
   Bool_t HasOwnFont() const;

   void  SetWrapLength(Int_t wl) { fWrapLength = wl; Layout(); }
   Int_t GetWrapLength() const { return fWrapLength; }

   void  Set3DStyle(Int_t style) { f3DStyle = style; fClient->NeedRedraw(this); }
   Int_t Get3DStyle() const { return f3DStyle; }

   void SetMargins(Int_t left=0, Int_t right=0, Int_t top=0, Int_t bottom=0)
      { fMLeft = left; fMRight = right; fMTop = top; fMBottom = bottom; }
   Int_t GetLeftMargin() const { return fMLeft; }
   Int_t GetRightMargin() const { return fMRight; }
   Int_t GetTopMargin() const { return fMTop; }
   Int_t GetBottomMargin() const { return fMBottom; }

   GContext_t GetNormGC() const { return fNormGC; }
   FontStruct_t GetFontStruct() const { return fFont->GetFontStruct(); }
   TGFont      *GetFont() const  { return fFont; }

   virtual void Layout();
   virtual void SavePrimitive(ostream &out, Option_t *option = "");

   ClassDef(TGLabel,0)  // A label GUI element
};

#endif
