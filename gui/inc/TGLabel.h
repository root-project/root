// @(#)root/gui:$Name:  $:$Id: TGLabel.h,v 1.3 2000/09/30 11:24:12 rdm Exp $
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



class TGLabel : public TGFrame {

friend class TGClient;

protected:
   TGString      *fText;         // label text
   UInt_t         fTWidth;       // text width
   UInt_t         fTHeight;      // text height
   Int_t          fTMode;        // text drawing mode (ETextJustification)
   Bool_t         fTextChanged;  // has text changed
   GContext_t     fNormGC;       // graphics context used for drawing label
   FontStruct_t   fFontStruct;   // font to draw label

   virtual void DoRedraw();

   static FontStruct_t   fgDefaultFontStruct;
#ifdef R__SUNCCBUG
public:
#endif
   static TGGC           fgDefaultGC;

public:
   TGLabel(const TGWindow *p, TGString *text,
           GContext_t norm = fgDefaultGC(),
           FontStruct_t font = fgDefaultFontStruct,
           UInt_t options = kChildFrame,
           ULong_t back = fgDefaultFrameBackground);
   TGLabel(const TGWindow *p, const char *text,
           GContext_t norm = fgDefaultGC(),
           FontStruct_t font = fgDefaultFontStruct,
           UInt_t options = kChildFrame,
           ULong_t back = fgDefaultFrameBackground);
   virtual ~TGLabel();

   virtual TGDimension GetDefaultSize() const { return TGDimension(fTWidth, fTHeight+1); }
   const TGString *GetText() const { return fText; }
   void SetText(TGString *newText);
   void SetText(const char *newText) { SetText(new TGString(newText)); }
   void SetTextJustify(Int_t tmode) { fTMode = tmode; }

   static FontStruct_t  GetDefaultFontStruct();
   static const TGGC   &GetDefaultGC();

   ClassDef(TGLabel,0)  // A label GUI element
};

#endif
