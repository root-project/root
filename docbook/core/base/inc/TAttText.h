// @(#)root/base:$Id$
// Author: Rene Brun   12/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TAttText
#define ROOT_TAttText


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAttText                                                             //
//                                                                      //
// Text attributes.                                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif
#ifndef ROOT_Riosfwd
#include "Riosfwd.h"
#endif


class TAttText {

protected:
   Float_t    fTextAngle;           //Text angle
   Float_t    fTextSize;            //Text size
   Short_t    fTextAlign;           //Text alignment
   Color_t    fTextColor;           //Text color index
   Font_t     fTextFont;            //Text font number

public:
   TAttText();
   TAttText(Int_t align, Float_t angle, Color_t color, Style_t font, Float_t tsize);
   virtual ~TAttText();
           void     Copy(TAttText &atttext) const;
   virtual Short_t  GetTextAlign() const {return fTextAlign;}
   virtual Float_t  GetTextAngle() const {return fTextAngle;}
   virtual Color_t  GetTextColor() const {return fTextColor;}
   virtual Font_t   GetTextFont()  const {return fTextFont;}
   virtual Float_t  GetTextSize()  const {return fTextSize;}
   virtual void     Modify();
   virtual void     ResetAttText(Option_t *toption="");
   virtual void     SaveTextAttributes(ostream &out, const char *name, Int_t alidef=12, Float_t angdef=0, Int_t coldef=1, Int_t fondef=61, Float_t sizdef=1);
   virtual void     SetTextAttributes();  // *MENU*
   virtual void     SetTextAlign(Short_t align=11) { fTextAlign = align;}
   virtual void     SetTextAngle(Float_t tangle=0) { fTextAngle = tangle;}  // *MENU*
   virtual void     SetTextColor(Color_t tcolor=1) { fTextColor = tcolor;}
   virtual void     SetTextFont(Font_t tfont=62) { fTextFont = tfont;}
   virtual void     SetTextSize(Float_t tsize=1) { fTextSize = tsize;}
   virtual void     SetTextSizePixels(Int_t npixels);

   ClassDef(TAttText,1)  //Text attributes
};

#endif

