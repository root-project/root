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


#include "Rtypes.h"

class TAttText {

protected:
   Float_t    fTextAngle;           ///< Text angle
   Float_t    fTextSize;            ///< Text size
   Short_t    fTextAlign;           ///< Text alignment
   Color_t    fTextColor;           ///< Text color
   Font_t     fTextFont;            ///< Text font

public:
   TAttText();
   TAttText(Int_t align, Float_t angle, Color_t color, Style_t font, Float_t tsize);
   virtual ~TAttText();
           void     Copy(TAttText &atttext) const;
   virtual Short_t  GetTextAlign() const {return fTextAlign;} ///< Return the text alignment
   virtual Float_t  GetTextAngle() const {return fTextAngle;} ///< Return the text angle
   virtual Color_t  GetTextColor() const {return fTextColor;} ///< Return the text color
   virtual Font_t   GetTextFont()  const {return fTextFont;}  ///< Return the text font
   virtual Float_t  GetTextSize()  const {return fTextSize;}  ///< Return the text size
   virtual Float_t  GetTextSizePercent(Float_t size);         ///< Return the text in percent of the pad size
   virtual void     Modify();
   virtual void     ResetAttText(Option_t *toption="");
   virtual void     SaveTextAttributes(std::ostream &out, const char *name, Int_t alidef=12, Float_t angdef=0, Int_t coldef=1, Int_t fondef=61, Float_t sizdef=1);
   virtual void     SetTextAttributes();  // *MENU*
   virtual void     SetTextAlign(Short_t align=11) { fTextAlign = align;}  ///< Set the text alignment
   virtual void     SetTextAngle(Float_t tangle=0) { fTextAngle = tangle;} ///< Set the text angle
   virtual void     SetTextColor(Color_t tcolor=1) { fTextColor = tcolor;} ///< Set the text color
   virtual void     SetTextColorAlpha(Color_t tcolor, Float_t talpha);
   virtual void     SetTextFont(Font_t tfont=62) { fTextFont = tfont;}     ///< Set the text font
   virtual void     SetTextSize(Float_t tsize=1) { fTextSize = tsize;}     ///< Set the text size
   virtual void     SetTextSizePixels(Int_t npixels);                      ///< Set the text size in pixel

   ClassDef(TAttText,2)  //Text attributes
};

   enum ETextAlign {kHAlignLeft=10, kHAlignCenter=20, kHAlignRight=30,
                    kVAlignBottom=1, kVAlignCenter=2, kVAlignTop=3};

#endif

