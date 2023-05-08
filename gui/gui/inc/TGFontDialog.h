// @(#)root/gui:$Id$
// Author: Bertrand Bellenot + Fons Rademakers   23/04/03

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGFontDialog
#define ROOT_TGFontDialog

#include "TGFrame.h"

class TGButton;
class TGLabel;
class TGListBox;
class TGComboBox;
class TGColorSelect;
class TGFont;

class TGFontDialog : public TGTransientFrame {

public:
   struct FontProp_t {
      TString     fName;               ///< font name
      Int_t       fSize;               ///< font size
      UInt_t      fAlign;              ///< text alignment
      Pixel_t     fColor;              ///< text color
      Bool_t      fBold;               ///< bold flag
      Bool_t      fItalic;             ///< italic flag
   };

protected:
   TGListBox           *fFontNames;    ///< list of font names
   TGListBox           *fFontSizes;    ///< list of font sizes
   TGListBox           *fFontStyles;   ///< list of font styles
   TGComboBox          *fTextAligns;   ///< font alignment selection
   TGLabel             *fSample;       ///< sample of selected font
   TGColorSelect       *fColorSelect;  ///< color selection dialog
   TString              fName;         ///< font name
   TString              fLName;        ///< logical font name
   FontProp_t          *fFontProp;     ///< font info structure
   Bool_t               fItalic;       ///< italic flag
   Bool_t               fBold;         ///< bold flag
   Int_t                fSize;         ///< font size
   Int_t                fTextAlign;    ///< text alignment
   Pixel_t              fTextColor;    ///< text color
   Pixel_t              fInitColor;    ///< initial value of text color
   Int_t                fInitAlign;    ///< initial value of  text align
   TGFont              *fInitFont;     ///< initial font
   TString              fSampleText;   ///< string used for sample
   TGGC                *fSampleTextGC; ///< GC used for sample text
   TGFont              *fLabelFont;    ///< TGFont used for sample text
   Bool_t               fHitOK;        ///< flag = kTRUE if user press the Ok button
   Int_t                fNumberOfFonts;///< total number of fonts
   Bool_t               fWaitFor;      ///< if kTRUE WaitForUnmap is called in constructor.

   Bool_t               Build(char **fontList, Int_t cnt);
   void                 GetFontName();
   void                 CloseWindow() override;
   Bool_t               ProcessMessage(Longptr_t msg, Longptr_t parm1, Longptr_t parm2) override;

public:
   TGFontDialog(const TGWindow *parent = nullptr, const TGWindow *t = nullptr,
                FontProp_t *fontProp = nullptr, const TString &sample = "",
                char **fontList = nullptr, Bool_t wait = kTRUE);
   virtual ~TGFontDialog();

   virtual void SetFont(TGFont *font);
   virtual void SetColor(Pixel_t color);
   virtual void SetAlign(Int_t align);
   virtual void EnableAlign(Bool_t on = kTRUE);
   virtual void UpdateStyleSize(const char *family);

   virtual void FontSelected(char *font)
            { Emit("FontSelected(char*)", font); }  //*SIGNAL*
   virtual void AlignSelected(Int_t a)
            { Emit("AlignSelected(Int_t)", a); }   //*SIGNAL*
   virtual void ColorSelected(Pixel_t c)
            { Emit("ColorSelected(Pixel_t)", c); }  //*SIGNAL*

   ClassDefOverride(TGFontDialog,0)  // Font selection dialog
};

#endif
