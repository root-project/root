// @(#)root/gui:$Name:  $:$Id: TGFontDialog.h,v 1.4 2004/09/14 09:22:57 rdm Exp $
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


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGFontDialog.                                                        //
//                                                                      //
// The TGFontDialog allows easy font and font attribute selection.      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif

#ifndef ROOT_TGButton
#include "TGButton.h"
#endif

#ifndef ROOT_TGLabel
#include "TGLabel.h"
#endif

#ifndef ROOT_TGListBox
#include "TGListBox.h"
#endif

#ifndef ROOT_TGComboBox
#include "TGComboBox.h"
#endif

#ifndef ROOT_TGColorSelect
#include "TGColorSelect.h"
#endif


class TGFontDialog : public TGTransientFrame {

public:
   struct FontProp_t {
      TString     fName;               // font name
      Int_t       fSize;               // font size
      UInt_t      fAlign;              // text alignment
      Pixel_t     fColor;              // text color
      Bool_t      fBold;               // bold flag
      Bool_t      fItalic;             // italic flag
   };

protected:
   TGListBox           *fFontNames;    // list of font names
   TGListBox           *fFontSizes;    // list of font sizes
   TGListBox           *fFontStyles;   // list of font styles
   TGComboBox          *fTextAligns;   // font alignment selection
   TGLabel             *fSample;       // sample of selected font
   TString              fName;         // font name
   TString              fLName;        // logical font name
   FontProp_t          *fFontProp;     // font info structure
   Bool_t               fItalic;       // italic flag
   Bool_t               fBold;         // bold flag
   Int_t                fSize;         // font size
   UInt_t               fTextAlign;    // text aligment
   Pixel_t              fTextColor;    // text color
   TString              fSampleText;   // string used for sample
   TGGC                *fSampleTextGC; // GC used for sample text
   TGFont              *fLabelFont;    // TGFont used for sample text
   Bool_t               fHitOK;        // flag = kTRUE if user press the Ok button

   Bool_t               GetFontProperties(const char *fontFamily = 0);
   void                 GetFontName();
   virtual void         CloseWindow();
   virtual Bool_t       ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);

public:
   TGFontDialog(const TGWindow *parent = 0, const TGWindow *t = 0,
                FontProp_t *fontProp = 0, const TString &sample = "",
                const char **fontList = 0);
   virtual ~TGFontDialog();

   ClassDef(TGFontDialog,0)  // Font selection dialog
};

#endif
