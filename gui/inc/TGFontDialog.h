// @(#)root/gui:$Name:  $:$Id: TGFontDialog.h,v 1.0 2003/04/23 16:38:03 rdm Exp $
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
      TString     fName;
      Int_t       fSize;
      UInt_t      fAlign;
      ULong_t     fColor;
      Bool_t      fBold;
      Bool_t      fItalic;
   };

protected:
   TGListBox           *fFontNames;
   TGListBox           *fFontSizes;
   TGListBox           *fFontStyles;
   TGComboBox          *fTextAligns;
   TGLabel             *fSample;
   TString              fName;
   TString              fLName;
   FontProp_t          *fFontProp;
   Bool_t               fItalic;
   Bool_t               fBold;
   Int_t                fSize;
   UInt_t               fTextAlign;
   ULong_t              fTextColor;
   TString              fSampleText;
   TGGC                *fSampleTextGC;
   TGFont              *fLabelFont;

   Bool_t               GetFontProperties(const char *fontFamily = 0);
   void                 GetFontName();
   virtual Bool_t       ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);

public:
   TGFontDialog(const TGWindow *parent, const TGWindow *t,
                FontProp_t *fontProp, const TString &sample = "",
                const char **fontList = 0);
   virtual ~TGFontDialog();

   ClassDef(TGFontDialog,0)  // Font selection dialog
};

#endif
