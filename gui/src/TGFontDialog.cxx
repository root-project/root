// @(#)root/gui:$Name:  $:$Id: TGFontDialog.cxx,v 1.0 2003/04/23 16:38:03 rdm Exp $
// Author: Bertrand Bellenot + Fons Rademakers   23/04/03

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
/**************************************************************************

    This file is part of xclass.
    Copyright (C) 1996-2000 David Barth, Hector Peraza.

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

**************************************************************************/

#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include <stdlib.h>

#include "TGFontDialog.h"
#include "TCanvas.h"
#include "TROOT.h"
#include "TString.h"

enum {
    kFDLG_OK           = 1,
    kFDLG_CANCEL,

    kFDLG_FONTNAMES    = 21,
    kFDLG_FONTSIZES,
    kFDLG_FONTSTYLES,
    kFDLG_FONTALIGNS,

    kFDLG_COLORSEL     = 31
};

static const char *kFDLG_DEFAULTSAMPLE = "AaBbCcYyZz 1234567890";


static const char *gFontSizes[] = {
    "8",  "9", "10", "11", "12", "13", "14", "16",
   "18", "20", "22", "24", "26", "28", "30", "32",
   "34", "36", "48", "72", 0
};

static const char *gFontStyles[] = {
    "Normal", "Bold", "Italic", "Bold Italic", 0
};

static const char *gAlignTypes[] = {
    "top left", "top center", "top right",
    "middle left", "middle center", "middle right",
    "bottom left", "bottom center", "bottom right",
    0
};

static const UInt_t gAlignValues[] = {
    kTextTop     | kTextLeft,
    kTextTop     | kTextCenterX,
    kTextTop     | kTextRight,
    kTextCenterY | kTextLeft,
    kTextCenterY | kTextCenterX,
    kTextCenterY | kTextRight,
    kTextBottom  | kTextLeft,
    kTextBottom  | kTextCenterX,
    kTextBottom  | kTextRight,
    0
};

static const char *gFontList[] = {
    "Arial",
    "Comic Sans MS",
    "Courier New",
    "Georgia",
    "Impact",
    "Monotype",
    "Symbol",
    "Times New Roman",
    "Trebuchet MS",
    "Verdana",
    "Webdings",
    "Wingdings",
    0
};


//________________________________________________________________________________
TGFontDialog::TGFontDialog(const TGWindow *p, const TGWindow *t,
                           FontProp_t *fontProp, const TString &sample,
                           const char **fontList) :
              TGTransientFrame(p, t, 100, 100)
{
   // Create font dialog.

   TGLabel *lbl;
   TGHorizontalFrame *hf, *hf2;
   TGVerticalFrame *vf;
   Int_t i, w;
   UInt_t width = 0, height = 0;

   fSampleTextGC = 0;
   fLabelFont    = 0;
   fSample       = 0;

   if (!fontProp) {
      Error("TGFontDialog", "fontProp argument may not be 0");
      return;
   }

   TGLayoutHints *lh1 = new TGLayoutHints(kLHintsLeft | kLHintsExpandY);
   TGLayoutHints *lh2 = new TGLayoutHints(kLHintsLeft | kLHintsTop, 2, 0, 2, 2);

   fFontProp = fontProp;
   if (fontProp->fName == "") {
      fName      = fontList ? fontList[0] : "Arial";
      fItalic    = kFALSE;
      fBold      = kFALSE;
      fSize      = 12;
      fTextAlign = gAlignValues[6];   //kTextBottom  | kTextLeft
      fTextColor = TGFrame::GetBlackPixel();
   } else {
      fName      = fontProp->fName;
      fItalic    = fontProp->fItalic;
      fBold      = fontProp->fBold;
      fSize      = fontProp->fSize;
      fTextAlign = fontProp->fAlign;
      fTextColor = fontProp->fColor;
   }

   hf = new TGHorizontalFrame(this, 10, 10);
   AddFrame(hf, new TGLayoutHints(kLHintsNormal, 5, 5, 5, 5));

   //--------------------- font names listbox

   vf = new TGVerticalFrame(hf, 10, 10);

   lbl = new TGLabel(vf, new TGString("Font"));
   vf->AddFrame(lbl, lh2);

   fFontNames = new TGListBox(vf, kFDLG_FONTNAMES);
   fFontNames->Resize(120, 120);
   fFontNames->Associate(this);
   vf->AddFrame(fFontNames, lh1);

   hf->AddFrame(vf, new TGLayoutHints(kLHintsLeft | kLHintsExpandY,
                                      0, 10, 2, 2));

   //--------------------- font styles listbox

   vf = new TGVerticalFrame(hf, 10, 10);

   lbl = new TGLabel(vf, new TGString("Style"));
   vf->AddFrame(lbl, lh2);

   fFontStyles = new TGListBox(vf, kFDLG_FONTSTYLES);
   fFontStyles->Resize(80, fFontStyles->GetDefaultHeight());
   fFontStyles->Associate(this);
   vf->AddFrame(fFontStyles, lh1);

   hf->AddFrame(vf, new TGLayoutHints(kLHintsLeft | kLHintsExpandY,
                                      0, 10, 2, 2));

   //--------------------- font sizes listbox

   vf = new TGVerticalFrame(hf, 10, 10);

   lbl = new TGLabel(vf, new TGString("Size"));
   vf->AddFrame(lbl, lh2);

   fFontSizes = new TGListBox(vf, kFDLG_FONTSIZES);
   fFontSizes->Resize(50, fFontSizes->GetDefaultHeight());
   fFontSizes->Associate(this);
   vf->AddFrame(fFontSizes, lh1);

   hf->AddFrame(vf, new TGLayoutHints(kLHintsLeft | kLHintsExpandY,
                                      0, 0, 2, 2));

   //--------------------- Text align and color

   vf = new TGVerticalFrame(hf, 10, 10, kFixedWidth);

   lbl = new TGLabel(vf, new TGString("Text Align :"));
   vf->AddFrame(lbl, new TGLayoutHints(kLHintsLeft, 5, 5, 5, 5));

   fTextAligns = new TGComboBox(vf, kFDLG_FONTALIGNS);
   fTextAligns->Associate(this);
   vf->AddFrame(fTextAligns, new TGLayoutHints(kLHintsCenterX | kLHintsTop |
                kLHintsExpandX, 5, 5, 0, 5));

   fTextAligns->Resize(110,20);

   vf->Layout();
   vf->Resize(120,vf->GetDefaultHeight());

   w = hf->GetDefaultWidth();

   hf2 = new TGHorizontalFrame(vf, 10, 10);
   vf->AddFrame(hf2, new TGLayoutHints(kLHintsNormal, 0, 0, 5, 5));

   lbl = new TGLabel(hf2, new TGString("Text Color"));
   hf2->AddFrame(lbl, new TGLayoutHints(kLHintsLeft, 5, 5, 20, 5));

   TGColorSelect *colorsel = new TGColorSelect(hf2, fTextColor, kFDLG_COLORSEL);
   colorsel->Associate(this);

   hf2->AddFrame(colorsel, new TGLayoutHints(kLHintsRight | kLHintsTop, 5, 5, 20, 5));

   vf->Resize(vf->GetDefaultSize());

   hf->AddFrame(vf, new TGLayoutHints(kLHintsLeft | kLHintsExpandY,
                                      10, 0, lbl->GetDefaultHeight()+6, 0));

   //--------------------- initialize controls

   Resize(GetDefaultSize());

   if (!fontList)
      fontList = gFontList;

   for (i = 0; fontList[i] != 0; ++i)
      if (GetFontProperties(fontList[i]))
         fFontNames->AddEntry(new TGString(fontList[i]), i);

   for (i = 0; gFontSizes[i] != 0; ++i)
      fFontSizes->AddEntry(new TGString(gFontSizes[i]), i);

   for (i = 0; gAlignTypes[i] != 0; ++i)
      fTextAligns->AddEntry(new TGString(gAlignTypes[i]), i);

   for (i = 0; gAlignValues[i] != 0; ++i)
      if (gAlignValues[i] == fTextAlign)
         fTextAligns->Select(i);

   //--------------------- sample box

   hf = new TGHorizontalFrame(this, 10, 10);
   AddFrame(hf, new TGLayoutHints(kLHintsNormal | kLHintsExpandX, 5, 5, 5, 5));

   TGGroupFrame *gf = new TGGroupFrame(hf, new TGString("Sample"), kFixedSize);
   hf->AddFrame(gf, new TGLayoutHints(kLHintsNormal, 3, 3, 3, 3));

   TGCompositeFrame *cf = new TGCompositeFrame(gf, 200, 100, kSunkenFrame);
   gf->AddFrame(cf, new TGLayoutHints(kLHintsCenterX | kLHintsCenterY |
                                      kLHintsExpandX | kLHintsExpandY,
                                      1, 1, 3, 1));

   if (sample == "")
      fSampleText = kFDLG_DEFAULTSAMPLE;
   else
      fSampleText = sample;

   for (i = 0; fontList[i] != 0; ++i) {
      if (strstr(fName, fontList[i])) {
         fFontNames->Select(i);
         GetFontProperties();   // sets the supported font styles
         break;
      }
   }

   for (i = 0; gFontSizes[i] != 0; ++i) {
      if (fSize == atoi(gFontSizes[i])) {
         fFontSizes->Select(i);
         break;
      }
   }

   if ((fBold == kFALSE) && (fItalic == kFALSE))
      fFontStyles->Select(0);
   if ((fBold == kTRUE) && (fItalic == kFALSE))
      fFontStyles->Select(1);
   if ((fBold == kFALSE) && (fItalic == kTRUE))
      fFontStyles->Select(2);
   if ((fBold == kTRUE) && (fItalic == kTRUE))
      fFontStyles->Select(3);

   GetFontName();   //sets fLName and other data members

   GCValues_t gcval;
   gcval.fMask = kGCForeground | kGCFont;
   gcval.fForeground = fTextColor;
   gcval.fFont = fLabelFont->GetFontHandle();
   fSampleTextGC = fClient->GetGC(&gcval, kTRUE);
   fSample = new TGLabel(cf, fSampleText, (*fSampleTextGC)(), (*fLabelFont)());
   fSample->SetTextJustify(gAlignValues[fTextAligns->GetSelected()]);
   cf->AddFrame(fSample, new TGLayoutHints(kLHintsCenterX | kLHintsCenterY |
                kLHintsExpandX | kLHintsExpandY, 1, 1, 1, 1));
   cf->Layout();

   gf->Resize(w, 80);

   //--------------------- OK and Cancel buttons

   vf = new TGVerticalFrame(hf, 10, 10, kFixedWidth);

   TGTextButton *ok = new TGTextButton(vf, new TGHotString("&OK"), kFDLG_OK);
   ok->Associate(this);
   vf->AddFrame(ok, new TGLayoutHints(kLHintsTop | kLHintsExpandX, 5, 5, 20, 5));

   TGTextButton *cancel = new TGTextButton(vf, new TGHotString("&Cancel"), kFDLG_CANCEL);
   cancel->Associate(this);
   vf->AddFrame(cancel, new TGLayoutHints(kLHintsTop | kLHintsExpandX, 5, 5, 5, 5));

   vf->Resize(cancel->GetDefaultWidth()+70, vf->GetDefaultHeight());

   hf->AddFrame(vf, new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 5, 5, 5, 5));

   MapSubwindows();

   width  = GetDefaultWidth();
   height = GetDefaultHeight();

   Resize(width, height);

   //---- position the dialog relative to the parent's window

   if (t) {
      int ax, ay;
      Window_t wdummy;
      gVirtualX->TranslateCoordinates(t->GetId(), GetParent()->GetId(),
                                      50, 50, ax, ay, wdummy);
      Move(ax, ay);
      SetWMPosition(ax, ay);
   }

   //---- make the dialog box non-resizable

   SetWMSize(width, height);
   SetWMSizeHints(width, height, width, height, 0, 0);

   SetWindowName("Font Selection");
   SetIconName("Font Selection");
   SetClassHints("FontDialog", "FontDialog");

   SetMWMHints(kMWMDecorAll | kMWMDecorResizeH  | kMWMDecorMaximize |
                              kMWMDecorMinimize | kMWMDecorMenu,
               kMWMFuncAll  | kMWMFuncResize    | kMWMFuncMaximize |
                              kMWMFuncMinimize,
               kMWMInputModeless);


   MapWindow();

   fFontNames->GetScrollBar()->SetPosition(fFontNames->GetSelected()-3);
   fFontSizes->GetScrollBar()->SetPosition(fFontSizes->GetSelected()-3);
   fFontSizes->Layout();
   fFontNames->Layout();

   fClient->WaitFor(this);
}

//________________________________________________________________________________
TGFontDialog::~TGFontDialog()
{
   // Delete all widgets.

   Cleanup();
   fClient->FreeFont(fLabelFont);
   fClient->FreeGC(fSampleTextGC);
}

//________________________________________________________________________________
Bool_t TGFontDialog::ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2)
{
   // Handle dialog events.

   switch (GET_MSG(msg)) {
      case kC_COMMAND:
         switch (GET_SUBMSG(msg)) {
            case kCM_BUTTON:
               switch (parm1) {

                  case kFDLG_OK:
                     fFontProp->fName   = fName;
                     fFontProp->fItalic = fItalic;
                     fFontProp->fBold   = fBold;
                     fFontProp->fSize   = fSize;
                     fFontProp->fColor  = fTextColor;
                     fFontProp->fAlign  = fTextAlign;
                     CloseWindow();
                     break;
                  case kFDLG_CANCEL:
                     CloseWindow();
                     break;
               }
               break;

            case kCM_LISTBOX:
               switch (parm1) {
                  case kFDLG_FONTNAMES:
                     GetFontProperties();
                  break;
               }
               GetFontName();
               break;

            case kCM_COMBOBOX:
               GetFontName();
               break;

            default:
               break;
         }
         break;

      case kC_COLORSEL:
         switch (GET_SUBMSG(msg)) {

            case kCOL_SELCHANGED:
               fTextColor = parm2;
               GetFontName();
               break;

            default:
               break;
         }
         break;

      default:
         break;
   }
   return kTRUE;
}

//________________________________________________________________________________
Bool_t TGFontDialog::GetFontProperties(const char *fontFamily)
{
   // Get possible styles and sizes for the selected font family if
   // fontFamily = 0, otherwise check if fontFamily exists. Returns
   // kFALSE if fontFamily does not exist.

   TString family;

   if (fontFamily) {
      family = fontFamily;
   } else {
      TGTextLBEntry *e = (TGTextLBEntry *) fFontNames->GetSelectedEntry();
      family = e ? e->GetText()->GetString() : "";
   }

   if (family == "") {
      Error("GetFontProperties", "no font selected or specified, should not happen");
      return kFALSE;
   }

   // only the MS versions of these fonts work good
   TString rgstry = "*";
   if (family == "Symbol" || family == "Webdings" || family == "Wingdings")
      rgstry = "microsoft";

   TString font = Form("-*-%s-*-*-*-*-*-*-*-*-*-*-%s-*", family.Data(),
                       rgstry.Data());

   Int_t cnt;
   char **fontList = gVirtualX->ListFonts(font, 1024, cnt);
   if (!fontList) {
      if (!fontFamily)
         Warning("GetFontProperties", "font %s not found", font.Data());
      return kFALSE;
   }
   if (fontFamily) {
      gVirtualX->FreeFontNames(fontList);
      return kTRUE;
   }

   Bool_t styles[4] = { kFALSE, kFALSE, kFALSE, kFALSE };
   Int_t  i;
   for (i = 0; i < cnt; i++) {
      if (strstr(fontList[i], "-medium-r-")) styles[0]  = kTRUE;
      if (strstr(fontList[i], "-bold-r-"))   styles[1]  = kTRUE;
      if (strstr(fontList[i], "-medium-i-")) styles[2]  = kTRUE;
      if (strstr(fontList[i], "-medium-o-")) styles[2]  = kTRUE;
      if (strstr(fontList[i], "-bold-i-"))   styles[3]  = kTRUE;
      if (strstr(fontList[i], "-bold-o-"))   styles[3]  = kTRUE;
   }
   gVirtualX->FreeFontNames(fontList);

   // update style list box
   fFontStyles->RemoveEntries(0, 1000);
   for (i = 0; gFontStyles[i] != 0; ++i)
      if (styles[i])
         fFontStyles->AddEntry(new TGString(gFontStyles[i]), i);
   fFontStyles->MapSubwindows();
   fFontStyles->Layout();
   fFontStyles->Select(0);

   return kTRUE;
}

//________________________________________________________________________________
void TGFontDialog::GetFontName()
{
   UInt_t  width = 0, height = 0;
   TGTextLBEntry *e;
   const char *size, *name;
   char st1[12];
   char st2;
   Int_t sel;

   e = (TGTextLBEntry *) fFontNames->GetSelectedEntry();
   name = e ? e->GetText()->GetString() : "";
   fName = name;

   e = (TGTextLBEntry *) fFontSizes->GetSelectedEntry();
   size = e ? e->GetText()->GetString() : "";
   fSize = atoi(size);

   sel = fFontStyles->GetSelected();

   switch(sel) {
      case 0:
         sprintf(st1, "medium");
         st2     = 'r';
         fItalic = kFALSE;
         fBold   = kFALSE;
         break;
      case 1:
         sprintf(st1, "bold");
         st2     = 'r';
         fItalic = kFALSE;
         fBold   = kTRUE;
         break;
      case 2:
         sprintf(st1, "medium");
         st2     = 'i';
         fItalic = kTRUE;
         fBold   = kFALSE;
         break;
      case 3:
         sprintf(st1, "bold");
         st2     = 'i';
         fItalic = kTRUE;
         fBold   = kTRUE;
         break;
      default:
         sprintf(st1, "medium");
         st2     = 'r';
         fItalic = kFALSE;
         fBold   = kFALSE;
         break;
   }

   TString rgstry = "*";
   if (fName == "Symbol" || fName == "Webdings" || fName == "Wingdings")
      rgstry = "microsoft";

   TString oldFont = fLName;
   fLName = Form("-*-%s-%s-%c-*-*-%s-*-*-*-*-*-%s-*", name, st1, st2, size,
                 rgstry.Data());
   if (oldFont != fLName) {
      Int_t cnt;
      char **fontList = gVirtualX->ListFonts(fLName, 1, cnt);
      if (!fontList) {
         if (fItalic) {
            // try oblique
            fLName = Form("-*-%s-%s-o-*-*-%s-*-*-*-*-*-%s-*", name, st1, size,
                          rgstry.Data());
            fontList = gVirtualX->ListFonts(fLName, 1, cnt);
         }
         if (!fontList) {
            Warning("GetFontName", "font %s not found", fLName.Data());
            fLName = oldFont;
            return;
         }
      }
      gVirtualX->FreeFontNames(fontList);
      if (fLabelFont) {
         fClient->FreeFont(fLabelFont);
         fLabelFont = fClient->GetFont(fLName, kFALSE);
      } else
         fLabelFont = fClient->GetFont(fLName);
      if (!fLabelFont) {
         // should not happen
         fLName = oldFont;
         return;
      }
      if (fSample) fSample->SetTextFont(fLabelFont);
   }
   fTextAlign = gAlignValues[fTextAligns->GetSelected()];
   if (fSample) fSample->SetTextJustify(fTextAlign);
   if (fSample) fSample->SetTextColor(fTextColor);
   fClient->NeedRedraw(this);
}
