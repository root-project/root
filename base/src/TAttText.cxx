// @(#)root/base:$Name:  $:$Id: TAttText.cxx,v 1.11 2002/01/24 11:39:27 rdm Exp $
// Author: Rene Brun   12/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "Riostream.h"
#include "TROOT.h"
#include "Strlen.h"
#include "TAttText.h"
#include "TVirtualPad.h"
#include "TStyle.h"
#include "TVirtualX.h"
#include "TError.h"

ClassImp(TAttText)

//______________________________________________________________________________
//*-*-*-*-*-*-*-*-*-*-*-*-*Text Attributes class*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      =====================
//*-*  Text attributes are:
//*-*    Text Alignment
//*-*    Text Color
//*-*    Text Font
//*-*    Text Size
//*-*    Text Angle
//*-*
//*-*  This class is used (in general by secondary inheritance)
//*-*
//*-*  by many other classes (graphics, histograms).
//*-*    align : Text alignment = 10*HorizontalAlign + VerticalAlign
//*-*            For Horizontal alignment the following convention applies:
//*-*               1=left adjusted, 2=centered, 3=rigth adjusted
//*-*            For Vertical alignment the following convention applies:
//*-*               1=bottom adjusted, 2=centered, 3=top adjusted
//*-*            For example align = 11 = left adjusted and bottom adjusted
//*-*                        align = 32 = right adjusted and vertically centered
//*-*    angle : Text angle in degrees
//*-*    color : Text Color Index
//*-*    font  : Text font code = 10*fontnumber + precision
//*-*             Font numbers must be between 1 and 14
//*-*             precision = 0 fast hardware fonts (steps in the size)
//*-*             precision = 1 scalable and rotatable hardware fonts (see below)
//*-*             precision = 2 scalable and rotatable hardware fonts
//*-*             precision = 3 scalable and rotatable hardware fonts. Text size
//*-*                           is given in pixels.
//*-*    size  : Character size expressed in percentage of the current pad height
//*-*            The textsize in pixels (say charheight) will be:
//*-*             charheight = textsize*canvas_height if current pad is horizontal.
//*-*             charheight = textsize*canvas_width  if current pad is vertical.
//*-*             charheight = number of pixels if font precision is greater than 2
//*-*
//*-* Font quality and speed
//*-* ======================
//*-*  When precision 0 is used, only the original non-scaled system fonts
//*-*  are used. The fonts have a minimum (4)  and maximum (37) size in pixels.
//*-*  These fonts are fast and are of good quality. Their size varies with
//*-*  large steps and they cannot be rotated.
//*-*  Precision 1 and 2 fonts have a different behaviour depending if the
//*-*  True Type Fonts are used or not. If TTF are used, you always get
//*-*  very good quality scalable and rotatable fonts. However TTF are slow.
//*-*  Precision 1 and 2 fonts have a different behaviour for Postscript
//*-*  in case of TLatex objects. With precision 1, the Postscript text uses
//*-*  the old convention (see TPostScript) for some special characters
//*-*  to draw sub and superscripts or greek text.
//*-*  With precision 2 the "PostScript" special characters are drawn as such.
//*-*  To draw sub and superscripts it is highly recommended to use TLatex
//*-*  objects instead.
//*-*
//*-*  How to use True Type Fonts
//*-*  ==========================
//*-*  You can activate the TTF by adding (or activating) the following line
//*-*  in your .rootrc file.
//*-*  Unix.*.Root.UseTTFonts:     true
//*-*  WinNT  for NT
//*-*  You can check that you indeed use the TTF in your Root session.
//*-*  When the TTF is active, you get the following message at the start
//*-*  of a session:
//*-*    "FreeType Engine v1.x used to render TrueType fonts."
//*-*  You can also check with the command gEnv->Print().
//*-*
//*-*  List of the currently supported fonts (screen and PostScript)
//*-*  =============================================================
//*-*   Font ID       X11                       Win32 TTF       lfItalic  lfWeight  x 10
//*-*        1 : times-medium-i-normal      "Times New Roman"      1           4
//*-*        2 : times-bold-r-normal        "Times New Roman"      0           7
//*-*        3 : times-bold-i-normal        "Times New Roman"      1           7
//*-*        4 : helvetica-medium-r-normal  "Arial"                0           4
//*-*        5 : helvetica-medium-o-normal  "Arial"                1           4
//*-*        6 : helvetica-bold-r-normal    "Arial"                0           7
//*-*        7 : helvetica-bold-o-normal    "Arial"                1           7
//*-*        8 : courier-medium-r-normal    "Courier New"          0           4
//*-*        9 : courier-medium-o-normal    "Courier New"          1           4
//*-*       10 : courier-bold-r-normal      "Courier New"          0           7
//*-*       11 : courier-bold-o-normal      "Courier New"          1           7
//*-*       12 : symbol-medium-r-normal     "Symbol"               0           6
//*-*       13 : times-medium-r-normal      "Times New Roman"      0           4
//*-*       14 :                            "Wingdings"            0           4
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

//______________________________________________________________________________
TAttText::TAttText()
{
//*-*-*-*-*-*-*-*-*-*-*-*-*AttText default constructor*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      ===========================
//*-*  Default text attributes are taking from the current style
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   if (!gStyle) {
      ResetAttText();
      return;
   }
   fTextAlign = gStyle->GetTextAlign();
   fTextAngle = gStyle->GetTextAngle();
   fTextColor = gStyle->GetTextColor();
   fTextFont  = gStyle->GetTextFont();
   fTextSize  = gStyle->GetTextSize();
}

//______________________________________________________________________________
TAttText::TAttText(Int_t align, Float_t angle, Color_t color, Style_t font, Float_t tsize)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*AttText normal constructor*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      ===========================
//*-*  Text attributes are taking from the argument list
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   fTextAlign = align;
   fTextAngle = angle;
   fTextColor = color;
   fTextFont  = font;
   fTextSize  = tsize;
}

//______________________________________________________________________________
TAttText::~TAttText()
{
//*-*-*-*-*-*-*-*-*-*-*-*-*AttText destructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      =================

}

//______________________________________________________________________________
void TAttText::Copy(TAttText &atttext) const
{
//*-*-*-*-*-*-*-*-*Copy this text attributes to a new atttext*-*-*-*-*-*-*-*-*
//*-*              ==========================================
   atttext.fTextAlign  = fTextAlign;
   atttext.fTextAngle  = fTextAngle;
   atttext.fTextColor  = fTextColor;
   atttext.fTextFont   = fTextFont;
   atttext.fTextSize   = fTextSize;
}

//______________________________________________________________________________
void TAttText::Modify()
{
//*-*-*-*-*-*-*-*-*-*Change current text attributes if necessary*-*-*-*-*-*-*
//*-*                ===========================================

   if (!gPad) return;
//*-*- Do we need to change font?
   if (!gPad->IsBatch()) {
      gVirtualX->SetTextAngle(fTextAngle);
      Float_t wh = (Float_t)gPad->XtoPixel(gPad->GetX2());
      Float_t hh = (Float_t)gPad->YtoPixel(gPad->GetY1());
      Float_t tsize;
      if (wh < hh)  tsize = fTextSize*wh;
      else          tsize = fTextSize*hh;
      if (fTextFont%10 > 2) tsize = fTextSize;

#ifndef WIN32
again:
#ifndef R__TTFROT
      if (gVirtualX->HasTTFonts() && gVirtualX->GetTextSize() != -10.0) {
         if (fTextAngle != 0) {
            gVirtualX->SetTextSize(-10.0);
            goto again;
         }
#else
      if (gVirtualX->HasTTFonts()) {
#endif
         if (gVirtualX->GetTextFont() != fTextFont) {
            gVirtualX->SetTextFont(fTextFont);
            if (!gVirtualX->HasTTFonts()) goto again;
            gVirtualX->SetTextSize(tsize);
         }
         if (gVirtualX->GetTextSize() != tsize)
            gVirtualX->SetTextSize(tsize);

      } else if (gVirtualX->GetTextFont() != fTextFont ||
                 gVirtualX->GetTextSize() != tsize) {
         char     fx11[64];
         Int_t   fpx11 = fTextFont; if (fpx11 < 0) fpx11 = -fpx11;
         Int_t  ifpx11 = fpx11/10;
         Int_t      ih = Int_t(tsize); // to be set to IH = INT(WKSC(IWKNB)*YRATIO*RCHH*1.5)
         Float_t rsize = Float_t(ih);
         if (ih > 37) ih = 37;
         if (ih <= 0) ih = 1;
         if (ifpx11 <= 0 || ifpx11 > 13) ifpx11 = 6;
//*-*-          Set Font name
         switch (ifpx11) {
           case  1 : strcpy(fx11, "-*-times-medium-i-normal--");     break;
           case  2 : strcpy(fx11, "-*-times-bold-r-normal--");       break;
           case  3 : strcpy(fx11, "-*-times-bold-i-normal--");       break;
           case  4 : strcpy(fx11, "-*-helvetica-medium-r-normal--"); break;
           case  5 : strcpy(fx11, "-*-helvetica-medium-o-normal--"); break;
           case  6 : strcpy(fx11, "-*-helvetica-bold-r-normal--");   break;
           case  7 : strcpy(fx11, "-*-helvetica-bold-o-normal--");   break;
           case  8 : strcpy(fx11, "-*-courier-medium-r-normal--");   break;
           case  9 : strcpy(fx11, "-*-courier-medium-o-normal--");   break;
           case 10 : strcpy(fx11, "-*-courier-bold-r-normal--");     break;
           case 11 : strcpy(fx11, "-*-courier-bold-o-normal--");     break;
           case 12 : strcpy(fx11, "-*-symbol-medium-r-normal--");    break;
           case 13 : strcpy(fx11, "-*-times-medium-r-normal--");     break;
         };
         char *buffer;
         Int_t il = strlen(fx11);
//*-*           Check if closest font has already been computed
         static Int_t first = 1;
         static Int_t fontchecked[13][40];
         if (first) {
            for (int ifont=0;ifont<13;ifont++) {
               for (int isize=0;isize<40;isize++) fontchecked[ifont][isize] = 0;
            }
            first = 0;
         }
         Int_t ihh = fontchecked[ifpx11-1][ih-1];
         if (ihh) {
            buffer = fx11 + il;
            sprintf(buffer,"%d-*",ihh);
            gVirtualX->SetTextFont(fx11, TVirtualX::kLoad);
         } else {
//*-*-           Find smallest size available
            Int_t isxfnt;
            for (isxfnt=8; isxfnt<17; isxfnt++) {
               buffer = fx11 + il;
               sprintf(buffer,"%d-*",isxfnt);
               if (!gVirtualX->SetTextFont(fx11, TVirtualX::kCheck) ) break;
               if (isxfnt == 16) Warning("TAttText::Modify", "cannot find the right size font");
            }
//*-*-           Find the closest size available
            ihh = ih;
            while (1) {
               if (ihh < isxfnt) ihh = isxfnt;
               buffer = fx11 + il;
               sprintf(buffer,"%d-*",ihh);
               if (!gVirtualX->SetTextFont(fx11, TVirtualX::kLoad) ) {
                  fontchecked[ifpx11-1][ih-1] = ihh;
                  break;
               }
               if (ihh == isxfnt) {
                  Warning("TAttText::Modify", "cannot find the right size font");
                  return;
               }
               ihh--;
            }
         }

//*-*-           ready to draw text
         Float_t mgn = rsize/Float_t(ihh);
         if (mgn > 100) mgn = 100;
         if (mgn <0)    mgn = 1;
         if (fTextFont%10 == 0 || fTextFont%10 > 2) mgn = 1;
         gVirtualX->SetTextMagnitude(mgn);
         gVirtualX->DrawText(0,0,0,-1.,0,TVirtualX::kClear);
         gVirtualX->SetTextFont(fTextFont);
         gVirtualX->SetTextSize(tsize);
      }
#else
      if (gVirtualX->GetTextFont() != fTextFont)   gVirtualX->SetTextFont(fTextFont);
      if (gVirtualX->GetTextSize() != tsize)       gVirtualX->SetTextSize(tsize);
#endif
      gVirtualX->SetTextAlign(fTextAlign);
      gVirtualX->SetTextColor(fTextColor);
   }

   gPad->SetAttTextPS(fTextAlign,fTextAngle,fTextColor,fTextFont,fTextSize);
}

//______________________________________________________________________________
void TAttText::ResetAttText(Option_t *)
{
//*-*-*-*-*-*-*-*-*Reset this text attributes to default values*-*-*-*-*-*-*
//*-*              ============================================

   fTextAlign  = 11;
   fTextAngle  = 0;
   fTextColor  = 1;
   fTextFont   = 62;
   fTextSize   = 0.05;
}

//______________________________________________________________________________
void TAttText::SaveTextAttributes(ofstream &out, const char *name, Int_t alidef, Float_t angdef, Int_t coldef, Int_t fondef, Float_t sizdef)
{
    // Save text attributes as C++ statement(s) on output stream out

   if (fTextAlign != alidef) {
      out<<"   "<<name<<"->SetTextAlign("<<fTextAlign<<");"<<endl;
   }
   if (fTextColor != coldef) {
      out<<"   "<<name<<"->SetTextColor("<<fTextColor<<");"<<endl;
   }
   if (fTextFont != fondef) {
      out<<"   "<<name<<"->SetTextFont("<<fTextFont<<");"<<endl;
   }
   if (fTextSize != sizdef) {
      out<<"   "<<name<<"->SetTextSize("<<fTextSize<<");"<<endl;
   }
   if (fTextAngle != angdef) {
      out<<"   "<<name<<"->SetTextAngle("<<fTextAngle<<");"<<endl;
   }
}

//______________________________________________________________________________
void TAttText::SetTextAttributes()
{
//*-*-*-*-*-*-*-*-*Invoke the DialogCanvas Text attributes*-*-*-*-*-*-*
//*-*              =======================================

   if (gPad) gROOT->SetSelectedPad(gPad->GetSelectedPad());

   TList *lc = (TList*)gROOT->GetListOfCanvases();
   if (!lc->FindObject("R__atttext")) {
      gROOT->ProcessLine("TAttTextCanvas *R__atttext = "
                         "new TAttTextCanvas(\"R__atttext\",\"Text Attributes\","
                         "400,600);");
   }
   gROOT->ProcessLine(Form("R__atttext->UpdateTextAttributes(%d,%f,%d,%d,%f);"
                           "R__atttext->Show();",fTextAlign,fTextAngle,
                           fTextColor,fTextFont,fTextSize));
}

//______________________________________________________________________________
void TAttText::SetTextSizePixels(Int_t npixels)
{
// Set the text size in pixels.
// If the font precision is greater than 2, the text size is set to npixels,
// otherwise the text size is computed as a per cent of the pad size.

   if (fTextFont%10 > 2) {
      fTextSize = Float_t(npixels);
   } else {
      TVirtualPad *pad = gROOT->GetSelectedPad();
      if (!pad) return;
      Float_t dy = pad->AbsPixeltoY(0) - pad->AbsPixeltoY(npixels);
      fTextSize = dy/(pad->GetY2() - pad->GetY1());
   }
}
