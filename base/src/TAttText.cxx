// @(#)root/base:$Name:  $:$Id: TAttText.cxx,v 1.1.1.1 2000/05/16 17:00:38 rdm Exp $
// Author: Rene Brun   12/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <fstream.h>

#include "TROOT.h"
#include "Strlen.h"
#include "TAttText.h"
#include "TVirtualPad.h"
#include "TStyle.h"
#include "TVirtualX.h"
#include "TError.h"

static Float_t x11factor[13] ={1.000,1.000,1.010,0.910,0.920,0.920,0.925,1.204,
                               1.204,1.168,1.166,1.007,1.026};

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
//*-*  by many other classes (graphics, histograms).
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
   if (!gStyle) return;
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
//*-*             precision = 1 fast hardware fonts (steps in the size)
//*-*             precision = 2 scalable and rotatable hardware fonts
//*-*    size  : Character size expressed in percentage of the current pad height
//*-*            The textsize in pixels (say charheight) will be:
//*-*             charheight = textsize*canvas_height if current pad is horizontal.
//*-*             charheight = textsize*canvas_width  if current pad is vertical.
//*-*
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
void TAttText::Copy(TAttText &atttext)
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
         Float_t mgn = x11factor[ifpx11-1]*rsize/Float_t(ihh);
         if (mgn <0)    mgn = 1;
         if (mgn > 100) mgn = 100;
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
