// Author: Valeri Fine   21/01/2002
/****************************************************************************
** $Id$
**
** Copyright (C) 2002 by Valeri Fine. Brookhaven National Laboratory.
**                                    All rights reserved.
**
**
*****************************************************************************/


/////////////////////////////////////////////////////////////////////////////////
//
// TQtPadFont class is Qt QFont class with TAttText ROOT class interface
//
/////////////////////////////////////////////////////////////////////////////////

#include "TQtPadFont.h"
#include "TSystem.h"
#include "TMath.h"
#include <QFontMetrics>
#include <QDebug>

TString TQtPadFont::fgRomanFontName   = "Times New Roman";
TString TQtPadFont::fgArialFontName   = "Arial";
TString TQtPadFont::fgCourierFontName = "Courier New";
TString TQtPadFont::fgSymbolFontFamily= "Symbol";


//______________________________________________________________________________
static float CalibrateFont()
{
    // Use the ROOT font with ID=1 to calibrate the current font on fly;
    // Environment variable ROOTFONTFACTOR allows to set the factor manually
    static float fontCalibFactor = -1;
    if (fontCalibFactor  < 0 ) {

       const char * envFactor = gSystem->Getenv("ROOTFONTFACTOR");
       bool ok=false;
       if (envFactor && envFactor[0])
          fontCalibFactor= QString(envFactor).toFloat(&ok);
       if (!ok) {
          TQtPadFont pattern;
          pattern.SetTextFont(62);

          int w,h;
          QFontMetrics metrics(pattern);
          w = metrics.width("This is a PX distribution");
          h = metrics.height();

//  X11 returns          h = 12
//  XFT returns          h = 14
//  WIN32 TTF returns    h = 16
//  Nimbus Roman returns h = 18
//  Qt4 XFT              h = 21

          qDebug() << " Font metric w = " << w <<" h = "<< h
                   << "points=" << pattern.pointSize()
                   << "pixels=" << pattern.pixelSize()
                   << pattern;
// graf2d/graf/src/TTF.cxx:const Float_t kScale = 0.93376068
          const Float_t kScale = 0.95; //0.93376068;
          float f;
          switch (h) {
             case 12: f = 1.10;  break;// it was  f = 1.13 :-(;
             case 14: f = 0.915; break;// it was f = 0.94  :-(;
             case 16: f = 0.965; break;// to be tested yet
             case 18: f = 0.92;  break;// to be tested yet
             case 20: f = 0.99;  break;// to be tested yet
             case 21: f = 0.90;  break;// to be tested yet
             default: f = kScale;  break;
          }
          fontCalibFactor = f;
       }
    }
    return fontCalibFactor;
}

//______________________________________________________________________________
static inline float FontMagicFactor(float size)
{
   // Adjust the font size to match that for Postscipt format
   static float calibration =0;
   if (calibration == 0) calibration = CalibrateFont();
   return TMath::Max(calibration*size,Float_t(1.0));
}

//______________________________________________________________________________
TQtPadFont::TQtPadFont(): TAttText()
{fTextFont = -1;fTextSize = -1; }

//______________________________________________________________________________
void  TQtPadFont::SetTextFont(const char *fontname, int italic_, int bold_)
{

   //*-*    mode              : Option message
   //*-*    italic   : Italic attribut of the TTF font
   //*-*    bold     : Weight attribute of the TTF font
   //*-*    fontname : the name of True Type Font (TTF) to draw text.
   //*-*
   //*-*    Set text font to specified name. This function returns 0 if
   //*-*    the specified font is found, 1 if not.

   this->setWeight((long) bold_*10);
   this->setItalic((Bool_t)italic_);
   this->setFamily(fontname);

   if (!strcmp(fontname,RomanFontName())) {
      this->setStyleHint(QFont::Serif);
   } else if (!strcmp(fontname,ArialFontName())) {
      this->setStyleHint(QFont::SansSerif);
   } else if (!strcmp(fontname,CourierFontName())){
      this->setStyleHint(QFont::TypeWriter);
   }
   this->setStyleStrategy(QFont::PreferDevice);
#if 0
   qDebug() << "TQtPadFont::SetTextFont:" << fontname
         << this->lastResortFamily ()
         << this->lastResortFont ()
         << this->substitute (fontname)
         << "ROOT  font number=" << fTextFont;
#endif

#if 0
   qDebug() << "TGQt::SetTextFont font:"    << fontname
            << " bold="  << bold_
            << " italic="<< italic_;
#endif
}

//______________________________________________________________________________
void  TQtPadFont::SetTextFont(Font_t fontnumber)
{
   //*-*-*-*-*-*-*-*-*-*-*-*-*Set current text font number*-*-*-*-*-*-*-*-*-*-*-*
   //*-*                      ===========================
   //*-*  List of the currently supported fonts (screen and PostScript)
   //*-*  =============================================================
   //*-*   Font ID       X11                       Win32 TTF       lfItalic  lfWeight x 10
   //*-*        1 : times-medium-i-normal      "Times New Roman"      1           5
   //*-*        2 : times-bold-r-normal        "Times New Roman"      0           8
   //*-*        3 : times-bold-i-normal        "Times New Roman"      1           8
   //*-*        4 : helvetica-medium-r-normal  "Arial"                0           5
   //*-*        5 : helvetica-medium-o-normal  "Arial"                1           5
   //*-*        6 : helvetica-bold-r-normal    "Arial"                0           8
   //*-*        7 : helvetica-bold-o-normal    "Arial"                1           8
   //*-*        8 : courier-medium-r-normal    "Courier New"          0           5
   //*-*        9 : courier-medium-o-normal    "Courier New"          1           5
   //*-*       10 : courier-bold-r-normal      "Courier New"          0           8
   //*-*       11 : courier-bold-o-normal      "Courier New"          1           8
   //*-*       12 : symbol-medium-r-normal     "Symbol"               0           6
   //*-*       13 : times-medium-r-normal      "Times New Roman"      0           5
   //*-*       14 :                            "Wingdings"            0           5

   if ( (fTextFont == fontnumber)  || (fontnumber <0) ) return;
   TAttText::SetTextFont(fontnumber);

   int it, bld;
   const char *fontName;

   switch(fTextFont/10) {

   case  1:
      it  = 1;
      bld = 5;
      fontName = RomanFontName();
      break;
   case  2:
      it  = 0;
      bld = 8;
      fontName = RomanFontName();
      break;
   case  3:
      it  = 1;
      bld = 8;
      fontName = RomanFontName();
      break;
   case  4:
      it  = 0;
      bld = 5;
      fontName = ArialFontName();
      break;
   case  5:
      it  = 1;
      bld = 5;
      fontName = ArialFontName();
      break;
   case  6:
      it  = 0;
      bld = 8;
      fontName = ArialFontName();
      break;
   case  7:
      it  = 1;
      bld = 8;
      fontName = ArialFontName();
      break;
   case  8:
      it   = 0;
      bld  = 5;
      fontName = CourierFontName();
      break;
   case  9:
      it  = 1;
      bld = 5;
      fontName = CourierFontName();
      break;
   case 10:
      it  = 0;
      bld = 8;
      fontName = CourierFontName();
      break;
   case 11:
      it  = 1;
      bld = 8;
      fontName = CourierFontName();
      break;
   case 12:
      it  = 0;
      bld = 5;
      fontName = SymbolFontFamily();
      break;
   case 13:
      it  = 0;
      bld = 5;
      fontName = RomanFontName();
      break;
   case 14:
      it  = 0;
      bld = 5;
      fontName = "Wingdings";
      break;
   default:
      it  = 0;
      bld = 5;
      fontName = RomanFontName();
      break;

   }
   SetTextFont(fontName, it, bld);
}

//______________________________________________________________________________
void  TQtPadFont::SetTextSize(Float_t textsize)
{
   //*-*-*-*-*-*-*-*-*-*-*-*-*Set current text size*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   //*-*                      =====================
   if ( fTextSize != textsize ) {
      TAttText::SetTextSize(textsize);
      if (fTextSize > 0) {
         Int_t   tsize =(Int_t)( textsize+0.5);
         this->setPixelSize(static_cast<int>(FontMagicFactor(tsize)));
      }
   }
}
//______________________________________________________________________________
 void  TQtPadFont::SetTextSizePixels(Int_t npixels)
 {
    // Set the text pixel size
    SetTextSize(static_cast<float>(npixels));
 }
//______________________________________________________________________________
const char *TQtPadFont::RomanFontName()
{
   // Get the system name for the "Roman" font
   return fgRomanFontName;
}

//______________________________________________________________________________
const char *TQtPadFont::ArialFontName()
{
   // Get the system name for the "Arial" font
   return fgArialFontName;
}

//______________________________________________________________________________
const char *TQtPadFont::CourierFontName()
{
   // Get the system name for the "Courier" font
   return fgCourierFontName;
}
//______________________________________________________________________________
const char *TQtPadFont::SymbolFontFamily()
{
   // Get the system name for the "Symbol" font
   return fgSymbolFontFamily;
}
//______________________________________________________________________________
void TQtPadFont::SetSymbolFontFamily(const char *symbolFnName)
{
   // Set the system name for the "Symbol" font
   fgSymbolFontFamily = symbolFnName;  // we need the TString here !!!
}

//______________________________________________________________________________
void   TQtPadFont::SetTextMagnify(Float_t  mgn)
{
   //
   // Scale the font accroding the inout mgn magnification factor
   // mgn        : magnification factor
   // -------
   // see: TVirtualX::DrawText(int x, int y, float angle, float mgn, const char *text, TVirtualX::ETextMode /*mode*/)
   //
    Int_t tsize = (Int_t)(fTextSize+0.5);
    if (TMath::Abs(mgn-1) >0.05)  {
       int pxSize = int(mgn*FontMagicFactor(tsize));
       if(pxSize<=0) pxSize=1;
       this->setPixelSize(pxSize);
    }
}
