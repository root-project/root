// Author: Valeri Fine   21/01/2002
/****************************************************************************
** $Id$
**
** Copyright (C) 2002 by Valeri Fine. Brookhaven National Laboratory.
**                                    All rights reserved.
**
** This file may be distributed under the terms of the Q Public License
** as defined by Trolltech AS of Norway and appearing in the file
** LICENSE.QPL included in the packaging of this file.
**
*****************************************************************************/


/////////////////////////////////////////////////////////////////////////////////
//
// TQtPadFont creates the QFort object to map to ROOT  TAttText attributes
//
/////////////////////////////////////////////////////////////////////////////////

#include "TQtPadFont.h"
#include "TSystem.h"
#include "TMath.h"
#include <qfontmetrics.h>
#if QT_VERSION >= 0x40000
#  include <QDebug>
#endif

const char *TQtPadFont::fgRomanFontName   = "Times New Roman";
const char *TQtPadFont::fgArialFontName   = "Arial";
const char *TQtPadFont::fgCourierFontName = "Courier New";
const char *TQtPadFont::fgSymbolFontFamily= "Symbol";


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
          pattern.SetTextFont(6);

          int w,h;
          QFontMetrics metrics(pattern);
          w = metrics.width("This is a PX distribution");
          h = metrics.height();

// I found 0.94 matches well what Rene thinks it should be
// for TTF and XFT and it should be 1.1 for X Fonts
//
//  X11 returns          h = 12
//  XFT returns          h = 14
//  WIN32 TTF returns    h = 16
//  Nimbus Roman returns h = 18
//  Qt4 XFT              h = 20

          // printf(" Font metric w = %d , h = %d\n", w,h);
          float f;
          switch (h) {
             case 12: f = 1.10;  break;// it was  f = 1.13 :-(;
             case 14: f = 0.915; break;// it was f = 0.94  :-(;
             case 16: f = 0.965; break;// to be tested yet
             case 18: f = 0.92;  break;// to be tested yet
             case 20: f = 0.99;  break;// to be tested yet
             default: f = 1.10;  break;
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
{}

//______________________________________________________________________________
void  TQtPadFont::SetTextFont(const char *fontname, int italic, int bold)
{

   //*-*    mode              : Option message
   //*-*    italic   : Italic attribut of the TTF font
   //*-*    bold     : Weight attribute of the TTF font
   //*-*    fontname : the name of True Type Font (TTF) to draw text.
   //*-*
   //*-*    Set text font to specified name. This function returns 0 if
   //*-*    the specified font is found, 1 if not.

   this->setWeight((long) bold*10);
   this->setItalic((Bool_t)italic);
   this->setFamily(fontname);
#if QT_VERSION >= 0x40000
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
#endif
   // fprintf(stderr, "TGQt::SetTextFont font: <%s> bold=%d italic=%d\n",fontname,bold,italic);
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

   int italic, bold;
   const char *fontName = RomanFontName();

   switch(fTextFont/10) {

   case  1:
      italic = 1;
      bold   = 5;
      fontName = RomanFontName();
      break;
   case  2:
      italic = 0;
      bold   = 8;
      fontName = RomanFontName();
      break;
   case  3:
      italic = 1;
      bold   = 8;
      fontName = RomanFontName();
      break;
   case  4:
      italic = 0;
      bold   = 5;
      fontName = ArialFontName();
      break;
   case  5:
      italic = 1;
      bold   = 5;
      fontName = ArialFontName();
      break;
   case  6:
      italic = 0;
      bold   = 8;
      fontName = ArialFontName();
      break;
   case  7:
      italic = 1;
      bold   = 8;
      fontName = ArialFontName();
      break;
   case  8:
      italic = 0;
      bold   = 5;
      fontName = CourierFontName();
      break;
   case  9:
      italic = 1;
      bold   = 5;
      fontName = CourierFontName();
      break;
   case 10:
      italic = 0;
      bold   = 8;
      fontName = CourierFontName();
      break;
   case 11:
      italic = 1;
      bold   = 8;
      fontName = CourierFontName();
      break;
   case 12:
      italic = 0;
      bold   = 5;
      fontName = SymbolFontFamily();
      break;
   case 13:
      italic = 0;
      bold   = 5;
      fontName = RomanFontName();
      break;
   case 14:
      italic = 0;
      bold   = 5;
      fontName = "Wingdings";
      break;
   default:
      italic = 0;
      bold   = 5;
      fontName = RomanFontName();
      break;

   }
   SetTextFont(fontName, italic, bold);
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
         SetTextSizePixels(int(FontMagicFactor(tsize)));
      }
   }
}
//______________________________________________________________________________
 void  TQtPadFont::SetTextSizePixels(Int_t npixels)
 {
    // Set the text pixel size
    TAttText::SetTextSizePixels(npixels);
    this->setPixelSize(int(FontMagicFactor(npixels)));
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
void   TQtPadFont::SetTextMaginfy(Float_t  mgn)
{
   //
   // Scale the font accroding the inout mgn magnification factor
   // mgn        : magnification factor
   // -------
   // see: TVirtualX::DrawText(int x, int y, float angle, float mgn, const char *text, TVirtualX::ETextMode /*mode*/)
   //
    Int_t tsize = (Int_t)(fTextSize+0.5);
    if (TMath::Abs(mgn-1) >0.05)  this->setPixelSizeFloat(mgn*FontMagicFactor(tsize));
}
