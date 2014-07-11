// @(#)root/qt:$Name:  $:$Id$
// Author: Valeri Fine   21/01/2002
/****************************************************************************
**
** Copyright (C) 2002 by Valeri Fine.  All rights reserved.
**
*****************************************************************************/

#ifndef ROOT_TQtPadFont
#define ROOT_TQtPadFont

#include "TAttText.h"
#include "TString.h"

#ifndef __CINT__
#  include <QFont>
#else
   class  QFont;
#endif
   //
   // TQtPadFont class is Qt QFont class with TAttText ROOT class interface
   //
class TQtPadFont : public QFont, public TAttText
{
private:
   static TString fgRomanFontName;
   static TString fgArialFontName;
   static TString fgCourierFontName;
   static TString fgSymbolFontFamily;

public:
   TQtPadFont();
   TQtPadFont(const TQtPadFont &src):QFont(src),TAttText(src) {}
   virtual ~TQtPadFont(){;}
   void  SetTextFont(const char *fontname, int italic, int bold);
   void  SetTextFont(Font_t fontnumber=62);
   void  SetTextSize(Float_t textsize=1);
   void  SetTextSizePixels(Int_t npixels);
   void  SetTextMagnify(Float_t mgn);
   static const char *RomanFontName();
   static const char *ArialFontName();
   static const char *CourierFontName();
   static const char *SymbolFontFamily();
   static void SetSymbolFontFamily(const char *symbolFnName="Symbol");

   ClassDef(TQtPadFont,0) //< Create Qt QFont object based on ROOT TAttText attributes
};

#endif
