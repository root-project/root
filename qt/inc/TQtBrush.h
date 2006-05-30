// @(#)root/qt:$Name:  $:$Id: TQtBrush.h,v 1.4 2006/03/24 15:31:10 antcheva Exp $
// Author: Valeri Fine   21/01/2002

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * Copyright (C) 2002 by Valeri Fine.                                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TQtBrush
#define ROOT_TQtBrush

#ifndef __CINT__
#  include <qbrush.h>
#  include <qcolor.h>
#  include <qpixmap.h>
#else
   class  QColor;
   class  QBrush;
   class  QPixmap;
#endif

#include "Rtypes.h"
   //
   // TQtBrush creates the QBrush Qt object based on the ROOT "fill" attributes 
   //
class TQtBrush : public QBrush
{
protected:
  QColor fBackground;
  int fStyle;
  int fFasi;
#ifdef R__WIN32
  QPixmap fCustomPixmap; // shadow transparent pixmap fro WIN32
#endif
public:
   TQtBrush();
   TQtBrush(const TQtBrush &src):QBrush(src)
   {
      fBackground=src.fBackground;
      fStyle=src.fStyle;
      fFasi=src.fFasi;
   }
   virtual ~TQtBrush(){;}
   void SetStyle(int style=1000){  SetStyle(style/1000,style%1000); };
   void SetStyle(int style, int fasi);
   void SetColor(const QColor &color);
   const QColor &GetColor() const { return fBackground;}
   int   GetStyle()         const { return 1000*fStyle + fFasi; }
   ClassDef(TQtBrush,0); // create QBrush object based on the ROOT "fill" attributes 
};

#endif
