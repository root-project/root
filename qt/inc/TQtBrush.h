// @(#)root/qt:$Name:$:$Id:$
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

#include "qbrush.h"
#include "qcolor.h"

class TQtBrush : public QBrush
{
protected:
  QColor fBackground;
  int fStyle;
  int fFasi;
public:
   TQtBrush();
   TQtBrush(const TQtBrush &src):QBrush(src)
   {
      fBackground=src.fBackground;
      fStyle=src.fStyle;
      fFasi=src.fFasi;
   }
   ~TQtBrush(){;}
   void SetStyle(int style=1000){  SetStyle(style/1000,style%1000); };
   void SetStyle(int style, int fasi);
   void SetColor(QColor &color);
   const QColor &GetColor() const { return fBackground;}
   int    GetStyle() const  { return 1000*fStyle + fFasi; }
};

#endif
