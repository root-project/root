// Author: Valeri Fine   21/01/2002
/****************************************************************************
** $Id: TQtBrush.h,v 1.3 2004/06/08 21:13:36 fine Exp $
**
** Copyright (C) 2002 by Valeri Fine.  All rights reserved.
**
** This file may be distributed under the terms of the Q Public License
** as defined by Trolltech AS of Norway and appearing in the file
** LICENSE.QPL included in the packaging of this file.
*****************************************************************************/

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
