// @(#)root/qt:$Name:  $:$Id$
// Author: Valeri Fine   21/01/2002
/****************************************************************************
**
** Copyright (C) 2002 by Valeri Fine.  All rights reserved.
**
** This file may be distributed under the terms of the Q Public License
** as defined by Trolltech AS of Norway and appearing in the file
** LICENSE.QPL included in the packaging of this file.
*****************************************************************************/
#ifndef ROOT_TQtPen
#define ROOT_TQtPen

#include "Riostream.h"
#include "TAttLine.h"

#ifndef __CINT__
#  include <QtGui/QPen>
#else
   class  QPen;
#endif
   //
   // TQtPen creates the QPen object to map to ROOT  TAttLine attributes
   //
class TQtPen : public QPen, public TAttLine
{

public:
   TQtPen();
   TQtPen(const TQtPen &src):QPen(src),TAttLine(src) {}
   TQtPen(const TAttLine &line);
   TQtPen &operator=(const TAttLine &line);
   virtual ~TQtPen(){;}
   void  SetLineAttributes() { TAttLine::SetLineAttributes(); }
   void  SetLineAttributes(const TAttLine &lineAttributes);
   void  SetLineColor(Color_t cindex);
   void  SetLineType(int n, int*dash);
   void  SetLineStyle(Style_t linestyle);
   void  SetLineWidth(Width_t width);
};

#endif
