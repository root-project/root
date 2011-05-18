// @(#)root/qt:$Id$
// Author: Valeri Fine   06/01/2006
/****************************************************************************
**
** Copyright (C) 2002 by Valeri Fine.  All rights reserved.
**
** This file may be distributed under the terms of the Q Public License
** as defined by Trolltech AS of Norway and appearing in the file
** LICENSE.QPL included in the packaging of this file.
*****************************************************************************/

#ifndef ROOT_TQtEmitter
#define ROOT_TQtEmitter


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TQtEmitter - is a proxy class tyo emti the Qt signals on behalf      //
// of TGQt non-Qt class                                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TQtRConfig.h"
#ifndef __CINT__
#include <qobject.h>
class QPixmap;

class TQtEmitter : public QObject {
  Q_OBJECT
private:
   friend class TGQt;
  void EmitPadPainted(QPixmap *p)  { emit padPainted(p);}
protected:
  TQtEmitter& operator=(const TQtEmitter&); // AXEL: intentionally not implementedpublic:
  TQtEmitter(){};
signals:
  void padPainted(QPixmap *p);
};
#endif
#endif 
