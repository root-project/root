// @(#)root/qt:$Name:  $:$Id: TQtTimer.h,v 1.2 2004/07/28 00:12:40 brun Exp $
// Author: Valeri Fine   09/08/2004
/****************************************************************************
** $Id: TQtTimer.h,v 1.1 2004/08/10 01:55:36 fine Exp $
**
** Copyright (C) 2004 by Valeri Fine.  All rights reserved.
**
** This file may be distributed under the terms of the Q Public License
** as defined by Trolltech AS of Norway and appearing in the file
** LICENSE.QPL included in the packaging of this file.
*****************************************************************************/

#ifndef ROOT_TQtTimer
#define ROOT_TQtTimer

#include <qtimer.h>  

//___________________________________________________________________
class  TQtTimer  : public QTimer {
 Q_OBJECT
protected:
  static TQtTimer *fgQTimer;
  int fCounter;     
  TQtTimer (QObject *parent=0, const char *name=0): QTimer(parent,name),fCounter(0){}
 protected slots:
  virtual void AwakeRootEvent();
 
public:
  static TQtTimer *Create(QObject *parent=0, const char *name=0);
};


#endif
