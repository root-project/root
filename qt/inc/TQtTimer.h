// @(#)root/qt:$Name:  $:$Id: TQtTimer.h,v 1.1 2004/08/13 06:21:09 brun Exp $
// Author: Valeri Fine   09/08/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * Copyright (C) 2002 by Valeri Fine.                                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

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
