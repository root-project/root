// @(#)root/qt:$Name:  $:$Id: TQtTimer.cxx,v 1.3 2004/09/12 11:00:22 brun Exp $
// Author: Valery Fine  09/08/2004

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
  static TQtTimer *QtTimer();
};
inline TQtTimer *TQtTimer::QtTimer(){ return fgQTimer; }


#endif
