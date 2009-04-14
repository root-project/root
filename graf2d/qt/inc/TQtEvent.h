// @(#)root/qt:$Id$
// Author: Valeri Fine   21/01/2002

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * Copyright (C) 2002 by Valeri Fine.                                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TQtEvent
#define ROOT_TQtEvent

#include <QEvent>

class TQtObject;
class TWaitCondition;
//______________________________________________________________________________
class TQtEvent : public QEvent 
{

private:
    TWaitCondition *fCondition;
    unsigned long *fResult;   // QApplication owns QEvent and will destroy it
    QObject *fReceiver;
    QEvent  *fThatEvent;

public:
    TQtEvent(int code);
    TQtEvent(QObject *o, QEvent *e);
    virtual ~TQtEvent(){}
    void SetWait(TWaitCondition &condition, unsigned long  &result);
    void SetWait(TWaitCondition &condition);
    void SetResult(unsigned long e=0);
 //   QEvent *WaitResult(); too dangerous
    bool Notify();
    virtual void ExecuteCB(){;}
};

#endif
