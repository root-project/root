// Author: Valeri Fine   21/01/2002
/****************************************************************************
** $Id: TQtEvent.h,v 1.1.1.1 2002/03/27 18:17:02 fine Exp $
**
** Copyright (C) 2002 by Valeri Fine.  All rights reserved.
**
** This file may be distributed under the terms of the Q Public License
** as defined by Trolltech AS of Norway and appearing in the file
** LICENSE.QPL included in the packaging of this file.
*****************************************************************************/

#ifndef ROOT_TQtEvent
#define ROOT_TQtEvent

#include "qevent.h"

#include "TVirtualX.h"

class TQtObject;
class TWaitCondition;
//______________________________________________________________________________
class TQtEvent : public QCustomEvent 
{

private:
    TWaitCondition *fCondition;
    void   **fResult; // QApplication owns QEvent and will destroy it
    QObject *fReceiver;
    QEvent  *fThatEvent;

public:
    TQtEvent(int code);
    TQtEvent(QObject *o, QEvent *e);
    virtual ~TQtEvent(){}
    void SetWait(TWaitCondition &condition,void *&result);
    void SetWait(TWaitCondition &condition);
    void SetResult(void *e=0);
 //   QEvent *WaitResult(); too dangerous
    bool Notify();
    virtual void ExecuteCB(){;}
};

#endif
