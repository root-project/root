#ifndef ROOT_TQTROOTTHREAD
#define ROOT_TQTROOTTHREAD

// Author: Valeri Fine   21/01/2002
/****************************************************************************
** $Id: TQtRootThread.h,v 1.5 2004/06/28 20:16:54 fine Exp $
**
** Copyright (C) 2002 by Valeri Fine.  All rights reserved.
**
** This file may be distributed under the terms of the Q Public License
** as defined by Trolltech AS of Norway and appearing in the file
** LICENSE.QPL included in the packaging of this file.
*****************************************************************************/

#include "qthread.h"
#include "TQtRConfig.h"

class TWaitCondition;
class QEvent;

class TQtRootThread : public QThread  {

  protected:

    TWaitCondition *fCondition;
    Qt::HANDLE      fThreadHandleId;  // current thread id

  protected:
    void SetThreadId();           // must be called by any derived class from its run method only

  public:
    TQtRootThread(TWaitCondition *condition=0):fCondition(condition),fThreadHandleId(0) {;}
    virtual ~TQtRootThread();
    Qt::HANDLE GetThreadId(){ return  fThreadHandleId;}
    bool   IsThisThread();
    void   SetWait(TWaitCondition &condition);
    bool   Wait(unsigned long time= ULONG_MAX);
    void   WakeOne();

    virtual void run();
    virtual void Run() = 0;
};
//______________________________________________________________________________
inline bool TQtRootThread::IsThisThread()
{
#ifdef R__QTGUITHREAD
  return (fThreadHandleId == QThread::currentThread());
#else
  return TRUE;
#endif
}
//______________________________________________________________________________
inline void TQtRootThread::SetWait(TWaitCondition &condition) {fCondition = &condition;}

#endif
