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

#ifndef ROOT_TQtRootThread
#define ROOT_TQtRootThread

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
