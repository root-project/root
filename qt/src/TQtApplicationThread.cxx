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

#include "TQtApplicationThread.h"
#include "TQtApplication.h"
#include "qapplication.h"
#include "TApplication.h"
#include "TQtEvent.h"

//______________________________________________________________________________
class TQtThreadDispatcher : public QObject {
protected:
  friend class TQtApplicationThread;
  TQtApplicationThread *fThread;
  TQtThreadDispatcher(TQtApplicationThread *that) : fThread(that) {}
  bool event(QEvent *e) {return fThread->eventCB((TQtEvent *)e); }
};

//______________________________________________________________________________
TQtApplicationThread::TQtApplicationThread(int argc, char **argv)
                     :  TQtRootThread(),fArgc(argc),fArgv(argv)
{ }
//______________________________________________________________________________
TQtApplicationThread::~TQtApplicationThread()
{ }
//______________________________________________________________________________
void TQtApplicationThread::Run()
{
  // QApplicatioin must be created  within the proper thread

  TQtApplication::CreateQApplication(gApplication->Argc(),gApplication->Argv(),kTRUE );
  WakeOne();
  connect(qApp,SIGNAL(aboutToQuit ()),this,SLOT(AboutToQuit()));
  while(qApp->exec());
}
//______________________________________________________________________________
bool TQtApplicationThread::eventCB(TQtEvent *evt)
{
  if (evt) evt->Notify();
  return TRUE;
}

//______________________________________________________________________________
void TQtApplicationThread::AboutToQuit ()
{
  // no GUI anymore - replace the pointer
  gVirtualX = gGXBatch;
}
