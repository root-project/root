// Author: Valeri Fine   21/01/2002
/****************************************************************************
** $Id: TQtApplicationThread.cxx,v 1.5 2002/09/15 03:33:10 fine Exp $
**
** Copyright (C) 2002 by Valeri Fine. Brookhaven National Laboratory.
**                                    All rights reserved.
**
** This file may be distributed under the terms of the Q Public License
** as defined by Trolltech AS of Norway and appearing in the file
** LICENSE.QPL included in the packaging of this file.
**
*****************************************************************************/

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
