// Author: Valery Fine   21/01/2002
/****************************************************************************
** $Id: TQtRootThread.cxx,v 1.8 2004/06/28 20:16:55 fine Exp $
**
** Copyright (C) 2002 by Valeri Fine. Brookhaven National Laboratory.
**                                    All rights reserved.
**
** This file may be distributed under the terms of the Q Public License
** as defined by Trolltech AS of Norway and appearing in the file
** LICENSE.QPL included in the packaging of this file.
**
*****************************************************************************/

#include "TQtRootThread.h"
#include "Rtypes.h"
#include "qobject.h"
#include <assert.h>

#include "TWaitCondition.h"
#include "TQtRConfig.h"

////////////////////////////////////////////////////////////////////////////////
//
// TQtRootThread is a subclass of QThread
// to synchronize the ROOT thread with the dedicated 
// GUI thread if needed
//
// Class has no WIN32 specific.
// WIN32 merely means the ROOT Qt layer does use the dedicted GUI thread
// That can be the case on UNIX also
//
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
TQtRootThread::~TQtRootThread(){}

//______________________________________________________________________________
void TQtRootThread::run()
{
  // Enter this thread event loop
#ifdef R__QTGUITHREAD
  fThreadHandleId = QThread::currentThread();
  // Call an user defined thread body
  Run();
#else
  assert(0);
  fThreadHandleId = 0;
  // UNIX has its own event loop
#endif

}
//______________________________________________________________________________
bool TQtRootThread::Wait(unsigned long time)
{
   // Wait for "time" msec if the fCiondition provided
   return fCondition ? fCondition->wait (time) : TRUE;
}
//______________________________________________________________________________
void TQtRootThread::WakeOne()
{
  // Wake thread if the fCiondition provided
  if(fCondition) fCondition->wakeOne();
}
