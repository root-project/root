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

#include "TQtRootThread.h"
#include "Rtypes.h"
#include "qobject.h"
#include <assert.h>

#include "TWaitCondition.h"
#include "TQtRConfig.h"

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
