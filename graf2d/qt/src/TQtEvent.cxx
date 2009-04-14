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

#include <QObject>
#include "TQtEvent.h"
#include "TWaitCondition.h"
#include <assert.h>

//////////////////////////////////////////////////////////////////////////////
//
//  class TQtEvent to send an event between two Qt threads
//
//////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
TQtEvent::TQtEvent(int code):QEvent(QEvent::Type(QEvent::User+code)), fCondition(0), fResult(0)
         , fReceiver(0),fThatEvent(0)
{ }
//______________________________________________________________________________
TQtEvent::TQtEvent(QObject *o, QEvent *e): QEvent(QEvent::User), fCondition(0)
         , fResult(0), fReceiver(o),fThatEvent(e)
{ }
//______________________________________________________________________________
bool TQtEvent::Notify()
{
  bool r = FALSE;
  if (fReceiver)
  {
    r = fReceiver->event(fThatEvent);
    SetResult();
  }
  return r;
}
//______________________________________________________________________________
void TQtEvent::SetResult(unsigned long e)
{
  if (fResult)   *fResult = e;
  if (fCondition) fCondition->wakeOne();
}
//______________________________________________________________________________
void TQtEvent::SetWait(TWaitCondition &condition)
{
  fCondition = &condition;
}
//______________________________________________________________________________
void TQtEvent::SetWait(TWaitCondition &condition,unsigned long &result)
{
  SetWait(condition);
  fResult    = &result;
}
