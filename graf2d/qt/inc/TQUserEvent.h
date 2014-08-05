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

#ifndef ROOT_TQUserEvent
#define ROOT_TQUserEvent

#include <qglobal.h>
#if QT_VERSION < 0x40000
  #include <qevent.h>
#endif /* QT_VERSION */
#include "GuiTypes.h"

class TQUserEvent : public
#if QT_VERSION < 0x40000
   QCustomEvent
#else
   QEvent
#endif
{
#if QT_VERSION >= 0x40000
private:
   Event_t *fEvent;
#endif
public:
#if QT_VERSION >= 0x40000
   Event_t *data() const { return fEvent;}
   void setData(Event_t *ev) { delete data(); fEvent=ev;}
   TQUserEvent(const Event_t &pData) : QEvent(Type(QEvent::User+Type(1))),
                                       fEvent(0)
#else
   TQUserEvent(const Event_t &pData) : QCustomEvent(Id(),0)
#endif
   {   setData(new Event_t); *(Event_t *)data() = pData;  }
   ~TQUserEvent() { delete (Event_t *)data(); }
   void getData( Event_t &user) const { user = *(Event_t*)data(); }
   static Type Id() { return Type(QEvent::User + Type(1) /*kQTGClientEvent*/) ;}
};

#endif
