// @(#)root/qt:$Name:$:$Id:$
// Author: Valeri Fine   25/03/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * Copyright (C) 2004 by Valeri Fine.                                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TQtEventQueue
#define ROOT_TQtEventQueue

#include "GuiTypes.h"
#include <qptrlist.h>

//////////////////////////////////////////////////////////////////////////
//
//  TQtEventQueue is a queue container of the pointers of Event_t
//  structures created by TQtClientFilter class.
//  If auto-deleting is turned on, all the items in a collection are
//  deleted when the collection itself is deleted.
//  For the full list of the members see:
//  http://doc.trolltech.com/3.3/qptrlist.html
//
//////////////////////////////////////////////////////////////////////////


class TQtEventQueue : public QPtrList<Event_t> {
   public:
      TQtEventQueue(bool autoDelete=true);
      TQtEventQueue(const TQtEventQueue &src): QPtrList<Event_t>(src) {;}
      virtual ~TQtEventQueue(){}
      void     enqueue(const Event_t *);
      Event_t *dequeue();
      int      RemoveItems(const Event_t *ev);

   protected:
      virtual int compareItems(QPtrCollection::Item item1, QPtrCollection::Item item2);
};
//______________________________________________________________________________
inline void TQtEventQueue::enqueue(const Event_t *ev)
{    append(ev);                              }
//______________________________________________________________________________
inline Event_t *TQtEventQueue::dequeue()
{       return isEmpty() ? 0 : take(0);                       }

#endif
