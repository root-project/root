// Author: Valeri Fine   25/03/2004
#ifndef ROOT_TQtEventQueue
#define ROOT_TQtEventQueue

// @(#)root/qt:$Id$
// Author: Valeri Fine   25/03/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * Copyright (C) 2004 by Valeri Fine.                                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "GuiTypes.h"
#include <qglobal.h> 
#if QT_VERSION < 0x40000
#  include <qptrlist.h> 
#else /* QT_VERSION */
#  include <QQueue> 
#endif /* QT_VERSION */

/////////////////////////////////////////////////////////////////////////////////
//
//  TQtEventQueue is a queue container of the pointers of Event_t structures 
//  created by TQtClientFilter class
//  If auto-deleting is turned on, all the items in a collection are deleted when 
//  the collection itself is deleted.
//  (for the full list of the members see: 
//  http://doc.trolltech.com/3.3/qptrlist.html)
//
/////////////////////////////////////////////////////////////////////////////////


#if QT_VERSION < 0x40000
class TQtEventQueue : public QPtrList<Event_t> {
#else /* QT_VERSION */
class TQtEventQueue : public QQueue<const Event_t *> {
#endif /* QT_VERSION */
   public:
      TQtEventQueue(bool autoDelete=true);
#if QT_VERSION < 0x40000
      TQtEventQueue(const TQtEventQueue &src): QPtrList<Event_t>(src) {;}
#else /* QT_VERSION */
      TQtEventQueue(const TQtEventQueue &src): QQueue<const Event_t *>(src) {;}
#endif /* QT_VERSION */
      virtual ~TQtEventQueue();
      void     enqueue(const Event_t *);
      const Event_t *dequeue();
      int      RemoveItems(const Event_t *ev);

   protected:
#if 0
#if QT_VERSION < 0x40000
      virtual int compareItems(QPtrCollection::Item item1, QPtrCollection::Item item2);
#else /* QT_VERSION */
      virtual int compareItems(Q3PtrCollection::Item item1, Q3PtrCollection::Item item2);
#endif /* QT_VERSION */
#endif 
};
//______________________________________________________________________________
inline void TQtEventQueue::enqueue(const Event_t *ev)
{    
#if QT_VERSION < 0x40000
   append(ev);  
#else
   QQueue<const Event_t *>::enqueue(ev);
#endif
}
//______________________________________________________________________________
inline const Event_t *TQtEventQueue::dequeue()
{
   return isEmpty() ? 0 : 
#if QT_VERSION < 0x40000
            take(0);                       
#else
            QQueue<const Event_t *>::dequeue();
#endif
}

#endif
