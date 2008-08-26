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

#include "TQtEventQueue.h"
#include "TQtLock.h"
#include <qapplication.h>
#include <cassert>


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

//______________________________________________________________________________
#if QT_VERSION < 0x40000
TQtEventQueue::TQtEventQueue(bool autoDelete): QPtrList<Event_t> ()
{ 
  //  If auto-deleting is turned on, all the items in a collection 
  //  are deleted when the collection itself is deleted.
   setAutoDelete(autoDelete); 
}
#else /* QT_VERSION */
TQtEventQueue::TQtEventQueue(bool /*autoDelete*/): QQueue<const Event_t *> ()
{ 
  //  If auto-deleting is turned on, all the items in a collection 
  //  are deleted when the collection itself is deleted.
}
#endif /* QT_VERSION */

//______________________________________________________________________________
TQtEventQueue::~TQtEventQueue()
{
#if QT_VERSION >= 0x40000
    qDeleteAll(*this); 
#endif
}
#if 0
//______________________________________________________________________________
#if QT_VERSION < 0x40000
int TQtEventQueue::compareItems(QPtrCollection::Item i1, QPtrCollection::Item i2)
#else /* QT_VERSION */
int TQtEventQueue::compareItems(Q3PtrCollection::Item i1, Q3PtrCollection::Item i2)
#endif /* QT_VERSION */
{
//   This virtual function compares two list items. 
//   Returns:   zero if item1 == item2  
//   --------   nonzero if item1 != item2 
//             
//   This function returns int rather than bool 
//   so that reimplementations can return three values 
//   and use it to sort by: 
//       0 if item1 == item2 
//     > 0 (positive integer) if item1 > item2 
//     < 0 (negative integer) if item1 < item2 
   Event_t &ev1 = *(Event_t *)i1;
   Event_t &ev2 = *(Event_t *)i2;
   return ev1.fWindow - ev2.fWindow;
}
#endif

//______________________________________________________________________________
int TQtEventQueue::RemoveItems(const Event_t *ev)
{ 
   // Removes all items matching ev->fWindow 
   // The removed item is deleted if auto-deletion (by default) is enabled
   // with class ctor
   
   // This method is used to debug the application only (by far)
   int counter = 0;
   assert(0);
   if (ev) {
#if 0
      TQtLock lock;
#if QT_VERSION < 0x40000
      int next = find(ev);
      while(next != -1) {
         remove();            // The removed item is deleted also
         next = findNext(ev);
         counter++;
      }
#else
      // to be done yet.  
      assert(0);
      int next = remove_if(begin(),end(),ev);
      while(next != -1) {
         remove();            // The removed item is deleted also
         next = findNext(ev);
         counter++;
      }
#endif
#endif 
   }
   return counter;
}
