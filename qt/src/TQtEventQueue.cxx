// Author: Valeri Fine   25/03/2004

#include "TQtEventQueue.h"
#include <qapplication.h> 

/****************************************************************************
** $Id: TQtEventQueue.cxx,v 1.3 2004/03/25 17:56:07 fine Exp $
**
** Copyright (C) 2004 by Valeri Fine. Brookhaven National Laboratory.
**                                    All rights reserved.
**
** This file may be distributed under the terms of the Q Public License
** as defined by Trolltech AS of Norway and appearing in the file
** LICENSE.QPL included in the packaging of this file.
**
*****************************************************************************/

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
TQtEventQueue::TQtEventQueue(bool autoDelete): QPtrList<Event_t> ()
{ 
  //  If auto-deleting is turned on, all the items in a collection 
  //  are deleted when the collection itself is deleted.
   setAutoDelete(autoDelete); 
}

//______________________________________________________________________________
int TQtEventQueue::compareItems(QPtrCollection::Item i1, QPtrCollection::Item i2)
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

//______________________________________________________________________________
int TQtEventQueue::RemoveItems(const Event_t *ev)
{ 
   // Removes all items matching ev->fWindow 
   // The removed item is deleted if auto-deletion (by default) is enabled
   // with class ctor
   
   int counter = 0;
   if (ev) {
      qApp->lock();
      int next = find(ev);
      while(next != -1) {
         remove();            // The removed item is deleted also
         next = findNext(ev);
         counter++;
      }
      qApp->unlock();
   }
   return counter;
}
