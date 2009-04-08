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
#include <QApplication>
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
TQtEventQueue::TQtEventQueue(): QQueue<const Event_t *> ()
{
   // Create the ROOT event queue
}

//______________________________________________________________________________
TQtEventQueue::~TQtEventQueue()
{
    // Remove all remaining events if any
    qDeleteAll(*this); 
}

//______________________________________________________________________________
int TQtEventQueue::RemoveItems(const Event_t *ev)
{ 
   // Removes all items matching ev->fWindow 
   // The removed item is deleted if auto-deletion (by default) is enabled
   // with class ctor
   
   // This method is used to debug the application only (by far)
   int counter = 0;
   assert(0);
   if (ev) { }
   return counter;
}

