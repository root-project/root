// @(#)root/gui:$Id: TGEventHandler.cxx
// Author: Bertrand Bellenot   29/01/2008

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGEventHandler                                                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGEventHandler.h"
#include "TGWindow.h"

ClassImp(TGEventHandler)

//______________________________________________________________________________
Bool_t TGEventHandler::HandleEvent(Event_t *ev)
{
   // Handle the event. Returns true if the event has been handled,
   // false otherwise.

   return fWindow->HandleEvent(ev);
}

//______________________________________________________________________________
void TGEventHandler::SendMessage(const TGWindow *w, Long_t msg, Long_t parm1,
                                 Long_t parm2)
{
   // Send message (i.e. event) to window w. Message is encoded in one long
   // as message type and up to two long parameters.

   Event_t event;

   if (w) {
      event.fType   = kClientMessage;
      event.fFormat = 32;
      event.fHandle = gROOT_MESSAGE;

      event.fWindow  = w->GetId();
      event.fUser[0] = msg;
      event.fUser[1] = parm1;
      event.fUser[2] = parm2;
      event.fUser[3] = 0;
      event.fUser[4] = 0;

      gVirtualX->SendEvent(w->GetId(), &event);
   }
}
