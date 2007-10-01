// @(#)root/gui:$Id$
// Author: Fons Rademakers   2/8/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGIdleHandler                                                        //
//                                                                      //
// Handle idle events, i.e. process GUI actions when there is nothing   //
// else anymore to do.                                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGIdleHandler.h"
#include "TGWindow.h"


ClassImp(TGIdleHandler)

//______________________________________________________________________________
TGIdleHandler::TGIdleHandler(TGWindow *w)
{
   // Create idle handler.

   if (w) {
      fWindow = w;
      if (fWindow->GetClient())
         fWindow->GetClient()->AddIdleHandler(this);
   } else
      Error("TGIdleHandler", "window cannot be 0");
}

//______________________________________________________________________________
TGIdleHandler::~TGIdleHandler()
{
   // Delete idle handler.

   if (fWindow->GetClient())
      fWindow->GetClient()->RemoveIdleHandler(this);
}

//______________________________________________________________________________
Bool_t TGIdleHandler::HandleEvent()
{
   // Handle the idle event. Returns true if the event has been handled,
   // false otherwise.

   return fWindow->HandleIdleEvent(this);
}
