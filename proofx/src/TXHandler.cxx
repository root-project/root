// @(#)root/proofx:$Id$
// Author: G. Ganis Mar 2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TXHandler                                                            //
//                                                                      //
// Handler of asynchronous events for xproofd sockets.                  //
// Classes needing this should inherit from this and overload the       //
// relevant methods.                                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TError.h"
#include "TXHandler.h"

ClassImp(TXHandler)

//________________________________________________________________________
Bool_t TXHandler::HandleInput(const void *)
{
   // Handler of asynchronous input events

   AbstractMethod("HandleInput");
   return kTRUE;
}

//________________________________________________________________________
Bool_t TXHandler::HandleError(const void *)
{
   // Handler of asynchronous error events

   AbstractMethod("HandleError");
   return kTRUE;
}
