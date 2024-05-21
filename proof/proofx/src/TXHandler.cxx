// @(#)root/proofx:$Id$
// Author: G. Ganis Mar 2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TXHandler
\ingroup proofx

Handler of asynchronous events for XProofD sockets.
Classes needing this should inherit from this and overload the relevant methods.

*/

#include "TError.h"
#include "TXHandler.h"

ClassImp(TXHandler);

////////////////////////////////////////////////////////////////////////////////
/// Handler of asynchronous input events

Bool_t TXHandler::HandleInput(const void *)
{
   AbstractMethod("HandleInput");
   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Handler of asynchronous error events

Bool_t TXHandler::HandleError(const void *)
{
   AbstractMethod("HandleError");
   return kTRUE;
}
